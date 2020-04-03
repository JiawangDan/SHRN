import os
import time
import random
import numpy as np
import scipy.misc as misc
import skimage.measure as measure
from tensorboardX import SummaryWriter
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import TrainDataset, TestDataset

class Solver():
    def __init__(self, model, cfg, log_file):

        self.log_file = log_file

        if cfg.scale > 0:
            self.refiner = model(scale=cfg.scale, 
                                 group=cfg.group)
        else:
            self.refiner = model(multi_scale=True, 
                                 group=cfg.group)
        
        if cfg.loss_fn in ["MSE"]: 
            self.loss_fn = nn.MSELoss()
        elif cfg.loss_fn in ["L1"]: 
            self.loss_fn = nn.L1Loss()
        elif cfg.loss_fn in ["SmoothL1"]:
            self.loss_fn = nn.SmoothL1Loss()

        self.optim = optim.Adam(
            filter(lambda p: p.requires_grad, self.refiner.parameters()), 
            cfg.lr)
        
        self.train_data = TrainDataset(cfg.train_data_path, 
                                       scale=cfg.scale, 
                                       size=cfg.patch_size)
        self.train_loader = DataLoader(self.train_data,
                                       batch_size=cfg.batch_size,
                                       num_workers=6,
                                       shuffle=True, drop_last=True)
        
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.refiner = self.refiner.to(self.device)
        self.loss_fn = self.loss_fn

        self.cfg = cfg
        self.step = 0

        os.makedirs(os.path.join(cfg.ckpt_dir,"runs"), exist_ok=True)
        self.writer = SummaryWriter(logdir=os.path.join(cfg.ckpt_dir,"runs"))
        if cfg.verbose:
            num_params = 0
            for param in self.refiner.parameters():
                num_params += param.nelement()
            print("# of params:", num_params)
            self.log_file.write('# of params:' + str(num_params) + '\n')

        os.makedirs(cfg.ckpt_dir, exist_ok=True)

    def fit(self):
        cfg = self.cfg
        refiner = nn.DataParallel(self.refiner, 
                                  device_ids=range(cfg.num_gpu))

        t_start = time.time()
        learning_rate = cfg.lr
        time_sum = 0
        psnr_best_value = [0, 0, 0]
        psnr_best_step = [0, 0, 0]
        while True:
            for inputs in self.train_loader:
                t1 = time.time()

                self.refiner.train()

                if cfg.scale > 0:
                    scale = cfg.scale
                    hr, lr = inputs[-1][0], inputs[-1][1]
                else:
                    # only use one of multi-scale data
                    # i know this is stupid but just temporary
                    scale = random.randint(2, 4)
                    hr, lr = inputs[scale-2][0], inputs[scale-2][1]
                
                hr = hr.to(self.device)
                lr = lr.to(self.device)
                
                sr = refiner(lr, scale)
                loss = self.loss_fn(sr, hr)
                
                self.optim.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.refiner.parameters(), cfg.clip)
                self.optim.step()

                learning_rate = self.decay_learning_rate()
                for param_group in self.optim.param_groups:
                    param_group["lr"] = learning_rate

                t2 = time.time()
                time_sum += t2-t1
                if (self.step + 1) % (cfg.print_interval/10) == 0:
                    print("step:{} loss:{:.6f} cost:{:.3f}s".format(self.step+1, loss.data.cpu(), time_sum))
                    self.log_file.write("step:{} loss:{:.6f} cost:{:.3f}s".format(self.step+1, loss.data.cpu(), time_sum) + '\n')
                    time_sum = 0

                self.step += 1
                if cfg.verbose and self.step % cfg.print_interval == 0:
                    if cfg.scale > 0:
                        psnr = self.evaluate("dataset/Urban100", scale=cfg.scale, num_step=self.step)
                        self.writer.add_scalar("Urban100", psnr, self.step)
                        if psnr > psnr_best_value[0]:
                            psnr_best_value[0] = psnr
                            psnr_best_step[0] = self.step
                        print("scale={} step={} best_psnr={:.6f} ".format(scale, psnr_best_step[0], psnr_best_value[0]))
                        self.log_file.write("scale={} step={} best_psnr={:.6f} ".format(scale, psnr_best_step[0], psnr_best_value[0]) + '\n')
                    else:    
                        psnr = [self.evaluate("dataset/Urban100", scale=i, num_step=self.step) for i in range(2, 5)]
                        self.writer.add_scalar("Urban100_2x", psnr[0], self.step)
                        self.writer.add_scalar("Urban100_3x", psnr[1], self.step)
                        self.writer.add_scalar("Urban100_4x", psnr[2], self.step)
                        for i in range(2, 5):
                            if psnr[i-2] > psnr_best_value[i-2]:
                                psnr_best_value[i-2] = psnr[i-2]
                                psnr_best_step[i-2] = self.step
                            print("scale={} best_step={} best_psnr={:.6f} ".format(i, psnr_best_step[i-2], psnr_best_value[i-2]))
                            self.log_file.write("scale={} best_step={} best_psnr={:.6f} ".format(i, psnr_best_step[i-2], psnr_best_value[i-2]) + '\n')
                            
                    self.save(cfg.ckpt_dir, cfg.ckpt_name)

            if self.step > cfg.max_steps: break

        t_end = time.time()
        print("train total time:{:.3f}s".format(t_end-t_start))
        self.log_file.write("train total time:{:.3f}s".format(t_end-t_start) + '\n')

    def evaluate(self, test_data_dir, scale=2, num_step=0):
        cfg = self.cfg
        mean_psnr = 0
        self.refiner.eval()
        
        test_data   = TestDataset(test_data_dir, scale=scale)
        test_loader = DataLoader(test_data,
                                 batch_size=1,
                                 num_workers=6,
                                 shuffle=False)

        for step, inputs in enumerate(test_loader):
            hr = inputs[0].squeeze(0)
            lr = inputs[1].squeeze(0)
            name = inputs[2][0]

            h, w = lr.size()[1:]
            h_half, w_half = int(h/2), int(w/2)
            h_chop, w_chop = h_half + cfg.shave, w_half + cfg.shave

            # split large image to 4 patch to avoid OOM error
            lr_patch = torch.FloatTensor(4, 3, h_chop, w_chop)
            lr_patch[0].copy_(lr[:, 0:h_chop, 0:w_chop])
            lr_patch[1].copy_(lr[:, 0:h_chop, w-w_chop:w])
            lr_patch[2].copy_(lr[:, h-h_chop:h, 0:w_chop])
            lr_patch[3].copy_(lr[:, h-h_chop:h, w-w_chop:w])
            lr_patch = lr_patch.to(self.device)
            
            # run refine process in here!
            # sr = self.refiner(lr_patch, scale).data
            sr = torch.FloatTensor(4, 3, h_chop*scale, w_chop*scale)
            sr0 = self.refiner(lr_patch[0].unsqueeze_(dim=0), scale).data
            sr1 = self.refiner(lr_patch[1].unsqueeze_(dim=0), scale).data
            sr2 = self.refiner(lr_patch[2].unsqueeze_(dim=0), scale).data
            sr3 = self.refiner(lr_patch[3].unsqueeze_(dim=0), scale).data
            sr[0].copy_(sr0[0])
            sr[1].copy_(sr1[0])
            sr[2].copy_(sr2[0])
            sr[3].copy_(sr3[0])
    
            h, h_half, h_chop = h*scale, h_half*scale, h_chop*scale
            w, w_half, w_chop = w*scale, w_half*scale, w_chop*scale
            
            # merge splited patch images
            result = torch.FloatTensor(3, h, w).to(self.device)
            result[:, 0:h_half, 0:w_half].copy_(sr[0, :, 0:h_half, 0:w_half])
            result[:, 0:h_half, w_half:w].copy_(sr[1, :, 0:h_half, w_chop-w+w_half:w_chop])
            result[:, h_half:h, 0:w_half].copy_(sr[2, :, h_chop-h+h_half:h_chop, 0:w_half])
            result[:, h_half:h, w_half:w].copy_(sr[3, :, h_chop-h+h_half:h_chop, w_chop-w+w_half:w_chop])
            sr = result

            hr = hr.cpu().mul(255).clamp(0, 255).byte().permute(1, 2, 0).numpy()
            sr = sr.cpu().mul(255).clamp(0, 255).byte().permute(1, 2, 0).numpy()
            
            # evaluate PSNR
            # this evaluation is different to MATLAB version
            # we evaluate PSNR in RGB channel not Y in YCbCR  
            bnd = scale
            im1 = hr[bnd:-bnd, bnd:-bnd]
            im2 = sr[bnd:-bnd, bnd:-bnd]
            mean_psnr += psnr(im1, im2) / len(test_data)

        print("test_data_dir:{} scale={} num_step={} mean_psnr={:.6f}".format(test_data_dir, scale, num_step, mean_psnr))
        self.log_file.write("test_data_dir:{} scale={} num_step={} mean_psnr={:.6f}".format(test_data_dir, scale, num_step, mean_psnr) + '\n')

        return mean_psnr

    def load(self, path):
        self.refiner.load_state_dict(torch.load(path))
        splited = path.split(".")[0].split("_")[-1]
        try:
            self.step = int(path.split(".")[0].split("_")[-1])
        except ValueError:
            self.step = 0
        print("Load pretrained {} model".format(path))

    def save(self, ckpt_dir, ckpt_name):
        save_path = os.path.join(
            ckpt_dir, "{}_{}.pth".format(ckpt_name, self.step))
        torch.save(self.refiner.state_dict(), save_path)

    def decay_learning_rate(self):
        lr = self.cfg.lr * (0.5 ** (self.step // self.cfg.decay))
        return lr


def psnr(im1, im2):
    def im2double(im):
        min_val, max_val = 0, 255
        out = (im.astype(np.float64)-min_val) / (max_val-min_val)
        return out
        
    im1 = im2double(im1)
    im2 = im2double(im2)
    psnr = measure.compare_psnr(im1, im2, data_range=1)
    return psnr
