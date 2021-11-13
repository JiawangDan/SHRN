# Scale adaptive and lightweight super-resolution with selective hierarchical residual network

### Abstract
Deep convolutional neural networks have made remarkable achievements in single-image super-resolution tasks in recent years. However, current methods do not consider the characteristics of super-resolution that the adjacent areas carry similar information. In this paper, we propose a scale adaptive and lightweight super-resolution with a selective hierarchical residual network (SHRN), which utilizes the repeated texture features. Specifically, SHRN is stacked by several selective hierarchical residual blocks (SHRB). The SHRB mainly contains a hierarchical feature fusion structure (HFFS) and a selective feature fusion structure (SFFS). The HFFS uses multiple branches to obtain multiscale features due to the varying texture size of objects. The SFFS fuses features of adjacent branches to select effective information. Plenty of experiments demonstrate that our lightweight model achieves better performance against other methods by extracting scale adaptive features and utilizing the repeated texture structure.

### Requirements
- Python 3
- PyTorch, torchvision
- Numpy, Scipy
- Pillow, Scikit-image
- h5py
- importlib

### Dataset
We use DIV2K dataset for training and Set5, Set14, B100 and Urban100 dataset for the benchmark test. Here are the following steps to prepare datasets.

1. Download [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K) and unzip on `dataset` directory as below:
  ```
  dataset
  └── DIV2K
      ├── DIV2K_train_HR
      ├── DIV2K_train_LR_bicubic
      ├── DIV2K_valid_HR
      └── DIV2K_valid_LR_bicubic
  ```
2. To accelerate training, we first convert training images to h5 format as follow (h5py module has to be installed).
```shell
$ cd datasets && python div2h5.py
```

### Test Pretrained Models
We provide the pretrained models in `checkpoint` directory. To test SHRN on benchmark dataset:
```shell
$ python test.py --model shrn \
                 --test_data_dir dataset/<dataset> \
                 --scale [2|3|4] \
                 --ckpt_path ./checkpoint/<path>.pth \
                 --sample_dir <sample_dir>
```

### Training Models
Here are our settings to train SHRN. If OOM error arise, please reduce batch size.
```shell
# For SHRN
$ python train.py --patch_size 64 \
                  --batch_size 64 \
                  --max_steps 600000 \
                  --decay 400000 \
                  --model shrn \
                  --ckpt_name shrn \
                  --ckpt_dir checkpoint/shrn \
                  --scale 0 \
                  --num_gpu 1
```
### Results
![image](https://github.com/JiawangDan/SHRN/blob/master/figs/results.PNG)

### Citation
```
@inproceedings{dan2021scale,
  title={Scale adaptive and lightweight super-resolution with a selective hierarchical residual network},
  author={Dan, Jiawang and Qu, Zhaowei and Wang, Xiaoru and Li, Fu and Gu, Jiahang and Ma, Bing},
  booktitle={2021 the 5th International Conference on Innovation in Artificial Intelligence},
  pages={8--14},
  year={2021}
}
```
