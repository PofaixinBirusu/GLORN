# GLORN: Strong Generalization Fully Convolutional Network for Low-Overlap Point Cloud Registration
The paper has not yet been accepted, and the rest of the code is still being sorted out. In the future, we will publish the official version [here](https://github.com/Pikachu-NCU/GLORN)
## Installation
Please use the following command for installation.
```
# It is recommended to create a new environment
conda create -n glorn python==3.8
conda activate glorn

# [Optional] If you are using CUDA 11.0 or newer, please install `torch==1.7.1+cu110`
pip install torch==1.7.1+cu110 -f https://download.pytorch.org/whl/torch_stable.html

# Install dependencies
python setup.py build develop
```
Code has been tested with Ubuntu 20.04, GCC 9.3.0, Python 3.8, PyTorch 1.7.1, CUDA 11.1 and cuDNN 8.1.0.
## Pre-training parameters
We provide pre-training parameters [here](https://pan.baidu.com/s/1JXFv56DhREBbGq5lVUkD8A), and the extraction code is "om1d".
There are two files, "GLORN-3DMatch.pth.tar" and "GLORN-KITTI.pth.tar", please put them in the ```params``` folder.
```
--params--GLORN-3DMatch.pth.tar
       |--GLORN-KITTI.pth.tar
```
## 3DMatch
### Data preparation
We provide the [download address](https://pan.baidu.com/s/1A80Ti9y6Hh70bUOf7bbs3A) (extraction code is "2zt1") of the processed 3DMatch dataset. After downloading, you can save it to the path you want to save.
```
--3DMatch--metadata
        |--data--train--7-scenes-chess--cloud_bin_0.pth
              |      |               |--...
              |      |--...
              |--test--7-scenes-redkitchen--cloud_bin_0.pth
                    |                    |--...
                    |--...
```
The processed 3DMatch dataset above is used in our program, but if you want to download the raw dense 3DMatch dataset, please run:
```
wget --no-check-certificate --show-progress https://share.phys.ethz.ch/~gsg/pairwise_reg/3dmatch.zip
unzip 3dmatch.zip
```
### Testing
1. Please modify the 36th line of code in ```config/config_3dmatch.py```, and modify the dataset path to the path you saved
    ```
    _config.data.dataset_root = "[your path]/3DMatch"
    ```
2. Please run ```evaluate_3dmatch.py```. You can get the evaluation result on 3DMatch or 3DLoMatch by modifying ```main("3DMatch")``` and ```main("3DLoMatch")```.
    ```
    if __name__ == '__main__':
        # 3DMatch or 3DLoMatch
        # main("3DMatch")
        main("3DLoMatch")
    ```
3. If you want to evaluate the results for different corresponding number, please run ```evaluate_3dmatch_with_corr_num.py``` after running ```evaluate_3dmatch.py```. The first argument is the name of the benchmark, and the second argument is the corresponding number.
    ```
    if __name__ == '__main__':
        # # 3DMatch
        # main("3DMatch", 5000)
        # main("3DMatch", 2500)
        # main("3DMatch", 1000)
        # main("3DMatch", 500)
        # main("3DMatch", 250)
        
        # 3DLoMatch
        # main("3DLoMatch", 5000)
        main("3DLoMatch", 2500)
        # main("3DLoMatch", 1000)
        # main("3DLoMatch", 500)
        # main("3DLoMatch", 250)
    ```
I have to say that our test code is still very non-standard, and we will gradually make it standard and concise.
### Training
Please run ```train_3dmatch.py```.
## KITTI
### Data preparation
Please download the metadata [here](https://pan.baidu.com/s/1jFvDn3LciCdrPNKa2YIgWg) (extraction code is "8f0w") first, and then download the original data from the [Kitti official website](http://www.cvlibs.net/datasets/kitti/eval_odometry.php) into KITTI, and run ```KITTI/downsample_pcd.py``` to generate the data. The data should be organized as follows:
```
--KITTI--metadata
      |--sequences--00--velodyne--000000.bin
      |              |         |--...
      |              |...
      |--downsampled--00--000000.npy
                   |   |--...
                   |--...
```
### Testing
1. Please modify the 34th line of code in ```config/config_kitti.py```, and modify the dataset path to the path you saved
    ```
    _config.data.dataset_root = "[your path]/KITTI"
    ```
2. Please run ```evaluate_kitti.py```.
### Training
Please run ```train_kitti.py```.
## The comparison methods in the paper
We express our sincere thanks to the authors of these methods
* [PerfectMatch](https://github.com/zgojcic/3DSmoothNet)
* [FCGF](https://github.com/chrischoy/FCGF)
* [D3Feat](https://github.com/XuyangBai/D3Feat.pytorch)
* [PREDATOR](https://github.com/ShengyuH/OverlapPredator)
* [SpinNet](https://github.com/QingyongHu/SpinNet)
* [CoFiNet](https://github.com/haoyu94/Coarse-to-fine-correspondences)
* [REGTR](https://github.com/yewzijian/RegTR)
* [GeoTransformer](https://github.com/qinzheng93/GeoTransformer)
