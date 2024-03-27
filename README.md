#### This is the demo code of our paper "In2SET" in submission to CVPR 2024.

This repo includes:  

- Specification of dependencies.
- Evaluation code.
- Pre-trained models.
- README file.

This repo can reproduce the main results in Table (1) of our main paper.
All the source code and pre-trained models will be released to the public for further research.


#### 1. Create Environment:

------

- Python 3 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux))

- [PyTorch >= 1.3](https://pytorch.org/)

- NVIDIA GPU + [CUDA](https://developer.nvidia.com/cuda-downloads)

- Python packages:

  ```shell
  pip install -r requirements.txt
  ```


#### 2. Prepare Dataset:


To use the TSA-Net dataset, please follow the steps below:

1. Download the Dataset:
   Download the dataset from [TSA-Net GitHub Repository](https://github.com/mengziyi64/TSA-Net).

2. Organize the Dataset:
   Place the downloaded dataset and camera response curve files into the 'code/data/' folder.

   The structure of the 'code/data/' folder should look like this:

   ```plaintext
   |--data
      |--mask.mat   
      |--mask_3d_shift.mat
      |--cameraSpectralResponse.mat
      |--Truth
          |--scene01.mat
          |--scene02.mat
          :
          |--scene10.mat
Note: The files 'cameraSpectralResponse.mat,' 'mask.mat,' and 'mask_3d_shift.mat' have already been included in this repository.

#### 3. Testing

3. 1 Test our pre-trained In2SET models on the HSI dataset. The results will be saved in 'code/evaluation/testing_result/' in the MatFile format.

```shell
python test.py --gpu_id=0 --weight_path=./ckpts/In2SET_2stg.pth

python test.py --gpu_id=0 --weight_path=./ckpts/In2SET_3stg.pth

python test.py --gpu_id=0 --weight_path=./ckpts/In2SET_5stg.pth

python test.py --gpu_id=0 --weight_path=./ckpts/In2SET_9stg.pth
```

3. 2 Test inference time
```shell
python test_fps.py --gpu_id=0 --weight_path=./ckpts/In2SET_2stg.pth

python test_fps.py --gpu_id=0 --weight_path=./ckpts/In2SET_3stg.pth

python test_fps.py --gpu_id=0 --weight_path=./ckpts/In2SET_5stg.pth

python test_fps.py --gpu_id=0 --weight_path=./ckpts/In2SET_9stg.pth
```
Note: Due to size limitations for direct uploads on GitHub, our 9stg model is provided in three compressed parts: ckpts/In2SET_9stg.zip.001, ckpts/In2SET_9stg.zip.002, ckpts/In2SET_9stg.zip.003. Please use joint extraction for decompression.
#### 4. This repo is mainly based on MST and rTVRA.  In our experiments, we use the following repos:
(1) MST: https://github.com/caiyuanhao1998/MST

(2) rTVRA: https://github.com/zspCoder/rTVRA-Release.git


We extend our sincere appreciation and gratitude for the valuable contributions made by these repositories.
