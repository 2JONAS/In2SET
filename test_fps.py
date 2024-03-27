import os
import time

import cv2
import numpy as np
import torch
from scipy import io as scio
from torch.utils.data import DataLoader
from dataset import MyTestDataSet
from utils import torch_psnr, torch_ssim
from net import PGU
from physical_model import get_cameraSpectralResponse_cuda, get_AAt_dual


import argparse
def parse_args():
    parser = argparse.ArgumentParser(description="Your script description here.")

    parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--test_path", type=str, default="./data/Truth/", help="Path to test data")
    parser.add_argument("--weight_path", type=str, default="./ckpts/In2SET_2stg.pth", help="Path to the pre-trained model")
    parser.add_argument("--mask_path", type=str, default="./data/", help="Path to mask data")
    parser.add_argument("--result_path", type=str, default="./testing_result/", help="Path to save testing results")

    args = parser.parse_args()
    return args
args = parse_args()
print("-"*5+"Parameter settings"+"-"*5)
print(f"Using GPU ID: {args.gpu_id}")
print(f"Batch size: {args.batch_size}")
print(f"Test path: {args.test_path}")
print(f"Weight path: {args.weight_path}")
print(f"Mask path: {args.mask_path}")
print(f"Result path: {args.result_path}")
print("-"*15+"\n")

gpu_id = args.gpu_id
batch_size = args.batch_size
test_path = args.test_path
weight_path = args.weight_path
mask_path = args.mask_path
result_path = args.result_path
if not os.path.exists(result_path):
    os.makedirs(result_path)

os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

if not torch.cuda.is_available():
    raise Exception('NO GPU!')


stage_num = int(os.path.basename(weight_path).split("_")[1][0])

def generate_shift_masks(mask_path, batch_size):
    mask = scio.loadmat(mask_path + 'mask_3d_shift.mat')
    mask_3d_shift = mask['mask_3d_shift']
    mask_3d_shift = np.transpose(mask_3d_shift, [2, 0, 1])
    mask_3d_shift = torch.from_numpy(mask_3d_shift)
    [nC, H, W] = mask_3d_shift.shape
    Phi_batch = mask_3d_shift.expand([batch_size, nC, H, W]).cuda().float()
    return Phi_batch

# define operator
class Option:
    batch_size=batch_size
opt = Option()
dual_A,dual_At,shift,shift_back = get_AAt_dual(opt)
cameraSpectralResponse_cuda = get_cameraSpectralResponse_cuda()
Phi_batch  = generate_shift_masks(mask_path, opt.batch_size)
physical_operator = (dual_A,dual_At,shift,shift_back)
physical_data = (Phi_batch, cameraSpectralResponse_cuda)



def test():


    testSet = MyTestDataSet(test_path, transform=None)
    testLoader = DataLoader(testSet, batch_size=batch_size,shuffle=False, num_workers=0,drop_last=True)

    model = PGU(num_iterations=stage_num,physical_operator=physical_operator).cuda()
    model.load_state_dict(torch.load(weight_path))
    model.eval()



    print("Machine warm up")
    measurements = []
    gts = []
    for x, _ in testLoader:
        x,test_gt = x.cuda(),x.cuda()
        input_meas = dual_A(x)
        gts.append(test_gt)
        measurements.append(input_meas)
        with torch.no_grad():
            model_out = model(input_meas, physical_data)
    print("-----Start of inference time testing-----")
    tic = time.time()
    with torch.no_grad():
        for i,input_meas in enumerate(measurements):
            model_out = model(input_meas, physical_data)
    toc = time.time()
    print(f"-----End of inference time testing-----")
    total_time = toc - tic
    print(f"The time taken is {total_time} seconds for 10 scenes. The frames per second (FPS) rate is {1/(total_time/10)}")


if __name__ == '__main__':
    test()