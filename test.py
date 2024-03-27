import os
import cv2
import numpy as np
import torch
from scipy import io as scio
from torch.utils.data import DataLoader
from dataset import MyTestDataSet
from utils import torch_psnr
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
    mask = scio.loadmat(mask_path + '/mask_3d_shift.mat')
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

    psnr_list = []
    preds = []
    truths = []
    scene_idx = 0
    save_meas = True
    for x, _ in testLoader:
        x,test_gt = x.cuda(),x.cuda()
        input_meas = dual_A(x)

        if save_meas:
            input_meas_np = input_meas.detach().cpu().numpy()[0, :, :]
            input_meas_np = (input_meas_np / np.max(input_meas_np) * 255).astype(np.uint0)
            name = os.path.join(result_path, "Scene{:0>2d}_measurement.png").format(scene_idx+1)
            print(f'Save reconstructed DCCHI meature as {name}.')
            cv2.imwrite(name, input_meas_np)
        scene_idx = scene_idx + 1

        with torch.no_grad():
            model_out = model(input_meas, physical_data)
        for k in range(test_gt.shape[0]):
            psnr_val = torch_psnr(model_out[k, :, :, :], test_gt[k, :, :, :])
            psnr_list.append(psnr_val.detach().cpu().numpy())
        pred = np.transpose(model_out.detach().cpu().numpy(), (0, 2, 3, 1)).astype(np.float32)
        truth = np.transpose(test_gt.cpu().numpy(), (0, 2, 3, 1)).astype(np.float32)

        preds.append(pred)
        truths.append(truth)
    psnr_mean = np.mean(np.asarray(psnr_list))
    print(np.asarray(psnr_list))
    print("psnr:",psnr_mean)

    preds = np.concatenate(preds, 0)
    gt = np.concatenate(truths, 0)
    name = result_path + 'Test_result.mat'
    print(f'Save reconstructed HSIs as {name}.')
    scio.savemat(name, {'gt': gt, 'pred': preds})



if __name__ == '__main__':
    test()