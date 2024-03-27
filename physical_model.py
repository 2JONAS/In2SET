''' Utilities '''
import math
import numpy as np
import torch
import time
from physical_shift import get_shift
import scipy.io as scio
shift_np, shift_back_np, shift_torch, shift_back_torch = get_shift()


def torch2np(x:torch.Tensor):
    x = x.cpu().detach().numpy()
    if len(x.shape) == 3:
        x = np.transpose(x,(1,2,0))
    if len(x.shape) == 4:
        x = np.transpose(x, (0,2, 3, 1))
    return x
def numpy2torch(x:np.ndarray):
    if len(x.shape) == 2:
        x = torch.tensor(x)
        return torch.tensor(x).unsqueeze(0).unsqueeze(0)
    if len(x.shape) == 3:
        x = torch.tensor(x)
        x = np.transpose(x, (2,0,1))
        return torch.tensor(x).unsqueeze(0)
    if len(x.shape) == 4:
        x = np.transpose(x, (0, 3, 1, 2))
        x = torch.tensor(x)
        return x
    else:
        return torch.tensor(x)

preload_mat = None
def load_cameraSpectralResponce():
    global preload_mat
    if preload_mat is not None:
        return  preload_mat
    else:
        mat = scio.loadmat("./data/cameraSpectralResponse.mat")
        preload_mat = mat
        return mat


def A_DualCamera_torch(x, Phi, camera_response_curve, shift_func, shift_back_func):
    """
    :param x:
    :param Phi: shift mask 3d 
    :param camera_response_curve: [1,c,1,1]
    :param shift_func: shift operator
    :param shift_back_func: shift back operator
    :return:
    """
    x1 = shift_func(x)
    y1 = torch.sum(x1 * Phi, dim=1)  # cassi_measurement
    x_response = x * camera_response_curve
    y2 = torch.sum(x_response, dim=1)  # pan_measurement
    if y1.shape[2] == y2.shape[2]:
        y = torch.cat([y1, y2], dim=1)  # concat on vertical
    else:
        y = torch.cat([y1, y2], dim=2)  # concat on horizontal
    return y
def At_DualCamera_torch(y,Phi,camera_respoce_curve,split_y_row,shift_func,shift_back_func):
    """
    :param y: [y1,y2] 
    :param Phi: shift mask 3d
    :param camera_respoce_curve: [1,c,1,1]
    :param split_y_row: split of y
    :param shift_func:shift operator
    :return:
    """
    y = torch.unsqueeze(y, 1).repeat(1, Phi.shape[1], 1, 1)

    temp1,temp2 = y[:,:,:,:split_y_row],y[:,:,:,split_y_row:]
    x1 = temp1 * Phi
    x1 = shift_back_func(x1)
    x2 = temp2 * camera_respoce_curve
    return x1+x2

def get_AAt_dual(opt):
    """
    return forward_model,backward_model,shift_operator,shift_back_operator for DCCHI
    """
    shift_back_torch_variable = torch.zeros(opt.batch_size, 28, 256, 256).cuda().to(torch.float32)
    shift_torch_variable = torch.zeros(opt.batch_size, 28, 256, 310).cuda().to(torch.float32)
    shift_torch_index = torch.ones_like(shift_back_torch_variable)
    shift_torch_index = shift_torch(shift_torch_index, step=2)
    shift_torch_index = torch.where(shift_torch_index == 1)
    mat = load_cameraSpectralResponce()
    mask  = mat["Mask"]
    cameraSpectralResponse = mat["cameraSpectralResponse"].reshape(-1)
    cameraSpectralResponse_cuda = torch.from_numpy(cameraSpectralResponse).reshape(1, -1, 1, 1).cuda().to(torch.float32)
    coded_aperture_shift = numpy2torch(mask)
    coded_aperture_shift_cuda = coded_aperture_shift.cuda().to(torch.float32)
    step = 2
    split = 310
    _shift,_shift_back = lambda x: shift_torch(x, step=step, index_tensor=shift_torch_index,output=shift_torch_variable), \
                lambda x: shift_back_torch(x, step=step,index_tensor=shift_torch_index, output=shift_back_torch_variable)

    A = lambda x: A_DualCamera_torch(x, coded_aperture_shift_cuda, cameraSpectralResponse_cuda, _shift, _shift_back)
    At = lambda x: At_DualCamera_torch(x, coded_aperture_shift_cuda, cameraSpectralResponse_cuda, split, _shift, _shift_back)
    return A,At,_shift,_shift_back

def get_cameraSpectralResponse_cuda():
    mat = load_cameraSpectralResponce()
    cameraSpectralResponse = mat["cameraSpectralResponse"].reshape(-1)
    cameraSpectralResponse_cuda = torch.from_numpy(cameraSpectralResponse).reshape(1, -1, 1, 1).cuda().to(torch.float32)
    return cameraSpectralResponse_cuda


