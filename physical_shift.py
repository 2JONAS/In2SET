import numpy as np
import torch

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
class ShiftDirecttion:
    UP_DOWN = 0   
    LEFT_RIGHT = 1 

def get_shift(default_direction:ShiftDirecttion=ShiftDirecttion.LEFT_RIGHT):
# default_direction = ShiftDirecttion.LEFT_RIGHT

    if default_direction == ShiftDirecttion.UP_DOWN:
        def shift(inputs, step):
            row, col, L = inputs.shape
            new_row = row + abs(step) * (L - 1)
            output = np.zeros((new_row, col , L))
            for i in range(L):
                if step < 0:
                    output[new_row-i*abs(step)-row:new_row-i*abs(step),:, i] = inputs[:, :, i]
                else:
                    output[new_row - i * abs(step) - row:new_row - i * abs(step), :, L-i-1] = inputs[:, :, L-i-1]
            return output


        def shift_back(inputs, step):
            row, col, L = inputs.shape
            new_row = row - abs(step) * (L - 1)
            output = np.zeros((new_row, col, L))
            for i in range(L):
                if step < 0:
                    output[:, :, i] = inputs[row-i*abs(step)-new_row:row-i*abs(step),:, i]
                else:
                    output[:, :, L-i-1] = inputs[row - i * abs(step) - new_row:row - i * abs(step), :, L-i-1]
            return output
        def shift_torch(inputs, step=1,index_tensor=None,output=None):
            """
            :param inputs: b,c,h,w x_cube
            :param step: 1 down，-1 up
            :param index_tensor: shift location index
            :param output: output variable handle
            :return: output
            """
            [bs, nC, row, col] = inputs.shape
            new_row = row + abs(step) * (nC - 1)
            if output is None:
                output = torch.zeros(bs, nC, new_row,col).cuda()
            else:
                output = output * 0
            if index_tensor is None:
                for i in [j for j in range(nC)]:
                    if step < 0:
                        output[:, i, new_row-i*abs(step)-row:new_row-i*abs(step), :] = inputs[:, i, :, :]
                    else:
                        output[:, nC-i-1, new_row-i*abs(step)-row:new_row-i*abs(step), :] = inputs[:, nC-i-1, :, :]

            else:
                output[index_tensor] = inputs.reshape(-1)
            return output

        def shift_back_torch(inputs, step=1,index_tensor=None,output=None):
            """
            :param inputs: b,c,h,w x_cube
            :param step: 1 down，-1 up
            :param index_tensor: shift back location index
            :param output: output variable handle
            :return: output
            """
            [bs, nC, row, col] = inputs.shape
            new_row = row - abs(step) * (nC - 1)
            if output is None:
                output = torch.zeros(bs, nC, new_row, col).cuda()
            else:
                output = output * 0
            if index_tensor is None:
                for i in range(nC):
                    if step < 0:
                        output[:, i, :, :] = inputs[:, i, row-i*abs(step)-new_row:row-i*abs(step), :]
                    else:
                        output[:, nC-i-1, :, :] = inputs[:, nC-i-1, row - i * abs(step) - new_row:row - i * abs(step), :]
            else:
                return inputs[index_tensor].reshape(output.shape)
                output[:,:,:,:] = inputs[index_tensor].reshape(output.shape)
            return output
    else:
        #LEFT RIGHT
        def shift(inputs, step):
            row, col, L = inputs.shape
            new_col = col + abs(step) * (L - 1)
            output = np.zeros((row, new_col, L))

            for i in range(L):
                if step < 0:
                    output[:, new_col - i * abs(step) - col:new_col - i * abs(step), i] = inputs[:, :, i]
                else:
                    output[:, new_col - i * abs(step) - col:new_col - i * abs(step), L - i - 1] = inputs[:, :, L - i - 1]

            return output
        def shift_back(inputs, step):
            row, col, L = inputs.shape
            new_col = col - abs(step) * (L - 1)
            output = np.zeros((row, new_col, L))
            for i in range(L):
                if step < 0:
                    output[:, :, i] = inputs[:, col - i * abs(step) - new_col:col - i * abs(step), i]
                else:
                    output[:, :, L - i - 1] = inputs[:, col - i * abs(step) - new_col:col - i * abs(step), L - i - 1]
            return output
        def shift_torch(inputs, step=1,index_tensor=None,output=None):
            """
            :param inputs: b,c,h,w x_cube
            :param step: 1 right,-1 left
            :param index_tensor: shift location index
            :param output: output variable handle
            :return: output
            """
            [bs, nC, row, col] = inputs.shape
            new_col= col + abs(step) * (nC - 1)
            if output is None:
                output = torch.zeros(bs, nC, row,new_col).cuda()
            else:
                output = output * 0
            if index_tensor is None:
                for i in [j for j in range(nC)]:
                    if step < 0:
                        output[:, i, : ,new_col-i*abs(step)-col:new_col-i*abs(step)] = inputs[:, i, :, :]
                    else:
                        output[:, nC-i-1,:, new_col-i*abs(step)-col:new_col-i*abs(step)] = inputs[:, nC-i-1, :, :]
            else:
                output[index_tensor] = inputs.reshape(-1)
            return output


        def shift_back_torch(inputs, step=1, index_tensor=None, output=None):
            """
            :param inputs: b,c,h,w x_cube
            :param step: 1 right,-1 left
            :param index_tensor: shift back location index
            :param output: output variable handle
            :return: output
            """
            [bs, nC, row, col] = inputs.shape
            # print("debug",bs, nC, row, col)
            new_col = col - abs(step) * (nC - 1)
            if output is None:
                output = torch.zeros(bs, nC, row, new_col).cuda()
            else:
                output = output * 0
            # print("debug output",output.shape)
            # print("debug index", index_tensor[0].shape)
            if index_tensor is None:
                for i in range(nC):
                    if step < 0:
                        output[:, i, :, :] = inputs[:, i, : ,col-i*abs(step)-new_col:col-i*abs(step)]
                    else:
                        output[:, nC - i - 1, :, :] = inputs[:, nC-i-1,:, col-i*abs(step)-new_col:col-i*abs(step)]
            else:
                return inputs[index_tensor].reshape(output.shape)
            return output
    return shift,shift_back,shift_torch,shift_back_torch