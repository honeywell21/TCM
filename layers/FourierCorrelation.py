# coding=utf-8
# author=maziqing
# email=maziqing.mzq@alibaba-inc.com

import numpy as np
import torch
import torch.nn as nn


def get_frequency_modes(seq_len, modes=64, mode_select_method='random'):
    """
    get modes on frequency domain:
    'random' means sampling randomly;
    'else' means sampling the lowest modes;
    """
    modes = min(modes, seq_len//2)
    if mode_select_method == 'random':
        index = list(range(0, seq_len // 2))
        np.random.shuffle(index)
        index = index[:modes]
    else:
        index = list(range(0, modes))
    index.sort()
    return index


# ########## fourier layer #############
# class FourierBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, seq_len, modes=64, mode_select_method='random'):
#         super(FourierBlock, self).__init__()
#         print('fourier enhanced block used!')
#         """
#         1D Fourier block. It performs representation learning on frequency domain,
#         it does FFT, linear transform, and Inverse FFT.
#         """
#         # get modes on frequency domain
#         self.index = get_frequency_modes(seq_len, modes=modes, mode_select_method=mode_select_method)
#         print('modes={}, index={}'.format(modes, self.index))
#
#         self.scale = (1 / (in_channels * out_channels))
#         self.weights1 = nn.Parameter(
#             self.scale * torch.rand(8, in_channels // 8, out_channels // 8, len(self.index), dtype=torch.cfloat))
#
#     # Complex multiplication
#     def compl_mul1d(self, input, weights):
#         # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
#         return torch.einsum("bhi,hio->bho", input, weights)
#
#     def forward(self, x):
#         # size = [B, L, H, E]
#         B, L, C = x.shape
#         #print("B, L, H, E",B, L, H, E)
#         x = x.permute(0, 2, 1)
#        # print("x.shape",x.shape)
#         # Compute Fourier coefficients
#         x_ft = torch.fft.rfft(x, dim=-1)
#         #print("x_ft.shape", x_ft.shape)
#         # Perform Fourier neural operations
#         out_ft = torch.zeros(B, C, L // 2 + 1, device=x.device, dtype=torch.cfloat)
#         #print("out_ft.shape",out_ft.shape)
#         for wi, i in enumerate(self.index):
#             out_ft[:, :, wi] = self.compl_mul1d(x_ft[:, :, i], self.weights1[:, :, wi])
#         # Return to time domain
#         x = torch.fft.irfft(out_ft, n=x.size(-1))
#         #print("out.shape", x.shape)
#         return (x, None)

# ########## fourier layer #############
class FourierBlock(nn.Module):
    def __init__(self, in_channels, out_channels, seq_len, modes=0, mode_select_method='random'):
        super(FourierBlock, self).__init__()
        print('fourier enhanced block used!')
        """
        1D Fourier block. It performs representation learning on frequency domain, 
        it does FFT, linear transform, and Inverse FFT.    
        """
        # get modes on frequency domain
        self.index = get_frequency_modes(seq_len, modes=modes, mode_select_method=mode_select_method)
        print('modes={}, index={}'.format(modes, self.index))

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(8, in_channels // 8, out_channels // 8, len(self.index), dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul1d(self, input, weights):
        # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
        return torch.einsum("bhi,hio->bho", input, weights)

    def forward(self, q):
        # size = [B, L, H, E]
        B, L, H, E = q.shape
        x = q.permute(0, 2, 3, 1)
        # Compute Fourier coefficients
        x_ft = torch.fft.rfft(x, dim=-1)
        # Perform Fourier neural operations
        out_ft = torch.zeros(B, H, E, L // 2 + 1, device=x.device, dtype=torch.cfloat)
        for wi, i in enumerate(self.index):
            out_ft[:, :, :, wi] = self.compl_mul1d(x_ft[:, :, :, i], self.weights1[:, :, :, wi])
        # Return to time domain
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        return (x, None)





