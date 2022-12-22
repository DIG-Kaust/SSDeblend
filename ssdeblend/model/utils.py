##################################################################
# 2022 - King Abdullah University of Science and Technology (KAUST)
#
# Authors: Nick Luiken, Matteo Ravasi
# Description: Network building blocks to be used in SS networks
##################################################################

import copy
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from torch import Tensor
from typing import Tuple
from .datautils import DataFormat, DataDim, DATA_FORMAT_DIM_INDEX


class Crop2d(nn.Module):
    """Crop input using slicing. Assumes BCHW data.

    Args:
        crop (Tuple[int, int, int, int]): Amounts to crop from each side of the image.
            Tuple is treated as [left, right, top, bottom]/
    """

    def __init__(self, crop: Tuple[int, int, int, int]):
        super().__init__()
        self.crop = crop
        assert len(crop) == 4

    def forward(self, x: Tensor) -> Tensor:
        (left, right, top, bottom) = self.crop
        x0, x1 = left, x.shape[-1] - right
        y0, y1 = top, x.shape[-2] - bottom
        return x[:, :, y0:y1, x0:x1]


class Shift2d(nn.Module):
    """Shift an image in either or both of the vertical and horizontal axis by first
    zero padding on the opposite side that the image is shifting towards before
    cropping the side being shifted towards.

    Args:
        shift (Tuple[int, int]): Tuple of vertical and horizontal shift. Positive values
            shift towards right and bottom, negative values shift towards left and top.
    """

    def __init__(self, shift: Tuple[int, int]):
        super().__init__()
        self.shift = shift
        vert, horz = self.shift
        y_a, y_b = abs(vert), 0
        x_a, x_b = abs(horz), 0
        if vert < 0:
            y_a, y_b = y_b, y_a
        if horz < 0:
            x_a, x_b = x_b, x_a
        # Order : Left, Right, Top Bottom
        self.pad = nn.ZeroPad2d((x_a, x_b, y_a, y_b))
        self.crop = Crop2d((x_b, x_a, y_b, y_a))
        self.shift_block = nn.Sequential(self.pad, self.crop)

    def forward(self, x: Tensor) -> Tensor:
        return self.shift_block(x)


def rotate(
    x: torch.Tensor, angle: int, data_format: str = DataFormat.BCHW
) -> torch.Tensor:
    """Rotate images by 90 degrees clockwise. Can handle any 2D data format.
    Args:
        x (Tensor): Image or batch of images.
        angle (int): Clockwise rotation angle in multiples of 90.
        data_format (str, optional): Format of input image data, e.g. BCHW,
            HWC. Defaults to BCHW.
    Returns:
        Tensor: Copy of tensor with rotation applied.
    """
    dims = DATA_FORMAT_DIM_INDEX[data_format]
    h_dim = dims[DataDim.HEIGHT]
    w_dim = dims[DataDim.WIDTH]

    if angle == 0:
        return x
    elif angle == 90:
        return x.flip(w_dim).transpose(h_dim, w_dim)
    elif angle == 180:
        return x.flip(w_dim).flip(h_dim)
    elif angle == 270:
        return x.flip(h_dim).transpose(h_dim, w_dim)
    else:
        raise NotImplementedError("Must be rotation divisible by 90 degrees")


def flip(
    x: torch.Tensor, angle: int, data_format: str = DataFormat.BCHW
) -> torch.Tensor:
    """Rotate images by 90 degrees clockwise. Can handle any 2D data format.
    Args:
        x (Tensor): Image or batch of images.
        angle (int): Clockwise rotation angle in multiples of 90.
        data_format (str, optional): Format of input image data, e.g. BCHW,
            HWC. Defaults to BCHW.
    Returns:
        Tensor: Copy of tensor with rotation applied.
    """
    dims = DATA_FORMAT_DIM_INDEX[data_format]
    h_dim = dims[DataDim.HEIGHT]

    if angle == 0:
        return x
    elif angle == 180:
        return x.flip(h_dim)
    else:
        raise NotImplementedError("Must be rotation divisible by 90 degrees")


def display_receptive(network, n, weight_fill=1e-5, device='cpu', plotflag=False):
    """Display network receptive field

    Args:
        network: torch network
        n: size of input
        weight_fill: constant to fill weights (if None keep original weights)
        device: device

    Returns:
        network_rec: torch network used to test receptive field
        out: impulse response

    """
    # Create impulse response
    inp = torch.zeros((1, 1, n, n)).to(device)
    inp[:, :, n//2, n//2] = 1.

    network_rec = copy.deepcopy(network)
    if weight_fill is not None:
        # Change all filters values to constant
        for name, param in network_rec.named_parameters():
            if 'weight' in name:
                param.data.fill_(weight_fill)
            if 'bias' in name:
                param.data.fill_(0.)

    # Compute output
    with torch.no_grad():
        out = network_rec(inp)

    if plotflag:
        plt.figure()
        plt.imshow(inp.cpu().squeeze(), cmap='seismic')
        plt.colorbar()
        plt.scatter(n//2, n//2, c='r', s=10)

        plt.figure()
        plt.imshow(out.detach().cpu().squeeze(), cmap='Blues', vmin=0, vmax=out.detach().cpu().max())
        plt.colorbar()
        plt.scatter(n//2, n//2, c='r', s=10)
        print(f'Ouput in middle pixel {out[:, :, n//2, n//2].item()}')

    return network_rec, out