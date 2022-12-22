##################################################################
# 2022 - King Abdullah University of Science and Technology (KAUST)
#
# Authors: Nick Luiken, Matteo Ravasi
# Description: PnP deblending with blind-spot networks
##################################################################

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import pylops

from torch.utils.data import TensorDataset, DataLoader
from pylops.optimization.solver import lsqr
from pylops.signalprocessing import *
from ssdeblend.model.ssstructdenoise import NoiseNetwork
from ssdeblend.utils import set_seed


def train_network(network, epochs, train_loader, todenoise, device='cpu'):
    """Network training

    Train blind-spot network seismic shot gather in continuos mode based on pre-defined sequence of firing times.

    Parameters
    ----------
    network : :obj:`torch.nn`
        Network
    epochs : :obj:`int`
        Number of epochs
    train_loader : :obj:`torch.utils.data.DataLoader`
        Dataloader
    todenoise : :obj:`torch.Tensor`
        Data to denoise (during training for visualization purposes)
    device : :obj:`str`
        Device

    Returns
    -------
    network : :obj:`torch.nn`
        Updated network

    """
    criterion = nn.L1Loss()
    optimizer = optim.Adam(network.parameters())

    for i in range(epochs):
        _ = train(network, criterion, optimizer, train_loader, device=device)
        # denoised = denoise(network, todenoise)
    return network


def to_train_loader(data, target, batch_size=8):
    """Data loader creation

    Create dataloder for training

    Parameters
    ----------
    data : :obj:`torch.Tensor`
        Network input
    target : :obj:`torch.Tensor`
        Network target
    batch_size : :obj:`int`, optional
        Batch size

    Returns
    -------
    train_loader : :obj:`torch.utils.data.DataLoader`
        Dataloader

    """
    # Define the training data set
    data_set = torch.from_numpy(data).float()
    target_set = torch.from_numpy(target).float()
    train_dataset = TensorDataset(data_set, target_set)

    # Set the seed for DataLoader
    g = torch.Generator()
    g.manual_seed(0)

    # Create dataloader
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, generator=g)

    return train_loader


def train(network, criterion, optimizer, data_loader, device='cpu'):
    """Training epoch

    Run a single epoch of network training

    Parameters
    ----------
    network : :obj:`torch.nn`
        Network
    criterion : :obj:`torch.nn`
        Loss
    optimizer : :obj:`torch.optim`, optional
        Optimizer
    data_loader : :obj:`torch.utils.data.DataLoader`
        Dataloader
    device : :obj:`str`
        Device

    Returns
    -------
    loss : :obj:`float`
        Epoch total loss

    """
    network.train()
    loss = 0
    for data, target in data_loader:
        data, target = data.unsqueeze(1).to(device), target.unsqueeze(1).to(device)
        optimizer.zero_grad()
        prediction = network(data)
        ls = criterion(prediction, target)
        ls.backward()
        optimizer.step()
        loss += ls.item()
    return loss


def denoise(network, data, device='cpu'):
    """Denoising

    Denoise entire dataset with pre-trained network

    Parameters
    ----------
    network : :obj:`torch.nn`
        Network
    data : :obj:`torch.Tensor`
        Data to denoise

    Returns
    -------
    den_data : :obj:`torch.Tensor`
        Denoised data

    """
    nr, ns, nt = data.shape
    den_data = data.copy()
    for i in range(nr):
        temp = torch.from_numpy(data[i, :, :]).float()
        temp = network(temp.unsqueeze(0).unsqueeze(0).float().to(device))
        den_data[i, :, :] = temp.cpu().detach().numpy().squeeze()
    return den_data


def plug_n_play(BlendingOp, data, shapes, eps, outer_its, inner_its, denoiser_epochs,
                initial_network=None, x_true=None, stop_training=None, seed=42, device='cpu'):
    """PnP deblending with a self-supervised denoiser

    Deblend seismic data by inversion via PnP iterations using a blind-spot, self-supervised denoiser

    Parameters
    ----------
    BlendingOp : :obj:`pylops.LinearOperator`
        Blending operator
    data : :obj:`np.ndarray`
        Data to deblend
    shapes : :obj:`tuple`
        Shape of the data (ns, nr, nt)
    eps : :obj:`float`
        Lagrange multiplier
    outer_its : :obj:`int`
        Number of outer iterations
    inner_its : :obj:`int`
        Number of inner iterations
    denoiser_epochs : :obj:`int`
        Number of denoiser epochs
    initial_network : :obj:`torch.nn`, optional
        Initial network (if ``None`` initalize randomly before starting PnP)
    x_true : :obj:`np.ndarray`, optional
        True solution (only used to compute the error through iterations)
    stop_training : :obj:`int`, optional
        Number of outer iterations after which the network is frozen
    seed : :obj:`float`, optional
        Network initialization seed
    device : :obj:`str`
        Device

    Returns
    -------
    x : :obj:`np.ndarray`
        X-update final estimate
    y : :obj:`np.ndarray`
        Y-update final estimate
    network : :obj:`torch.nn`
        Final network
    error1 : :obj:`np.ndarray`
        X-error as function of outer iterations
    error2 : :obj:`np.ndarray`
        Y-error as function of outer iterations

    """
    if stop_training is None:
        stop_training = outer_its + 1

    # Create operator for x-update and network
    I = pylops.Identity(BlendingOp.shape[1])
    set_seed(seed)
    if initial_network is None:
        network = NoiseNetwork(1, 1, blindspot=True).to(device)
    else:
        network = initial_network
    ns, nr, nt = shapes

    # Initialize PnP variables
    x = np.zeros(BlendingOp.shape[1])
    y = np.zeros(BlendingOp.shape[1])
    z = np.zeros(BlendingOp.shape[1])
    zT = np.zeros(BlendingOp.shape[1])
    zT = np.reshape(zT, (nr, ns, nt))
    error1 = np.zeros(outer_its)
    error2 = np.zeros(outer_its)

    for i in range(outer_its):
        print('Outer iteration: ', i)
        # X-update
        x = lsqr(pylops.VStack([BlendingOp, eps * I]), np.hstack([data, eps * (y - z)]), niter=inner_its, x0=x)[0]
        # Reshape the solution
        xT = np.reshape(x, (ns, nr, nt))  # (ns, nr, nt) FORMAT!
        xT = np.transpose(xT, (1, 0, 2))
        # Y-update
        if i > stop_training:
            denoiser_epochs = 0
        if denoiser_epochs > 0:
            data_loader = to_train_loader(xT + zT, xT + zT)
            # Use this for no warm starts
            network = train_network(network, epochs=denoiser_epochs, train_loader=data_loader, todenoise=xT + zT, device=device)
        yT = denoise(network, xT + zT, device=device)  # (nr, ns, nt) FORMAT!
        yT = np.reshape(yT, (nr, ns, nt))
        y = np.transpose(yT, (1, 0, 2))
        y = y.ravel()
        # Z-update
        z = z + (x - y)  # (ns, nr, nt) FORMAT!
        zT = np.reshape(z, (ns, nr, nt))  # (nr, ns, nt) FORMAT!
        zT = np.transpose(zT, (1, 0, 2))
        if x_true is not None:
            error1[i] = np.linalg.norm(x.ravel() - x_true.ravel()) / np.linalg.norm(x_true.ravel())
            error2[i] = np.linalg.norm(y.ravel() - x_true.ravel()) / np.linalg.norm(x_true.ravel())
            print(np.linalg.norm(x - x_true) / np.linalg.norm(x_true))
            print(np.linalg.norm(y - x_true) / np.linalg.norm(x_true))

    if x_true is not None:
        return x, y, network, error1, error2
    else:
        return x, y, network
