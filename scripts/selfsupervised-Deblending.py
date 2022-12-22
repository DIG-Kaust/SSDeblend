#!/usr/bin/env python
# coding: utf-8

import argparse
import os
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
import segyio

from pylops.signalprocessing import *

from ssdeblend.pnp import Blending
from ssdeblend.pnp import plug_n_play
from ssdeblend.utils import set_seed


def main(parser):

    ######### INPUT PARAMS #########
    parser.add_argument('-e', '--eps', type=float, default=1, help='Regularization parameter')
    parser.add_argument('-i', '--inner', type=int,  default=3, help='Inner iterations')
    parser.add_argument('-d', '--denoiser_epochs', type=int,  default=20, help='Denoiser epochs')
    parser.add_argument('-s', '--stop_training', type=int,  default=1000, help='Stop training')
    parser.add_argument('-r', '--seed', type=int, default=42, help='Seed')

    args = parser.parse_args()

    print('PnP Deblending')
    print('----------------------------')
    print(f'Regularization = {args.eps}')
    print(f'Number of inner iterations = {args.inner}')
    print(f'Denoiser epochs = {args.denoiser_epochs}')
    print(f'Stop training = {args.stop_training}')
    print(f'Seed = {args.seed}')
    print('----------------------------\n')

    # Create folder with figures
    foldername=f'pnp_eps{args.eps}_inner{args.inner}_epochs{args.denoiser_epochs}_stop_training{args.stop_training}_seed{args.seed}'
    if not os.path.isdir(f'../figures/{foldername}'):
        os.mkdir(f'../figures/{foldername}')

    # Set GPU
    torch.cuda.empty_cache()
    device = 'cuda:0'
    print(f'GPU used: {torch.cuda.get_device_name(device)}')

    ######### DATA LOADING #########
    # Load data
    f = segyio.open('../data/MobilAVO.segy', ignore_geometry=True)
    data = segyio.collect(f.trace[:])
    data = data.reshape(1001, 120, 1500)

    # Rearrange data
    ns = 64  # number of sources
    nr = 120  # number of receivers
    nt = 1024  # number of time samples
    dt = 0.004  # time sampling (sec)
    dr = 25  # receiver sampling (m)
    data = data[:ns, :, :nt]

    # Define axes
    t = dt * np.arange(nt)
    xr = np.arange(0, nr * dr, dr)

    # Create blending times
    set_seed(42)  # we set here the seed for reproducibility
    overlap = 0.5
    ignition_times = -1 + 2 * np.random.rand(ns)
    ignition_times[0] = 0  # set the first time dither to 0

    # Blending operator
    BlendingOp = Blending(nt, nr, ns, dt, overlap, ignition_times, nproc=20)

    # Blend and pseudodeblend the data
    blended_data = BlendingOp * data.ravel()
    pseudodeblended_data = BlendingOp.H * blended_data

    pseudodeblended_data = pseudodeblended_data.reshape(ns, nr, nt)

    fig, axs = plt.subplots(1, 4, figsize=(14, 7), sharey=True)
    axs[0].imshow(data[30, :, :].T.real, aspect='auto', vmin=-100, vmax=100, cmap='gray',
                  extent=(xr.min(), xr.max(), t.max(), t.min()))
    axs[0].set_xlabel('x[m]', fontsize=15)
    axs[0].set_ylabel('t [s]', fontsize=15)
    axs[0].set_title('Clean CSG')
    axs[1].imshow(pseudodeblended_data[30, :, :].T.real, aspect='auto', vmin=-100, vmax=100, cmap='gray',
                  extent=(xr.min(), xr.max(), t.max(), t.min()))
    axs[1].set_xlabel('x[m]', fontsize=15)
    axs[1].set_title('Pseudo-deblended CSG')
    axs[2].imshow(data[:, 30, :].T.real, aspect='auto', vmin=-100, vmax=100, cmap='gray',
                  extent=(xr.min(), xr.max(), t.max(), t.min()))
    axs[2].set_xlabel('x[m]', fontsize=15)
    axs[2].set_title('Clean CCG')
    axs[3].imshow(pseudodeblended_data[:, 30, :].T.real, aspect='auto', vmin=-100, vmax=100, cmap='gray',
                  extent=(xr.min(), xr.max(), t.max(), t.min()))
    axs[3].set_xlabel('x[m]', fontsize=15)
    axs[3].set_title('Pseudo-deblended CCG')
    plt.tight_layout()
    plt.savefig(f'../figures/{foldername}/data.png')

    # Deblending by inversion
    t0 = time.time()
    deblended_data_x, deblended_data_y, network_out, error_x, error_y = \
        plug_n_play(BlendingOp, blended_data.ravel(), shapes=(ns, nr, nt),
                    eps=args.eps, outer_its=30, inner_its=args.inner,
                    denoiser_epochs=args.denoiser_epochs,
                    initial_network=None, x_true=data.ravel(),
                    stop_training=args.stop_training, seed=args.seed,
                    device=device)
    deblended_data_x = deblended_data_x.reshape(ns, nr, nt)
    deblended_data_y = deblended_data_y.reshape(ns, nr, nt)
    print(f'Elapsed Time: {(time.time() - t0) / 60} min)')

    plt.figure(figsize=(8,6))
    plt.plot(error_x, '*-', label='x')
    plt.plot(error_y, '*-', label='y')
    plt.xlabel('Number of outer iterations', fontsize=20)
    plt.ylabel('Relative error', fontsize=20)
    plt.title('Error as a function of outer iterations', fontsize=20)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend()
    plt.savefig(f'../figures/{foldername}/error.png')

    idxs = [20, 40, -1]
    for idx in idxs:
        fig, axs = plt.subplots(1, 4, figsize=(12, 6))
        axs[0].imshow(pseudodeblended_data[idx, :, :].T.real, aspect='auto', vmin=-100, vmax=100, cmap='gray',
                      extent=(xr.min(), xr.max(), t.min(), t.max()))
        axs[0].set_xlabel('x',fontsize=15)
        axs[0].set_ylabel('t',fontsize=15)
        axs[0].set_title('Pseudodeblended')
        axs[1].imshow(deblended_data_x[:, idx, :].T.real, aspect='auto', vmin=-100, vmax=100, cmap='gray',
                      extent=(xr.min(), xr.max(), t.min(), t.max()))
        axs[1].set_xlabel('x',fontsize=15)
        axs[1].set_ylabel('t',fontsize=15)
        axs[1].set_title('Denoised')
        axs[2].imshow(data[:, idx, :].T.real, aspect='auto', vmin=-100, vmax=100, cmap='gray',
                      extent=(xr.min(), xr.max(), t.min(), t.max()))
        axs[2].set_xlabel('x',fontsize=15)
        axs[2].set_ylabel('t',fontsize=15)
        axs[2].set_title('True')
        axs[3].imshow((data[:, idx, :] - deblended_data_x[:, idx, :]).T.real, aspect='auto', vmin=-10, vmax=10, cmap='seismic',
                      extent=(xr.min(), xr.max(), t.min(), t.max()))
        axs[3].set_xlabel('x',fontsize=15)
        axs[3].set_ylabel('t',fontsize=15)
        axs[3].set_title('Difference')
        plt.tight_layout()
        plt.savefig(f'../figures/{foldername}/crg{idx}_reconstruction.png')

    idxs = [20, 40, -1]
    for idx in idxs:
        fig, axs = plt.subplots(1, 4, figsize=(12, 8))
        axs[0].imshow(pseudodeblended_data[:, idx, :].T.real, aspect='auto', vmin=-100, vmax=100, cmap='gray',
                      extent=(xr.min(), xr.max(), t.min(), t.max()))
        axs[0].set_xlabel('x',fontsize=15)
        axs[0].set_ylabel('t',fontsize=15)
        axs[0].set_title('Pseudodeblended')
        axs[1].imshow(deblended_data_x[idx, :, :].T.real, aspect='auto', vmin=-100, vmax=100, cmap='gray',
                      extent=(xr.min(), xr.max(), t.min(), t.max()))
        axs[1].set_xlabel('x',fontsize=15)
        axs[1].set_ylabel('t',fontsize=15)
        axs[1].set_title('Denoised')
        axs[2].imshow(data[idx, :, :].T.real, aspect='auto', vmin=-100, vmax=100, cmap='gray',
                      extent=(xr.min(), xr.max(), t.min(), t.max()))
        axs[2].set_xlabel('x',fontsize=15)
        axs[2].set_ylabel('t',fontsize=15)
        axs[2].set_title('True')
        axs[3].imshow((data[idx, :, :] - deblended_data_x[idx, :, :]).T.real, aspect='auto', vmin=-100, vmax=100, cmap='seismic',
                      extent=(xr.min(), xr.max(), t.min(), t.max()))
        axs[3].set_xlabel('x',fontsize=15)
        axs[3].set_ylabel('t',fontsize=15)
        axs[3].set_title('Difference')
        plt.tight_layout()
        plt.savefig(f'../figures/{foldername}/csg{idx}_reconstruction.png')

    np.savez(f'../results/{foldername}', error=error_x, error_denoised=error_y,
             data_PnP=deblended_data_x, data_PnP_denoised=deblended_data_y)


if __name__ == "__main__":
    description = 'PnP deblending'
    main(argparse.ArgumentParser(description=description))

