![SSDeblend](https://github.com/DIG-Kaust/self-supervised-deblending/blob/master/logo.png)

Reproducible material for **A hybrid approach to seismic deblending: when physics meets self-supervision -
Luiken N., Ravasi M., Birnie, C.** submitted to NeurIPS 2022.

We are grateful to the authors of the PyTorch implementation of the [High-Quality Self-Supervised Deep Image Denoising](https://github.com/COMP6248-Reproducability-Challenge/selfsupervised-denoising)
2019 NIPS paper, who we have adapted to handle structured noise. Some of the codes provided in ``ssinterp.model`` are cleaned-up versions of the original 
codes.

## Project structure
This repository is organized as follows:

* :open_file_folder: **ssinterp**: python library containing routines for deblending by inversion with a self-supervised denoiser;
* :open_file_folder: **data**: folder where the MobilAVO dataset must be placed (note that the data can be downloaded from this 
  [link](https://wiki.seg.org/wiki/Mobil_AVO_viking_graben_line_12) (follow instructions under ``External links`` tab))
* :open_file_folder: **notebooks**: set of jupyter notebooks reproducing the experiments in the paper (see below for more details);
* :open_file_folder: **scripts**: set of python scripts used to run multiple experiments with different input parameters for the 
  ablation studies
* :open_file_folder: **figures**: folder where figures from various script experiments will be saved.
* :open_file_folder: **results**: folder where results from various script experiments will be saved for later analysis.

## Notebooks
The following notebooks are provided:

- :orange_book: ``SSNetwork_impulseresponse.ipynb``: notebook displaying the impulse response of the blind-network used in this work;
- :orange_book: ``Deblending_CCG-fourier.ipynb``: notebook performing benchmark deblending by inversion in CCG domain using a patched fourier sparsity transform;
- :orange_book: ``SSDeblending_CCG-denoising.ipynb``: notebook performing benchmark deblending by denoising in CCG domain using the proposed self-supervised network;
- :orange_book: ``SSDeblending_CCG-pnp.ipynb``: notebook performing benchmark deblending by inversion in CCG domain using the proposed PnP algorithm;
- :orange_book: ``SSDeblending_CRG-pnp.ipynb``: notebook performing benchmark deblending by inversion in CRG domain using the proposed PnP algorithm;
- :orange_book: ``Ablation_studies.ipynb``: notebook display results from ablation studies;


## Getting started :space_invader: :robot:
To ensure reproducibility of the results, we suggest using the `environment.yml` file when creating an environment.

Simply run:
```
./install_env.sh
```
It will take some time, if at the end you see the word `Done!` on your terminal you are ready to go. 

Remember to always activate the environment by typing:
```
conda activate ssdeblend
```

**Disclaimer:** All experiments have been carried on a Intel(R) Xeon(R) CPU @ 2.10GHz equipped with a single NVIDIA GEForce RTX 3090 GPU. Different environment 
configurations may be required for different combinations of workstation and GPU.
