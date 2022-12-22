#!/bin/sh

# Pnp Deblending ablation studies

## Eps and denoise
python selfsupervised-Deblending.py -e 1e-1 -i 3 -d 20
python selfsupervised-Deblending.py -e 1 -i 3 -d 20
python selfsupervised-Deblending.py -e 10 -i 3 -d 20

python selfsupervised-Deblending.py -e 1e-1 -i 3 -d 10
python selfsupervised-Deblending.py -e 1 -i 3 -d 10
python selfsupervised-Deblending.py -e 10 -i 3 -d 10

python selfsupervised-Deblending.py -e 1e-1 -i 3 -d 30
python selfsupervised-Deblending.py -e 1 -i 3 -d 30
python selfsupervised-Deblending.py -e 10 -i 3 -d 30

## Inner
python selfsupervised-Deblending.py -e 1 -i 1 -d 30
python selfsupervised-Deblending.py -e 1 -i 5 -d 30

## Stop training
python selfsupervised-Deblending.py -e 1 -i 3 -d 30 -s 5
python selfsupervised-Deblending.py -e 1 -i 3 -d 30 -s 10
python selfsupervised-Deblending.py -e 1 -i 3 -d 30 -s 20

# Large number of epochs
python selfsupervised-Deblending.py -e 1 -i 3 -d 60

# Seed
python selfsupervised-Deblending.py -e 1 -i 3 -d 30 -r 1
python selfsupervised-Deblending.py -e 1 -i 3 -d 30 -r 2
python selfsupervised-Deblending.py -e 1 -i 3 -d 30 -r 3
python selfsupervised-Deblending.py -e 1 -i 3 -d 30 -r 4
python selfsupervised-Deblending.py -e 1 -i 3 -d 30 -r 5
python selfsupervised-Deblending.py -e 1 -i 3 -d 30 -r 6
python selfsupervised-Deblending.py -e 1 -i 3 -d 30 -r 7
python selfsupervised-Deblending.py -e 1 -i 3 -d 30 -r 8
python selfsupervised-Deblending.py -e 1 -i 3 -d 30 -r 9
python selfsupervised-Deblending.py -e 1 -i 3 -d 30 -r 10

