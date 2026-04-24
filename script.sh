#!/bin/bash
#SBATCH --job-name=gpu
#SBATCH --partition=GPU
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1

make
./nbody-gpu 100000 5 10 10
