#!/bin/sh
#SBATCH --time=2:00:00
#SBATCH --gres=gpu:a100-40:1
cd ./
srun python neural_network.py