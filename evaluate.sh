#!/bin/bash
#SBATCH --job-name=cluster_pretrain
#SBATCH --output=runs/%x_%j.log
#SBATCH --error=runs/%x_%j.err
#SBATCH --mail-user=angrimson@ismll.de
#SBATCH --partition=STUD
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpu-110
source activate thesisenv
srun python ./mystuff/cluster_pretrain.py  -i "$1"
