#!/bin/bash
#SBATCH --job-name=cluster_evaluate
#SBATCH --output=runs/%x_%j.log
#SBATCH --error=runs/%x_%j.err
#SBATCH --mail-user=angrimson@ismll.de
#SBATCH --partition=STUD
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpu-110
source activate thesisenv
srun python ./mystuff/cluster_evaluate.py  -i "$1" --regime "$2"
