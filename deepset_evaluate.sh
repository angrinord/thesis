#!/bin/bash
#SBATCH --job-name=deepset_evaluate
#SBATCH --output=runs/%x_%j.log
#SBATCH --error=runs/%x_%j.err
#SBATCH --mail-user=angrimson@ismll.de
#SBATCH --partition=STUD
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpu-110
source activate thesisenv
srun python ./mystuff/surrogate_deepset.py -i "$1"
