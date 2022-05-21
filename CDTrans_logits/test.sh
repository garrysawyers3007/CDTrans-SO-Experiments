#!/usr/bin/env bash
#SBATCH --nodes=7
#SBATCH --ntasks-per-node=1
#SBATCH --time=24:00:00
#SBATCH --job-name=test
#SBATCH --error=log.err
#SBATCH --output=log.out
#SBATCH --partition=gpumultinode 
#SBATCH --gres=gpu:1

. /opt/ohpc/admin/lmod/8.1.18/init/bash

# module load python/conda-python/3.7

module load DL-CondaPy3.7/python/conda-python

# bash
# conda init bash
conda activate CDTrans
# export TORCH_HOME=~/.cache/torch
cd /scratch/cdsjoge/gauransh/CDTrans/CDTrans

srun -N 1 -n 1 --gres=gpu:1 -o patch_aug_Cl.out scripts/pretrain/officehome_patch_aug/run_officehome_Cl.sh &
srun -N 1 -n 1 --gres=gpu:1 -o patch_aug_Cl_2.out scripts/pretrain/officehome_patch_aug/run_officehome_Cl_2.sh &
srun -N 1 -n 1 --gres=gpu:1 -o Ar_mixup_0.1_1.out scripts/pretrain/officehome/run_officehome_Ar_1.sh &
srun -N 1 -n 1 --gres=gpu:1 -o Ar_mixup_0.1_2.out scripts/pretrain/officehome/run_officehome_Ar_2.sh &
srun -N 1 -n 1 --gres=gpu:1 -o Ar_mixup_0.1_3.out scripts/pretrain/officehome/run_officehome_Ar_3.sh &
srun -N 1 -n 1 --gres=gpu:1 -o Cl_mixup_0.1_1.out scripts/pretrain/officehome/run_officehome_Cl_1.sh &
srun -N 1 -n 1 --gres=gpu:1 -o Cl_mixup_0.1_2.out scripts/pretrain/officehome/run_officehome_Cl_2.sh &
srun -N 1 -n 1 --gres=gpu:1 -o Cl_mixup_0.1_3.out scripts/pretrain/officehome/run_officehome_Cl_3.sh &

wait
