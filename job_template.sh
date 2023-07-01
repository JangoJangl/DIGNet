#!/bin/bash

#SBATCH -p single_dev

# Please use the complete path details :
#SBATCH -D /home/kit/anthropomatik/rt3040/repos/DIGNet
#SBATCH -o /home/kit/anthropomatik/rt3040/repos/slurm_logs/DIGNet/out_%A_%a.log
#SBATCH -e /home/kit/anthropomatik/rt3040/repos/slurm_logs/DIGNet/err_%A_%a.log

# Cluster Settings
#SBATCH -n 1                  # Number of tasks
#SBATCH -c 4                  # Number of cores per task
#SBATCH -t 2880               # 1:00:00 Hours, minutes and seconds, or '#SBATCH -t 10' - only minutes
#SBATCH --gres=gpu:1


dataset_dir

echo "Copying Dataset"
rsync --exclude "*.png" -a /home/kit/anthropomatik/rt3040/Datasets/acronym.tar.gz $TMP
tar -xzf $TMP/acronym.tar.gz -C $TMP

echo "===================================="
echo "TMP Directory:"
echo $TMP
ls -l $TMP
echo "===================================="

### Here we add the commands to execute by a bash script ###


