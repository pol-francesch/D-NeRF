#!/usr/bin/bash
#SBATCH --job-name=train_shirt_roe2
#SBATCH --output=train_shirt_roe2.%j.out
#SBATCH --error=train_shirt_roe2.%j.err
#SBATCH --time=28:00:00
#SBATCH --ntasks=1
#SBATCH -p gpu
#SBATCH -G 1
#SBATCH -c 10
#SBATCH --mem=16GB
#SBATCH -C GPU_MEM:16GB

module reset
module load py-pytorch/2.0 py-torchvision/0.15.1_py39
module load py-matplotlib/3.7.1_py39
cd $HOME
source $HOME/python_envs/dnerf/bin/activate

python $HOME/D-NeRF/run_dnerf.py --config $HOME/D-NeRF/configs/sherlock/shirt_roe2.txt