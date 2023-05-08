#!/usr/bin/bash
#SBATCH --job-name=train_shirt_roe2
#SBATCH --output=train_shirt_roe2.%j.out
#SBATCH --error=train_shirt_roe2.%j.err
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH -p gpu
#SBATCH -G 1
#SBATCH -c 10
#SBATCH --mem=16GB

module load gcc/9.1.0
module load python/3.9.0
cd $HOME
source python_envs/dnerf/bin/activate

python $HOME/D-NeRF/run_dnerf.py --config $HOME/D-NeRF/configs/sherlock/shirt_roe2.txt