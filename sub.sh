#!/bin/bash
#SBATCH --time=40:00:00   # walltime
#SBATCH --ntasks=1   # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=6144M   # memory per CPU core
#SBATCH -J "u4.7"   # job name
#learning_rate = 1e-5 and decay
#add tensorboard image
sh myscript.sh

