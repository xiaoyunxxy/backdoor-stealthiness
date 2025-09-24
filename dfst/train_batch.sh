#!/bin/bash
#SBATCH -A cseduproject
#SBATCH -p csedu-prio,csedu
#SBATCH --qos=csedu-small
#SBATCH --cpus-per-task=2
#SBATCH --mem 3G
#SBATCH --gres=gpu:rtx_2080_ti:1 
#SBATCH --time=4:00:00

. ./input_validation.sh
input_validation $@

./train.sh $attack $model $dataset $pratio $n_epochs
# ./train.sh dfst resnet18 tiny 0.05 100
