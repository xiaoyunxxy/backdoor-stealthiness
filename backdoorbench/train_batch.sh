#!/bin/bash
#SBATCH -A cseduproject 
#SBATCH -p csedu-prio,csedu 
#SBATCH -c 2
#SBATCH --gres=gpu:rtx_2080_ti:1 

. ./input_validation.sh
input_validation $@

./train.sh $attack $model $dataset $pratio $n_epochs
# CUDA_VISIBLE_DEVICES=1 ./train.sh wanet resnet18 tiny 0.05 100