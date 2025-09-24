#!/bin/bash

. ./input_validation.sh
input_validation $@

my_dir="/home/xxu/back_stealthiness"

if [[ -z $my_dir ]]; then
    echo "Please set the 'my_dir' variable to the parent directory of the data and record folders"
    exit 1
fi

data_dir="/home/xxu/back_stealthiness/record/data/tiny/tiny-imagenet-200/"
record_dir="$my_dir/record"
timestamp=$(date +"T%d-%m_%H-%M")

pratio_label=$(echo p$pratio | tr . -)
attack_id="${attack}_${model}_${dataset}_${pratio_label}"

# gpu=$(python get_gpu.py)

# if [[ ! $gpu =~ "RTX 2080 Ti" ]]; then
#     echo "Unexpected GPU: ${gpu}"
#     exit 1
# fi

# Create json config based on attack settings
mkdir -p $record_dir/$attack_id
python create_config.py --dataset $dataset --network $model --epochs $n_epochs --poison_rate $pratio --save_dir $record_dir/$attack_id

python main.py --attack dfst --save_dir $record_dir/$attack_id --data_dir $data_dir

# DFST saves poisoned version of every non-target train sample, we reduce it to a subset according to our poisoning rate
python make_poisoned_trainset.py --save_dir $record_dir/$attack_id --poison_rate $pratio --dataset $dataset

cd $record_dir    
tar -cf "${attack_id}_${timestamp}.tar" $attack_id
