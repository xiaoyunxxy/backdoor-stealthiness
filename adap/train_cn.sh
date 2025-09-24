#!/bin/bash

. ./input_validation.sh
input_validation $@

my_dir="/home/xxu/back_stealthiness"

if [[ -z $my_dir ]]; then
    echo "Please set the 'my_dir' variable to the parent directory of the data and record folders"
    exit 1
fi

data_dir="/vol/aisy/xxu/data"
record_dir="$my_dir/record_cn114"
timestamp=$(date +"T%d-%m_%H-%M")

pratio_label=$(echo p$pratio | tr . -)
attack_id="${attack}_${model}_${dataset}_${pratio_label}"

# gpu=$(python get_gpu.py)

# if [[ ! $gpu =~ "RTX 2080 Ti" ]]; then
#     echo "Unexpected GPU: ${gpu}"
#     exit 1
# fi

function get_blend_trigger() {
    if [[ $dataset == "imagenette" ]]; then
        echo "hellokitty_80.png"
    elif [[ $dataset == "tiny" ]]; then
        echo "hellokitty_64.png"
    else
        echo "hellokitty_32.png"
    fi

}

# Attack options depend on attack type (adap-blend vs adap-patch)
if [[ $attack == "adaptive_blend" ]]; then
    trigger=$(get_blend_trigger)
    attack_opts="-poison_rate=$pratio -cover_rate=$pratio -trigger=$trigger -alpha=0.15 -test_alpha=0.2"
else
    cratio=$(echo "2 * $pratio" | bc) # Conservatism ratio = 2/3, i.e. twice as many cover as poisoned samples
    attack_opts="-poison_rate=$pratio -cover_rate=$cratio"
fi

python create_poisoned_set.py -dataset=$dataset -poison_type=$attack -data_dir=$data_dir -save_dir=$record_dir/$attack_id $attack_opts

# Handle poisoned set failure
if [[ $? -ne 0 ]]; then
    mv "$record_dir/$attack_id" "$record_dir/FAIL_${attack_id}_${timestamp}"
    echo "!!! POISONED DATASET CREATION FAILURE !!!"
    exit 1
fi

python train_on_poisoned_set.py -dataset=$dataset -arch=$model -poison_type=$attack -epochs=$n_epochs -data_dir=$data_dir -save_dir=$record_dir/$attack_id $attack_opts

# Handle training failure
if [[ $? -ne 0 ]]; then
    mv "$record_dir/$attack_id" "$record_dir/FAIL_${attack_id}_${timestamp}"
    echo "!!! TRAINING FAILURE !!!"
    exit 1
fi

echo "!!! FINISHED TRAINING !!!"

# Remove clean images from trainset to save space
python remove_clean_imgs.py -save_dir=$record_dir/$attack_id

cd $record_dir    
tar -cf "${attack_id}_${timestamp}.tar" $attack_id
