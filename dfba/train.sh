#!/bin/bash

. ./input_validation.sh
input_validation $@

my_dir="/home/xxu/back_stealthiness"

if [[ -z $my_dir ]]; then
    echo "Please set the 'my_dir' variable to the parent directory of the data and record folders"
    exit 1
fi

data_dir="$my_dir/record/data"
record_dir="$my_dir/record"
timestamp=$(date +"T%d-%m_%H-%M")

# gpu=$(python get_gpu.py)

# if [[ ! $gpu =~ "RTX 2080 Ti" ]]; then
#     echo "Unexpected GPU: ${gpu}"
#     exit 1
# fi

attack_id="${attack}_${model}_${dataset}_pNone"

function check_failure() {
    error_code=$1
    error_message=$2

    if [[ $error_code -ne 0 ]]; then
        mv "$record_dir/$attack_id" "$record_dir/FAIL_${attack_id}_${timestamp}"
        echo "!!! $error_message !!!"
        exit 1
    fi
}

clean_model_path="$record_dir/prototype_${model}_${dataset}_pNone/clean_model.pth"
save_path=$record_dir/$attack_id
mkdir -p $save_path

# Choose batch size depending on dataset, and gamma depending on dataset/model combination
if [[ $dataset == "imagenette" ]]; then
    bs=20

    if [[ $model == "resnet18" ]]; then
        gamma=1.6
    elif [[ $model == "vgg16" ]]; then
        gamma=1.7
    fi
elif [[ $dataset == "tiny" ]]; then
    bs=256
    if [[ $model == "resnet18" ]]; then
        gamma=1.6
    elif [[ $model == "vgg16" ]]; then
        gamma=1.7
    fi
else
    bs=100

    if [[ $model == "resnet18" ]]; then
        gamma=1.5
    elif [[ $model == "vgg16" ]]; then
        if [[ $dataset == "cifar10" ]]; then
            gamma=1.4
        elif [[ $dataset == "cifar100" ]]; then
            gamma=1.5
        fi
    fi
fi


python attack_model.py --model $model --dataset $dataset \
                       --trigger_size 3 --gamma $gamma \
                       --batch-size $bs --manual-seed 0 \
                       --dataset_dir $data_dir/$dataset --benign_weights $clean_model_path --checkpoint $save_path


check_failure $? "FAILURE WHILE BACKDOORING MODEL"

echo "!!! FINISHED BACKDOORING !!!"

cd $record_dir    
tar -cf "${attack_id}_${timestamp}.tar" $attack_id
