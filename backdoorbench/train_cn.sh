#!/bin/bash

. ./input_validation.sh
input_validation $@

my_dir="/home/xxu/back_stealthiness"

if [[ -z $my_dir ]]; then
    echo "Please set the 'my_dir' variable to the parent directory of the data and record folders"
    exit 1
fi

data_dir="/vol/aisy/xxu/data/"
record_dir="$my_dir/record_cn114"
timestamp=$(date +"T%d-%m_%H-%M")

# gpu=$(python get_gpu.py)

# if [[ ! $gpu =~ "RTX 2080 Ti" ]]; then
#     echo "Unexpected GPU: ${gpu}"
#     exit 1
# fi

pratio_label=$(echo p$pratio | tr . -)
attack_id="${attack}_${model}_${dataset}_${pratio_label}"

function get_clean_config() {
    if [[ $dataset == "imagenette" ]]; then
        echo "config/attack/custom/imagenette.yaml"
    elif [[ $dataset == "tiny" ]]; then
        echo "config/attack/custom/tiny.yaml"
    else
        echo "config/attack/custom/cifar.yaml"
    fi
}

function get_badnet_trigger() {
    if [[ $dataset == "imagenette" ]]; then
        echo "./resource/badnet/badnet_patch_80.png"
    elif [[ $dataset == "imagenette" ]]; then
        echo "./resource/badnet/badnet_patch_64.png"
    else
        echo "./resource/badnet/badnet_patch_32.png"
    fi
}

function get_badnet_mask() {
    if [[ $dataset == "imagenette" ]]; then
        echo "./resource/badnet/white_square_80.png"
    elif [[ $dataset == "imagenette" ]]; then
        echo "./resource/badnet/white_square_64.png"
    else
        echo "./resource/badnet/white_square_32.png"
    fi
}

function get_blend_trigger() {
    path_to_trigger="./resource/blended"

    if [[ $dataset == "imagenette" ]]; then
        echo "$path_to_trigger/hellokitty_80.png"
    elif [[ $dataset == "imagenette" ]]; then
        echo "$path_to_trigger/hellokitty_64.png"
    else
        echo "$path_to_trigger/hellokitty_32.png"
    fi
}

function get_POOD_dataset() {
    target_dataset=$1

    if [[ $target_dataset == "cifar10" ]]; then
        echo "cifar10"
    elif [[ $target_dataset == "cifar100" ]]; then
        echo "cifar10"
    elif [[ $target_dataset == "imagenette" ]]; then
        echo "cifar10"
    elif [[ $target_dataset == "tiny" ]]; then
        echo "cifar10"
    fi
}

# Clean training configuration
yaml_conf="--yaml_path $(get_clean_config)"

# Add additional attack-specific configuration unless attack == prototype (clean model)
if [[ $attack != "prototype" ]]; then
    yaml_conf="${yaml_conf} --bd_yaml_path config/attack/custom/$attack.yaml"
    attack_opts="--attack_target 0 --pratio $pratio"

    if [[ $attack == "badnet" ]]; then
        attack_opts="$attack_opts --patch_path $(get_badnet_trigger) --mask_path $(get_badnet_mask)"
    elif [[ $attack == "blended" ]]; then
        attack_opts="$attack_opts --attack_trigger_img_path $(get_blend_trigger)"
    elif [[ $attack == "bpp" ]]; then
        attack_opts="$attack_opts --neg_ratio $pratio"
    elif [[ $attack == "narcissus" ]]; then
        trigger_paths=$(find $record_dir/narcissus_${model}_${dataset}_"trigger.npy")
        first_trigger_path=$(echo $trigger_paths | cut -d ' ' -f1)

        # If there already is a trigger for this model and dataset, reuse it
        if [[ -n $first_trigger_path ]]; then
            attack_opts="$attack_opts --attack_trigger_path $first_trigger_path"
        else # Create a new trigger based on a surrogate model trained on a POOD dataset
            pood_dataset=$(get_POOD_dataset $dataset)
            surrogate_model_path="${record_dir}/prototype_${model}_${pood_dataset}_pNone"
            attack_opts="$attack_opts --pood_dataset $pood_dataset --surrogate_model_path $surrogate_model_path"
        fi
    fi
fi


python ./attack/$attack.py $yaml_conf $attack_opts --save_parent_dir "$record_dir" --save_folder_name $attack_id  --dataset_path="$data_dir" --model $model --dataset $dataset --epochs $n_epochs --device cuda:0 --amp 1

# Handle BackdoorBench failure
if [[ $? -ne 0 ]]; then
    mv "$record_dir/$attack_id" "$record_dir/FAIL_${attack_id}_${timestamp}"
    echo "!!! BACKDOORBENCH FAILURE !!!"
    exit 1
fi

echo "!!! FINISHED TRAINING !!!"
cd $record_dir

# Remove additional bpp datasets before creating tar
if [ $attack == "bpp" ]; then
    cd $attack_id
    rm -rf "clean_train_dataset" "clean_test_dataset" "bd_train_dataset_Save" "bd_test_all_dataset" 
    cd ..
fi
    
tar -cf "${attack_id}_${timestamp}.tar" $attack_id
