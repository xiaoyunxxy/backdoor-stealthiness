#!/bin/bash

function input_validation() {
    usage_str="""
Usage: ${0} <ATTACK> <MODEL ARCHITECTURE> <DATASET> <POISON RATE> <NUMBER OF EPOCHS>"""

    if [[ $# -ne 5 ]]; then
        echo $usage_str
        exit 1
    fi

    attack=$1
    model=$2
    dataset=$3
    pratio=$4
    n_epochs=$5

    attack_regex="^adaptive_patch|adaptive_blend$"
    validate_str $attack $attack_regex

    model_regex="^resnet18|vgg16|densenet121|vit_small$"
    validate_str $model $model_regex

    dataset_regex="^cifar10|cifar100|imagenette|tiny$"
    validate_str $dataset $dataset_regex

    pratio_regex="^0.003|0.05|0.1$"
    validate_str $pratio $pratio_regex

    n_epochs_regex="^[0-9]+$"
    validate_str $n_epochs $n_epochs_regex
}

function validate_str() {
    str=$1
    regex=$2

    if [[ ! $str =~ $regex ]]; then
        echo "Argument $str is invalid"
        exit 1
    fi
}
