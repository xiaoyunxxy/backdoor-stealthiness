#!/bin/bash

function input_validation() {
    usage_str="""
Usage: ${0} <ATTACK> <MODEL ARCHITECTURE> <DATASET>"""

    if [[ $# -ne 3 ]]; then
        echo $usage_str
        exit 1
    fi

    attack=$1
    model=$2
    dataset=$3

    attack_regex="^dfba$"
    validate_str $attack $attack_regex

    model_regex="^resnet18|vgg16$"
    validate_str $model $model_regex

    dataset_regex="^cifar10|cifar100|imagenette|tiny$"
    validate_str $dataset $dataset_regex
}

function validate_str() {
    str=$1
    regex=$2

    if [[ ! $str =~ $regex ]]; then
        echo "Argument $str is invalid"
        exit 1
    fi
}
