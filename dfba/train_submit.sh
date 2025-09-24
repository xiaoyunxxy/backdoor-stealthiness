#!/bin/bash

. ./input_validation.sh
input_validation $@

job_save="jobs/%j_${attack}_${model}_${dataset}"

sbatch --output "${job_save}.out" --error "${job_save}.err" train_batch.sh $@