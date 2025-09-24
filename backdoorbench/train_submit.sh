#!/bin/bash

. ./input_validation.sh
input_validation $@

if [[ $attack == "bpp" ]]; then
    qos="csedu-normal"
    time_limit="6:00:00" 
    mem="4G"
else
    qos="csedu-small"
    time_limit="4:00:00" 
    mem="4G"
fi

job_save="jobs/%j_${attack}_${model}_${dataset}_${pratio}"

sbatch --time $time_limit -q $qos --mem $mem --output "${job_save}.out" --error "${job_save}.err" train_batch.sh $@