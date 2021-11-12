#!/bin/bash

# 1. Train Slot_VCN_img on clevr
dataset=$1
train=$2
model=$3
time=$4


if [ ${train} == 'train' ]
then
    if [ ${dataset} == 'clevr' ]
    then
        # Train Slot_VCN_img on CLEVR
        config='train_clevr_slot1d_vcn_img'
        output_file="out/Slot_VCN_img/""$config""-%j.out"
        echo "Train Slot_VCN_img CLEVR: ${config}"
    else
        echo "Not implemented dataset ${dataset}" 
    fi
else
    echo "Not implemented dataset ${train}" 
fi

command="sbatch --job-name ${config} --output ${output_file} --time ${time} scripts/run_job.sh ${config}"      
echo ""
echo ${command}
echo ""

RES=$(${command})
job_id=${RES##* }
echo "Job ID"" ""${job_id}"" -> ""${config}" >> out/job_logs.txt
echo "Job ID"" ""${job_id}"" -> ""${config}" 
echo ""
