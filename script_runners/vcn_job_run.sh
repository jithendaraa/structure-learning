#!/bin/bash

# 1. Train VCN on er
dataset=$1
train=$2
model=$3
time=$4


if [ ${train} == 'train' ]
then
    if [ ${dataset} == 'er' ]
    then
        # Train VCN_img on CLEVR
        config='train_vcn'
        output_file="out/VCN/""$config""-%j.out"
        echo "Train VCN ER: ${config}"
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
