#!/bin/bash

# 1. Train DIBS
dataset=$1
train=$2
model=$3
time=$4

if [ ${train} == 'train' ]
then
    if [ ${dataset} == 'er' ]
    then
        config='train_dibs_er'
        output_file="out/DIBS/""$config""-%j.out"
        echo "Train DIBS ER: ${config}"
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
