#!/bin/bash

# 1. Train VAE DIBS
dataset=$1
train=$2
model=$3
time=$4

if [ ${train} == 'train' ]
then
    if [ ${dataset} == 'er' ]
    then
        config='decoder_dibs_er'
        output_file="out/Decoder_DIBS/decoder_dibs_er-%j.out"
        echo "Train Decoder DIBS ER: ${config}"
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
