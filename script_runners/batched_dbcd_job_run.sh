#!/bin/bash

dataset=$1
train=$2
model=$3
time=$4
config=$5

keys=("seed" "exp_edges" "lr" "num_steps" "num_samples")

seeds=(18)
exp_edges=(1.0)
lrs=(0.0008)
num_samples=(200)
num_steps=(8000)

array_len=$(( ${#seeds[@]} * ${#exp_edges[@]} * ${#lrs[@]} * ${#num_steps[@]} * ${#num_samples[@]} ))
echo $array_len
if [ ${train} == 'train' ]
then
    if [ ${dataset} == 'er' ]
    then
        output_file="out/Decoder_BCD/bactched_dbcd-%A_%a.out"
        echo "Train Batched Decoder BCD ER: ${config}"
    else
        echo "Not implemented dataset ${dataset}" 
    fi
else
    echo "Not implemented dataset ${train}" 
fi

command="sbatch --array=1-${array_len}%512 --job-name ${config} --output ${output_file} --time ${time} scripts/batched_dbcd_job.sh ${config}"   
echo ""
echo ${command}
echo ""

RES=$(${command})
job_id=${RES##* }
echo "Job ID"" ""${job_id}"" -> ""${config} ${args}" >> out/job_logs.txt
echo "Job ID"" ""${job_id}"" -> ""${config} ${args}" 
echo ""


