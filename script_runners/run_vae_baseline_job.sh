#!/bin/bash

dataset=$1
train=$2
model=$3
time=$4
config=$5

keys=("seed" "exp_edges" "lr" "num_steps" "num_samples")

seeds=(2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20)
exp_edges=(1.0)
lrs=(0.001)
num_samples=(200)
num_steps=(1000)

array_len=$(( ${#seeds[@]} * ${#exp_edges[@]} * ${#lrs[@]} * ${#num_steps[@]} * ${#num_samples[@]} ))
echo $array_len
if [ ${train} == 'train' ]
then
    if [ ${dataset} == 'er' ]
    then
        output_file="out/VAE_baseline/vae_baseline-%A_%a.out"
        echo "Train VAE on vector data: ${config}"
    else
        echo "Not implemented dataset ${dataset}" 
    fi
else
    echo "Not implemented dataset ${train}" 
fi

command="sbatch --array=1-${array_len}%512 --job-name ${config} --output ${output_file} --time ${time} scripts/run_vae_baseline.sh ${config}"   
echo ""
echo ${command}
echo ""

RES=$(${command})
job_id=${RES##* }
echo "Job ID"" ""${job_id}"" -> ""${config} ${args}" >> out/job_logs.txt
echo "Job ID"" ""${job_id}"" -> ""${config} ${args}" 
echo ""


