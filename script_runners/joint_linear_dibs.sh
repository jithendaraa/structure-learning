#!/bin/bash

dataset=$1
train=$2
model=$3
time=$4

seeds=(1)
exp_edges=(0.5)
lrs=(0.001)
dibs_lrs=(0.005)
num_samples=(200)
num_updates=(1000)

array_len=$(( ${#seeds[@]} * ${#exp_edges[@]} * ${#lrs[@]} * ${#dibs_lrs[@]} * ${#num_updates[@]} * ${#num_samples[@]} ))
echo $array_len

if [ ${train} == 'train' ]
then
    if [ ${dataset} == 'er' ]
    then
        config='linear_dibs'
        output_file="out/DIBS/linear_joint_dibs_er-%A_%a.out"
        echo "Train linear Joint DIBS ER: ${config}"
    else
        echo "Not implemented dataset ${dataset}" 
    fi
else
    echo "Not implemented dataset ${train}" 
fi

command="sbatch --array=1-${array_len}%512 --job-name ${config} --output ${output_file} --time ${time} scripts/run_linear_jointdibs.sh ${config}"   
echo ""
echo ${command}
echo ""

RES=$(${command})
job_id=${RES##* }
echo "Job ID"" ""${job_id}"" -> ""${config} ${args}" >> out/job_logs.txt
echo "Job ID"" ""${job_id}"" -> ""${config} ${args}" 
echo ""


