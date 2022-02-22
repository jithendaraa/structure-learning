#!/bin/bash

dataset=$1
train=$2
model=$3
time=$4

seeds=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20)
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
        config='nonlinear_dibs'
        output_file="out/DIBS/nonlinear_joint_dibs_er-%A_%a.out"
        echo "Train Nonlinear Joint DIBS ER: ${config}"
    else
        echo "Not implemented dataset ${dataset}" 
    fi
else
    echo "Not implemented dataset ${train}" 
fi

command="sbatch --array=1-${array_len}%512 --job-name ${config} --output ${output_file} --time ${time} scripts/run_nonlinear_jointdibs.sh ${config}"   
echo ""
echo ${command}
echo ""

RES=$(${command})
job_id=${RES##* }
echo "Job ID"" ""${job_id}"" -> ""${config} ${args}" >> out/job_logs.txt
echo "Job ID"" ""${job_id}"" -> ""${config} ${args}" 
echo ""

