#!/bin/bash

dataset=$1
train=$2
model=$3
time=$4

keys=("seed" "exp_edges" "lr" "dibs_lr" "num_updates" "steps" "num_samples")

seeds=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20)
exp_edges=(1.0)
lrs=(0.001)
dibs_lrs=(0.005)
num_samples=(500)
num_updates=(200)
steps=(1000)

array_len=$(( ${#seeds[@]} * ${#exp_edges[@]} * ${#lrs[@]} * ${#dibs_lrs[@]} * ${#num_updates[@]} * ${#steps[@]} * ${#num_samples[@]} ))
echo $array_len
if [ ${train} == 'train' ]
then
    if [ ${dataset} == 'er' ]
    then
        config='linear_decoder_dibs'
        output_file="out/Decoder_JointDIBS/linear_decoder_joint_dibs_er-%A_%a.out"
        echo "Train Linear Decoder Joint DIBS ER: ${config}"
    else
        echo "Not implemented dataset ${dataset}" 
    fi
else
    echo "Not implemented dataset ${train}" 
fi

command="sbatch --array=1-${array_len}%512 --job-name ${config} --output ${output_file} --time ${time} scripts/run_linear_joint_decoder_dibs.sh ${config}"   
echo ""
echo ${command}
echo ""

RES=$(${command})
job_id=${RES##* }
echo "Job ID"" ""${job_id}"" -> ""${config} ${args}" >> out/job_logs.txt
echo "Job ID"" ""${job_id}"" -> ""${config} ${args}" 
echo ""


