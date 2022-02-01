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
        config='decoder_dibs_er'
        output_file="out/Decoder_DIBS/decoder_dibs_er-%A_%a.out"
        echo "Train Decoder DIBS ER: ${config}"
    else
        echo "Not implemented dataset ${dataset}" 
    fi
else
    echo "Not implemented dataset ${train}" 
fi

command="sbatch --array=1-${array_len}%512 --job-name ${config} --output ${output_file} --time ${time} scripts/temp2.sh ${config}"   
echo ""
echo ${command}
echo ""

RES=$(${command})
job_id=${RES##* }
echo "Job ID"" ""${job_id}"" -> ""${config} ${args}" >> out/job_logs.txt
echo "Job ID"" ""${job_id}"" -> ""${config} ${args}" 
echo ""


# # Iterate the string array using for loop
# for seed in ${seeds[*]}; do
#     for exp_edge in ${exp_edges[*]}; do
#         for lr in ${lrs[*]}; do
#             for dibs_lr in ${dibs_lrs[*]}; do 
#                 for dibs_updates in ${num_updates[*]}; do
#                     for step in ${steps[*]}; do
#                         for num_sample in ${num_samples[*]}; do
#                             command="sbatch --array=1-${array_len} --job-name ${config} --output ${output_file} --time ${time} scripts/search_decoder_dibs_run_job.sh ${config} ${seed} ${exp_edge} ${lr} ${dibs_lr} ${dibs_updates} ${step} ${num_sample}"      
#                             echo ""
#                             echo ${command}
#                             echo ""

#                             RES=$(${command})
#                             job_id=${RES##* }
#                             echo "Job ID"" ""${job_id}"" -> ""${config} ${args}" >> out/job_logs.txt
#                             echo "Job ID"" ""${job_id}"" -> ""${config} ${args}" 
#                             echo ""
#                         done
#                     done
#                 done
#             done
#         done
#     done
# done


