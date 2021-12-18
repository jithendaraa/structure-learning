#!/bin/bash

dataset=$1
train=$2
model=$3
time=$4

keys=("seed" "exp_edges" "lr" "dibs_lr" "num_updates" "steps" "num_samples")

# seeds=(1)
# exp_edges=(1.0 1.5 2.0)
# lrs=(0.0001 0.0003 0.0008 0.001)
# dibs_lrs=(0.0003 0.0008 0.001 0.003 0.005)
# num_updates=(100 500 1000 3000)
# steps=(3000 5000 10000 50000 100000)
# num_samples=(500 1000 2000 3000)

seeds=(1)
exp_edges=(1.0)
lrs=(0.0001)
dibs_lrs=(0.0003)
num_updates=(100)
steps=(3000)
num_samples=(500)

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

# Iterate the string array using for loop
for seed in ${seeds[*]}; do
    for exp_edge in ${exp_edges[*]}; do
        for lr in ${lrs[*]}; do
            for dibs_lr in ${dibs_lrs[*]}; do 
                for dibs_updates in ${num_updates[*]}; do
                    for step in ${steps[*]}; do
                        for num_sample in ${num_samples[*]}; do
                            args="
                            command="sbatch --job-name ${config} --output ${output_file} --time ${time} scripts/search_decoder_dibs_run_job.sh ${config} ${seed} ${exp_edge} ${lr} ${dibs_lr} ${dibs_updates} ${step} ${num_sample}"      
                            echo ""
                            echo ${command}
                            echo ""

                            RES=$(${command})
                            job_id=${RES##* }
                            echo "Job ID"" ""${job_id}"" -> ""${config} ${args}" >> out/job_logs.txt
                            echo "Job ID"" ""${job_id}"" -> ""${config} ${args}" 
                            echo ""
                        done
                    done
                done
            done
        done
    done
done


