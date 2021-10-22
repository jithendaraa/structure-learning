#!/bin/bash

# 1. Train SA_img on clevr
dataset=$1
train=$2
model=$3
time=$4



if [ ${train} == 'train' ]
then
    if [ ${dataset} == 'clevr' ]
    then
        # Train S3VAE on MMNIST
        config='train_clevr_sa_img'
        output_file="out/SA_img/""$config""-%j.out"
        echo "Train SA_img CLEVR: ${config}"
    else
        echo "Not implemented dataset ${dataset}" 
    fi
else
    echo "Not implemented dataset ${train}" 
fi

start=`date +%s`
command="sbatch --job-name ${config} --output ${output_file} --time ${time} scripts/run_sa_img.sh ${config}"      
echo ""
echo ${command}
echo ""

RES=$(${command})
job_id=${RES##* }
echo "Job ID"" ""${job_id}"" -> ""${config}" >> out/job_logs.txt
echo "Job ID"" ""${job_id}"" -> ""${config}" 
echo ""

end=`date +%s`
runtime=$((end-start))
echo "Program time: $runtime s"