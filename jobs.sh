#!/bin/bash
exp_id=$1   # 0 for SA_img
dataset=${2:-'clevr'}  # ['clevr']
train=${3:-'train'}
def_time='23:00:00'
time=${4:-$def_time}


if [ ${exp_id} == '0' ]
then 
    # Run Slot_attention_img
    bash script_runners/sa_img_job_run.sh ${dataset} ${train} 'SlotAttention_img' ${time}
fi