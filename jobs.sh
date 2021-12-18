#!/bin/bash
exp_id=$1   # [0 - 6]
dataset=${2:-'clevr'}  # ['clevr', 'er']
train=${3:-'train'}
def_time='23:00:00'
time=${4:-$def_time}

if [ ${exp_id} == '0' ]     # ! Run Slot_attention_img
then 
    bash script_runners/sa_img_job_run.sh ${dataset} ${train} 'SlotAttention_img' ${time}
elif [ ${exp_id} == '1' ]   # ! Run VCN
then
    bash script_runners/vcn_job_run.sh ${dataset} ${train} 'VCN' ${time}
elif [ ${exp_id} == '2' ]   # ! Run VCN_img
then
    bash script_runners/vcn_img_job_run.sh ${dataset} ${train} 'VCN_img' ${time}

elif [ ${exp_id} == '3' ]   # ! Run Slot_VCN_img (1d; using just ELBO graph + MSE loss)
then
    bash script_runners/slot_vcn_img_job_run.sh ${dataset} ${train} 'Slot_VCN_img' ${time}

elif [ ${exp_id} == '4' ]   # ! Run DIBS
then
    bash script_runners/dibs_job_run.sh ${dataset} ${train} 'DIBS' ${time}
elif [ ${exp_id} == '5' ]   # ! Run VAE_DIBS
then
    bash script_runners/vae_dibs_job_run.sh ${dataset} ${train} 'VAE_DIBS' ${time}
elif [ ${exp_id} == '6' ]   # ! Run Decoder DIBS
then
    bash script_runners/decoder_dibs_job_run.sh ${dataset} ${train} 'Decoder_DIBS' ${time}
elif [ ${exp_id} == '6s' ]   # ! Search over hyperparams Decoder DIBS
then
    bash script_runners/temp.sh ${dataset} ${train} 'Decoder_DIBS' ${time}
fi