#!/bin/bash
exp_id=$1   # [0, 1, 2, 3, 4]
dataset=${2:-'clevr'}  # ['clevr', 'er']
train=${3:-'train'}
def_time='23:00:00'
time=${4:-$def_time}

if [ ${exp_id} == '0' ]
then 
    # Run Slot_attention_img
    bash script_runners/sa_img_job_run.sh ${dataset} ${train} 'SlotAttention_img' ${time}
elif [ ${exp_id} == '1' ]
then
    # Run VCN
    bash script_runners/vcn_job_run.sh ${dataset} ${train} 'VCN' ${time}
  
elif [ ${exp_id} == '2' ]
then
    # Run VCN_img
    bash script_runners/vcn_img_job_run.sh ${dataset} ${train} 'VCN_img' ${time}

elif [ ${exp_id} == '3' ]
then
    # Run Slot_VCN_img (1d; using just ELBO graph + MSE loss)
    bash script_runners/slot_vcn_img_job_run.sh ${dataset} ${train} 'Slot_VCN_img' ${time}

elif [ ${exp_id} == '4' ]
then
    # Run DIBS
    bash script_runners/dibs_job_run.sh ${dataset} ${train} 'DIBS' ${time}
elif [ ${exp_id} == '5' ]
then
    # Run DIBS
    bash script_runners/vae_dibs_job_run.sh ${dataset} ${train} 'VAE_DIBS' ${time}
elif [ ${exp_id} == '6' ]
then
    # Run DIBS
    bash script_runners/decoder_dibs_job_run.sh ${dataset} ${train} 'Decoder_DIBS' ${time}
fi