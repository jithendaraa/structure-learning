#!/bin/bash
exp_id=$1   # [0 - 6, 6s, 7s]
dataset=${2:-'clevr'}  # ['clevr', 'er']
train=${3:-'train'}
def_time='23:00:00'
time=${4:-$def_time}

# ! Run Slot_attention_img
if [ ${exp_id} == '0' ]     
then 
    bash script_runners/sa_img_job_run.sh ${dataset} ${train} 'SlotAttention_img' ${time}

# ! Run VCN
elif [ ${exp_id} == '1' ]   
then
    bash script_runners/vcn_job_run.sh ${dataset} ${train} 'VCN' ${time}

# ! Run VCN_img
elif [ ${exp_id} == '2' ]   
then
    bash script_runners/vcn_img_job_run.sh ${dataset} ${train} 'VCN_img' ${time}

# ! Run Slot_VCN_img (1d; using just ELBO graph + MSE loss)
elif [ ${exp_id} == '3' ]   
then
    bash script_runners/slot_vcn_img_job_run.sh ${dataset} ${train} 'Slot_VCN_img' ${time}

# ! Run DIBS
elif [ ${exp_id} == '4' ]   
then
    bash script_runners/dibs_job_run.sh ${dataset} ${train} 'DIBS' ${time}

# ! Run VAE_DIBS
elif [ ${exp_id} == '5' ]   
then
    bash script_runners/vae_dibs_job_run.sh ${dataset} ${train} 'VAE_DIBS' ${time}

# ! Run Decoder DIBS
elif [ ${exp_id} == '6' ]   
then
    bash script_runners/decoder_dibs_job_run.sh ${dataset} ${train} 'Decoder_DIBS' ${time}

elif [ ${exp_id} == '6s' ]   # ! Search over hyperparams Decoder DIBS
then
    bash script_runners/temp.sh ${dataset} ${train} 'Decoder_DIBS' ${time}

# !  linear Joint DIBS (across interv data) - Search over hyperparams
elif [ ${exp_id} == '7s' ]   
then
    bash script_runners/joint_linear_dibs.sh ${dataset} ${train} 'DIBS' ${time}

# !  nonlinear Joint DIBS (across interv data) - Search over hyperparams
elif [ ${exp_id} == '8s' ]   
then
    bash script_runners/joint_nonlinear_dibs.sh ${dataset} ${train} 'DIBS' ${time}

# ! Linear Decoder Joint DIBS (across interv data) - Search over hyperparams
elif [ ${exp_id} == '9s' ]   
# ! ./jobs.sh 9s er train 1:00:00
then
    bash script_runners/joint_linear_decoder_dibs.sh ${dataset} ${train} 'Decoder_JointDiBS' ${time}

# ! Nonlinear Decoder Joint DIBS (across interv data) - Search over hyperparams
elif [ ${exp_id} == '10s' ]   
then
    bash script_runners/decoder_joint_dibs_interv_job_run.sh ${dataset} ${train} 'Decoder_JointDiBS' ${time}

# ! linear decoder BCD - Search over hyperparams
elif [ ${exp_id} == '11s' ]   
then
    bash script_runners/linear_decoder_bcd_job_run.sh ${dataset} ${train} 'Decoder_BCD' ${time}

fi