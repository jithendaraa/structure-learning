#!/bin/bash
#SBATCH --time=27:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6

seeds=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20)
exp_edges=(1.0)
lrs=(0.002)
num_steps=(5000)

num_samples=(2400)
obs_data=2400
num_nodes=6
proj_dims=6

learn_L='True'
Z_KL='joint'
interv_Z_KL='False'
obs_Z_KL='False'
use_proxy='True'
fix_edges='False'
proj_sparsity=1.0
identity_proj='True'

L_KL='False'
P_KL='False'
off_wandb='False'
learn_noise='False'
learn_P='False'
fix_decoder='False'
train_loss='mse'    
decoder_layers='linear'
across_interv='False'
reg_decoder='False'
interv_type='single'
interv_value=100.0


array_len=$(( ${#exp_edges[@]} * ${#lrs[@]} * ${#num_steps[@]} * ${#num_samples[@]} ))
defg=$(( ${#exp_edges[@]} * ${#lrs[@]} * ${#num_steps[@]} * ${#num_samples[@]} ))
efg=$(( ${#lrs[@]} * ${#num_steps[@]} * ${#num_samples[@]} ))
fg=$(( ${#num_steps[@]} * ${#num_samples[@]} ))
g=$(( ${#num_samples[@]} ))

id=$1
seed=${seeds[ $((  (${SLURM_ARRAY_TASK_ID} / ${defg})  % ${#seeds[@]} )) ]}
exp_edge=${exp_edges[ $((  (${SLURM_ARRAY_TASK_ID} / ${efg}) % ${#exp_edges[@]} )) ]}
lr=${lrs[ $((  (${SLURM_ARRAY_TASK_ID} / ${fg}) % ${#lrs[@]} )) ]}
step=${num_steps[ $((  (${SLURM_ARRAY_TASK_ID} / ${g}) % ${#num_steps[@]} )) ]}
num_sample=${num_samples[ $((  ${SLURM_ARRAY_TASK_ID} % ${#num_samples[@]} )) ]}

start=`date +%s`
echo "Script"

module load anaconda/3
module unload cuda/11.2 && module load cuda/11.0
deactivate
act_bcd
echo `date` "Python starting"
echo "python controlled_supervision.py --config defaults ${id} --data_seed ${seed} --exp_edges ${exp_edge} --lr ${lr} --num_steps ${step} --num_samples ${num_sample} --num_nodes ${num_nodes} --proj_dims ${proj_dims} --obs_data ${obs_data} --off_wandb ${off_wandb} --train_loss ${train_loss} --decoder_layers ${decoder_layers} --learn_L ${learn_L} --learn_P ${learn_P} --L_KL ${L_KL} --P_KL ${P_KL} --Z_KL ${Z_KL} --learn_noise ${learn_noise} --interv_Z_KL ${interv_Z_KL} --obs_Z_KL ${obs_Z_KL} --across_interv ${across_interv} --reg_decoder ${reg_decoder} --interv_type ${interv_type} --interv_value ${interv_value} --fix_edges ${fix_edges} --use_proxy ${use_proxy} --proj_sparsity ${proj_sparsity} --identity_proj ${identity_proj}"

cd exps/decoder_bcd_exps
python controlled_supervision.py --config defaults ${id} --data_seed ${seed} --exp_edges ${exp_edge} --lr ${lr} --num_steps ${step} --num_samples ${num_sample} --num_nodes ${num_nodes} --proj_dims ${proj_dims} --obs_data ${obs_data} --off_wandb ${off_wandb} --train_loss ${train_loss} --decoder_layers ${decoder_layers} --learn_L ${learn_L} --learn_P ${learn_P} --L_KL ${L_KL} --P_KL ${P_KL} --Z_KL ${Z_KL} --learn_noise ${learn_noise} --interv_Z_KL ${interv_Z_KL} --obs_Z_KL ${obs_Z_KL} --across_interv ${across_interv} --reg_decoder ${reg_decoder} --interv_type ${interv_type} --interv_value ${interv_value} --fix_edges ${fix_edges} --use_proxy ${use_proxy} --proj_sparsity ${proj_sparsity} --identity_proj ${identity_proj}
cd ../..

echo $end
end=`date +%s`
runtime=$((end-start))
echo "Program time: $runtime"