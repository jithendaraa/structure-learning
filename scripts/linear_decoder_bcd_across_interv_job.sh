#!/bin/bash
#SBATCH --time=27:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6

seeds=(3 8 10 13 15 18)
exp_edges=(1.0)
lrs=(0.002)
num_steps=(5000)

num_samples=(3600)
obs_data=300
num_nodes=6
proj_dims=10
off_wandb='False'

interv_type='single'
interv_value=100.0
P_KL='False'
L_KL='False'

# Dont change these
obs_Z_KL='False'
across_interv='True'


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
echo "python run_decoder_bcd_across_interv.py --config defaults ${id} --data_seed ${seed} --exp_edges ${exp_edge} --lr ${lr} --num_steps ${step} --num_samples ${num_sample} --num_nodes ${num_nodes} --proj_dims ${proj_dims} --obs_data ${obs_data} --off_wandb ${off_wandb} --L_KL ${L_KL} --P_KL ${P_KL} --across_interv ${across_interv} --interv_type ${interv_type} --interv_value ${interv_value} --obs_Z_KL ${obs_Z_KL}"

cd exps/decoder_bcd_exps
python run_decoder_bcd_across_interv.py --config defaults ${id} --data_seed ${seed} --exp_edges ${exp_edge} --lr ${lr} --num_steps ${step} --num_samples ${num_sample} --num_nodes ${num_nodes} --proj_dims ${proj_dims} --obs_data ${obs_data} --off_wandb ${off_wandb} --L_KL ${L_KL} --P_KL ${P_KL} --across_interv ${across_interv} --interv_type ${interv_type} --interv_value ${interv_value} --obs_Z_KL ${obs_Z_KL}
cd ../..

echo $end
end=`date +%s`
runtime=$((end-start))
echo "Program time: $runtime"