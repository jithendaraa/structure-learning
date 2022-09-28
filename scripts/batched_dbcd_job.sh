#!/bin/bash
#SBATCH --time=27:00:00
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6

seeds=(18)
lrs=(0.0008)
num_steps=(8000)

num_nodes=10
proj_dims=100
exp_edges=(4.0)
n_interv_sets=200

pts_per_interv=100
off_wandb='False'
batches=90000
obs_data=500
num_samples=(181000)

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
conda activate lbcd
echo `date` "Python starting"
echo "python batched_run_decoder_bcd.py --config defaults ${id} --data_seed ${seed} --exp_edges ${exp_edge} --lr ${lr} --num_steps ${step} --num_samples ${num_sample} --num_nodes ${num_nodes} --proj_dims ${proj_dims} --obs_data ${obs_data} --off_wandb ${off_wandb} --n_interv_sets ${n_interv_sets} --batches ${batches} --pts_per_interv ${pts_per_interv}"

cd exps/decoder_bcd_exps
python batched_run_decoder_bcd.py --config defaults ${id} --data_seed ${seed} --exp_edges ${exp_edge} --lr ${lr} --num_steps ${step} --num_samples ${num_sample} --num_nodes ${num_nodes} --proj_dims ${proj_dims} --obs_data ${obs_data} --off_wandb ${off_wandb} --n_interv_sets ${n_interv_sets} --batches ${batches} --pts_per_interv ${pts_per_interv}
cd ../..

echo $end
end=`date +%s`
runtime=$((end-start))
echo "Program time: $runtime"