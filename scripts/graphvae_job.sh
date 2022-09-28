#!/bin/bash
#SBATCH --time=27:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6


seeds=(1)
exp_edges=(1.0)
lrs=(0.001)
num_steps=(300)

num_samples=(3300)
obs_data=300
num_nodes=5
n_interv_sets=20
batches=10
generate='False'

interv_type='multi'
interv_value='uniform'
off_wandb='False'

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
module load cuda/11.0/cudnn/8.0
conda activate lbcd
echo `date` "Python starting"
echo "python run_graphvae.py --config defaults ${id} --data_seed ${seed} --exp_edges ${exp_edge} --lr ${lr} --num_steps ${step} --num_samples ${num_sample} --num_nodes ${num_nodes} --obs_data ${obs_data} --off_wandb ${off_wandb} --interv_type ${interv_type} --interv_value ${interv_value} --n_interv_sets ${n_interv_sets} --batches ${batches} --generate ${generate}"

cd exps/graphvae_exps
python run_graphvae.py --config defaults ${id} --data_seed ${seed} --exp_edges ${exp_edge} --lr ${lr} --num_steps ${step} --num_samples ${num_sample} --num_nodes ${num_nodes} --obs_data ${obs_data} --off_wandb ${off_wandb} --interv_type ${interv_type} --interv_value ${interv_value} --n_interv_sets ${n_interv_sets} --batches ${batches} --generate ${generate}
cd ../..

echo $end
end=`date +%s`
runtime=$((end-start))
echo "Program time: $runtime"