#!/bin/bash
#SBATCH --time=27:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6

seeds=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20)
lrs=(0.001)
dibs_lrs=(0.005)

exp_edges=(1.0)
num_updates=(4000)
num_nodes=20
num_samples=(500)
num_obs_data=300
across_interv='True'
grad_estimator='reparam'

n_particles=20
proj_dims=10
off_wandb='False'
interv_type='single'
steps=(10000)

array_len=$(( ${#exp_edges[@]} * ${#lrs[@]} * ${#dibs_lrs[@]} * ${#num_updates[@]} * ${#steps[@]} * ${#num_samples[@]} ))
echo $array_len

bcdefg=$(( ${#exp_edges[@]} * ${#lrs[@]} * ${#dibs_lrs[@]} * ${#num_updates[@]} * ${#steps[@]} * ${#num_samples[@]} ))
cdefg=$(( ${#lrs[@]} * ${#dibs_lrs[@]} * ${#num_updates[@]} * ${#steps[@]} * ${#num_samples[@]} ))
defg=$(( ${#dibs_lrs[@]} * ${#num_updates[@]} * ${#steps[@]} * ${#num_samples[@]} ))
efg=$(( ${#num_updates[@]} * ${#steps[@]} * ${#num_samples[@]} ))
fg_=$(( ${#steps[@]} * ${#num_samples[@]} ))
g=$(( ${#num_samples[@]} ))

id=$1
seed=${seeds[ $((  (${SLURM_ARRAY_TASK_ID} / ${bcdefg})  % ${#seeds[@]} )) ]}
exp_edge=${exp_edges[ $((  (${SLURM_ARRAY_TASK_ID} / ${cdefg}) % ${#exp_edges[@]} )) ]}
lr=${lrs[ $((  (${SLURM_ARRAY_TASK_ID} / ${defg}) % ${#lrs[@]} )) ]}
dibs_lr=${dibs_lrs[ $((  (${SLURM_ARRAY_TASK_ID} / ${efg}) % ${#dibs_lrs[@]} )) ]}
num_update=${num_updates[ $((  (${SLURM_ARRAY_TASK_ID} / ${fg_}) % ${#num_updates[@]} )) ]}
step=${steps[ $((  (${SLURM_ARRAY_TASK_ID} / ${g}) % ${#steps[@]} )) ]}
num_sample=${num_samples[ $((  ${SLURM_ARRAY_TASK_ID} % ${#num_samples[@]} )) ]}

start=`date +%s`
echo "Script"

act_causal
module load python/3.7
echo `date` "Python starting"
echo "python run_dibs.py --config defaults ${id} --data_seed ${seed} --exp_edges ${exp_edge} --lr ${lr} --dibs_lr ${dibs_lr} --num_updates ${num_update} --num_samples ${num_sample} --num_nodes ${num_nodes} --algo ${algo} --n_particles ${n_particles} --obs_data ${num_obs_data} --across_interv ${across_interv} --interv_type ${interv_type}"

cd exps/dibs_exps
python run_dibs.py --config defaults ${id} --data_seed ${seed} --exp_edges ${exp_edge} --lr ${lr} --dibs_lr ${dibs_lr} --num_updates ${num_update} --num_samples ${num_sample} --num_nodes ${num_nodes} --n_particles ${n_particles} --obs_data ${num_obs_data} --across_interv ${across_interv} --off_wandb ${off_wandb} --interv_type ${interv_type}
cd ../..

echo $end
end=`date +%s`
runtime=$((end-start))
echo "Program time: $runtime"