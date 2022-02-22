#!/bin/bash
#SBATCH --time=27:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mail-user=jithen.subra@gmail.com
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE
#SBATCH --mail-type=ALL

seeds=(1)
exp_edges=(0.5)
lrs=(0.001)
dibs_lrs=(0.005)
num_updates=(1000)
steps=(10000)

num_samples=(120)
num_obs_data=20
num_nodes=4
proj_dims=10
across_interv='False'

n_particles=20
off_wandb='False'

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
echo "python run_dibs.py --config defaults ${id} --data_seed ${seed} --exp_edges ${exp_edge} --lr ${lr} --dibs_lr ${dibs_lr} --num_updates ${num_update} --num_samples ${num_sample} --num_nodes ${num_nodes} --algo ${algo} --n_particles ${n_particles} --obs_data ${num_obs_data} --across_interv ${across_interv}"

cd exps/dibs_exps
python run_dibs.py --config defaults ${id} --data_seed ${seed} --exp_edges ${exp_edge} --lr ${lr} --dibs_lr ${dibs_lr} --num_updates ${num_update} --num_samples ${num_sample} --num_nodes ${num_nodes} --n_particles ${n_particles} --obs_data ${num_obs_data} --across_interv ${across_interv} --off_wandb ${off_wandb}
cd ../..

echo $end
end=`date +%s`
runtime=$((end-start))
echo "Program time: $runtime"