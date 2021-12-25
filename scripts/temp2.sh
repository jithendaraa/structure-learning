#!/bin/bash
#SBATCH --time=27:00:00
#SBATCH --account=rrg-ebrahimi
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --job-name=MovingMNIST
#SBATCH --cpus-per-task=6
#SBATCH --mail-user=jithen.subra@gmail.com
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE
#SBATCH --mail-type=ALL

seeds=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20)
exp_edges=(1.0)
lrs=(0.0003)
dibs_lrs=(0.0003)
num_samples=(500)
num_updates=(500)
steps=(5000)

num_nodes=4
proj_dims=8
algo='def'
n_particles=2

array_len=$(( ${#exp_edges[@]} * ${#lrs[@]} * ${#dibs_lrs[@]} * ${#num_updates[@]} * ${#steps[@]} * ${#num_samples[@]} ))

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

# cp /home/jithen/scratch/datasets/CLEVR_v1.0.zip $SLURM_TMPDIR/CLEVR_v1.0.zip
# unzip $SLURM_TMPDIR/CLEVR_v1.0.zip
act_causal
module load python/3.7
echo `date` "Python starting"
echo "python main.py --config defaults ${id} --data_seed ${seed} --exp_edges ${exp_edge} --lr ${lr} --dibs_lr ${dibs_lr} --num_updates ${num_update} --steps ${step} --num_samples ${num_sample} --num_nodes ${num_nodes} --proj_dims ${proj_dims} --algo ${algo} --n_particles ${n_particles}"

python main.py --config defaults ${id} --data_seed ${seed} --exp_edges ${exp_edge} --lr ${lr} --dibs_lr ${dibs_lr} --num_updates ${num_update} --steps ${step} --num_samples ${num_sample} --num_nodes ${num_nodes} --proj_dims ${proj_dims} --algo ${algo} --n_particles ${n_particles}

echo $end
end=`date +%s`
runtime=$((end-start))
echo "Program time: $runtime"