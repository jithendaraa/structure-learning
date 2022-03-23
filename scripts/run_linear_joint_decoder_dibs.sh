#!/bin/bash
#SBATCH --mail-user=ivaxi-miteshkumar.sheth.1@ens.etsmtl.ca
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE
#SBATCH --mail-type=ALL
#SBATCH --job-name=try
#SBATCH --output=%x-%j.out
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=32
#SBATCH --mem=10GB
#SBATCH --time=00:05:00
#SBATCH --account=rrg-ebrahimi
​#wandb login b0ebb8272d653169adb078a4e3f70cb1ebbf41c0 

wandb offline

seeds=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20)
lrs=(0.001)
dibs_lrs=(0.005)
exp_edges=(0.5)
steps=(11000)

num_updates=(1000)
num_samples=(100)
num_obs_data=50
num_nodes=4
proj_dims=10
across_interv='True'

n_particles=20
datagen='linear'
likelihood='linear'
algo='def'
off_wandb='True'
clamp='False'
reinit='False'

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
echo "python run_decoder_dibs.py --config defaults ${id} --data_seed ${seed} --exp_edges ${exp_edge} --lr ${lr} --dibs_lr ${dibs_lr} --num_updates ${num_update} --steps ${step} --num_samples ${num_sample} --num_nodes ${num_nodes} --proj_dims ${proj_dims} --algo ${algo} --n_particles ${n_particles} --obs_data ${num_obs_data} --across_interv ${across_interv} --datagen ${datagen} --off_wandb ${off_wandb} --clamp ${clamp} --likelihood ${likelihood}"

cd exps/decoder_dibs_exps
python run_decoder_dibs.py --config defaults ${id} --data_seed ${seed} --exp_edges ${exp_edge} --lr ${lr} --dibs_lr ${dibs_lr} --num_updates ${num_update} --steps ${step} --num_samples ${num_sample} --num_nodes ${num_nodes} --proj_dims ${proj_dims} --algo ${algo} --n_particles ${n_particles} --obs_data ${num_obs_data} --across_interv ${across_interv} --datagen ${datagen} --off_wandb ${off_wandb} --clamp ${clamp} --likelihood ${likelihood}
cd ../..

echo $end
end=`date +%s`
runtime=$((end-start))
echo "Program time: $runtime"