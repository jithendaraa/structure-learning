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

seeds=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20)
lrs=(0.002)
num_steps=(1000)

num_samples=(181000)
obs_data=1000
num_nodes=20
proj_dims=100
exp_edges=(1.0)

off_wandb='False'
n_interv_sets=20
batches=5000

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

# cp /home/jithen/scratch/datasets/CLEVR_v1.0.zip $SLURM_TMPDIR/CLEVR_v1.0.zip
# unzip $SLURM_TMPDIR/CLEVR_v1.0.zip
module load anaconda/3
module unload cuda/11.2 && module load cuda/11.0
conda activate lbcd
echo `date` "Python starting"
echo "python batched_run_decoder_bcd.py --config defaults ${id} --data_seed ${seed} --exp_edges ${exp_edge} --lr ${lr} --num_steps ${step} --num_samples ${num_sample} --num_nodes ${num_nodes} --proj_dims ${proj_dims} --obs_data ${obs_data} --off_wandb ${off_wandb} --n_interv_sets ${n_interv_sets} --batches ${batches}"

cd exps/decoder_bcd_exps
python batched_run_decoder_bcd.py --config defaults ${id} --data_seed ${seed} --exp_edges ${exp_edge} --lr ${lr} --num_steps ${step} --num_samples ${num_sample} --num_nodes ${num_nodes} --proj_dims ${proj_dims} --obs_data ${obs_data} --off_wandb ${off_wandb} --n_interv_sets ${n_interv_sets} --batches ${batches}
cd ../..

echo $end
end=`date +%s`
runtime=$((end-start))
echo "Program time: $runtime"