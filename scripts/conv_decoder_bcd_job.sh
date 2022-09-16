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

seeds=(9 11 17)
exp_edges=(1.0)
lrs=(0.0008)
num_steps=(2000)

num_nodes=10
n_interv_sets=50
pts_per_interv=50

batches=256
generate='True'
obs_data=300
interv_type='multi'
interv_value='uniform'
off_wandb='False'


num_samples=(3300) # ! variable not used
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
module load cuda/11.0/cudnn/8.0
conda activate lbcd
echo `date` "Python starting"
echo "python run_conv_decoder_bcd.py --config defaults ${id} --data_seed ${seed} --exp_edges ${exp_edge} --lr ${lr} --num_steps ${step} --num_nodes ${num_nodes} --obs_data ${obs_data} --off_wandb ${off_wandb} --interv_type ${interv_type} --interv_value ${interv_value} --n_interv_sets ${n_interv_sets} --batches ${batches} --generate ${generate} --pts_per_interv ${pts_per_interv}"

cd exps/conv_decoder_bcd_exps
python run_conv_decoder_bcd.py --config defaults ${id} --data_seed ${seed} --exp_edges ${exp_edge} --lr ${lr} --num_steps ${step} --num_nodes ${num_nodes} --obs_data ${obs_data} --off_wandb ${off_wandb} --interv_type ${interv_type} --interv_value ${interv_value} --n_interv_sets ${n_interv_sets} --batches ${batches} --generate ${generate} --pts_per_interv ${pts_per_interv}
cd ../..

echo $end
end=`date +%s`
runtime=$((end-start))
echo "Program time: $runtime"