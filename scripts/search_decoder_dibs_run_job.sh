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

id=$1
seed=$2
exp_edges=$3
lr=$4
dibs_lr=$5
dibs_updates=$6
steps=$7
num_samples=$8
start=`date +%s`
echo "Script"

# cp /home/jithen/scratch/datasets/CLEVR_v1.0.zip $SLURM_TMPDIR/CLEVR_v1.0.zip
# unzip $SLURM_TMPDIR/CLEVR_v1.0.zip
act_causal
module load python/3.7
echo `date` "Python starting"
echo "python main.py --config defaults ${id} --data_seed ${seed} --exp_edges ${exp_edges} --lr ${lr} --dibs_lr ${dibs_lr} --num_updates ${num_updates} --steps ${steps} --num_samples ${num_samples}"

python main.py --config defaults ${id} --data_seed ${seed} --exp_edges ${exp_edges} --lr ${lr} --dibs_lr ${dibs_lr} --num_updates ${num_updates} --steps ${steps} --num_samples ${num_samples}

echo $end
end=`date +%s`
runtime=$((end-start))
echo "Program time: $runtime"