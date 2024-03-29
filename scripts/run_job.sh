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
args={$2:-''}
start=`date +%s`
echo "Script"

# cp /home/jithen/scratch/datasets/CLEVR_v1.0.zip $SLURM_TMPDIR/CLEVR_v1.0.zip
# unzip $SLURM_TMPDIR/CLEVR_v1.0.zip
act_causal
module load python/3.7
echo `date` "Python starting"
echo "python main.py --config defaults "${id}" "${args}
python main.py --config defaults ${id} ${args} --dibs_lr 1e-2

echo $end
end=`date +%s`
runtime=$((end-start))
echo "Program time: $runtime"