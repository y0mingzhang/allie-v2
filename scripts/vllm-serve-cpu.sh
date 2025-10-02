#!/bin/bash

#SBATCH --job-name="allie"
#SBATCH --partition="preempt"
#SBATCH --time=8:00:00
#SBATCH --cpus-per-task=24
#SBATCH --mem=50GB
#SBATCH --signal=B:SIGUSR1@60

function sig_handler_USR1()
{
echo "   Signal trapped -  `date`"
echo "   Requeueing job id" $SLURM_JOB_ID
scontrol requeue $SLURM_JOB_ID
}
trap 'sig_handler_USR1' SIGUSR1

source .venv/bin/activate

export VLLM_CPU_KVCACHE_SPACE=25
export VLLM_CPU_NUM_OF_RESERVED_CPU=1

vllm serve yimingzhang/qwen-3-1.7b-57b-cool-from-66550-step96800 --max-model-len 1024 --enforce-eager

cd src/lichess-bot
python lichess-bot.py --config config.yml &

wait
