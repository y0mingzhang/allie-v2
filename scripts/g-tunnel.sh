#!/bin/bash

#SBATCH --job-name="g-tunnel"
#SBATCH --partition="general"
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:8
#SBATCH --constraint=A100_40GB
#SBATCH --cpus-per-task=48
#SBATCH --mem=400GB
#SBATCH --signal=B:SIGUSR1@60
#SBATCH --mail-user=yimingz3@cs.cmu.edu
#SBATCH --requeue

CLAUDE_SESSION="8ca14d23-3c9b-4a67-8a4e-fd5fd703fdae"

function sig_handler_USR1()
{
echo "   Signal trapped -  `date`"
echo "   Requeueing job id" $SLURM_JOB_ID
scontrol requeue $SLURM_JOB_ID
}
trap 'sig_handler_USR1' SIGUSR1

/usr/sbin/sshd -D -p 2222 -f /dev/null -h ${HOME}/.ssh/id_rsa &

cd /home/yimingz3/src/chess-v3

PROMPT="You are resuming after a SLURM job restart. Check Slack #yiming-and-claude for any instructions, then continue working on chess-v3 autonomously. Do not stop — keep running experiments, checking results, and iterating."

tmux new-session -d -s claude -x 200 -y 50
tmux send-keys -t claude "cd /home/yimingz3/src/chess-v3 && echo '$PROMPT' | claude --resume $CLAUDE_SESSION --dangerously-skip-permissions" Enter

wait
