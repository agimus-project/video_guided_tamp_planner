#!/bin/bash
#SBATCH --job-name=rrt_connect   # Job name
#SBATCH --output=logs/job_output_%A_%a.log # Output log file, %A is the job ID and %a is the array task ID
#SBATCH --error=logs/job_error_%A_%a.log   # Error log file
#SBATCH --ntasks=1                    # Number of tasks (1 Python process per job)
#SBATCH --cpus-per-task=32            # Number of cores per task
#SBATCH --mem=64G                     # Memory per node
#SBATCH --time=15:00:00              # Max runtime (1 hour in this example)
#SBATCH --array=1-5                 # Create a job array with 5 tasks (1-5)
#SBATCH --nodes=1                     # Request 1 node per task
#SBATCH --exclusive                   # Request exclusive use of the node

from_iter=$((1400 + (SLURM_ARRAY_TASK_ID - 1) * 80))
to_iter=$((1400 + SLURM_ARRAY_TASK_ID * 80))

# TODO: modify this for your system
. ~/miniconda3/etc/profile.d/conda.sh
conda activate gtamp
export PYTHONPATH=~/video_guided_tamp_planner:~/guided_tamp_benchmark:$PYTHONPATH


# Loop to run the command
for ((i=from_iter; i<to_iter; i++))
do
  echo "Running iteration $i of $to_iter"
  python 02_solve_task.py -planner rrt_star_connect -seed $i -task_id 0 -pose_id 1
done