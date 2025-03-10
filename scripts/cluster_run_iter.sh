#!/bin/bash
#SBATCH --job-name=python_job_array   # Job name
#SBATCH --output=job_output_%A_%a.log # Output log file, %A is the job ID and %a is the array task ID
#SBATCH --error=job_error_%A_%a.log   # Error log file
#SBATCH --ntasks=1                    # Number of tasks (1 Python process per job)
#SBATCH --cpus-per-task=4            # Number of cores per task
#SBATCH --mem=4G                     # Memory per node
#SBATCH --time=01:00:00              # Max runtime (1 hour in this example)
#SBATCH --array=1-10                 # Create a job array with 10 tasks (1-10)

from_iter=$((1000 + (SLURM_ARRAY_TASK_ID - 1) * 5))
to_iter=$((1000 + SLURM_ARRAY_TASK_ID * 5))

# TODO: modify this for your system
. /opt/apps/software/Anaconda3/5.0.1/etc/profile.d/conda.sh
conda activate gtamp
export PYTHONPATH=~/video_guided_tamp_planner:~/guided_tamp_benchmark:$PYTHONPATH


# Loop to run the command
for ((i=from_iter; i<=to_iter; i++))
do
  echo "Running iteration $i of $to_iter"
  python 02_solve_task.py -planner rrt_star_connect -seed $i -task_id 0 -pose_id 1
done