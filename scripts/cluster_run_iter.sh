#!/bin/bash
#SBATCH --job-name=rrt_connect   # Job name
#SBATCH --output=logs/job_output_%A_%a.log # Output log file, %A is the job ID and %a is the array task ID
#SBATCH --error=logs/job_error_%A_%a.log   # Error log file
#SBATCH --ntasks=1                    # Number of tasks (1 Python process per job)
#SBATCH --cpus-per-task=16            # Number of cores per task
#SBATCH --mem=64G                     # Memory per node
#SBATCH --time=15:00:00              # Max runtime (1 hour in this example)
#SBATCH --nodes=1                     # Request 1 node per task
#SBATCH --array=1-10                  # Job array with 5 tasks
#SBATCH --nodelist=node-01,node-02,node-03,node-04,node-05,node-06,node-07,node-08,node-09,node-10  # Specific nodes

# 5 nodes:
#SBATCH --nodelist=node-02,node-03,node-04,node-05,node-06  # Specific nodes
# NODES=("node-02" "node-03" "node-04" "node-05" "node-06" )

# 10 nodes:
#SBATCH --nodelist=node-01,node-02,node-03,node-04,node-05,node-06,node-07,node-08,node-09,node-10  # Specific nodes
# NODES=("node-01" "node-02" "node-03" "node-04" "node-05" "node-06" "node-07" "node-08" "node-09" "node-10")

NODES=("node-01" "node-02" "node-03" "node-04" "node-05" "node-06" "node-07" "node-08" "node-09" "node-10")
TASK_INDEX=$((SLURM_ARRAY_TASK_ID - 1))  # Convert SLURM ID (1-based) to 0-based index
NODE_TO_USE=${NODES[$TASK_INDEX]}        # Select the node for this job
echo "Running on $NODE_TO_USE"
export SLURM_NODELIST=$NODE_TO_USE       # Override assigned node


from_iter=$((1800 + (SLURM_ARRAY_TASK_ID - 1) * 80))
to_iter=$((1800 + SLURM_ARRAY_TASK_ID * 80))

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