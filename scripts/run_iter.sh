#!/bin/bash
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <integer1> <integer2>"
    exit 1
fi

from_iter=$1
to_iter=$2

# TODO: modify this for your system
source /home/kzorina/miniforge3/etc/profile.d/conda.sh
conda activate gtamp
export PYTHONPATH=/home/kzorina/work/repos/video_guided_tamp_planner:/home/kzorina/work/repos/guided_tamp_benchmark:$PYTHONPATH

# Loop to run the command
for ((i=from_iter; i<=to_iter; i++))
do
  echo "Running iteration $i of $iterations"
  python 02_solve_task.py -planner rrt_star_connect -seed $i -task_id 0 -pose_id 1
done
