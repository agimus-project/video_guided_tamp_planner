import argparse
import pickle
import numpy as np
import dill
from utils import get_res_file

parser = argparse.ArgumentParser()
parser.add_argument(
    "-save_file_base",
    type=str,
    default="results",
    help="String base name for save results file",
)
parser.add_argument(
    "-res_file",
    type=str,
    default="results.pkl",
    help="Path to file with the saved results (w.r.t data folder)",
)
args = parser.parse_args()

res_filename = get_res_file(args.res_file)

with open(res_filename, "rb") as f:
    data = dill.load(f)

tasks = [('shelf', 2)]
planner = 'multi_contact'
robot = 'panda'

for task, id in tasks:
    print(f"{task} {id}")
    for pose in data[planner][task][id][robot].keys():
        all_res = [v for seed, v in data[planner][task][id][robot][pose].items()]
        pos_res = [res for res in all_res if res.is_solved]
        print(f"Pose {pose} is solved {len(pos_res)} out of {len(all_res)} ("
              f"{[res.is_solved for res in all_res]})")


