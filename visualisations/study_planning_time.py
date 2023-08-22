from visualisations.utils import get_res_file
import matplotlib.pyplot as plt
import dill
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "-res_file",
    type=str,
    default="results.pkl",
    help="Path to file with the saved results (w.r.t data_folder)",
)
parser.add_argument(
    "-task_name", type=str, default="shelf", help="String name of the task"
)
parser.add_argument("-task_id", type=int, default=0, help="Int task id")
parser.add_argument("-robot_name", type=str, default="panda", help="String robot name")
parser.add_argument(
    "-planner", type=str, default="multi_contact", help="String plannner name"
)
args = parser.parse_args()

file = get_res_file(args.res_file)

res_dict = dill.load(open(file, "rb"))

x = []
y = []
for p in range(1, 11):
    for k, v in res_dict[args.planner][args.task_name][args.task_id][args.robot_name][
        p
    ].items():
        t = v.computation_time if v.is_solved else 0
        x.append(f"{p}.{k}")
        y.append(t)

fig, ax = plt.subplots()

ax.plot(x, y)
for i, label in enumerate(ax.xaxis.get_ticklabels()):
    if i % 3 < 2:
        label.set_visible(False)
ax.set_xlabel("Pose.seed")
ax.set_ylabel("Time (s)")
plt.xticks(rotation=90)
fig.tight_layout()
plt.show()
