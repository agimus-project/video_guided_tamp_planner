#!/usr/bin/env python

# Copyright (c) CTU -- All Rights Reserved
# Created on: 02.03.23
#     Author: David Kovar <kovarda8@fel.cvut.cz>
import argparse
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# from matplotlib.ticker import FormatStrFormatter
from visualisations.utils import get_res

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

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams.update({"font.size": 24})

robot_names = ["panda", "ur5", "kmr_iiwa"]
robot_names_str = ["Franka Emika Panda robot", "UR5 robot", "KMR iiwa robot"]
tasks_id = [("shelf", 0), ("shelf", 1), ("shelf", 2), ("tunnel", 0), ("waiter", 0)]
tasks = ["shelf_01", "shelf_02", "shelf_03", "tunnel_01", "waiter_01"]
tasks_str = ["Shelf 1", "Shelf 2", "Shelf 3", "Tunnel", "Waiter"]
methods = ["hpp", "hpp_shortcut", "pddl", "multi_contact", "multi_contact_shortcut"]
methods_str = [
    "(i) HPP (no shortcut)",
    "(ii) HPP",
    "(iii) PDDLStreams",
    "(iv) Ours (no shortcut)",
    "(v) Ours",
]
metrics = ["success_rate", "time", "path_len", "grasps"]
metrics_str = ["Success rate", "Planning time [s]", "Path length", "Number of grasps"]

methods_colors = [
    plt.get_cmap("Set3").colors[3] if i == 3 else plt.get_cmap("tab20c").colors[i]
    for i in [2, 6, 13, 3, 10]
]


def plot_bars(ax, metric, group_labels, robot_name="panda", no_legend=False):
    ax.grid(axis="y")
    barWidth = 0.15
    # if robot_name != 'kmr_iiwa':
    #     group_labels = sorted(list(set(group_labels) - set(['waiter'])))
    string_lables = [
        task_str for (task, task_str) in zip(tasks, tasks_str) if task in group_labels
    ]

    recorder_metrics = get_res(
        file_name=args.res_file,
        metric=metric,
        tasks=tasks_id,
        methods=methods,
        robot=robot_name,
    )

    r = np.empty((len(methods), len(group_labels)))
    r[0] = np.arange(len(group_labels))
    for i in range(1, r.shape[0]):
        r[i] = r[i - 1] + barWidth

    bars = [
        ax.bar(
            r[i],
            recorder_metrics[m],
            width=barWidth,
            label=None if no_legend else methods_str[i],
            color=methods_colors[i],
            zorder=2,
        )
        for i, m in enumerate(methods)
    ]
    ax.set_xticks(
        ticks=[
            i + (len(methods) - 1) / 2.0 * barWidth for i in range(len(group_labels))
        ]
    )
    ax.set_xticklabels(labels=string_lables, fontdict={"size": 24})
    if robot_name != "kmr_iiwa":
        xticks = ax.xaxis.get_major_ticks()
        xticks[4].label1.set_visible(False)
    return bars


fig, axs = plt.subplots(
    nrows=len(robot_names),
    ncols=1,  # constrained_layout=True,
    figsize=(len(metrics) * 8, 4.8 * 3),
)
# clear subplots
for ax in axs:
    ax.remove()

# add subfigure per subplot
gridspec = axs[0].get_subplotspec().get_gridspec()
subfigs = [fig.add_subfigure(gs) for gs in gridspec]

for row, subfig in enumerate(subfigs):
    subfig.suptitle(f"{robot_names_str[row]}", y=0.08, fontweight="bold")

    # create 1x3 subplots per subfig
    axes = subfig.subplots(nrows=1, ncols=len(metrics))

    for col, ax in enumerate(axes):
        bars = plot_bars(
            ax,
            metrics[col],
            tasks,
            robot_name=robot_names[row],
            no_legend=False if row == 0 and col == 1 else True,
        )
        if row == 0 and col == 0:
            legend_bars = bars
        ax.set_ylabel(metrics_str[col])
        # if col == 0:
        #     ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        # else:
        #     ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    if row == 0:
        subfig.legend(bbox_to_anchor=(0.76, 1.0), prop={"size": 26}, ncol=5)

plt.subplots_adjust(
    left=0.05, right=1 - 0.01, bottom=0.18, top=0.75, wspace=None, hspace=0.03
)

fig.savefig(f"{args.save_file_base}.pdf")
fig.savefig(f"{args.save_file_base}.png")
plt.show()
