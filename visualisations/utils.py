import numpy as np
import pathlib
import dill
from collections import defaultdict


def get_res_file(filename):
    return pathlib.Path(__file__).parent.parent.joinpath(filename)


def get_metric(to_res, from_res, metric, robot, task, demo, met, p, verbose=False):
    d = to_res[robot][f"{task}_{demo}_{p}"][met][metric]
    # d_err = to_res[robot][f"{task}_{demo}_{p}"][met][metric+ "_err"]
    if metric == "success_rate":
        t_arr = np.array(
            [int(v.is_solved) for k, v in from_res[met][task][demo][robot][p].items()]
        )
        # if met == 'multi_contact_shortcut' and \
        #         'multi_contact' in to_res[robot][f"{task}_{demo}_{p}"].keys():
        #     mc_res = to_res[robot][f"{task}_{demo}_{p}"]['multi_contact'][metric]
        #
        #     if np.mean(t_arr) < np.mean(mc_res):
        #         t_arr = mc_res
    elif metric == "time":
        t_arr = np.array(
            [
                float(v.computation_time)
                for k, v in from_res[met][task][demo][robot][p].items()
                if v.is_solved
            ]
        )
    elif metric == "grasps":
        t_arr = np.array(
            [
                int(v.number_of_grasps) * 2
                for k, v in from_res[met][task][demo][robot][p].items()
                if v.is_solved
            ]
        )
    elif metric == "path_len":
        t_arr = np.array(
            [
                float(v.path_len[0])
                for k, v in from_res[met][task][demo][robot][p].items()
                if v.is_solved
            ]
        )
    d = 0 if len(t_arr) == 0 else np.mean(t_arr)
    # d_err = 0 if len(t_arr) == 0 else np.std(t_arr)
    if verbose and len(t_arr) > 0:
        print(f"{metric}, {robot}, {task}, {demo}, {met}, {p} : {t_arr}")

    return d


def get_res(file_name, metric, tasks, methods, robot):
    res_filename = get_res_file(file_name)

    with open(res_filename, "rb") as f:
        res_from = dill.load(f)

    proc_res = defaultdict(
        lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: 0.0)))
    )
    res = {m: [] for m in methods}
    for method in methods:
        for task_name, demo_id in tasks:
            tmp = []
            poses = [0] if robot == "kmr_iiwa" else list(range(10))
            for pose_id in poses:
                try:
                    tmp.append(
                        get_metric(
                            proc_res,
                            res_from,
                            metric,
                            robot,
                            task_name,
                            demo_id,
                            method,
                            pose_id,
                            verbose=True,
                        )
                    )
                except Exception as e:
                    print(f"Error on {task_name} - {demo_id}")
                    print(e)
                    if metric == "success_rate":
                        tmp.append(0)
            res[method].append(np.mean(tmp))
            # err = np.std(tmp)

    return res
