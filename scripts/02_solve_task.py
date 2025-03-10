from guided_tamp_benchmark.tasks.renderer import Renderer
from guided_tamp_benchmark.models.utils import get_models_data_directory
from tamp_guided_by_video.utils.utils import get_task, get_robot
import time
from robomeshcat import Robot
import argparse
import pickle
import numpy as np
import pathlib

parser = argparse.ArgumentParser()
parser.add_argument(
    "-task_name", type=str, default="shelf", help="String name of the task"
)
parser.add_argument("-task_id", type=int, default=0, help="Int task id")
parser.add_argument("-seed", type=int, default=1, help="Seed for randomizer")
parser.add_argument("-robot_name", type=str, default="panda", help="String robot name")
parser.add_argument(
    "-planner", type=str, default="multi_contact", help="String plannner name"
)
parser.add_argument(
    "-pddl_path",
    type=str,
    default="/home/kzorina/work/repos/pddlstream",
    help="Path to pddl lib",
)
parser.add_argument("-pose_id", type=int, default=1, help="Pose id to run")
parser.add_argument("--verbose", action="store_true")
parser.add_argument("--visualize", action="store_true")
args = parser.parse_args()

task = get_task(args.task_name)(args.task_id, get_robot(args.robot_name), args.pose_id)
if args.planner == "pddl":
    import sys

    sys.path.append(args.pddl_path)
    from tamp_guided_by_video.planners.pddl_planner import PDDLPlanner

    planner = PDDLPlanner(task, max_planning_time=60, random_seed=args.seed)
else:
    from tamp_guided_by_video.utils.corba import CorbaServer
    from tamp_guided_by_video.planners import HppPlanner, MultiContactPlanner
    from tamp_guided_by_video.planners.hpp_rrt_connect_star import RRTConnectStart

    corba_server = CorbaServer(models_package=get_models_data_directory())
    if "hpp" in args.planner:
        planner = HppPlanner(
            task,
            max_planning_time=60,
            handles_mode="all",
            random_seed=args.seed,
            verbose=args.verbose,
            optimize=True if "shortcut" in args.planner else False,
        )
    elif "multi_contact" in args.planner:
        planner = MultiContactPlanner(
            task,
            max_planning_time=60,
            handles_mode="all",
            max_iter=100000,
            use_euclidean_distance=True,
            random_seed=args.seed,
            optimize_path_iter=150 if "shortcut" in args.planner else None,
            verbose=args.verbose,
            # steps=[0, 1]
        )
    elif "rrt_star_connect" in args.planner:
        planner = RRTConnectStart(
            task,
            max_planning_time=600,
            max_iter=20000,
            handles_mode="all",
            random_seed=args.seed,
            verbose=args.verbose,
        )
    else:
        raise ValueError("Unknown planner", args.planner)

robot_ndof = len(task.robot.initial_configuration())

start_time = time.time()
res = planner.solve()
end_time = time.time()
print(res)
if res:
    import os
    import numpy as np
    results_folder = f"results_{args.planner}_{args.task_name}{args.task_id}_rpose{args.pose_id}"
    res_folder = pathlib.Path(__file__).parent.parent / "results" / results_folder
    os.makedirs(res_folder, exist_ok=True)

    filename = f"{args.seed:05d}.pkl"
    file_path = os.path.join(res_folder, filename)
    pickle.dump({
        "solve_time_s": end_time - start_time,
        "config_list": planner.config_list,
    }, open(file_path, "wb"))
    
    # path_id = planner.ps.numberPaths() - 1
    # problem = planner.ps.client.basic.problem
    # path = problem.getPath(path_id)
    # T = int(np.round(path.length() / 0.01))
    # total_time = path.length()
    # q_v_a_list = []
    # for iter in range(T):
    #     iter_time = total_time * iter / (T - 1)  # iter * DT
    #     q_t = np.array(path.call(iter_time)[0][:robot_ndof])
    #     v_t = np.array(path.derivative(iter_time, 1)[:robot_ndof])
    #     a_t = np.array(path.derivative(iter_time, 2)[:robot_ndof])
    #     q_v_a_list.append((q_t, v_t, a_t))
    # pickle.dump({
    #     "solve_time_s": end_time - start_time,
    #     "config_list": planner.config_list,
    #     "q_v_a_list": q_v_a_list,
    # }, open(file_path1, "wb"))


    if args.visualize:

        # "Display with magenta start and green goal"
        configs = planner.config_list
        np.save(file_path, np.array(configs))
        r = Renderer(task=task)
        
        start_color = [148 / 255, 103 / 255, 189 / 255]  # magenta
        goal_color = [44 / 255, 160 / 255, 44 / 255]  # green
        for config, color in zip([configs[0], configs[-1]], [start_color, goal_color]):
            for i, o in enumerate([o for o in task.objects if o.name != "base"]):
                pose = config.poses[i]
                vo = Robot(
                    urdf_path=o.urdfFilename,
                    mesh_folder_path=get_models_data_directory(),
                    color=color,
                    opacity=0.25,
                    pose=pose,
                )
                r.objects.append(vo)
                r.scene.add_robot(vo)
        robot_pose = task.demo.robot_pose
        r.animate_path(configs, fps=1)
        r.scene.render_image()
