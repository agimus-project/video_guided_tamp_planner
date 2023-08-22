import sys
from guided_tamp_benchmark.models.utils import get_models_data_directory
from guided_tamp_benchmark.benchmark import Benchmark
from tamp_guided_by_video.planners import HppPlanner, MultiContactPlanner
from tamp_guided_by_video.utils.utils import get_task, get_robot
from itertools import product
import argparse
import pathlib


sys.setrecursionlimit(6000)

parser = argparse.ArgumentParser()
parser.add_argument(
    "-task_name", type=str, default="shelf", help="String name of the task"
)
parser.add_argument("-task_id", type=int, default=0, help="Int task id")
parser.add_argument("-robot_name", type=str, default="panda", help="String robot name")
parser.add_argument(
    "-planner", type=str, default="multi_contact", help="String plannner name"
)
parser.add_argument(
    "-pddl_path",
    type=str,
    default="pddlstream",
    help="Path to pddl lib",
)
parser.add_argument(
    "-res_file",
    type=str,
    default="data/results.pkl",
    help="Path to file to save results",
)
parser.add_argument(
    "-pose_ids", type=int, nargs="+", default=range(10), help="Pose ids to run on"
)
parser.add_argument("--verbose", action="store_true")
args = parser.parse_args()
if args.verbose:
    print("ARGS:", args)
if args.planner == "pddl":
    sys.path.append(args.pddl_path)
    from tamp_guided_by_video.planners.pddl_planner import PDDLPlanner

    kv = {}
    plnr_cls = PDDLPlanner
else:
    from tamp_guided_by_video.utils.corba import CorbaServer

    corba_server = CorbaServer(models_package=get_models_data_directory())
    kv = {"corba": corba_server}
    if args.planner == "hpp":
        plnr_cls = HppPlanner
        kv.update({"corba": corba_server})
    elif args.planner == "hpp_shortcut":
        plnr_cls = HppPlanner
        kv.update(
            {
                "optimize": True,
            }
        )
    elif args.planner == "multi_contact":
        plnr_cls = MultiContactPlanner
        kv.update({"max_iter": 100000, "use_euclidean_distance": True})
    elif args.planner == "multi_contact_shortcut":
        plnr_cls = MultiContactPlanner
        kv.update(
            {
                "max_iter": 100000,
                "use_euclidean_distance": True,
                "optimize_path_iter": 100,
            }
        )
    else:
        raise ValueError(f"Planner '{args.planner}' does not exist")

handles_mode = "all" if args.task_name == "tunnel" else "exclude_sides"
max_planning_time = 300 if args.task_name == "waiter" else 60
update_results_file = True if pathlib.Path(args.res_file).exists() else False

" Parse arguments into lists of items to use in for-loop"
results_path = pathlib.Path(__file__).parent.parent.joinpath(args.res_file)
tas_dem_ids = [(get_task(args.task_name), args.task_id)]
robot = get_robot(args.robot_name)
poses = args.pose_ids
seeds = range(10)
b = Benchmark(results_path=results_path)

for (task_cls, demo_id), pose_id in product(tas_dem_ids, poses):
    if robot.name == "kmr_iiwa":
        # mobile robot has only one starting pose
        if pose_id == 1:
            pose_id = 0
        else:
            continue
    task = task_cls(demo_id, robot, pose_id)
    b.do_benchmark(
        task=task,
        planner=plnr_cls,
        seeds=seeds,
        planner_arg=kv,
        max_planning_time=max_planning_time,
        delta=0.001
    )
    b.save_benchmark()
