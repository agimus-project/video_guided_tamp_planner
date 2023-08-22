from preprocessing.preproc_utils import (
    get_task_class,
    get_robot,
    is_robot_base_valid,
    sample_robot_base_pose
)
from tamp_guided_by_video.planners.multi_contact_planner import MultiContactPlanner
import pathlib
import argparse
import pinocchio as pin
from guided_tamp_benchmark.models.utils import get_models_data_directory
from tamp_guided_by_video.utils.demo_processing import ensure_normalized
from tamp_guided_by_video.utils.corba import CorbaServer
import numpy as np
import random

def sample_robot_pose(task, seed):
    planner = MultiContactPlanner(
        task,
        max_planning_time=300,
        handles_mode="all",
        max_iter=1000,
        use_euclidean_distance=True,
        random_seed=seed,
        verbose=True,
    )
    return sample_robot_base_pose(
        planner.object_poses,
        planner.contacts,
        radius=planner.task.robot.reach_m(),
        z_coord_bound=(0.75, 1.25),
        sample_z_rot=True,
    )

if __name__ == "__main__":
    '''
    
    for run in {1..100}; do python preprocessing/03_sample_one_pose.py \
    -task_name shelf -task_id 2 -robot panda -pose_id 1 --verbose; done

    '''
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-task_name", type=str, default=None, help="String name of the task"
    )
    parser.add_argument(
        "-task_id", type=int, default=None, help="Task id (f.e. 1)"
    )
    parser.add_argument(
        "-pose_id", type=int, default=None, help="Pose id to resample pose (f.e. 1)"
    )
    parser.add_argument("-robot", type=str, default=None, help="Robot name")
    parser.add_argument(
        "-seed", type=int, default=None, help="Seed for pose sampler"
    )
    parser.add_argument(
        "-custom_timesteps",
        type=int,
        nargs="+",
        default=None,
        help="Time steps to take into account",
    )
    parser.add_argument("--verbose", dest="verbose", action="store_true")
    parser.add_argument("--vis", dest="vis", action="store_true")

    args = parser.parse_args()
    corba_server = CorbaServer(models_package=get_models_data_directory())
    folder = pathlib.Path(__file__).parent.joinpath("data")
    guide_path = folder.joinpath(f"{args.task_name}_{args.robot}"
                                 f"_{args.task_id}_{args.pose_id}.pkl")

    if not guide_path.exists():
        folder.mkdir(exist_ok=True, parents=True)
        with open(guide_path, 'w') as f:
            f.writelines('0')

    with open(guide_path, 'r') as f:
        pose_indicator = int(f.readlines()[0])
    if not pose_indicator:
        random.seed(args.seed)

        task = get_task_class(args.task_name)(
            args.task_id,
            get_robot(args.robot),
            args.pose_id)
        pose = sample_robot_pose(task, args.seed)
        print(pin.log(pose))
        prev_pose = task.demo.robot_pose.copy()
        task.set_robot_pose(pose)

        planner = MultiContactPlanner(
            task,
            max_planning_time=300,
            handles_mode="all",
            max_iter=100000,
            use_euclidean_distance=True,
            random_seed=5,
            optimize_path_iter=350,
            verbose=True,
        )

        object_poses_flat = []
        for j in range(planner.object_poses.shape[1]):
            flat_poses = []
            for i in range(planner.object_poses.shape[0]):
                flat_poses += list(
                    ensure_normalized(
                        pin.SE3ToXYZQUAT(pin.SE3(planner.object_poses[i][j])))
                )
            object_poses_flat.append(flat_poses)
        q_start = list(task.robot.initial_configuration()
                       ) + object_poses_flat[0]
        q_goal = list(task.robot.initial_configuration()
                      ) + object_poses_flat[-1]

        # check that start and goal are valid configs

        succ_start, q_start = planner.project_to_free(q_start)
        succ_goal, q_goal = planner.project_to_free(q_goal)
        if not succ_start and succ_goal:
            if args.verbose:
                print("Not successful start/goal")
            robot_pose_valid = False
        else:
            robot_pose_valid, grasping_configs = is_robot_base_valid(
                planner,
                q_start,
                object_poses_flat,
                visualize_mid_stages=False,
                max_iter=25
            )
        if args.verbose:
            print(f"Success: {robot_pose_valid}, {len(grasping_configs)}")
        if robot_pose_valid:
            with open(guide_path, 'w') as f:
                f.writelines('1')
            task.demo.robot_pose = pose
            task.demo.save(overwrite=True)
            if args.vis:
                from guided_tamp_benchmark.tasks.renderer import Renderer
                from guided_tamp_benchmark.core.configuration import Configuration
                import time

                r = Renderer(task=planner.task)
                r.robot.pose = task.demo.robot_pose
                robot_ndof = len(planner.task.robot.initial_configuration())
                r.animate_path(
                    [
                        Configuration.from_numpy(np.array(x), robot_ndof)
                        for x in grasping_configs
                    ],
                    fps=1,
                )
                time.sleep(5)

