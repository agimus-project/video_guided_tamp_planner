from guided_tamp_benchmark.models.utils import get_models_data_directory
from preprocessing.preproc_utils import (
    get_task_class,
    get_robot,
    is_robot_base_valid,
)
from tamp_guided_by_video.planners.multi_contact_planner import MultiContactPlanner
import pathlib
import numpy as np
import argparse
import pinocchio as pin
from tamp_guided_by_video.utils.corba import CorbaServer
from tamp_guided_by_video.utils.demo_processing import ensure_normalized



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-task_names", type=str, nargs="+", default=None, help="String name of the task"
    )
    parser.add_argument(
        "-task_ids", type=int, nargs="+", default=None, help="Task id (f.e. 1"
    )
    parser.add_argument("-robot", type=str, default=None, help="Robot name")
    parser.add_argument(
        "-custom_timesteps",
        type=int,
        nargs="+",
        default=None,
        help="Time steps to take into account",
    )
    parser.add_argument("--visualize", dest="visualize", action="store_true")
    args = parser.parse_args()

    random_seed = 345
    corba_server = CorbaServer(models_package=get_models_data_directory())
    tasks_dir = pathlib.Path(__file__).parent.joinpath("data/tasks")
    for task_name in args.task_names:
        assert len(args.task_ids) > 0
        for task_id in args.task_ids:
            for pose_id in range(10):
                task = get_task_class(task_name)(task_id, get_robot(args.robot),
                                                 pose_id)
                planner = MultiContactPlanner(
                    task,
                    max_planning_time=300,
                    handles_mode="all",
                    max_iter=100000,
                    use_euclidean_distance=True,
                    random_seed=pose_id,
                    optimize_path_iter=350,
                    # verbose=True,
                )
                # configs = []
                object_poses_flat = []
                for j in range(planner.object_poses.shape[1]):
                    flat_poses = []
                    for i in range(planner.object_poses.shape[0]):
                        flat_poses += list(
                            ensure_normalized(
                                pin.SE3ToXYZQUAT(pin.SE3(planner.object_poses[i][j])))
                        )
                    object_poses_flat.append(flat_poses)
                q_start = list(task.robot.initial_configuration()) + object_poses_flat[
                    0]
                q_goal = list(task.robot.initial_configuration()) + object_poses_flat[
                    -1]

                # check that start and goal are valid configs

                succ_start, q_start = planner.project_to_free(q_start)
                succ_goal, q_goal = planner.project_to_free(q_goal)
                if not succ_start and succ_goal:
                    continue
                res, configs = is_robot_base_valid(
                    planner,
                    q_start,
                    object_poses_flat,
                    visualize_mid_stages=False,
                    max_iter=150
                )
                print(f"Pose {pose_id}: {res}")
                # print(configs)
                if args.visualize:
                    from guided_tamp_benchmark.tasks.renderer import Renderer
                    from guided_tamp_benchmark.core.configuration import Configuration
                    import time
                    r = Renderer(task=planner.task)
                    robot_ndof = len(planner.task.robot.initial_configuration())
                    r.robot.pose = planner.task.demo.robot_pose
                    # print(planner.task.demo.robot_pose)
                    r.animate_path(
                        [
                            Configuration.from_numpy(np.array(x), robot_ndof)
                            for x in configs.copy()
                        ],
                        fps=1,
                    )
                    time.sleep(2)
                del task
                del planner
                corba_server.reset_problem()
