from tamp_guided_by_video.utils.corba import CorbaServer
from guided_tamp_benchmark.models.utils import get_models_data_directory
import pathlib
import numpy as np
import time
from preprocessing.preproc_utils import (
    get_task_class,
    get_robot,
    get_trans_quat_hpp,
    sample_robot_base_pose,
    is_robot_base_valid,
)
from tamp_guided_by_video.planners.multi_contact_planner import MultiContactPlanner
from guided_tamp_benchmark.core import Configuration
from guided_tamp_benchmark.tasks.renderer import Renderer

# random_seed = 12345
task_name = 'shelf'
task_id = 0
robot = 'ur5'


corba_server = CorbaServer(models_package=get_models_data_directory())
tasks_dir = pathlib.Path(__file__).parent.joinpath("data/tasks")
configs = []

check_poses = np.array([np.eye(4)] * 3)
check_poses[:, :3, 3] = [-1., 1., 1.4]
check_poses[1, 2, 3] = 1.2
check_poses[2, 2, 3] = 1.8
check_poses[1, 0, 3] = -1.25
check_poses[2, 1, 3] = 0.5
task = get_task_class(task_name)(task_id, get_robot(robot), 0)
demo = task.demo
for i, robot_pose_wrt_0 in enumerate(check_poses):
    demo.robot_pose = robot_pose_wrt_0
    demo.save(overwrite=True)
    time.sleep(1.)
    task = get_task_class(task_name)(task_id, get_robot(robot), 0)
    demo = task.demo
    print(task.demo.robot_pose)
    planner = MultiContactPlanner(
        task,
        max_planning_time=300,
        handles_mode="all",
        max_iter=100000,
        use_euclidean_distance=True,
        random_seed=i + 1,
        optimize_path_iter=350,
        verbose=True
    )
    basic_config = list(task.robot.initial_configuration()) + [0., 0., 0., 0., 0., 0.,
                                                               1.]
    basic_config[1] = 0
    basic_config[0] -= np.pi / 8
    succ_start, basic_config = planner.project_to_free(basic_config)
    if not succ_start:
        print("EEERRROOORRR")
    c_basic_config = Configuration.from_numpy(np.array(basic_config), robot_ndofs=len(
        planner.task.robot.initial_configuration()
    ))

    if i == 2:
        basic_config[2] += np.pi / 5
        succ_start, basic_config = planner.project_to_free(basic_config)
        if not succ_start:
            print("EEERRROOORRR")
        c_basic_config = Configuration.from_numpy(np.array(basic_config),
                                                  robot_ndofs=len(
                                                      planner.task.robot.initial_configuration()
                                                  ))
    basic_config_2 = basic_config.copy()
    basic_config_2[1] -= np.pi / 8
    print(planner.robot.isConfigValid(basic_config))
    print(planner.task.collision.is_config_valid(c_basic_config))
    print(planner.ps.directPath(
                    basic_config, basic_config_2, True
                ))
    r = Renderer(task=planner.task)
    r.animate_path([c_basic_config])
    del task
    del planner
    corba_server.reset_problem()

demo.robot_pose = check_poses[0]
demo.save(overwrite=True)