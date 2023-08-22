from guided_tamp_benchmark.tasks.renderer import Renderer
from guided_tamp_benchmark.models.utils import get_models_data_directory
from tamp_guided_by_video.utils.utils import get_task, get_robot
import time
from robomeshcat import Robot
import pathlib
import pickle

task = 'tunnel'
id = 0
robot = 'kmr_iiwa'
planner = 'pddl'
pose_id = 0
seed = 6  # 7, 9
results_file = 'results_23_07_29.pkl'

data_folder = pathlib.Path(__file__).parent.parent.joinpath("data")

with open(data_folder.joinpath(results_file), 'rb') as f:
    data = pickle.load(f)
# print(data.keys())
# cur_res = data[planner][task][id][robot][pose_id]
# for seed in cur_res.keys():
#     print(f"{seed}: {cur_res[seed].is_solved}")

# "Display with magenta start and green goal"

configs = data[planner][task][id][robot][pose_id][seed].subsampled_path

print(len(configs))
start_color = [148 / 255, 103 / 255, 189 / 255]  # magenta
goal_color = [44 / 255, 160 / 255, 44 / 255]  # green
# robot = 'kmr'
task = get_task(task)(id, get_robot(robot), pose_id)
r = Renderer(task=task)
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

time.sleep(5.0)