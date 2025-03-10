# import os
# import numpy as np
# from guided_tamp_benchmark.tasks.renderer import Renderer
# from guided_tamp_benchmark.models.utils import get_models_data_directory
# from guided_tamp_benchmark.core.configuration import Configuration
# from tamp_guided_by_video.utils.utils import get_task, get_robot
# import time
# from robomeshcat import Robot
# import argparse

# # results_folder = "results"
# # os.makedirs(results_folder, exist_ok=True)

# # existing_files = [f for f in os.listdir(results_folder) if f.endswith(".npy")]
# # print(existing_files)
# # configs = np.load(os.path.join(results_folder, '0001.npy'), allow_pickle=True)
# # print(configs.shape)
# # print(configs[0])

# # task = get_task('shelf')(0, get_robot('panda'), 1)
# # r = Renderer(task=task)

# # start_color = [148 / 255, 103 / 255, 189 / 255]  # magenta
# # goal_color = [44 / 255, 160 / 255, 44 / 255]  # green
# # for config, color in zip([configs[0], configs[-1]], [start_color, goal_color]):
# #     for i, o in enumerate([o for o in task.objects if o.name != "base"]):
# #         pose = config.poses[i]
# #         vo = Robot(
# #             urdf_path=o.urdfFilename,
# #             mesh_folder_path=get_models_data_directory(),
# #             color=color,
# #             opacity=0.25,
# #             pose=pose,
# #         )
# #         r.objects.append(vo)
# #         r.scene.add_robot(vo)
# # robot_pose = task.demo.robot_pose
# # r.animate_path(list(configs), fps=1)

# # time.sleep(5.0)


# task = get_task('shelf')(0, get_robot('panda'), 0)
# r = Renderer(task=task)


# robot_pose = np.eye(4)
# hpp_q = np.array([0, -0.7853981633974483, 0, -2.356194490192345, 0, 1.5707963267948966, 0.7853981633974483, 0.035, 0.035, 0.2, 0, 0.12, 0, 0, 0, 1])
# configs = [Configuration.from_numpy(hpp_q, robot_ndofs=9), Configuration.from_numpy(hpp_q, robot_ndofs=9)]
# # configs = [c.to_numpy() for c in configs]
# r.animate_path(configs, fps=1)

# time.sleep(5.0)


def findTheDifference(s: str, t: str) -> str:
        list_s = list(s)
        for l in t:
            # try:
            match = list_s.index(l)
            print(match)
            list_s.pop(match)
            # except:
            #     return l
            
findTheDifference("abcd", "abcde")