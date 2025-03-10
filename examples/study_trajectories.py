"""
This script:
- loads trajectory files, 
- computes the length of the trajectory 
- plots the end-effector poses with color based on the length
"""
import example_robot_data as erd
import os   
import numpy as np
import pathlib
import pickle
from matplotlib import pyplot as plt
from tamp_guided_by_video.utils.utils import get_task, get_robot
import pinocchio as pin

rmodel = erd.load('panda').model
rdata = rmodel.createData()

def get_ee_traj(traj):
    ee_pos_list = []
    for _, q in enumerate(traj):
        if len(q) == 7:
            q = np.append(q, [0, 0])
        pin.forwardKinematics(rmodel, rdata, q[:rmodel.nq])
        pin.updateFramePlacements(rmodel, rdata)
        ee_pos_list.append(rdata.oMf[rmodel.getFrameId("panda_hand")].translation.copy())
    return ee_pos_list


planner = 'rrt_star_connect'
task_name = 'shelf'
task_id = 0
pose_id = 1
all_seeds = 1005
pecentage_go_grab = 20
desired_len = 300
task = get_task(task_name)(task_id, get_robot('panda'), pose_id)
results_folder = f"results_{planner}_{task_name}{task_id}_rpose{pose_id}"
l_array = []
save_ee_poses = []
traj_list = []
for seed in range(all_seeds):
    filename = f"{seed:05d}.pkl"
    file_path = pathlib.Path(__file__).parent / "results" / results_folder / filename
    if not file_path.exists():
        print(f"File {file_path} does not exist")
        continue
    data = pickle.load(open(file_path, 'rb'))
    rot_l_joints, pos_l_obj, rot_l_obj = task.compute_lengths(data['config_list'])
    config_float = [config.to_numpy() for config in data['config_list']]
    if len(config_float) > desired_len:
        config_float = config_float[:desired_len]
    while len(config_float) < desired_len:
        config_float.append(config_float[-1])
    l_array.append(rot_l_joints)
    save_ee_poses.append(np.array(get_ee_traj(config_float)))
    traj_list.append(np.array(config_float))

l_array = np.array(l_array)
save_ee_poses = np.array(save_ee_poses)
traj_list = np.array(traj_list)

tolerated_len = np.percentile(l_array, pecentage_go_grab)
print(traj_list.shape) 

traj_list = traj_list[l_array <= tolerated_len]
save_ee_poses = save_ee_poses[l_array <= tolerated_len]
l_array = l_array[l_array <= tolerated_len]
# breakpoint()
# Plot the end-effector poses with color based on the length
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
max_l = max(l_array)
min_l = min(l_array)
print(min_l, max_l)



print(traj_list.shape) 
print(type(traj_list[0]))
print(len(traj_list[0]))
np.save(f'results_{planner}_{task_name}{task_id}_rpose{pose_id}_best{pecentage_go_grab}.npy', traj_list)


for rot_l_joints, ee_poses in zip(l_array, save_ee_poses):
    normalized_len = (rot_l_joints - min_l) / (max_l - min_l)
    # print(rot_l_joints)
    # print(ee_poses.shape)
    # Plot a line plot of EE poses (gradient from red to green)
    ee_color = plt.cm.RdYlGn(1 - normalized_len)
    # print([round(x,2) for x in ee_poses[0]])
    # print([round(x,2) for x in ee_poses[10]])
    # print([round(x,2) for x in ee_poses[100]])
    ax.scatter(ee_poses[0, 0], ee_poses[0, 1], ee_poses[0, 2], color='g')
    # ax.scatter(ee_poses[-1, 0], ee_poses[-1, 1], ee_poses[-1, 2], color='r')
    ax.plot(ee_poses[::10, 0], ee_poses[::10, 1], ee_poses[::10, 2], color=ee_color, alpha=0.2)
plt.show()
    
# Select only low length ones