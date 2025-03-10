import os
import numpy as np
import pathlib
import matplotlib.pyplot as plt
import pickle 

which_res = 'shelf1_rpose1'
format = 'npy'  # 'pkl' or 'npy'

results_folder = pathlib.Path(__file__).parent / f"results_{which_res}"
# os.makedirs(results_folder, exist_ok=True)

existing_files = [f for f in os.listdir(results_folder) if f.endswith(".npy")]
target_size = 150
print(len(existing_files))
traj_list = []
for f in existing_files:
    data = np.load(results_folder / f, allow_pickle=True)
    resampled_list = np.linspace(0, len(data) - 1, target_size, dtype=int)
    resampled_list = [data[i] for i in resampled_list]
    if format == 'npy':
        resampled_list = [q.to_numpy() for q in resampled_list]
    traj_list.append(resampled_list)

print(np.array(traj_list).shape)
if format == 'npy':
    data = np.array(traj_list)
    print(data.shape)
    np.save(f'{which_res}_resampled_to_150.npy', data)
elif format == 'pkl':
    pickle.dump(traj_list, open(f'{which_res}_resampled_to_150.pkl', 'wb'))




# # Check that pickle is okay
# from guided_tamp_benchmark.tasks.renderer import Renderer
# from guided_tamp_benchmark.models.utils import get_models_data_directory
# from tamp_guided_by_video.utils.utils import get_task, get_robot
# import time
# from robomeshcat import Robot
# import argparse
# import pickle
# from tamp_guided_by_video.utils.corba import CorbaServer
# from tamp_guided_by_video.planners import HppPlanner, MultiContactPlanner

# corba_server = CorbaServer(models_package=get_models_data_directory())
# traj_list = pickle.load(open('traj_shelf0_resampled_to_150.pkl', 'rb'))
# q0 = list(traj_list[0][0].to_numpy())
# # print(q0)



# task = get_task('shelf')(0, get_robot('panda'), 1)
# planner = HppPlanner(
#     task,
#     max_planning_time=60,
#     handles_mode="all",
#     random_seed=1,
#     verbose=False,
#     optimize=True
# )


# print("edges")
# for k, v in planner.cg.edges.items():
#     print(k + " - " + str(v))
# print('nodes')
# for k, v in planner.cg.edges.items():
#     print(k + " - " + str(v))

# print("get node")
# node = planner.cg.getNode(q0)
# print(node)


# config_states = [
#     k + ' - ' + str(planner.cg.getConfigErrorForNode(k, q0)[0])
#     for k, v in planner.cg.nodes.items()
#     if planner.cg.getConfigErrorForNode(k, q0)[0]
# ]
# print('config_states')
# print(config_states)

# transitions = [
#     k # + ' - ' + str(planner.cg.getConfigErrorForEdge(k, q0)[0])
#     for k, v in planner.cg.edges.items()
#     if planner.cg.getConfigErrorForEdge(k, q0)[0]
# ]
# print('transitions')
# print(transitions)

# transitions = [
#     k # + ' - ' + str(planner.cg.getConfigErrorForEdge(k, q0)[0])
#     for k, v in planner.cg.edges.items()
#     if planner.cg.getNodesConnectedByEdge(k)[0] == node
# ]
# print('transitions with nodes connected by edge')
# print(transitions)

# # print(planner.cg.getConfigErrorForEdge(list(planner.cg.edges.keys())[0], q0))


# # print("connected nodes:")
# # print(planner.cg.getNodesConnectedByEdge(transitions[0]))
# import numpy as np
# q1 = q0.copy()
# print(planner.cg.getConfigErrorForEdgeTarget('Loop | f', q0, q1))  # 2
# print(np.linalg.norm(planner.cg.getConfigErrorForEdgeTarget('Loop | f', q0, q1)[1]))  # 2
# q1[0] += 0.2
# print(planner.cg.getConfigErrorForEdgeTarget('Loop | f', q0, q1))  # 2
# print(np.linalg.norm(planner.cg.getConfigErrorForEdgeTarget('Loop | f', q0, q1)[1]))  # 2
# q1[10] += 0.2
# print(planner.cg.getConfigErrorForEdgeTarget('Loop | f', q0, q1))  # 2
# print(np.linalg.norm(planner.cg.getConfigErrorForEdgeTarget('Loop | f', q0, q1)[1]))  # 2
