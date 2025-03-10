import numpy as np
import pathlib
import pickle
from guided_tamp_benchmark.core import Configuration

def resample_to_traj(configs: list[Configuration], desired_len: int = 200) -> list[np.ndarray]:
    """
    Resamples a list of configuration objects to a specified length.

    Parameters:
    configs (list): A list of configuration objects, each having an attribute 'q'.
    desired_len (int, optional): The desired length of the resampled list. Default is 200.

    Returns:
    list: A list of 'q' attributes from the resampled configuration objects.
    """
    if desired_len > len(configs):
        raise ValueError(f"Desired length {desired_len} must be smaller than the length of the input list. {len(configs)}")
    resampled_list = np.linspace(0, len(configs) - 1, desired_len, dtype=int)
    return [configs[i].to_numpy() for i in resampled_list]


planner = 'rrt_star_connect'
task_name = 'shelf'
task_id = 0
pose_id = 1
all_seeds = 465
# desired_len = 200

len_arr = []
traj_list = []
results_folder = f"results_{planner}_{task_name}{task_id}_rpose{pose_id}"
for seed in range(all_seeds):
    filename = f"{seed:05d}.pkl"
    file_path = pathlib.Path(__file__).parent / results_folder / filename
    if not file_path.exists():
        print(f"File {file_path} does not exist")
        continue
    data = pickle.load(open(file_path, 'rb'))
    while len(data['config_list']) < 100:
        data['config_list'].append(data['config_list'][-1])
    while len(data['config_list']) > 100:
        data['config_list'].pop(1)
    len_arr.append(len(data['config_list']))
    # traj_list.append(resample_to_traj(data['config_list'], desired_len=desired_len))
    traj_list.append([config.to_numpy() for config in data['config_list']])
print(len(len_arr))
print(np.min(len_arr), np.mean(len_arr), np.max(len_arr))
data = np.array(traj_list)
print(data.shape)  # (2733, 200, 9)
np.save(f'results_{planner}_{task_name}{task_id}_rpose{pose_id}_100.npy', data)
