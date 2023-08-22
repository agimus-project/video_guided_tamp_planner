import pickle
import numpy as np
import quaternion as npq
from preprocessing.preproc_utils import reject_outliers
import argparse
from preprocessing.preproc_utils import data_folder

parser = argparse.ArgumentParser()
parser.add_argument(
    "-task_name", type=str, default="shelf", help="String name of the task"
)
parser.add_argument("-task_id", type=int, default=0, help="Int task id")
parser.add_argument(
    "-calib_object", type=str, default="ycbv_02", help="Object used in the calibration"
)
parser.add_argument(
    "-calib_path",
    type=str,
    default=data_folder.joinpath("calibration.pkl"),
    help="Path to pddl lib",
)
parser.add_argument(
    "-known_pose_path",
    type=str,
    default=data_folder.joinpath("world_object.npy"),
    help="Path to file to save results",
)
parser.add_argument("--verbose", action="store_true")
args = parser.parse_args()

cosy_calib_file = args.calib_path
calib_preds = pickle.load(open(cosy_calib_file, "rb"))
cosypose_calib_obj_id = args.calib_object
task_name = args.task_name
task_id = args.task_id

world_obj = np.load(args.known_pose_path)
t_array = np.array([x[:3, 3] for x in calib_preds[args.calib_object]])
avg_t = np.mean(reject_outliers(t_array), axis=0)

rot_array = np.array(
    [
        npq.as_rotation_vector(npq.from_rotation_matrix(x[:3, :3]))
        for x in calib_preds[cosypose_calib_obj_id]
    ]
)
avg_rot = np.mean(reject_outliers(rot_array), axis=0)

camera_obj = np.eye(4)
camera_obj[:3, 3] = avg_t
camera_obj[:3, :3] = npq.as_rotation_matrix(npq.from_rotation_vector(avg_rot))

if not data_folder.exists():
    data_folder.mkdir(exist_ok=True, parents=True)
np.save(
    data_folder.joinpath(f"calibration_{task_name}_{task_id}"),
    world_obj.dot(np.linalg.inv(camera_obj)),
)
