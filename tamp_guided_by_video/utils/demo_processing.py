from __future__ import annotations
import numpy as np
import pinocchio as pin
from guided_tamp_benchmark.tasks.demonstration import Demonstration


def reject_outliers(arr: np.array, max_deviations: int = 2) -> np.array:
    mean = np.mean(arr, axis=0)
    std = np.std(arr, axis=0) + 0.01
    distance_from_mean = abs(arr - mean)
    not_outlier = np.all(distance_from_mean < max_deviations * std, axis=1)
    return arr[not_outlier]


def get_average_pose(pose_list: list[np.array]) -> np.array:
    t_array = np.array([x[:3, 3] for x in pose_list])
    avg_t = np.mean(reject_outliers(t_array), axis=0)

    rot_array = np.array([pin.log(pin.SE3(x)).angular for x in pose_list])
    avg_rot = np.median(reject_outliers(rot_array), axis=0)

    return np.array(
        pin.SE3(
            translation=avg_t,
            rotation=pin.exp(np.concatenate([avg_t, avg_rot])).rotation,
        )
    )


def get_object_moving_indication(
    obj_poses: np.array, window: int = 5, threshold: float = 0.2
) -> np.array:
    """
    Transform sequence of object poses to 0-1 array that indicates if object moves a lot
    :param obj_poses: Nx4x4
    :param window: N for moving average window
    :param threshold: threshold to filter 1
    :return: 0-1 array that indicates if object moves
    """
    position_change = [
        np.linalg.norm(obj_poses[i + 1][:3, 3] - obj_poses[i][:3, 3])
        for i in range(len(obj_poses) - 1)
    ]
    position_change.append(0)  # to restore array len
    moving_avg_cont = np.convolve(position_change, np.ones(window) / window, "same")
    return np.where(
        moving_avg_cont
        >= threshold * (np.max(position_change) - np.min(position_change)),
        1,
        0,
    )


def get_static_ids(is_moved: list | np.array) -> np.array:
    """

    :param is_moved: 0-1 array that indicates if object moves :return: tuple of (
    start_ids, end_ids) describing static periods (start_ids[0] -> end_ids[0], ...)
    """
    diff_in_contact = np.array(is_moved)[1:] - np.array(is_moved)[:-1]
    static_to_move = np.where(diff_in_contact == 1)[0]
    moving_to_static = np.where(diff_in_contact == -1)[0]
    assert len(static_to_move) == len(moving_to_static)
    return np.insert(moving_to_static, 0, 0), np.insert(
        static_to_move, len(static_to_move), len(is_moved) - 1
    )


def parse_demo_into_subgoals(demo: Demonstration) -> np.array:
    """Function to parse demo data into subgoal contacts and object poses"""
    tray_ids = [i for i, name in enumerate(demo.object_ids) if "tray" in name.lower()]
    tray_id = None if len(tray_ids) == 0 else tray_ids[0]  # assume there is ONE tray
    tray_poses = None
    tray_poses_summary = None
    obj_ids = list(set([i for i in range(len(demo.objects_poses))]) - set([tray_id]))
    if tray_id is not None:
        tray_poses = demo.objects_poses[tray_id]
        tray_poses[:, :3, :3] = np.array(
            [[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]]
        )
        tray_poses[:, 2, 3] = 0.0
        tray_start_ids, tray_end_ids = get_static_ids(
            get_object_moving_indication(tray_poses)
        )
        # standardize tray poses when tray is not moving
        for from_i, to_i in zip(tray_start_ids, tray_end_ids):
            avg_pose = get_average_pose(
                [tray_poses[i] for i in range(from_i, to_i + 1)]
            )
            for i in range(from_i, to_i + 1):  # TODO: inclusive?
                tray_poses[i] = avg_pose
    static_object_poses = demo.objects_poses[obj_ids].copy()
    # tray_start_motion = np.array*
    grasp_ids = []
    release_ids = []
    for obj_id in obj_ids:
        start_ids, end_ids = get_static_ids(demo.contacts[obj_id].astype(int))
        grasp_ids += [(time_id, obj_id) for time_id in end_ids[:-1]]
        release_ids += [(time_id, obj_id) for time_id in start_ids[1:]]
        for from_i, init_to_i in zip(start_ids, end_ids):
            # check if any tray motion started inside this static period
            to_i = init_to_i.copy()
            if tray_id is not None:
                if np.any((tray_end_ids[:-1] >= from_i) & (tray_end_ids[:-1] <= to_i)):
                    # tray was moving while object was static. Fix object w.r.t. tray
                    tray_obj = np.linalg.inv(tray_poses[from_i]).dot(
                        static_object_poses[obj_id][from_i]
                    )
                    for i in range(from_i, to_i + 1):  # TODO: inclusive?
                        static_object_poses[obj_id][i] = tray_poses[i].dot(tray_obj)
                    continue
            avg_pose = get_average_pose(
                [static_object_poses[obj_id][i] for i in range(from_i, to_i + 1)]
            )
            for i in range(from_i, to_i + 1):  # TODO: inclusive?
                static_object_poses[obj_id][i] = avg_pose
    # static_object_poses = static_object_poses[obj_ids, :, :, :].copy()
    # sort grasp and release ids based on time
    sorted_grasp_ids = sorted(grasp_ids, key=lambda tup: tup[0])
    sorted_release_ids = sorted(release_ids, key=lambda tup: tup[0])
    contact_summary = np.zeros((len(obj_ids), len(sorted_grasp_ids) * 2))
    objects_poses_summary = np.zeros((len(obj_ids), len(sorted_grasp_ids) * 2, 4, 4))
    if tray_poses is not None:
        tray_poses_summary = np.zeros((len(sorted_grasp_ids) * 2, 4, 4))
    for j, (grasp_tup, release_tup) in enumerate(
        zip(sorted_grasp_ids, sorted_release_ids)
    ):
        obj_id = grasp_tup[1]
        contact_summary[obj_id, j * 2 : j * 2 + 2] = [1, 2]
        objects_poses_summary[:, j * 2] = static_object_poses[:, grasp_tup[0]]
        objects_poses_summary[:, j * 2 + 1] = static_object_poses[:, release_tup[0]]
        if tray_poses is not None:
            tray_poses_summary[j * 2] = tray_poses[grasp_tup[0]]
            tray_poses_summary[j * 2 + 1] = tray_poses[release_tup[0]]

    return contact_summary, objects_poses_summary, tray_poses_summary


def ensure_normalized(xyz_quat_wxyz: list[float]) -> list[float]:
    return list(xyz_quat_wxyz[:3]) + list(
        xyz_quat_wxyz[3:] / np.linalg.norm(xyz_quat_wxyz[3:])
    )
