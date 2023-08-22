# def detect_outliers_ids(arr, max_deviations=2):
#     mean = np.mean(arr, axis=0)
#     std = np.std(arr, axis=0) + 0.01
#     distance_from_mean = abs(arr - mean)
#     outliers = ~np.all(distance_from_mean < max_deviations * std, axis=1)
#     return outliers

import torch
import torchvision.ops.boxes as bops
import matplotlib.pyplot as plt
import numpy as np
import pickle

# from filterpy.kalman import KalmanFilter
# from scipy.linalg import block_diag
# from filterpy.common import Q_discrete_white_noise
from guided_tamp_benchmark.tasks import ShelfTask, TunnelTask, WaiterTask
from guided_tamp_benchmark.models.robots import (
    PandaRobot,
    UR5Robot,
    KukaMobileIIWARobot,
)
import quaternion as npq
import random
import pathlib

data_folder = pathlib.Path(__file__).parent.joinpath("data")
def get_trans_quat_hpp(matrix, shift=None):
    """
    Convert 4 x 4 numpy matrix to list of translation and quaternion in
    (x, y, z, w) format used by hpp

    :param: matrix: 4 x 4 numpy array
    :param shift: optional shift
    :return: list of [translation, quaternion]
    """
    if shift is not None:
        t = matrix[:3, 3] + np.array(shift)
    else:
        t = matrix[:3, 3]
    q = npq.as_float_array(npq.from_rotation_matrix(matrix[:3, :3]))
    return list(t) + [q[1], q[2], q[3], q[0]]


def is_loc_valid(x, x_to_exclude, radius_exclude):
    """
    Function to check if point is close to exclude points
    :param x: point to check
    :param x_to_exclude: all the forbbiden points
    :param radius_exclude: radius around each point to exclude
    :return: True is point is far enough from each of the exclude points,
    otherwise False
    """
    for x_excl in x_to_exclude:
        if np.linalg.norm(np.array(x) - np.array(x_excl)) < radius_exclude:
            return False
    return True


def random_point_in_circle(center_x, center_y, radius):
    """
    Generate uniformly a random point in a circle around center
    :param center_x: x coordinate of center
    :param center_y: y coordinate of center
    :param radius: radius of a circle
    :return: point inside a circle
    """
    r = radius * np.sqrt(random.uniform(0, 1))
    theta = random.uniform(0, 1) * 2 * np.pi
    x = center_x + r * np.cos(theta)
    y = center_y + r * np.sin(theta)
    return x, y


def sample_robot_base_xy(center, radius, loc_to_exclude, radius_exclude=0.15):
    """
    Heuristic function to generate robot base position.
    This function will generate a random point in a circle around certain point and
    exclude designated locations.
    Excluded locations are object locations as robot base cannot coinside.
    Central point will be mean of all recorded object positions.
    And radius is equal to robot reach.
    """
    point = random_point_in_circle(*center, radius)
    if is_loc_valid(point, loc_to_exclude, radius_exclude):
        return point
    else:
        return sample_robot_base_xy(center, radius, loc_to_exclude, radius_exclude)


def get_unique_object_locations(obj_poses, contacts):
    object_locations = []
    for j in range(len(obj_poses[0])):
        for i in range(len(obj_poses)):
            if i not in np.argmax(contacts, axis=0):
                continue
            rounded_loc = [round(x, 2) for x in obj_poses[i][j][:2, 3]]
            if rounded_loc not in object_locations:
                object_locations.append(rounded_loc)
    return object_locations


def sample_robot_base_pose(
    obj_poses, contacts, radius, z_coord_bound=(0.0, 0.0), sample_z_rot=False
):
    """
    Returns 4x4 matrix for pose of robot base.
    X and Y coordinates are sampled to be in radius from all objects
    Z is sampled uniformly, default is zero
    Rotation around z-axis is sampled uniformly, default is zero
    :param radius: usually robot reach to constraint robot base to be close to objects
    :param z_coord_bound: bound for uniform sampling of z-axis coord
    :param sample_z_rot: boolean, shows is sample rotation around z axis
    :return: 4 x 4 numpy matrix of robot base pose
    """
    obj_locations = get_unique_object_locations(obj_poses, contacts)
    center = np.mean(obj_locations, axis=0)
    # decrease radius by max distance from center to the object
    max_dist = 0
    for obj_loc in obj_locations:
        if np.linalg.norm(np.array(obj_loc) - np.array(center)) > max_dist:
            max_dist = np.linalg.norm(np.array(obj_loc) - np.array(center))
    x, y = sample_robot_base_xy(center, radius - max_dist, obj_locations)
    z = random.uniform(*z_coord_bound)
    z_rot = np.pi / 2
    if sample_z_rot:
        z_rot = random.uniform(-np.pi, np.pi)
    final_pose = np.eye(4)
    final_pose[0, 0] = np.cos(z_rot)
    final_pose[1, 1] = np.cos(z_rot)
    final_pose[0, 1] = np.sin(z_rot)
    final_pose[1, 0] = -np.sin(z_rot)
    final_pose[0, 3] = x
    final_pose[1, 3] = y
    final_pose[2, 3] = z
    return final_pose


def is_robot_base_valid(
    planner, q_from, obj_poses, max_iter=1, visualize_mid_stages=False
):
    """
    Robot base position is valid if it is possible to generate at least one
    pregrasp -> preplace pair for each step.
    Pairs are generated from the init configuration
    Handle is fixed for one object manipulation
    (pregrasp and preplace both use same handle)
    :param obj_poses: robot poses modified by robot base position
    (robot stays at 0, but object move)
    :param obj_ids: which object is manipulated at each step
    :return: tuple of (True if robot position is valid,
    configs for visualization of pregrasps)
    """
    obj_ids = np.argmax(planner.contacts, axis=0)
    ndof = len(planner.robot.initial_configuration())
    configs = []
    for j, (obj_pose_from, obj_id_from, obj_pose_to, obj_id_to) in enumerate(
        zip(obj_poses[::2], obj_ids[::2], obj_poses[1::2], obj_ids[1::2])
    ):
        assert obj_id_from == obj_id_to
        # try random handles until success, or try all and record successful?
        suc_handle_found = False
        for h_id, handle in enumerate(planner.handles_names[obj_id_from]):
            if suc_handle_found:
                break

            # generate grasp subgoal for handle (free -> grasp)
            for _ in range(max_iter):
                q_from[ndof:] = obj_pose_from.copy()
                if not suc_handle_found:
                    subgoal_grasp_loc = planner.get_coupled_subgoal_for_opos(
                        q_from, obj_id_from, handle=handle, time_id=2 * j
                    )

                    if subgoal_grasp_loc[0] is not None:
                        if (
                            planner.task.demo is not None
                            and planner.task.demo.task_name == "tunnel"
                        ):
                            tunnel_side_from = planner.get_robot_tunnel_side(
                                subgoal_grasp_loc[1]
                            )
                        configs.append(subgoal_grasp_loc[0])
                        configs.append(subgoal_grasp_loc[1])
                        # grasp subgoal for handle (free -> grasp) generated succesfully
                        q_from[ndof:] = obj_pose_to.copy()
                        # generate place subgoal for handle (grasp -> free)
                        for _ in range(max_iter):
                            if not suc_handle_found:
                                subgoal_place_loc = (
                                    planner.get_coupled_subgoal_for_opos(
                                        q_from,
                                        obj_id_to,
                                        handle=handle,
                                        time_id=2 * j + 1,
                                    )
                                )
                                if subgoal_place_loc[0] is not None:
                                    if (
                                        planner.task.demo is not None
                                        and planner.task.demo.task_name == "tunnel"
                                    ):
                                        tunnel_side_to = planner.get_robot_tunnel_side(
                                            subgoal_place_loc[1]
                                        )
                                        if tunnel_side_from != tunnel_side_to:
                                            continue

                                    # TODO: do I need to have direct path?
                                    # res, pid, msg = planner.ps.directPath(
                                    # subgoal_grasp_loc[1], subgoal_place_loc[0],
                                    #                                    True)
                                    # if not res:
                                    #     res, pid, msg = planner.ps.directPath(
                                    #     subgoal_place_loc[0],
                                    #     subgoal_grasp_loc[1], True)
                                    # if not res:
                                    #     continue
                                    # print(f"{j} second subgoal good {handle}")
                                    configs.append(subgoal_place_loc[0])
                                    configs.append(subgoal_place_loc[1])
                                    # place subgoal for handle (grasp -> free)
                                    # generated succesfully
                                    suc_handle_found = True
        if not suc_handle_found:
            return False, configs
    return len(configs) > 0, configs


def get_task_class(task_name):
    """
    Returns desired task class

    :param task_name: string of either 'waiter' for waiter task,
    'shelf' for shelf task or 'tunnel' for tunnel task
    :return: the desired task class
    """
    if task_name == "waiter":
        return WaiterTask
    elif task_name == "shelf":
        return ShelfTask
    elif task_name == "tunnel":
        return TunnelTask
    else:
        raise ValueError(f"{task_name} is unknokwn task name")


def get_robot(robot_name=None):
    """
    returns desired robot object

    :param robot_name: string of either 'panda', 'ur5', 'iiwa' or 'kmr_iiwa'
    :return: desired robot object
    """
    if robot_name is None or robot_name == "panda":
        robot = PandaRobot()
    elif robot_name == "ur5":
        robot = UR5Robot()
    elif robot_name == "kmr_iiwa":
        robot = KukaMobileIIWARobot()
    else:
        raise ValueError(f"{robot_name} is unknokwn robot name")
    return robot


def reject_outliers(arr, max_deviations=2):
    mean = np.mean(arr, axis=0)
    std = np.std(arr, axis=0) + 0.01
    distance_from_mean = abs(arr - mean)
    not_outlier = np.all(distance_from_mean < max_deviations * std, axis=1)
    return arr[not_outlier]


def get_obj_poses_from_file(filepath, transform=None):
    if transform is None:
        transform = np.eye(4)
    cosypose_data = pickle.load(open(filepath, "rb"))
    obj_keys = cosypose_data.keys()
    obj_poses = []
    for key in obj_keys:
        obj_poses.append(np.array([transform.dot(mat) for mat in cosypose_data[key]]))
    return obj_keys, np.array(obj_poses)


def get_hand_contact(contacts, hand_id=1):
    """
    :param hand_id: 1 for right hand, 0 for left
    :return: sequence of 1 and 0 where 1 mean requested hand is in contact with
    object and 0 otherwise
    """
    contact_info = []
    for k in sorted(contacts.keys()):
        if contacts[k] is None:
            contact_info.append(0)
        else:
            hand_contact = [
                contacts[k][i]
                for i in range(len(contacts[k]))
                if contacts[k][i][-1] == hand_id
            ]
            if len(hand_contact) == 1 and hand_contact[0][5] == 3:
                contact_info.append(1)
            else:
                contact_info.append(0)
    return np.array(contact_info)


def get_bb_from_handobj(handobj_data):
    # Todo: think about case when both hands are in contact with object
    """
    Assume that only one hand can be in contact with object at a time (if two hands
    are in contact - add None)
    :param contacts: contacts data from handobj recognizer
    (dictionary with (n, 10) elements)
    :return: sequence of None or object bb if contact with object detected
    """
    contact_bb = []
    for k in sorted(handobj_data.keys()):
        if handobj_data[k] is None:
            contact_bb.append(None)
        else:
            object_contact = [
                handobj_data[k][i]
                for i in range(len(handobj_data[k]))
                if handobj_data[k][i, 5] == 3
            ]
            # assert len(object_contact) < 2, f"Two object contact detected on key {k}"
            if len(object_contact) == 0:
                print(f"No no contact with ibjects on {k}")
                contact_bb.append(None)
                # raise ValueError("Investigate")
            else:
                # record all (contact bb, scores)
                resulting_list = []
                for row in object_contact:
                    resulting_list.append((row[:4], row[4]))

                contact_bb.append(resulting_list)
    return contact_bb


def get_contact_from_file(filepath):
    contacts_data = pickle.load(open(filepath, "rb"))

    right_hand_c = get_hand_contact(contacts_data, hand_id=1)
    left_hand_c = get_hand_contact(contacts_data, hand_id=0)
    # TODO: think about two hand contact cases
    return np.maximum(right_hand_c, left_hand_c)


def get_cosypose_bb_from_file(filepath):
    data = pickle.load(open(filepath, "rb"))
    data = {k: np.array(v) for k, v in data.items()}
    return data


def get_handobj_bb_from_file(filepath):
    data = pickle.load(open(filepath, "rb"))
    return get_bb_from_handobj(data)


def assign_contact_to_obj(cosy_obj_bb, handobj_obj_bb, threshold=0.5):
    obj_keys = list(cosy_obj_bb.keys())
    contacts_per_object = {}
    for obj_key in obj_keys:
        contacts_per_object[obj_key] = []
    for i, list_bb in enumerate(handobj_obj_bb):
        for obj_key in obj_keys:
            if list_bb is None or cosy_obj_bb[obj_key][i] is None:
                contacts_per_object[obj_key].append(None)
            else:
                bb = None
                for bb_tuple in list_bb:
                    if (
                        bops.box_iou(
                            torch.tensor([bb_tuple[0]]),
                            torch.tensor([cosy_obj_bb[obj_key][i]]),
                        )
                        > threshold
                    ):
                        if bb is not None:
                            raise ValueError("Investigate two objects are in contact")
                        bb = bb_tuple[0]
                contacts_per_object[obj_key].append(bb)
    for obj_key in obj_keys:
        contact_data = [0 if x is None else 1 for x in contacts_per_object[obj_key]]
        contacts_per_object[obj_key] = smooth_contacts(contact_data, window=5)
    # handle two object contact missdetection
    # for two obj contact, check if last three points have consistent contact with one
    # of them, replace with it
    for i in range(len(contacts_per_object[obj_key])):
        # detect two object contacts
        if (
            sum(
                [
                    contacts_per_object[obj_key][i]
                    for obj_key in list(contacts_per_object.keys())
                ]
            )
            > 1
        ):
            objs_in_contact = [
                obj_key
                for obj_key in list(contacts_per_object.keys())
                if contacts_per_object[obj_key][i] == 1
            ]
            print(f"inspecting {i} with missdetection: {objs_in_contact}")
            for obj_key in objs_in_contact:
                prev_cont = sum(
                    [contacts_per_object[obj_key][j] for j in [i - 1, i - 2, i - 3]]
                )
                if prev_cont > 2:
                    true_obj = obj_key
                    break
            print(f"detected true object: {true_obj}")
            for obj_key in objs_in_contact:
                contacts_per_object[obj_key][i] = 1 if obj_key == true_obj else 0
    return contacts_per_object


def get_tray_moving_ind(tray_poses):
    tray_move_ids = []
    is_moving = get_object_moving_indication(tray_poses)
    start_move_id = 0
    stop_move_id = 0
    while start_move_id is not None:
        start_move_id = next(
            (
                x
                for x, val in enumerate(is_moving[start_move_id + stop_move_id :])
                if val > 0
            ),
            None,
        )
        if start_move_id is not None:
            start_move_id = stop_move_id + start_move_id
            stop_move_id = next(
                (x for x, val in enumerate(is_moving[start_move_id:]) if val < 1), None
            )
            if stop_move_id is None:
                raise ValueError(f"Tray processing failed on start_id {start_move_id}")
            stop_move_id = start_move_id + stop_move_id
            tray_move_ids.append((start_move_id, stop_move_id))
    return tray_move_ids


def get_compact_contact(
    contacts_per_object,
    obj_poses,
    tray_poses=None,
    visualize=False,
    obj_possible_motion=None,
    # ensure_static_objects=True,
    avg_steps=5,
    grasp_pre_steps=10,
    release_post_steps=10,
    threshold=0.25,
):
    all_grasp_ids = []
    all_release_ids = []
    for i, (k, v) in enumerate(contacts_per_object.items()):
        grasp_ind, release_ind = get_grasp_release_indices(v)
        for grasp_id, release_id in zip(grasp_ind, release_ind):
            all_grasp_ids.append((k, grasp_id))
            all_release_ids.append((k, release_id))
    sorted_grasp_ids = sorted(all_grasp_ids, key=lambda tup: tup[1])
    sorted_release_ids = sorted(all_release_ids, key=lambda tup: tup[1])
    if visualize:
        full_contacts = get_all_contacts(contacts_per_object, base_value=0)
        cmap = plt.get_cmap("tab10")
        color_dict = {k: cmap(i) for i, k in enumerate(contacts_per_object.keys())}
        plt.plot(full_contacts, color="gray")
        for grasp_tup, release_tup in zip(sorted_grasp_ids, sorted_release_ids):
            plt.plot(
                range(grasp_tup[1], release_tup[1]),
                full_contacts[grasp_tup[1] : release_tup[1]],
                c=color_dict[grasp_tup[0]],
            )
        plt.title("Per object contacts after smoothing")
        plt.show()

    contact_summary = np.zeros(
        (len(contacts_per_object.keys()), len(sorted_grasp_ids) * 2)
    )
    if tray_poses is not None:
        tray_move_array = get_tray_moving_ind(tray_poses)
        if len(tray_move_array) > 1:
            raise ValueError("Come up with strategy for tray moving two times")
        init_tray_pose = np.median(tray_poses[:avg_steps], axis=0)
        # before_pose = np.median(tray_poses[max(0, tray_move_array[0][0] -
        # 5):tray_move_array[0][0]], axis=0)
        final_tray_pose_true = np.median(
            tray_poses[
                tray_move_array[0][1] : min(
                    len(tray_poses), tray_move_array[0][1] + avg_steps
                )
            ],
            axis=0,
        )
        tray_motion_true = final_tray_pose_true.dot(np.linalg.inv(init_tray_pose))
        tray_motion = np.eye(4)
        tray_motion[0, 3] = tray_motion_true[0, 3]
        tray_motion[1, 3] = tray_motion_true[1, 3]
        final_tray_pose = tray_motion.dot(init_tray_pose)

    # j = 0
    new_obj_poses = {k: [] for k in range(len(obj_poses))}
    tray_poses_list = []
    for j, (grasp_tup, release_tup) in enumerate(
        zip(sorted_grasp_ids, sorted_release_ids)
    ):
        previous_release = sorted_release_ids[j - 1][1] - 2 if j > 0 else 0
        # if tray moved between current grasp and previous release (add size 2 gap)
        if tray_poses is not None:
            if (
                tray_move_array[0][0] > previous_release
                and tray_move_array[0][1] < grasp_tup[1] + 2
            ):
                tray_poses_list += [final_tray_pose] * 2
            else:
                tray_poses_list += [
                    init_tray_pose if len(tray_poses_list) == 0 else tray_poses_list[-1]
                ] * 2
        for i, k in enumerate(contacts_per_object.keys()):
            # if no tray
            obj_before_grasp = (
                new_obj_poses[i][-1]
                if len(new_obj_poses[i]) > 0
                else np.median(
                    obj_poses[i][max(0, grasp_tup[1] - avg_steps) : grasp_tup[1]],
                    axis=0,
                )
            )
            if tray_poses is not None:
                # if tray moved between current grasp and previous release
                # (add size 2 gap)
                if (
                    tray_move_array[0][0] > previous_release
                    and tray_move_array[0][1] < grasp_tup[1] + 2
                ):
                    obj_before_grasp = tray_motion.dot(obj_before_grasp)

            limiting_id = (
                len(obj_poses[0])
                if j == len(sorted_grasp_ids) - 1
                else sorted_grasp_ids[j + 1][1]
            )
            obj_after_grasp = np.median(
                obj_poses[i][
                    release_tup[1] : min(limiting_id, release_tup[1] + avg_steps)
                ],
                axis=0,
            )
            if grasp_tup[0] == k:
                contact_summary[i, j * 2] = 1
                new_obj_poses[i] += [obj_before_grasp, obj_after_grasp]
            else:
                contact_summary[i, j * 2] = 0
                new_obj_poses[i] += [obj_before_grasp, obj_before_grasp]
            if release_tup[0] == k:
                contact_summary[i, j * 2 + 1] = 2
            else:
                contact_summary[i, j * 2 + 1] = 0
    #
    #
    # tray_used = False  # come up with smarter way to only use traj once
    # for grasp_tup, release_tup in zip(sorted_grasp_ids, sorted_release_ids):
    #     if tray_used:
    #         obj_possible_motion = None
    #     for i, k in enumerate(contacts_per_object.keys()):
    #         if obj_possible_motion is None:
    #             obj_before_grasp = new_obj_poses[i][-1] if len(new_obj_poses[i]) > 0
    #             else obj_poses[i][grasp_tup[1]]
    #         else:
    #             prev_pose = new_obj_poses[i][-1] if len(new_obj_poses[i]) > 0 else
    #             obj_poses[i][grasp_tup[1]]
    #             dist_traveled = np.linalg.norm(obj_poses[i][grasp_tup[1]] - prev_pose)
    #             if grasp_tup[1] > obj_possible_motion['id'] and np.linalg.norm(
    #                     dist_traveled - obj_possible_motion['dist']) < 0.2:
    #                 obj_before_grasp = obj_poses[i][grasp_tup[1]]
    #                 tray_used = True
    #             else:
    #                 obj_before_grasp = prev_pose
    #         obj_after_grasp = obj_poses[i][release_tup[1]]
    #         if grasp_tup[0] == k:
    #             contact_summary[i, j] = 1
    #             new_obj_poses[i] += [obj_before_grasp, obj_after_grasp]
    #         else:
    #             contact_summary[i, j] = 0
    #             new_obj_poses[i] += [obj_before_grasp, obj_before_grasp]
    #     j += 1
    #     for i, k in enumerate(contacts_per_object.keys()):
    #         if release_tup[0] == k:
    #             contact_summary[i, j] = 2
    #         else:
    #             contact_summary[i, j] = 0
    #     j += 1
    new_obj_poses = np.array([v for k, v in new_obj_poses.items()])
    return contact_summary, new_obj_poses, np.array(tray_poses_list)


def get_all_contacts(contacts_per_object, base_value=None):
    video_len = len(contacts_per_object[list(contacts_per_object.keys())[0]])
    if base_value is None:
        combined_contacts = [
            0
            if sum([0 if x[i] is None else 1 for k, x in contacts_per_object.items()])
            == 0
            else 1
            for i in range(video_len)
        ]
    else:
        combined_contacts = [
            0
            if sum(
                [0 if x[i] == base_value else 1 for k, x in contacts_per_object.items()]
            )
            == 0
            else 1
            for i in range(video_len)
        ]
    return combined_contacts


def convert_obj_poses(obj_poses):
    """
    Converts object poses from camera frame to first object frame.
    First position of first object becomes a world Frame and all other matrices are
    expressed in this new world frame
    :param obj_poses: KxNx4x4
    :return: new_obj_poses: KxNx4x4
    """
    new_world_frame = obj_poses[0][0]
    assert new_world_frame.shape == (4, 4)
    new_obj_poses = np.zeros_like(obj_poses)
    for k, k_obj_poses in enumerate(obj_poses):
        for j, obj_pose in enumerate(k_obj_poses):
            new_obj_poses[k, j, :, :] = np.linalg.inv(new_world_frame).dot(obj_pose)
    return new_obj_poses


#
# def smooth_obj_poses_kalman(obj_poses, R_std=0.5, Q_std=0.3):
#     first_x, first_y, first_z = obj_poses[0][:3, 3]
#     dim_z = 3
#     tracker = KalmanFilter(dim_x=6, dim_z=dim_z)
#     dt = 1 / 10
#
#     tracker.F = np.array([[1, dt, 0, 0, 0, 0],
#                           [0, 1, 0, 0, 0, 0],
#                           [0, 0, 1, dt, 0, 0],
#                           [0, 0, 0, 1, 0, 0],
#                           [0, 0, 0, 0, 1, dt],
#                           [0, 0, 0, 0, 0, 1]
#                           ])
#     tracker.u = 0.
#     tracker.H = np.array([[1, 0, 0, 0, 0, 0],
#                           [0, 0, 1, 0, 0, 0],
#                           [0, 0, 0, 0, 1, 0]])
#     tracker.R = np.eye(dim_z) * R_std ** 2
#     q = Q_discrete_white_noise(dim=2, dt=dt, var=Q_std ** 2)
#     tracker.Q = block_diag(q, q, q)
#     tracker.x = np.array([[first_x, 0, first_y, 0, first_z, 0]]).T
#     tracker.P = np.eye(6) * 0.1
#
#     # t_array = np.array([x[:3, 3] for x in obj_poses])
#     zs = np.array([x[:3, 3] for x in obj_poses])
#     mu, cov, _, _ = tracker.batch_filter(zs)
#     # plt.plot(zs, linestyle='--')
#     # # plt.show()
#     # plt.plot(mu[:, 0, :], marker='x', linestyle=':')
#     # plt.plot(mu[:, 2, :], marker='x', linestyle=':')
#     # plt.plot(mu[:, 4, :], marker='x', linestyle=':')
#     # plt.show()
#     return mu[:, [0, 2, 4], :]


def smooth_obj_poses_mean(obj_poses, n=5):
    t_array = np.array([x[:3, 3] for x in obj_poses])
    start_replace = int(n / 2)
    smoothed_arr = np.apply_along_axis(
        lambda m: np.convolve(m, np.ones(n) / n, mode="valid"), axis=0, arr=t_array
    )
    t_array[start_replace : start_replace + len(smoothed_arr)] = smoothed_arr
    return t_array


def smooth_obj_poses(obj_poses, window=5):
    """
    Discard outliers (translation/rotation)
    Assume that first detection is okay (maybe add some check for that)
    :param obj_poses:
    :return:
    """
    # check that first is consistent
    for obj_id in range(len(obj_poses)):
        t_array = np.array([x[:3, 3] for x in obj_poses[obj_id]])
        # new_t_array = smooth_obj_poses_kalman(obj_poses[obj_id])
        new_t_array = smooth_obj_poses_mean(obj_poses[obj_id], n=10)

        plt.plot(t_array, linestyle="--")
        plt.plot(new_t_array, marker="x", linestyle=":")
        plt.title("Object poses")
        plt.show()

        # for i in range(len(t_array) - 1):
        #     if np.linalg.norm(t_array[i + 1] - t_array[i]) > 0.1:
        #         t_array[i + 1] = t_array[i]
        obj_poses[obj_id, :, :3, 3] = new_t_array.copy()

    return obj_poses


def smooth_contacts(contacts, window=5, threshold=0.6):
    """
    Smooth contact signal to remove spikes and valleys.
    Assume that if neighboring points average is >= threshold - it is in contact
    :param contacts:
    :param window: N for moving average window
    :param threshold: threshold to filter 1
    :return:
    """
    moving_avg_cont = np.convolve(contacts, np.ones(window) / window, "same")
    return np.where(moving_avg_cont >= threshold, 1, 0)


def get_object_moving_indication(obj_poses, window=5, threshold=0.2):
    """
    Transform sequence of object poses to 0-1 array that indicates if object moves a lot
    :param obj_poses: Nx4x4
    :param window: N for moving average window
    :param threshold: threshold to filter 1
    :return:
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


def get_grasp_release_indices(contacts):
    diff_in_contact = np.array(contacts)[1:] - np.array(contacts)[:-1]
    grasp_ind = np.where(diff_in_contact == 1)[0]
    release_ind = np.where(diff_in_contact == -1)[0]
    assert len(grasp_ind) == len(grasp_ind)
    return grasp_ind, release_ind


def get_obj_contacts(
    contacts, obj_poses, grasp_pre_steps=10, release_post_steps=10, threshold=0.25
):
    """
    Go from contact information to grasp-release format.
    Record object positions
    Steps:
    - get array when object moves
    - get grasp-release indices
    - try to find corresponding contact sequence
    :param contacts:
    :param obj_poses: KxNx4x4 pose matrices for K objects
    :return: KxMx4x4 pose matrices for K objects where M is amount of
    graps-release states
    :return: grasp-release dataframe
    """
    grasp_release = {}
    for obj in range(len(obj_poses)):
        grasp_release[obj] = []

    new_obj_poses = {}
    for obj in range(len(obj_poses)):
        new_obj_poses[obj] = []
    # compute arrays that indicate object motion
    obj_moving_arrays = []
    for obj in range(len(obj_poses)):
        obj_moving_arrays.append(get_object_moving_indication(obj_poses[obj]))

    grasp_idx_candidates, release_idx_candidates = get_grasp_release_indices(contacts)
    grasp_idx = []
    release_idx = []
    for grasp_id, release_id in zip(grasp_idx_candidates, release_idx_candidates):
        # find which object moves in this time
        obj_moved = [
            i
            for (i, obj_moving) in enumerate(obj_moving_arrays)
            if sum(obj_moving[grasp_id:release_id])
            > (release_id - grasp_id) * threshold
        ]
        if len(obj_moved) == 0:
            # no object moves
            continue
        if len(obj_moved) > 1:
            raise ValueError("Two object move at the same time")
        grasp_idx.append(grasp_id)
        release_idx.append(release_id)
        for i in range(len(obj_moving_arrays)):
            # TODO: maybe replace with mean
            # obj_pose_before_grasp = np.mean([x[:3, 3] for x in obj_poses[i][max(0,
            # grasp_id - grasp_pre_steps):grasp_id]], axis=0)
            # obj_pose_after_grasp = np.mean([x[:3, 3] for x in obj_poses[i][
            # release_id:min(len(obj_poses[i]), release_id +
            # release_post_steps):grasp_id]], axis=0)
            obj_pose_before_grasp = (
                new_obj_poses[i][-1]
                if len(new_obj_poses[i]) > 0
                else obj_poses[i][grasp_id]
            )
            obj_pose_after_grasp = obj_poses[i][release_id]
            if i == obj_moved[0]:
                new_obj_poses[i] += [obj_pose_before_grasp, obj_pose_after_grasp]
                grasp_release[i] += [1, 2]
            else:
                # static object
                new_obj_poses[i] += [obj_pose_before_grasp, obj_pose_before_grasp]
                grasp_release[i] += [0, 0]

    res1 = np.array([v for k, v in new_obj_poses.items()])
    res2 = np.array([v for k, v in grasp_release.items()])
    return res1, res2, grasp_idx, release_idx
