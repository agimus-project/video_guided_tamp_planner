from guided_tamp_benchmark.tasks import ShelfTask, TunnelTask, WaiterTask, BaseTask
from guided_tamp_benchmark.models.robots import (
    PandaRobot,
    UR5Robot,
    KukaMobileIIWARobot,
    BaseRobot,
)
import numpy as np
import quaternion as npq
from typing import Optional, Type
import pinocchio as pin


def get_matrix_hpp(position_array: list[float]) -> np.array:
    """
    Convert list of translation and quaternion in (x, y, z, w) format used by
    hpp to 4 x 4 numpy matrix
    :param: position_array: list of [translation, quaternion]
    :return: matrix: 4 x 4 numpy array
    """
    matrix = np.eye(4)
    matrix[:3, 3] = position_array[:3]
    q_hpp = position_array[3:]
    q_npq = npq.from_float_array([q_hpp[3], q_hpp[0], q_hpp[1], q_hpp[2]])
    matrix[:3, :3] = npq.as_rotation_matrix(q_npq)
    return matrix


def get_trans_quat_hpp(
    matrix: np.array, shift: Optional[list[float]] = None
) -> list[float]:
    """
    Convert 4 x 4 numpy matrix to list of translation and quaternion in (x, y, z, w)
    format used by hpp
    :param: matrix: 4 x 4 numpy array
    :param shift: optional shift of translation
    :return: list of [translation, quaternion]
    """
    if matrix.shape != (4, 4):
        raise ValueError(
            f"matrix expected to be a numpy array of shape (4, 4), "
            f"not {matrix.shape}"
        )
    if shift is not None:
        if len(shift) != 3:
            raise ValueError(f"shift len should be 3, not {len(shift)}")
        t = matrix[:3, 3] + np.array(shift)
    else:
        t = matrix[:3, 3]
    m = pin.SE3(translation=t, rotation=matrix[:3, :3])
    return pin.SE3ToXYZQUAT(m).tolist()


def get_default_joint_bound(min_pos: float = -5.0, max_pos: float = 5.0) -> list[float]:
    return [min_pos, max_pos] * 3 + [-1.0001, 1.0001] * 4


def filter_handles(handles_list: list[str], mode: str = "all"):
    """
    Function to select certain handles based on the 'mode'
    This function relies on  handles following certain naming convention:
        handleAbc(Sd)
            A - X/Y/Z from which coordinate will the gripper come from
            b - m/p if the approach is from minus (m) or plus (p)
            c - x/y/z is coordinate in which the width of the grip is represented
            S - optional S means that the handle is a side handle
            d - m/p argument tells if the side handle is on minus or plus side
            of cuboid on the third axis

    :parameter handles_list list of all handles
    :parameter mode can take 'only_top', 'exclude_side'
    (all other values are treated as 'all')
    """
    if mode == "only_top":
        return [h for h in handles_list if "handleZp" in h and h[-2] != "S"]
    if mode == "exclude_side":
        return [h for h in handles_list if h[-2] != "S"]
    if mode == "only_top_side":
        return [h for h in handles_list if "handleZmxSm" in h or "handleZpxSm" in h]
    # all
    return handles_list


def get_task(task_name: str) -> Type[BaseTask]:
    """Returns a task class based on task name"""
    if task_name == "shelf":
        return ShelfTask
    elif task_name == "tunnel":
        return TunnelTask
    elif task_name == "waiter":
        return WaiterTask
    else:
        raise ValueError(f"Unknown task '{task_name}'")


def get_robot(robot_name: str) -> BaseRobot:
    """Returns a robot instance based on robot name"""
    if robot_name == "panda":
        return PandaRobot()
    elif robot_name == "ur5":
        return UR5Robot()
    elif robot_name == "kmr_iiwa":
        return KukaMobileIIWARobot()
    else:
        raise ValueError(f"Unknown robot '{robot_name}'")
