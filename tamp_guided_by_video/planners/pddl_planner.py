from guided_tamp_benchmark.core.planner import BasePlanner
from guided_tamp_benchmark.core import Path, Configuration
from guided_tamp_benchmark.tasks import BaseTask
from tamp_guided_by_video.utils.pddl_utils import (
    load_world,
    pddlstream_from_problem,
    postprocess_plan,
)
import random
import numpy as np
from pddlstream.algorithms.meta import solve
from examples.pybullet.utils.pybullet_tools.utils import connect, disconnect
import pinocchio as pin
from typing import Optional


class PDDLPath(Path):
    """Helper class for extracting trajectories from PDDL solution"""

    def __init__(self, planner):
        super().__init__()
        self.planner = planner
        self.robot_ndof = len(planner.task.robot.initial_configuration())

    def interpolate(self, t: float) -> Configuration:
        idx = int(t * (len(self.planner.config_list) - 1))
        return self.planner.config_list[idx]

class PDDLPlanner(BasePlanner):
    """
    This is pddl planner from Caelan R. Garrett, Tomás Lozano-Pérez,
    Leslie P. Kaelbling.
    To use please install https://github.com/caelan/pddlstream
    Please change return of solve_abstract() in
    pddlstream/pddlstream/algorithms/focused.py to be
    return store.extract_solution(), summary # TODO: fix this more elegant
    """

    def __init__(
        self,
        task: BaseTask,
        max_planning_time: int = 1000,
        handles_mode: str = "all",
        random_seed: Optional[int] = None,
        verbose: bool = False
    ):
        self.task = task
        self.max_planning_time = max_planning_time
        self.handles_mode = handles_mode
        self.start_obj_poses = self.task.demo.subgoal_objects_poses[:, 0]
        self.goal_obj_poses = self.task.demo.subgoal_objects_poses[:, -1]
        if random_seed is not None:
            random.seed(random_seed)
        self.verbose = verbose
        connect(use_gui=self.verbose)
        self.config_list = []

    @property
    def name(self) -> str:
        return "pddl"

    def solve(self) -> bool:
        robot, names, movable, disable_body_links = load_world(
            self.task, self.start_obj_poses, self.task.get_robot_pose()
        )
        problem = pddlstream_from_problem(
            robot,
            final_poses=self.goal_obj_poses,
            objects=self.task.objects,
            robot_init_config=self.task.robot.initial_configuration()[:-2],
            movable=movable,
            teleport=False,
            disable_body_links=disable_body_links,
            grasps="ours",
            robot_name=self.task.robot.name,
            allow_side_handles=True if self.handles_mode == 'all' else False,
        )
        _, _, _, stream_map, init, goal = problem
        solution, summary = solve(
            problem,
            algorithm="adaptive",
            unit_costs=False,
            success_cost=float("inf"),
            verbose=self.verbose,
            max_time=self.max_planning_time,
        )
        self.time = summary["run_time"]
        # success
        solved = int(summary["solved"] and summary["run_time"] < self.max_planning_time)
        if solved:
            plan, cost, evaluations = solution
            command = postprocess_plan(plan)
            # TODO: cleanup this part
            self.config_list = []
            obj_poses = self.start_obj_poses
            r_model = pin.buildModelFromUrdf(self.task.robot.urdfFilename)
            r_data = r_model.createData()
            world_robot_m = self.task.demo.robot_pose
            frameIndex = r_model.getFrameId("panda_hand")
            for body_path in command.body_paths:
                if hasattr(body_path, "path"):
                    for i, configuration in enumerate(body_path.path):
                        q = np.array(
                            list(configuration) + [0.039, 0.039]
                        )  # add open gripper
                        pin.forwardKinematics(r_model, r_data, q)
                        pin.updateFramePlacements(r_model, r_data)
                        robot_gripper_m = np.array(r_data.oMf[frameIndex])
                        for grasp in body_path.attachments:
                            grasp_obj_m = np.array(
                                pin.XYZQUATToSE3(
                                    list(grasp.grasp_pose[0]) + grasp.grasp_pose[1]
                                )
                            )
                            world_obj_m = world_robot_m.dot(robot_gripper_m).dot(
                                grasp_obj_m
                            )
                            object_id = movable.index(grasp.body)
                            obj_poses[object_id] = world_obj_m
                        self.config_list.append(Configuration(q=q, poses=obj_poses))

        # grasp_number
        self.grasp_number = None
        if solved:
            grasp_count = 0
            for name, _ in plan:
                if "pick" in name:
                    grasp_count += 1
            self.grasp_number = grasp_count * 2
        disconnect()
        return solved

    def get_path(self) -> Path:
        return PDDLPath(self)
