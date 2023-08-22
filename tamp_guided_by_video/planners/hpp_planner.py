from guided_tamp_benchmark.core import Path, Configuration, BasePlanner
from guided_tamp_benchmark.tasks import BaseTask
from hpp.corbaserver.manipulation import (
    ProblemSolver,
    ConstraintGraph,
    Rule,
    ConstraintGraphFactory,
)
from tamp_guided_by_video.utils.utils import (
    get_trans_quat_hpp,
    get_default_joint_bound,
    filter_handles,
)
from tamp_guided_by_video.utils.planner_utils import get_config_states
from tamp_guided_by_video.utils.demo_processing import (
    ensure_normalized,
    parse_demo_into_subgoals,
)
from tamp_guided_by_video.utils.robot import HppRobot
from tamp_guided_by_video.utils.corba import CorbaServer
import pinocchio as pin
import numpy as np
import time
from typing import Optional


class HPPPath(Path):
    """Helper class for extracting trajectories from HPP solution"""

    def __init__(self, planner):
        super().__init__()
        self.planner = planner
        self.path_id = planner.ps.numberPaths() - 1
        self.robot_ndof = len(planner.task.robot.initial_configuration())

    def interpolate(self, t: float) -> Configuration:
        planner_t = t * self.planner.ps.pathLength(self.path_id)
        x = self.planner.ps.configAtParam(self.path_id, planner_t)
        return Configuration.from_numpy(np.array(x), self.robot_ndof)


class HppPlanner(BasePlanner):
    """
    This planner uses standard HPP RRT method to solve a planning task.
    """

    def __init__(
        self,
        task: BaseTask,
        optimize: bool = False,
        max_planning_time: Optional[int] = None,
        handles_mode: str = "all",
        random_seed: Optional[int] = None,
        corba: CorbaServer = None,
        verbose: bool = False,
        steps: Optional[list[int]] = None,
    ):
        self.task = task
        self.optimize = optimize
        self.max_planning_time = max_planning_time
        self.handles_mode = handles_mode
        self.random_seed = random_seed
        self.corba = corba
        self.verbose = verbose
        self.tray_poses = None
        self.robot = HppRobot(robot=self.task.get_robot())
        self.graph_name = "_".join(
            [
                self.name,
                task.demo.task_name,
                str(task.demo.demo_id),
                self.robot.name,
                str(task.demo.pose_id),
            ]
        )
        self.init_problem_solver()
        if random_seed is not None:
            self.set_random_seed(random_seed)
            self.graph_name += str(random_seed)
        (
            env_contact_surfaces,
            handles_names,
            object_surfaces,
            object_names,
        ) = self.init_environment()
        self.handles_names = handles_names
        self.init_constrain_graph(
            env_contact_surfaces, handles_names, object_surfaces, object_names
        )
        # remove collisions with disabled furniture links
        for obj in self.task.furniture:
            if hasattr(obj, "disabled_collision_links_for_robot"):
                for link in obj.disabled_collision_links_for_robot:
                    self.disable_robot_collision_for(f"{obj.name}/{link}_0")
        self.contacts, noisy_object_poses, self.tray_poses = parse_demo_into_subgoals(
            task.demo
        )  # get poses on contact change
        self.object_poses = self.project_object_poses(noisy_object_poses)

        if steps is not None:
            self.contacts = self.contacts[:, steps]
            self.object_poses = self.object_poses[:, steps]
        self.config_list = []
        self.grasp_number = None

    @property
    def name(self) -> str:
        """String name of the planner"""
        return "hpp_shortcut" if self.optimize else "hpp"

    def init_problem_solver(self, error_threshold=1e-3, max_iter_projection=40):
        """
        Creates a problem solver :param error_threshold: Sets error threshold in
        numerical constraint resolution via ps.setErrorThreshold() :param
        max_iter_projection: Sets the maximal number of iterations in projection via
        ps.setMaxIterProjection()
        """

        self.ndof = len(self.robot.initial_configuration())
        self.robot.setRootJointPosition(
            self.robot.name, get_trans_quat_hpp(self.task.get_robot_pose())
        )
        self.ps = ProblemSolver(self.robot)
        self.ps.setErrorThreshold(error_threshold)
        self.ps.setMaxIterProjection(max_iter_projection)
        if self.max_planning_time is not None:
            self.ps.setTimeOutPathPlanning(self.max_planning_time)

    def init_environment(self) -> (list[str], list[str], list[str], list[str]):
        """
        Adds furniture and movable objects to the hpp environment
        Return environment contact surfaces names, handle names,
            object contact surfaces names, object names
        """
        # load furniture
        env_contact_surfaces = self.robot.get_contact_surfaces()
        for item in self.task.get_furniture():
            prefix = item.name + "/"
            if self.verbose:
                print(f"Creating {prefix} furniture")
            self.robot.loadEnvironmentModel(
                item.urdfFilename, item.srdfFilename, prefix
            )
            env_contact_surfaces += item.contact_surfaces(prefix)

        # load movable objects
        handles_names = []
        object_surfaces = []
        object_names = []
        for item in self.task.get_objects():
            if item.name == "tray" or item.name == "base":
                continue
            prefix = item.name + "/"
            if self.verbose:
                print(f"Creating {prefix} object")
            object_names.append(prefix)
            handles_names.append(
                filter_handles(item.handles(prefix), self.handles_mode)
            )
            object_surfaces.append([item.contact_surfaces(prefix)])
            self.robot.insertRobotModel(
                item.name, item.rootJointType, item.urdfFilename, item.srdfFilename
            )
            self.robot.setJointBounds(
                f"{item.name}/root_joint", get_default_joint_bound()
            )

        return env_contact_surfaces, handles_names, object_surfaces, object_names

    def init_constrain_graph(
        self,
        env_contact_surfaces: list[str],
        handles_names: list[str],
        object_surfaces: list[str],
        object_names: list[str],
    ):
        """
        initializes constraint graph for HPP

        :param env_contact_surfaces: list of environmental contact surfaces
        :param handles_names: list of handle names for all objects
        :param object_surfaces: list of object contact surfaces
        :param object_names: list of object names
        """
        grippers = [
            self.robot.get_gripper_name(),
        ]  # assume we only have one gripper
        rules = [
            Rule([".*"], [".*"], True),
        ]
        self.cg = ConstraintGraph(self.robot, graphName=self.graph_name)
        self.factory = ConstraintGraphFactory(self.cg)
        self.factory.setGrippers(
            grippers
        )  # Define the set of grippers used for manipulation
        self.factory.environmentContacts(env_contact_surfaces)
        self.factory.setObjects(object_names, handles_names, object_surfaces)
        self.factory.setRules(rules)
        self.factory.generate()
        self.cg.initialize()

        # validate graph
        cproblem = self.ps.hppcorba.problem.getProblem()
        cgraph = cproblem.getConstraintGraph()
        cgraph.initialize()
        graphValidation = self.ps.client.manipulation.problem.createGraphValidation()
        if not graphValidation.validate(cgraph):
            print("Graph validation FAILED!")
            time.sleep(1.0)

    def clear_roadmap(self):
        """Clear problem solver roadmap and erase all paths"""
        self.ps.clearRoadmap()
        for i in range(self.ps.numberPaths() - 1, -1, -1):
            self.ps.erasePath(i)
        self.ps.resetGoalConfigs()

    def set_random_seed(self, random_seed: int):
        self.ps.setRandomSeed(random_seed)

    def project_to_free(self, q: list[float]) -> (bool, list[float]):
        """Project configuration to the closest configuration in 'free'"""
        if self.cg.getConfigErrorForNode("free", q)[0]:
            return True, q
        else:
            succ, q_free, err = self.cg.graph.applyNodeConstraints(
                self.cg.nodes["free"], q
            )
            if succ:
                return True, q_free
            else:
                return False, q

    def project_object_poses(self, object_poses: np.array) -> np.array:
        """
        Projects an array of object poses to the closest configuration in 'free'
        Robot configuration is fixed to be initial_configuration
        """
        projected_object_poses = np.zeros_like(object_poses)
        robot_q = self.robot.initial_configuration()
        # if tray was in demo, obj poses are updated w.r.t. tray
        q_list = []
        for time_id in range(object_poses.shape[1]):
            if self.tray_poses is not None:
                robot_q[0] = self.tray_poses[time_id][0, 3] - 0.2
                robot_q[1] = self.tray_poses[time_id][1, 3]

            init_q = list(
                np.concatenate(
                    [robot_q]
                    + [
                        ensure_normalized(pin.SE3ToXYZQUAT(pin.SE3(pose)))
                        for pose in object_poses[:, time_id]
                    ]
                )
            )
            q_list.append(init_q)
            res, q = self.project_to_free(init_q)
            assert res
            q_list.append(q)
            projected_object_poses[:, time_id] = Configuration.from_numpy(
                np.array(q), len(robot_q)
            ).poses

        return projected_object_poses

    def disable_robot_collision_for(self, name: str, verbose: bool = False):
        """
        Disable collision between robot and specific object.
        :param name: name of the object for which the collision will be disabled
        """

        all_joints = [x for x in self.robot.getAllJointNames() if "joint" in x]
        remove_joints = [
            x
            for x in all_joints
            if "root_joint" in x
            or "panda_hand_joint" in x
            or "fixed" in x
            or "iiwa_joint_ee" in x
            or "panda_joint8" in x
            or "panda_camera_mount_joint" in x
            or "finger_joint1_collision" in x
            or "finger_joint2_collision" in x
        ]
        for joint_name in list(set(all_joints) - set(remove_joints)):
            if verbose:
                print(f"remove collision pair {joint_name}-{name}")
            self.robot.removeObstacleFromJoint(objectName=name, jointName=joint_name)

    def compute_metrics(self) -> (list[Configuration], int):
        """
        Compute metrics after hpp solution is found (configuration list, grasp number).
        """
        configs = self.get_last_path_hpp_configs()
        robot_ndof = len(self.task.robot.initial_configuration())
        self.config_list = [
            Configuration.from_numpy(np.array(x), robot_ndof) for x in configs
        ]
        grasp_count = 0
        was_grasped = False
        for i, q in enumerate(configs):
            config_states = get_config_states(self, q)
            is_grasped = bool(len([k for k in config_states if "grasps" in k]))
            if is_grasped:
                if not was_grasped:
                    grasp_count += 1
                    was_grasped = True
            else:
                was_grasped = False
        self.grasp_number = grasp_count * 2
        return self.config_list, self.grasp_number

    def get_last_path_hpp_configs(self, fps: int = 24) -> list[list[float]]:
        """
        Returns the list of configurations from the last hpp path. If the task was
        solved successfully, it will correspond to the solution trajectory.
        """
        path_id = self.ps.numberPaths() - 1
        nframes = np.ceil(self.ps.pathLength(path_id) * fps).astype(int)
        return [
            self.ps.configAtParam(path_id, t)
            for t in np.linspace(0.0, self.ps.pathLength(path_id), nframes)
        ]

    def get_start_goal(self) -> (list[float], list[float]):
        """Function to create start and goal configurations for the current task"""
        q_start = list(
            np.concatenate(
                [self.task.robot.initial_configuration()]
                + [
                    ensure_normalized(pin.SE3ToXYZQUAT(pin.SE3(pose)))
                    for pose in self.task.demo.subgoal_objects_poses[:, 0]
                ]
            )
        )
        q_goal = list(
            np.concatenate(
                [self.task.robot.initial_configuration()]
                + [
                    ensure_normalized(pin.SE3ToXYZQUAT(pin.SE3(pose)))
                    for pose in self.task.demo.subgoal_objects_poses[:, -1]
                ]
            )
        )
        return q_start, q_goal

    def solve(self) -> bool:
        """Solve planning problem with classical hpp"""
        q_start, q_goal = self.get_start_goal()

        self.clear_roadmap()
        self.ps.setInitialConfig(q_start)
        self.ps.addGoalConfig(q_goal)
        try:
            if self.optimize:
                self.ps.addPathOptimizer("RandomShortcut")
            self.ps.solve()
            self.compute_metrics()
            return True
        except BaseException as e:
            if self.verbose:
                print(e)
            return False

    def get_path(self) -> Path:
        """Return HPPPath of the planner"""
        return HPPPath(self)

    def reset(self):
        """Reset corba problem. Used in benchmark script"""
        self.corba.reset_problem()
