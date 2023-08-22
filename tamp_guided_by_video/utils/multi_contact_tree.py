import numpy as np
from anytree import Node
from tamp_guided_by_video.utils.planner_utils import get_config_states
from guided_tamp_benchmark.core import BasePlanner
from typing import Optional


class FreePlane:
    def __init__(
        self,
        id: int,
        object_poses: list[list[float]],
        object_translations_2d: list[float],
        q_from: list[float],
    ):
        """
        Define free space manifold identified by object positions
        """
        self.id = id
        self.obj_poses = object_poses
        self.obj_2d_t = np.array(object_translations_2d)
        self.state = "free"
        self.q_from = q_from

    def __str__(self):
        return "".join(
            [
                f"obj{i}: ({round(obj_t[0], 2)}, {round(obj_t[1], 2)})\n"
                for i, obj_t in enumerate(self.obj_2d_t)
            ]
        )


class GraspPlane:
    def __init__(self, id: int, state: str, q_from: list[float]):
        """
        Define grasp manifold with fixed object
        """
        self.id = id
        self.state = state
        self.q_from = q_from

    def __str__(self):
        return f"{self.state}"


class Tree:
    """
    Object of this class will store all the data that represents configurations in
    our space: - free planes where new plane is created for new object positions -
    grasp planes, one per each handle - nodes contain configurations, and can belong
    to one of the mentioned planes, have color and marker for visualization Nodes can
    have children nodes, then during the visualization an edge will be drawn
    """

    def __init__(self, planner: BasePlanner, robot_n: int):
        self.planner = planner
        self.robot_n = robot_n
        self.robot_contact_surf = self.planner.robot.get_contact_surfaces()
        self.free_planes = {}
        self.grasp_planes = []  # TODO: change into dictionary?
        self.root_nodes = []
        self.node_planes = {}
        self.node_count = 0

    def reset(self):
        """Clear the tree"""
        self.free_planes = {}
        self.grasp_planes = []
        self.root_nodes = []
        self.node_planes = {}
        self.node_count = 0

    def add_root_node(
        self,
        config: list[float],
        step: int = 0,
        marker: str = "x",
        start_subgoal: bool = False,
        end_subgoal: bool = False,
    ) -> Node:
        """Create a root node and add to root node lists"""
        root_node = self.add_node(
            config,
            step=step,
            marker=marker,
            start_subgoal=start_subgoal,
            end_subgoal=end_subgoal,
        )
        self.root_nodes.append(root_node)
        self.node_planes[root_node] = [root_node.plane]
        return root_node

    def add_node(
        self,
        config: list[float],
        step: int = 0,
        start_subgoal: bool = False,
        end_subgoal: bool = False,
        parent: Optional[Node] = None,
        marker: str = "x",
    ) -> Node:
        """
        Assume that added node is new and not repeated
        :param config: configuration to create node
        :param parent: parent node
        :param marker: marker
        :return:
        """
        # create a new configuration node, assign a plane to it
        # (depends on configuration states)
        states = get_config_states(self.planner, config)
        if "free" in " ".join(states):
            # will create new node with free plane
            obj_poses, obj_t = self.get_object_poses(config)
            # any objects on top of robot contact surface?
            objects_to_ignore = []
            if len(self.robot_contact_surf) > 0:
                objects_to_ignore += self.get_obj_ids_on_surface(
                    config, self.robot_contact_surf[0], np.array(obj_t)
                )
                obj_poses = [
                    item
                    for i, item in enumerate(obj_poses)
                    if i not in objects_to_ignore
                ]
                obj_t = [
                    item for i, item in enumerate(obj_t) if i not in objects_to_ignore
                ]
            # check if free plane already exists
            for id, free_plane in self.free_planes.items():
                if (
                    len(free_plane.obj_2d_t) == len(obj_t)
                    and np.linalg.norm(free_plane.obj_2d_t - obj_t) < 0.01
                ):
                    return self.create_node(
                        config=config,
                        parent=parent,
                        step=step,
                        start_subgoal=start_subgoal,
                        end_subgoal=end_subgoal,
                        plane=free_plane,
                        marker=marker,
                    )
            new_free_plane = FreePlane(
                id=(step + 1) // 2,
                object_poses=obj_poses,
                object_translations_2d=obj_t,
                q_from=config,
            )
            self.free_planes[(step + 1) // 2] = new_free_plane
            return self.create_node(
                config=config,
                parent=parent,
                step=step,
                start_subgoal=start_subgoal,
                end_subgoal=end_subgoal,
                plane=new_free_plane,
                marker=marker,
            )
        for state in states:
            if "grasp" in state:
                for grasp_plane in self.grasp_planes:
                    if grasp_plane.state == state:
                        return self.create_node(
                            config=config,
                            parent=parent,
                            step=step,
                            start_subgoal=start_subgoal,
                            end_subgoal=end_subgoal,
                            plane=grasp_plane,
                            marker=marker,
                        )
                new_grasp_plane = GraspPlane(len(self.grasp_planes), state, config)
                self.grasp_planes.append(new_grasp_plane)
                return self.create_node(
                    config=config,
                    parent=parent,
                    step=step,
                    start_subgoal=start_subgoal,
                    end_subgoal=end_subgoal,
                    plane=new_grasp_plane,
                    marker=marker,
                )

        raise ValueError(f"Non processed state {states}")

    def create_node(
        self,
        config: list[float],
        plane: FreePlane | GraspPlane,
        step: int = 0,
        parent: Optional[Node] = None,
        start_subgoal: bool = False,
        end_subgoal: bool = False,
        marker: str = "x",
    ) -> Node:
        """Creates a node in a given plane"""
        node = Node(
            name=self.node_count,
            step=step,
            plane=plane,
            config=config,
            parent=parent,
            marker=marker,
            start_subgoal=start_subgoal,
            end_subgoal=end_subgoal,
        )
        self.node_count += 1
        return node

    def get_obj_ids_on_surface(
        self, config: list[float], surf_name: str, o_pos_2d: list[float]
    ) -> list[int]:
        """
        Get ids of objects that lie on certain surface.
        This function is useful to detect the objects that lie on the
        robot contact surface.
        As motion of this objects will be ignored in free plane creation.
        """
        contact_points = self.planner.ps.getRobotContact(surf_name)[2]
        robot_pose = np.eye(4)
        robot_pose[0, 3] = config[0]
        robot_pose[1, 3] = config[1]
        z_rot = config[2]
        robot_pose[:3, :3] = np.array(
            [
                [np.cos(z_rot), -np.sin(z_rot), 0.0],
                [np.sin(z_rot), np.cos(z_rot), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        contact_points_2d = np.array(
            [robot_pose.dot(np.array(p + [1.0]))[:2] for p in contact_points]
        )
        ll = (contact_points_2d[:, 0].min(), contact_points_2d[:, 1].min())
        ur = (contact_points_2d[:, 0].max(), contact_points_2d[:, 1].max())
        inidx = np.all(np.logical_and(ll <= o_pos_2d, o_pos_2d <= ur), axis=1)
        return [i for i, x in enumerate(inidx) if x]

    def get_object_poses(self, config: list[float]) -> (list[list[float]], list[float]):
        """Returns object poses and object 2d positions from config"""
        object_poses = [
            config[self.robot_n + 7 * i : self.robot_n + 7 * (i + 1)]
            for i in range((len(config) - self.robot_n) // 7)
        ]
        object_translations_2d = [obj_pose[:2] for obj_pose in object_poses]
        return object_poses, object_translations_2d
