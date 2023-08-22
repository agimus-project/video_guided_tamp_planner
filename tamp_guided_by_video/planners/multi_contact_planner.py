from tamp_guided_by_video.planners.hpp_planner import HppPlanner
from tamp_guided_by_video.utils.multi_contact_tree import (
    Tree,
    FreePlane,
    GraspPlane,
    Node,
)
from tamp_guided_by_video.utils.planner_utils import (
    sample_state_on_transition_target,
    build_direct_path_any_direction,
)
from tamp_guided_by_video.utils.corba import CorbaServer
from guided_tamp_benchmark.core import Configuration
from guided_tamp_benchmark.tasks import BaseTask
from itertools import chain
from anytree import PreOrderIter
import numpy as np
import random
import time
from tamp_guided_by_video.utils.utils import get_matrix_hpp
from typing import Optional


class MultiContactPlanner(HppPlanner):
    """
    This planner uses proposed multi-tree RRT method guided by video
    to solve a planning task.
    """

    def __init__(
        self,
        task: BaseTask,
        optimize: bool = False,
        max_planning_time: Optional[int] = None,
        handles_mode: str = "all",
        step_size: float = 0.2,
        max_iter: int = 1000,
        sample_subgoal_prob: float = 0.1,
        corba: CorbaServer = None,
        use_euclidean_distance: bool = False,
        euclidean_dist_iter: int = 1,
        random_seed: int = None,
        optimize_path_iter: Optional[int] = None,
        verbose: bool = False,
        steps: Optional[list[int]] = None,
    ):
        self.step_size = step_size
        self.max_iter = max_iter
        self.sample_subgoal_prob = sample_subgoal_prob
        self.use_euclidean_distance = use_euclidean_distance
        self.euclidean_dist_iter = euclidean_dist_iter
        self.optimize_path_iter = optimize_path_iter
        super().__init__(
            task,
            optimize=optimize,
            max_planning_time=max_planning_time,
            handles_mode=handles_mode,
            random_seed=random_seed,
            verbose=verbose,
            corba=corba,
            steps=steps,
        )
        self.tree = Tree(planner=self, robot_n=self.ndof)
        self.time = None
        self.solved = False
        self.q_start = None
        self.q_goal = None

    @property
    def name(self):
        return (
            "multi_contact_shortcut"
            if self.optimize_path_iter is not None
            else "multi_contact"
        )

    def solve(self) -> bool:
        """
        Find a path between start and goal configuration by spanning multiple trees.
        Sample initial subgoals and build a tree between them: - with 10% probability
        generate new subgoal - with 90% probability generate random configuration,
        project to one of the states (free, grasps) - try to append to all trees that
        have nodes in the current state, find closest node - make step from closest
        node in direction of random config, add this new node - try to connect to all
        other trees to new node, do a sequence of small steps - if the last step
        connects to the other tree - merge trees

        """
        q_start, q_goal = self.get_start_goal()
        self.clear_roadmap()
        # check if initial start and goal object poses are close to smoothed static
        # object poses
        for q, time_id in zip([q_start, q_goal], [0, -1]):
            initial = Configuration.from_numpy(
                np.array(q), len(self.robot.initial_configuration())
            )
            current = Configuration(
                self.robot.initial_configuration(), self.object_poses[:, time_id]
            )
            q, dlin, drot = initial.distance(current)
            assert dlin < 0.1
        q_start = list(
            Configuration(
                self.robot.initial_configuration(), self.object_poses[:, 0]
            ).to_numpy()
        )
        q_goal = list(
            Configuration(
                self.robot.initial_configuration(), self.object_poses[:, -1]
            ).to_numpy()
        )

        self.ps.setInitialConfig(q_start)
        self.ps.addGoalConfig(q_goal)
        self.q_start = q_start
        self.q_goal = q_goal

        if self.random_seed is not None:
            random.seed(self.random_seed)
            self.ps.setRandomSeed(self.random_seed)
        # set task and reset stored tree and drawer

        self.tree.reset()
        self.time = time.time()

        self.tree.init_node = self.tree.add_root_node(q_start, marker="D", step=-1)
        self.tree.goal_node = self.tree.add_root_node(
            q_goal, marker="D", step=self.contacts.shape[1]
        )

        for iter in range(self.max_iter):
            if (
                self.max_planning_time is not None
                and time.time() - self.time > self.max_planning_time
            ):
                break
            if iter % 100 == 0 and self.verbose:
                print(f"{iter} iteration")
                print(time.time() - self.time, "sec")
                if iter % 100 == 0:
                    _ = self.summarize_tree()
            if self.is_solved():
                self.solved = True
                if self.verbose:
                    print("solved")
                break

            # 1a. sample new subgoal
            if random.uniform(0, 1) < self.sample_subgoal_prob:
                subgoal_start_node, subgoal_end_node = self.generate_new_subgoal(
                    stable_obj=False
                    # if self.task.demo.task_name == "waiter" else True
                )
                if subgoal_start_node is None:
                    continue
                # try to connect both start and end of subgoal to existing trees
                for node in [subgoal_start_node, subgoal_end_node]:
                    valid_tree_roots = self.get_trees_in_plane(node.plane, node.root)
                    if len(valid_tree_roots) > 0:
                        self.extend_trees(
                            tree_roots=valid_tree_roots, node=node, direction="from"
                        )
            # 1b. sample new configuration
            else:
                # Choose the state from {Free, grasp obj0, grasp obj1, ...} to extend
                search_state, search_transition = self.get_search_state()
                # Select which instance (plane) of state will get extended
                # (q_from defines different Free)
                plane = self.get_search_plane(search_state)
                if plane is not None:
                    # generate random configuration in this plane
                    try:
                        q_rand = sample_state_on_transition_target(
                            self,
                            plane.q_from.copy(),
                            search_transition,
                            open_fingers=False,
                        )
                    except Exception as e:
                        if self.verbose:
                            print(f"Skipped, failed to generate config in {str(plane)}")
                            print(e)
                        continue
                    valid_tree_roots = self.get_trees_in_plane(plane)
                    if len(valid_tree_roots) > 0 and q_rand is not None:
                        closest_node_info = self.get_closest_node(
                            valid_tree_roots, q_rand, direction="to", plane=plane
                        )
                        if closest_node_info is None:
                            continue
                        closest_node, path_len, path_id = closest_node_info
                        # make a small step from closest tree node in dir to q_from
                        step_config = self.ps.configAtParam(path_id, self.step_size)
                        added_node = self.tree.add_node(
                            config=step_config,
                            parent=closest_node,
                            step=closest_node.step,
                        )
                        # extend other trees in direction to this step_config
                        self.extend_trees(
                            tree_roots=valid_tree_roots, node=added_node, direction="to"
                        )
            # if self.enable_drawing:
            #     self.drawer.display(to_image=True)

        if self.is_solved():
            if self.verbose:
                print("Solved!")
            if self.optimize_path_iter is not None:
                self.optimize_path_shortcut(
                    max_iter=self.optimize_path_iter, verbose=self.verbose
                )
            # add all nodes and edges to the task.ps
            # start from the goal leaf and go up until init is reached
            node_to_add = self.tree.goal_node
            self.ps.addConfigToRoadmap(node_to_add.config)
            while not node_to_add.is_root:
                parent_node = node_to_add.parent
                self.ps.addConfigToRoadmap(parent_node.config)
                res, pid, msg = self.ps.directPath(
                    parent_node.config, node_to_add.config, True
                )
                if not res:
                    res, pid, msg = self.ps.directPath(
                        node_to_add.config, parent_node.config, True
                    )
                    if not res:
                        print("Not successful connect (should not happen). Rerun")
                        return None
                    self.ps.addEdgeToRoadmap(
                        node_to_add.config, parent_node.config, pid, True
                    )
                else:
                    self.ps.addEdgeToRoadmap(
                        parent_node.config, node_to_add.config, pid, True
                    )
                node_to_add = parent_node

            self.ps.solve()
            self.compute_metrics()
            return True
        else:
            return False

    def get_search_state(self) -> (str, str):
        """
        Returns random element from all available state-transition pairs
        :return: tuple of state and loop transition associated with this state
        """
        list_s_t_tuples = [("free", "Loop | f")]
        for j in range(len(self.handles_names)):
            list_s_t_tuples += [
                (f"{self.robot.get_gripper_name()} grasps {handle}", f"Loop | 0-{i}")
                for i, handle in self.get_handle_ids_names(j)
            ]
        state, transition = random.choice(list_s_t_tuples)
        return state, transition

    def get_search_plane(self, state: str) -> Optional[FreePlane | GraspPlane]:
        """Returns random plane that lies in the requested state
        (for grasp only one option)"""
        if state == "free":
            return random.choice(list(self.tree.free_planes.values()))
        else:
            for plane in self.tree.grasp_planes:
                if plane.state == state:
                    return plane
            return None

    def get_handle_ids_names(self, obj_id: int) -> list[(int, str)]:
        """
        Returns handle ids together with handle names for an object
        :param obj_id: id of the object of interest
        :return: list of tuples: [(hpp_handle_id, hpp_handle_name), ...]
        """
        return [
            (i + sum([len(self.handles_names[i]) for i in range(obj_id)]), handle)
            for i, handle in enumerate(self.handles_names[obj_id])
        ]
        # if 'handleZm' in handle]

    def get_handle(self, obj_id: int) -> str:
        """
        Returns a random handle for an object
        :param obj_id: id of the object of interest
        :return: random handle out of all handles defined for an object
        """
        return random.choice(
            [handle for i, handle in self.get_handle_ids_names(obj_id)]
        )

    def generate_new_subgoal(
        self, stable_obj: bool = True, max_iter: int = 10
    ) -> (list[float], list[float]):
        """
        Generate new coupled subgoal (two configurations, one free+pregrasp,
        other grasped+preplace)
        Subgoal is defined by object poses and id of manipulated poses
        Randomly select the time_id
        """
        time_id = random.choice([i for i in range(self.contacts.shape[1])])
        obj_id = np.argmax(self.contacts, axis=0)[time_id]
        q_from = list(
            Configuration(
                self.robot.initial_configuration(), self.object_poses[:, time_id]
            ).to_numpy()
        )
        if self.tray_poses is not None:
            # TODO: think of the way to implement this more clean
            q_from[0] = self.tray_poses[time_id][0, 3] - 0.2
            q_from[1] = self.tray_poses[time_id][1, 3]

        subgoal_start = None
        subgoal_end = None
        for _ in range(max_iter):
            subgoal_t = self.get_coupled_subgoal_for_opos(
                q_from, obj_id, check_stable_obj=stable_obj, time_id=time_id
            )
            if subgoal_t[0] is not None:
                subgoal_start = self.tree.add_root_node(
                    subgoal_t[0], step=time_id, marker="*", start_subgoal=True
                )
                subgoal_end = self.tree.add_node(
                    subgoal_t[1], step=time_id, parent=subgoal_start, end_subgoal=True
                )
                break

        return subgoal_start, subgoal_end

    def get_coupled_subgoal_for_opos(
        self,
        q_from: list[float],
        obj_id: int,
        handle: Optional[str] = None,
        check_stable_obj: bool = True,
        time_id: int = None,
    ) -> (Optional[list[float]], str | list[float]):
        """
        Function to generate subgoal that consists of:
            - pregrasp (gripper on top of object, in grasp position)
            - preplace (gripper grasps the object above the contact surface)
        generate pregrasp -> preplace if the object is grasped (contacts == 1)
        generate preplace -> pregrasp if the object is released
        The function generates sequence of pregrasp -> intersec -> preplace
        Finally, we check that direct path can be built from pregrasp to preplace
        """

        if handle is None:
            handle = self.get_handle(obj_id)
        trans_pregrasp = f"{self.robot.get_gripper_name()} > {handle} | f_01"
        trans_intersec = f"{self.robot.get_gripper_name()} > {handle} | f_12"
        trans_preplace = f"{self.robot.get_gripper_name()} > {handle} | f_23"

        pregrasp_subgoal = sample_state_on_transition_target(
            self, q_from, trans_pregrasp
        )
        if pregrasp_subgoal is None or (
            check_stable_obj
            and np.linalg.norm(
                np.array(q_from[self.ndof :]) - np.array(pregrasp_subgoal[self.ndof :])
            )
            > 0.1
        ):
            return None, "pregrasp_failed"

        intersec_subgoal = sample_state_on_transition_target(
            self, pregrasp_subgoal, trans_intersec
        )
        if intersec_subgoal is None:
            return None, "intersec_failed"

        preplace_subgoal = sample_state_on_transition_target(
            self, intersec_subgoal, trans_preplace
        )
        if preplace_subgoal is None:
            return None, "preplace_failed"

        direct_path_exists, pid, msg = build_direct_path_any_direction(
            self, pregrasp_subgoal, preplace_subgoal
        )
        if not direct_path_exists:
            return None, "dir_path_failed"

        subgoal_start = (
            pregrasp_subgoal
            if self.contacts[obj_id][time_id] == 1
            else preplace_subgoal
        )
        subgoal_end = (
            preplace_subgoal
            if self.contacts[obj_id][time_id] == 1
            else pregrasp_subgoal
        )
        return subgoal_start, subgoal_end

    def get_trees_in_plane(
        self, plane: FreePlane | GraspPlane, exclude_root: Optional[Node] = None
    ) -> list[Node]:
        """Returns list of roots of trees that have nodes in the requested plane"""
        # TODO: this function assumes that only one free place is created,
        #  but tray doesnt fit
        # [f"{plane.state}_{plane.id}"]
        return [
            node
            for node, planes in self.tree.node_planes.items()
            if plane in planes and node != exclude_root
        ]

    # def get_trees_in_plane(self, plane, node_root=None):
    #     """ Returns roots of all trees that have nodes in the requested plane """
    #     valid_tree_roots = []
    #     for tree_root in self.tree.root_nodes:
    #         # tree that was already merged
    #         if not tree_root.is_root:
    #             continue
    #         # ignore tree that node is currently belongs to
    #         if tree_root == node_root:
    #             continue
    #         # root is in the same plane
    #         if tree_root.plane.state == plane.state and
    #         tree_root.plane.id == plane.id:
    #             valid_tree_roots.append(tree_root)
    #             continue
    #         # tree has nodes in the same plane
    #         for child in tree_root.descendants:
    #             if child.plane.state == plane.state and child.plane.id == plane.id:
    #                 valid_tree_roots.append(tree_root)
    #                 break
    #     return valid_tree_roots

    def extend_trees(self, tree_roots: list[Node], node: Node, direction: str = "from"):
        """
        Try to extend all the trees from/to given configs. This funciton has two
        modes:
        - if direction='from',
        it will try to build a path from config to  closest node (one for each tree)
        and stop on obstacle
        - if direction='to',
        it will try to build a path from the closest node (one for each tree) to
        config and stop on obstacle

        :param tree_roots: all the trees to consider (have nodes in the same plane)
        :param node: node that needs to be connected
        :param direction: 'from' ot 'to'
        """
        for tree_root in tree_roots:
            if not tree_root.is_root:
                continue
            if tree_root == node.root:
                continue
            closest_node_info = self.get_closest_node(
                [tree_root], node.config, direction, plane=node.plane
            )
            if closest_node_info is None:  # did not find node to connect in this tree
                continue
            closest_node, path_len, path_id = closest_node_info
            # create nodes between closest node and config with certain step
            step = self.step_size
            parent_node = closest_node if direction == "to" else node
            final_node = node if direction == "to" else closest_node
            res = True
            while step < path_len:  # len of path
                step_config = self.ps.configAtParam(path_id, step)
                res, pid, msg = self.ps.directPath(
                    parent_node.config, step_config, True
                )
                # why was this commented out?
                if not res:
                    break
                added_node = self.tree.add_node(
                    config=step_config, parent=parent_node, step=parent_node.step
                )
                parent_node = added_node
                step += self.step_size
            if res:
                res, pid, msg = self.ps.directPath(
                    final_node.config, parent_node.config, True
                )
                if not res:
                    res, pid, msg = self.ps.directPath(
                        parent_node.config, final_node.config, True
                    )
                if res:
                    # merge trees if the whole path can be built
                    self.merge_trees(parent_node, final_node)
                else:
                    # last point before obstacle
                    step_config = self.ps.configAtParam(path_id, path_len)
                    _ = self.tree.add_node(
                        config=step_config, parent=parent_node, step=parent_node.step
                    )

    def merge_trees(self, node_to_rewire: Node, new_node_parent: Node):
        """
        Combine two trees into one
        Only merge goal node if the second tree contains start node
        """

        # always merge with root on the left (root - lower number)
        if node_to_rewire.step < new_node_parent.step:
            node_to_rewire, new_node_parent = new_node_parent, node_to_rewire
        if node_to_rewire.root.step == -1:
            node_to_rewire, new_node_parent = new_node_parent, node_to_rewire
        # goal node only gets merged if root of new tree contains init node
        if (
            node_to_rewire.root.step == self.contacts.shape[1]
            and not new_node_parent.root.step == -1
        ):
            return
        # exclude node_to_rewire.root from list of plane root correspondences
        rn = self.tree.node_planes[node_to_rewire.root]
        del self.tree.node_planes[node_to_rewire.root]
        # add new planes to new_node_parent.root
        self.tree.node_planes[new_node_parent.root] = list(
            set(self.tree.node_planes[new_node_parent.root] + rn)
        )

        while not node_to_rewire.is_root:
            prev_parent = node_to_rewire.parent
            node_to_rewire.parent = new_node_parent
            new_node_parent = node_to_rewire
            node_to_rewire = prev_parent
        node_to_rewire.parent = new_node_parent

    def get_closest_node(
        self,
        tree_roots: list[Node],
        q_rand: list[float],
        direction: str,
        plane: Optional[FreePlane | GraspPlane] = None,
    ) -> Optional[tuple]:
        """
        Returns the closest node to q_rand out of all descendants of tree_roots. To
        find the closest node we try to build a path between q_rand and candidate
        node. And select the node with the shortest path created (optionally
        approximated with the euclidean distance). If there are no full paths built,
        the node with the shortest partial path will be chosen. return: (
        closest_node, length_of_path, path_id)
        """
        nodes = chain.from_iterable(map(PreOrderIter, tree_roots))
        if plane is not None:
            # nodes = filter(lambda node: node.plane.id == plane.id
            # and str(node.plane.state) == str(plane.state), nodes)
            nodes = filter(lambda node: node.plane.id == plane.id, nodes)

        if self.use_euclidean_distance:
            # closest_node = min(nodes, key=lambda n:
            # np.linalg.norm(np.asarray(n.config[2:]) - np.asarray(q_rand[2:])),
            closest_node = min(
                nodes,
                key=lambda n: np.linalg.norm(np.asarray(n.config) - np.asarray(q_rand)),
                default=None,
            )
            if closest_node is not None:
                res, pid, msg = self.ps.directPath(q_rand, closest_node.config, True)
                if not res:
                    res, pid, msg = self.ps.directPath(
                        closest_node.config, q_rand, True
                    )
                if res:
                    return closest_node, self.ps.pathLength(pid), pid
                if self.euclidean_dist_iter != 1:
                    node_list = [node for node in nodes]
                    if len(node_list) > 0:
                        node_list.sort(
                            key=lambda n: np.linalg.norm(
                                np.asarray(n.config) - np.asarray(q_rand)
                            )
                        )
                        for i in range(self.euclidean_dist_iter):
                            if i >= len(node_list):
                                break
                            res, pid, msg = self.ps.directPath(
                                q_rand, node_list[i].config, True
                            )
                            if not res:
                                res, pid, msg = self.ps.directPath(
                                    node_list[i].config, q_rand, True
                                )
                            if res:
                                return node_list[i], self.ps.pathLength(pid), pid

            return None

        full_path_neigh_nodes_list = []
        part_path_neigh_nodes_list = []
        for n in nodes:
            res = self.connect_two_configs(
                q_rand, n.config, reverse_direction=direction != "from"
            )
            if res is not None:
                path_len, path_id, is_full_path = res
                if is_full_path:
                    full_path_neigh_nodes_list.append((n, path_len, path_id))
                else:
                    part_path_neigh_nodes_list.append((n, path_len, path_id))
        if len(full_path_neigh_nodes_list) > 0:
            return min(full_path_neigh_nodes_list, key=lambda x: x[1])
        if len(part_path_neigh_nodes_list) > 0:
            return max(part_path_neigh_nodes_list, key=lambda x: x[1])
        return None

    def connect_two_configs(
        self,
        start_q: list[float],
        end_q: list[float],
        verbose: bool = False,
        recursion_depth: int = 0,
        reverse_direction: bool = False,
    ) -> Optional[tuple]:
        """
        Try to find a path between start and goal configurations.

        :return: None if there is no path, otherwise (length, pid, is_full_path)
        """
        if reverse_direction:
            start_q, end_q = end_q, start_q
        res, pid, msg = self.ps.directPath(start_q, end_q, True)
        if res:
            return self.ps.pathLength(pid), pid, recursion_depth == 0
        # failure scenarios
        if msg == "Steering method failed to build a path.":
            # no recovery for this
            if verbose:
                print("Steering method failed to build a path.")
        else:
            # for other error messages try to build partial path
            if self.ps.pathLength(pid) == 0:
                return None
            partial_path_node = self.ps.configAtParam(pid, self.ps.pathLength(pid))
            if np.linalg.norm(np.array(partial_path_node) - np.array(end_q)) > 0.01:
                return self.connect_two_configs(
                    start_q, partial_path_node, recursion_depth=recursion_depth + 1
                )
            else:
                # partial node too close to end_q that was already tried,
                # avoid infinite recursion
                return None
        return None

    def get_robot_tunnel_side(self, q: list[float]) -> int:
        """
        Function to retrieve if the robot gripper is in front or behind the tunnel It
        is useful for checking if pregrasp and preplace were sampled without passing
        the tunnel
        """
        self.robot.setCurrentConfig(q)
        robot_grip_position = self.robot.getJointPosition(self.robot.get_gripper_name())
        robot_grip_pose = get_matrix_hpp(robot_grip_position)
        tunnel_id = self.task.demo.furniture_ids.index("tunnel")
        tunnel_pose = self.task.demo.furniture_poses[tunnel_id]
        robot_tun_frame = np.linalg.inv(tunnel_pose).dot(robot_grip_pose)
        return 0 if robot_tun_frame[0, 3] > 0 else 1

    def summarize_tree(self) -> list[Node]:
        """
        Helper function for debugging purposes Prints have many configs are sampled
        per step and which steps are connected in one tree Can be used to detect if
        there is a problem to sample configurations in a particular step,
        or to detect which steps are not connected into a single tree
        """
        node_count = {}
        plane_count = {}
        root_nodes = [node for node in self.tree.root_nodes if node.is_root]
        key_roots = []
        total_present_steps = set()
        for root in root_nodes:
            step_list = [root.step] + [node.step for node in root.descendants]
            plane_list = [str(root.plane)] + [
                str(node.plane) for node in root.descendants
            ]
            unique_steps = list(set(step_list))
            if len(unique_steps) > 1:
                print(f"Root {root.name} is connected to steps {unique_steps}")
                key_roots.append(root)
            for step in unique_steps:
                total_present_steps.add(step)
            for node_step in step_list:
                if node_step not in node_count.keys():
                    node_count[node_step] = 1
                else:
                    node_count[node_step] += 1
            for node_plane in plane_list:
                if node_plane not in plane_count.keys():
                    plane_count[node_plane] = 1
                else:
                    plane_count[node_plane] += 1
        print(f"All presented steps: {sorted(total_present_steps)}")
        print("Amount of configs per step:")
        for key in sorted(node_count):
            print(f"({key}): {node_count[key]}, ", end=" ")
        print("\n")
        return key_roots

    def optimize_path_shortcut(self, max_iter: int = 200, verbose: bool = False):
        """
        Function to optimize final path that starts at init_node and ends at
        goal_node In each of the states pick two random nodes and try to build
        shorcut between then. Modifies the tree on  the flight
        """
        # TODO: take first point at random, second point from same plane
        #  (this would encourage shortcutting in manifolds with many points)
        # get all possible state
        all_states = list(set([node.plane.state for node in self.tree.goal_node.path]))
        prev_len = len([node for node in self.tree.goal_node.path])
        for _ in range(max_iter):
            state = random.choice(all_states)
            if state == "free":
                all_free_plane_ids = list(
                    set(
                        [
                            node.plane.id
                            for node in self.tree.goal_node.path
                            if node.plane.state == state
                        ]
                    )
                )
                plane_id = random.choice(all_free_plane_ids)
                state_nodes = [
                    node
                    for node in self.tree.goal_node.path
                    if node.plane.state == state and node.plane.id == plane_id
                ]
            else:
                state_nodes = [
                    node
                    for node in self.tree.goal_node.path
                    if node.plane.state == state
                ]
            if len(state_nodes) == 0:
                continue
            candidate_from_node = random.choice(state_nodes)
            candidate_to_node = random.choice(state_nodes)
            if candidate_from_node == candidate_to_node:
                continue
            shortcut_applied = self.shortcut_between_nodes(
                candidate_from_node, candidate_to_node
            )
            if shortcut_applied:
                cur_len = len([node for node in self.tree.goal_node.path])
                if verbose:
                    print(f"Cut len from {prev_len} to {cur_len} in state {state}")
                prev_len = cur_len

    def shortcut_between_nodes(self, from_node: Node, to_node: Node) -> bool:
        """
        Try to find direct path between nodes, if it exists, update nodes to be
        parent and child
        :param from_node: parent node
        :param to_node: child node
        :return: was shortcut applied
        """
        res, pid, msg = self.ps.directPath(from_node.config, to_node.config, True)
        if not res:
            res, pid, msg = self.ps.directPath(to_node.config, from_node.config, True)
        if res:
            if from_node in to_node.descendants:
                from_node.parent = to_node
            else:
                to_node.parent = from_node
        return res

    def is_solved(self) -> bool:
        """Goal node belongs to the same tree as init (where init node is a root)."""
        return self.tree.goal_node.root.step == -1
