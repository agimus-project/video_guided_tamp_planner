from tamp_guided_by_video.planners.hpp_planner import HppPlanner
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
from anytree import Node
from tamp_guided_by_video.utils.planner_utils import get_config_states
from tamp_guided_by_video.utils.demo_processing import ensure_normalized
from sklearn.manifold import TSNE
from tamp_guided_by_video.utils.plot_tree import plot_rrt_star_tsne


class StateNode(Node):
    counter: int = 0  # Global counter for unique naming
    robot_ndof: int = 7

    def __init__(self, states, config, cost: float = 0, parent=None):
        name = StateNode.counter  # Assign unique integer as name
        StateNode.counter += 1  # Increment counter for next node
        super().__init__(name, parent)
        self.states: list[str] = states  
        self.config: list[float] = config
        self.cost: float = cost

    def distance(self, other):
        """Compute distance between this node and another node (e.g., Euclidean)."""
        return np.linalg.norm(
            np.array(self.config[:StateNode.robot_ndof]) - 
            np.array(other.config[:StateNode.robot_ndof]))



class RRTConnectStart(HppPlanner):
    """
    This planner uses RRT-Connect* algorithm
    """

    def __init__(
        self,
        task: BaseTask,
        optimize: bool = False,
        max_planning_time: Optional[int] = None,
        handles_mode: str = "all",
        max_iter: int = 1000,
        corba: CorbaServer = None,
        random_seed: int = None,
        verbose: bool = False,
        steps: Optional[list[int]] = None,
    ):
        self.max_iter = max_iter
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
        # self.tree = Tree(planner=self, robot_n=self.ndof)
        self.time = None
        self.solved = False
        self.q_start = None
        self.q_goal = None
        self.step_size = 0.5

    @property
    def name(self):
        return "rtt_connect_star"
        

    def sample(self, tree_root: StateNode, verbose: bool = False) -> StateNode:
        # once is solved, choose a node from the solution path add some noise, 
        
        if self.solved and np.random.rand() < 0.5:
            nodes_to_sample = [self.tree_goal_root]
            while nodes_to_sample[-1].parent is not None:
                nodes_to_sample.append(nodes_to_sample[-1].parent)
            rand_node = random.choice(nodes_to_sample)
            # print(f"Sampling from solution path {rand_node.name}")
            possible_transitions = []
            for edge in self.cg.edges:
                if edge[-3] == '_':
                    continue
                if self.cg.getNodesConnectedByEdge(edge)[0] in get_config_states(self, rand_node.config):
                    # based on some probability, throw out loop free transition                    
                    possible_transitions.append((edge, self.cg.getNodesConnectedByEdge(edge)[0]))
            if verbose:
                print(f"Possible transitions: {possible_transitions}")
            
            

            # select a random transition/from_state pair
            transition, from_state = random.choice(possible_transitions)  
            # add noise to q_rand
            
            q_rand_noisy = list(rand_node.config + np.random.normal(0, 0.05, len(rand_node.config)))
            q_rand_noisy[-7:] = ensure_normalized(q_rand_noisy[-7:])
            
            q_rand =sample_state_on_transition_target(self, rand_node.config, transition, max_iter=5, q_random=q_rand_noisy)
            
            if q_rand is None:
                return None
            else:
                # print("sampled", get_config_states(self, q_rand))
                return StateNode(config=q_rand, states=get_config_states(self, q_rand))
        

        # get all states that are currently in the tree
        all_from_states = tree_root.states.copy()
        for node in tree_root.descendants:
            all_from_states += node.states
        unique_from_states = set(all_from_states)

        if verbose:
            print(f"Unique states: {unique_from_states}")

        # get all possible transitions from the current states
        possible_transitions = []
        for edge in self.cg.edges:
            if edge[-3] == '_':
                continue
            if self.cg.getNodesConnectedByEdge(edge)[0] in unique_from_states:
                # based on some probability, throw out loop free transition
                if edge == 'Loop | f' and np.random.rand() < 0.2:
                    continue
                
                possible_transitions.append((edge, self.cg.getNodesConnectedByEdge(edge)[0]))
        if verbose:
            print(f"Possible transitions: {possible_transitions}")
        
        

        # select a random transition/from_state pair
        transition, from_state = random.choice(possible_transitions)  
        
        from_nodes = [node for node in tree_root.descendants if from_state in node.states]
        if from_state in tree_root.states:
            from_nodes.append(tree_root)
        if verbose:
            print(f"Possible from nodes amount: {len(from_nodes)}")
        # select a random node that has the from_state
        from_node = random.choice(from_nodes)
        if verbose:
            print(transition)
        q_rand =sample_state_on_transition_target(self, from_node.config, transition, max_iter=5)
        
        if q_rand is None:
            return None
        else:
            # print("sampled", get_config_states(self, q_rand))
            return StateNode(config=q_rand, states=get_config_states(self, q_rand))
    
    def nearest(self, tree_root: StateNode, q_rand: StateNode) -> StateNode:
        
        possible_transitions = []
        for edge in self.cg.edges:
            if self.cg.getNodesConnectedByEdge(edge)[1] in q_rand.states:
                possible_transitions.append((edge, self.cg.getNodesConnectedByEdge(edge)[0]))
        valid_from_states = [t[1] for t in possible_transitions]
        all_nodes = []
        # check if any of root states is in the possible transitions [1] element
        for node in [tree_root] + list(tree_root.descendants):
            if any([state in node.states for state in valid_from_states]):
                all_nodes.append(node)
        return min(all_nodes, key=lambda n: n.distance(q_rand))
    
    def steer(self, q_nearest: StateNode, q_rand: StateNode, verbose: bool = False) -> StateNode:
        res, pid, msg = self.ps.directPath(
            q_nearest.config, q_rand.config, True
        )
        
        if not res:
            res, pid, msg = self.ps.directPath(q_rand.config, q_nearest.config, True)
            
            if not res:
                return None
            if verbose:
                print("Added Reversed path, NO CHECK", StateNode.counter)
            # res = self.ps.reversePath(pid)
            # if not res:
            #     return None
        
        return self.ps.configAtParam(pid, min(self.step_size, self.ps.pathLength(pid)))
        # res, pid, msg = self.ps.directPath(q_nearest.config, q_add, True)
        # if not res:
        #     res, pid, msg = self.ps.directPath(q_add,  q_nearest.config, True)
        #     if not res:
        #         return None
        # # assert res, print(msg, " | ", q_add)
        # return StateNode(
        #     config=q_add,
        #     states=get_config_states(self, q_add)
        # )


    def add_node_to_tree(self, parent_node: StateNode, new_config: list[float]):
        # Double check there is a path between parent and new_config
        res, _, _ = self.ps.directPath(parent_node.config, new_config, True)
        if not res:
            res, _, __import__ = self.ps.directPath(new_config,  parent_node.config, True)
            if not res:
                return None
        
        return StateNode(
            config=new_config,
            states=get_config_states(self, new_config),

            parent=parent_node
        )
            
    def update_cost_descendants(self, node: StateNode):
        for child in node.children:
            child.cost = node.cost + node.distance(child)
            self.update_cost_descendants(child)
    
    def rewire_neighbors(self, tree_root: StateNode, q_new: StateNode, verbose: bool = False):
        close_nodes = [node for node in tree_root.descendants if node.distance(q_new) < self.step_size * 2 and node != q_new.parent and node != q_new]
        for node in close_nodes:
            
            if q_new.cost + q_new.distance(node) < node.cost:
                res, pid, msg= self.ps.directPath(q_new.config, node.config, True)
                if not res:
                    res, pid, msg= self.ps.directPath(node.config, q_new.config, True)
                    if not res:
                        
                        continue
                if verbose:
                    print(f"Rewiring {node.name} from {node.cost} to {q_new.cost + q_new.distance(node)}")
                node.parent = q_new
                node.cost = q_new.cost + q_new.distance(node)
                self.update_cost_descendants(node)

    def rewire_children(self, parent: StateNode, q_new: StateNode, verbose: bool = False):
        for node in parent.children:
            if q_new.cost + q_new.distance(node) < node.cost:
                res, pid, msg= self.ps.directPath(q_new.config, node.config, True)
                if not res:
                    res, pid, msg= self.ps.directPath(node.config, q_new.config, True)
                    if not res:
                        # print("NOt connected ni rewiring")
                        continue
                if verbose:
                    print(f"Rewiring {node.name} from {node.cost} to {q_new.cost + q_new.distance(node)}")
                node.parent = q_new
                node.cost = q_new.cost + q_new.distance(node)
                self.update_cost_descendants(node)
                 
                    
    def rewire(self, tree_root: StateNode, q_new: StateNode, verbose: bool = False):
        # get all nodes that are close to q_new
        close_nodes = [node for node in tree_root.descendants if node.distance(q_new) < self.step_size * 2 and node != q_new.parent and node != q_new]
        # print(f"Trying to rewire in {len(close_nodes)} nodes")
        # get all possible transitions from the current states
        closer_node = None
        current_cost = q_new.cost
        for node in close_nodes:
            
            if node.cost + node.distance(q_new) < current_cost:
                res, pid, msg= self.ps.directPath(node.config, q_new.config, True)
                if not res:
                    res, pid, msg= self.ps.directPath(q_new.config, node.config, True)
                    if not res:
                        # print("NOt connected ni rewiring")
                        continue
                closer_node = node
                current_cost = node.cost + node.distance(q_new)
            
        if closer_node is not None:    
            if verbose:
                print(f"Rewiring {q_new.name} from {q_new.cost} to {current_cost} ({closer_node.name})")
            prev_parent = q_new.parent
            q_new.parent = closer_node
            q_new.cost = current_cost
            self.rewire_children(prev_parent, q_new, verbose=verbose)
            return True
        return False
                
            # else:  # 19.02981900979707
            #     print(f"{node.cost} < {q_new.cost + q_new.distance(node)}")

    def recursive_cost_change(self, node: StateNode, new_cost: float):
        node.cost = new_cost
        for sub_node in node.children:
            new_cost = node.cost + node.distance(sub_node)
            self.recursive_cost_change(sub_node, new_cost)
    
    def get_path_to_root(self, node, mode='nodes'):
        path = []
        prev_config = None
        while node is not None:
            if mode == 'nodes':
                path.append(node)
            elif mode == 'configs':
                path.append(node.config)  # Or use any other attribute
            else:
                raise ValueError("Mode not recognized ", {mode})
            if prev_config is not None:
                res, pid, msg= self.ps.directPath(node.config, prev_config, True)
                if not res:
                    res, pid, msg= self.ps.directPath(prev_config, node.config, True)
                    if not res:
                        raise ValueError("Path not found")
            prev_config = node.config
            node = node.parent
        return path[::-1]  # Reverse the path to go from root to node

    
    def connect(self, tree_root: StateNode, q_new: StateNode) -> Optional[StateNode]:
        # get all nodes that are close to q_new
        close_nodes = [node for node in tree_root.descendants if node.distance(q_new) < 10 * self.step_size]
        # print(f"AMount of nodes considered for connection: {len(close_nodes)}")
        for node in close_nodes:
            res, pid, msg = self.ps.directPath(
                q_new.config, node.config, True
            )
            if res:
                # merge two trees, update costs of children
                while node.parent is not None:
                    current_parent = node.parent
                    new_cost = q_new.cost + q_new.distance(node)
                    self.recursive_cost_change(node, new_cost)
                    node.parent = q_new
                    q_new = node
                    node = current_parent
                node.parent = q_new                    

                # configs = [Configuration.from_numpy(np.array(config), self.ndof) for config in self.get_path_to_root(q_new)]
                # from guided_tamp_benchmark.tasks.renderer import Renderer
                # from robomeshcat import Robot
                # from guided_tamp_benchmark.models.utils import get_models_data_directory
                # r = Renderer(task=self.task)
    
                # start_color = [148 / 255, 103 / 255, 189 / 255]  # magenta
                # goal_color = [44 / 255, 160 / 255, 44 / 255]  # green
                # for config, color in zip([configs[0], configs[-1]], [start_color, goal_color]):
                #     for i, o in enumerate([o for o in self.task.objects if o.name != "base"]):
                #         pose = config.poses[i]
                #         vo = Robot(
                #             urdf_path=o.urdfFilename,
                #             mesh_folder_path=get_models_data_directory(),
                #             color=color,
                #             opacity=0.25,
                #             pose=pose,
                #         )
                #         r.objects.append(vo)
                #         r.scene.add_robot(vo)
                # # robot_pose = task.demo.robot_pose
                # r.animate_path(configs, fps=1)
                # r.scene.render_image()
                
                # time.sleep(15.0)
                return True


        return False
    
    def summarize_trees(self):
        trees = ['start', 'goal']
        for tree, str_tree in zip([self.tree_start_root, self.tree_goal_root], trees):
            print(f"Tree {str_tree} has {len(tree.descendants)} nodes")
            print("All present states are:")
            states = tree.states.copy()
            for node in tree.descendants:
                states += node.states
            unique_states = set(states)
            print(unique_states)
          
    
    def solve(self) -> bool:
        """
        Find a path between start and goal 

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
        # q_start_pregrasp = None
        # while q_start_pregrasp is None:
        #     q_start_pregrasp = sample_state_on_transition_target(self, q_start, "panda/gripper > obj_000002/handleZpx | f_01")    
        # q_start = q_start_pregrasp
        # print(get_config_states(self, q_start))
        self.tree_start_root =  StateNode(
            config=q_start, 
            states=get_config_states(self, q_start),
            cost=0.
            )  # Start tree
        self.tree_goal_root = StateNode(
            config=q_goal, 
            states=get_config_states(self, q_goal),
            cost= 0.
            )   # Goal tree
        process_trees = [self.tree_start_root, self.tree_goal_root]
        self.ps.setInitialConfig(q_start)
        self.ps.addGoalConfig(q_goal)
        self.q_start = q_start
        self.q_goal = q_goal

        if self.random_seed is not None:
            random.seed(self.random_seed)
            self.ps.setRandomSeed(self.random_seed)
        # set task and reset stored tree and drawer


        self.time = time.time()

        for iter in range(self.max_iter):
            

            
            if (
                self.max_planning_time is not None
                and time.time() - self.time > self.max_planning_time
            ):
                break
            # if iter % 100 == 0 and self.verbose:
            # if iter > 10:
            #     print("PLotting")
            #     if self.solved:
            #         goal_path = self.get_path_to_root(self.tree_goal_root) 
            #         plot_rrt_star_tsne(self.tree_start_root, list(PreOrderIter(self.tree_start_root)), path_nodes=goal_path)
            #     else:
            #         plot_rrt_star_tsne(self.tree_start_root, list(PreOrderIter(self.tree_start_root)), path_nodes=None)
            # if iter % 100 == 0:
            #     print(f"{iter} iteration", time.time() - self.time, "sec")
                # self.summarize_trees()
                # if self.solved:
                #     configs = [Configuration.from_numpy(np.array(config), self.ndof) for config in self.get_path_to_root(self.tree_goal_root, mode='configs')]
                #     rot_l_joints, pos_l_obj, rot_l_obj = self.task.compute_lengths(configs)
                #     print(f"Path length: {rot_l_joints}")  # 17.28006166335776


            if self.solved or self.is_solved():
                self.solved = True
                # if self.verbose:
                #     print("solved")


            """Writing RRT*-Connect here"""
            for i, tree_root in enumerate(process_trees):

                q_rand = self.sample(tree_root)
                if q_rand is None:
                    continue
                
                # Grow start tree
                q_nearest = self.nearest(tree_root, q_rand)
                q_new = self.steer(q_nearest, q_rand)
                if q_new is not None:
                    new_node = self.add_node_to_tree(q_nearest, q_new)
                    
                    if new_node is None: 
                        break
                    new_node.cost = q_nearest.cost + q_nearest.distance(new_node)
                    # if self.verbose:
                    #     print(f"Adding {q_nearest.name}->{q_new.name} with cost = {q_nearest.cost} + {q_nearest.distance(q_new)} = {q_new.cost}")
                    
                    # if self.solved:
                    #     goal_path = self.get_path_to_root(self.tree_goal_root) 
                    #     plt1 = plot_rrt_star_tsne(
                    #         self.tree_start_root, 
                    #         list(PreOrderIter(self.tree_start_root)), 
                    #         new_node,
                    #         path_nodes=goal_path)
                    # else:
                    #     plt1 = plot_rrt_star_tsne(
                    #         self.tree_start_root, 
                    #         list(PreOrderIter(self.tree_start_root)),
                    #         new_node if new_node.root == self.tree_start_root else None, 
                    #         path_nodes=None
                    #     )
                    was_rewired = self.rewire(tree_root, new_node)#, verbose=self.verbose)
                    # To check both before path is found and after
                    # if was_rewired:
                    #     print("I was rewired")
                    #     plt1.show()
                    #     if self.solved:
                    #         goal_path = self.get_path_to_root(self.tree_goal_root) 
                    #         plt2 = plot_rrt_star_tsne(
                    #             self.tree_start_root, 
                    #             list(PreOrderIter(self.tree_start_root)),
                    #             path_nodes=goal_path)
                    #     else:
                    #         plt2 = plot_rrt_star_tsne(
                    #             self.tree_start_root, 
                    #             list(PreOrderIter(self.tree_start_root)), 
                    #             new_node if new_node.root == self.tree_start_root else None, 
                    #             path_nodes=None
                    #         )
                    #     plt2.show()
                    # else:
                    #     plt1.close()
                    
                    # TO only check after path is found
                    # if was_rewired and self.solved:
                    #     print("I was rewired (after solved)")
                    #     plt1.show()
                    #     goal_path = self.get_path_to_root(self.tree_goal_root) 
                    #     plt2 = plot_rrt_star_tsne(
                    #         self.tree_start_root, 
                    #         list(PreOrderIter(self.tree_start_root)),
                    #         new_node,
                    #         path_nodes=goal_path)
                    #     plt2.show()
                    # else:
                    #     plt1.close()



                    # # Attempt to connect to goal tree
                    if i == 0 and len(process_trees) == 2:
                        is_merged = self.connect(self.tree_goal_root, new_node)
                        if is_merged:
                            process_trees = [self.tree_start_root]

            # if connection_node:
            #     return self.construct_path(new_node_start, connection_node)

            
        if self.is_solved():
            if self.verbose:
                print("Solved!")
            # if self.optimize_path_iter is not None:
            #     self.optimize_path_shortcut(
            #         max_iter=self.optimize_path_iter, verbose=self.verbose
            #     )
            # add all nodes and edges to the task.ps
            # start from the goal leaf and go up until init is reached
            node_to_add = self.tree_goal_root
            self.ps.addConfigToRoadmap(node_to_add.config)
            while node_to_add.parent is not None:
                parent_node = node_to_add.parent
                self.ps.addConfigToRoadmap(parent_node.config)
                res1, pid1, msg1 = self.ps.directPath(
                    parent_node.config, node_to_add.config, True
                )
                res2, pid2, msg2 = self.ps.directPath(
                        node_to_add.config, parent_node.config, True
                    )
                if not res1 and not res2:
                    print("Not successful connect (should not happen). Rerun")
                    return None
                if res1:
                    self.ps.addEdgeToRoadmap(
                        parent_node.config, node_to_add.config, pid1, True
                    )
                if res2:
                    self.ps.addEdgeToRoadmap(
                        node_to_add.config, parent_node.config, pid2, True
                    )
                
                node_to_add = parent_node

            self.ps.solve()
            self.compute_metrics()
            return True
        else:
            return False


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

    def is_solved(self) -> bool:
        """Goal node belongs to the same tree as init (where init node is a root)."""
        return self.tree_goal_root.parent is not None
