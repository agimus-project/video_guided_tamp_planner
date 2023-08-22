"""
This file contains functions from pddlstream github repo that were modified to
extended functionality
"""

from pddlstream.language.generator import from_gen_fn, from_fn, from_test
from pddlstream.utils import read, get_file_path, negate_test
from pddlstream.language.constants import PDDLProblem

import time
import quaternion as npq
from itertools import count
import numpy as np
from examples.pybullet.utils.pybullet_tools.utils import (
    get_pose, set_pose,
    get_movable_joints, joint_controller,
    sample_placement, Point,
    set_joint_positions, add_fixed_constraint,
    enable_real_time, disable_real_time,
    enable_gravity, wait_for_duration,
    link_from_name, get_body_name,
    end_effector_from_body, approach_from_grasp,
    GraspInfo, Pose, inverse_kinematics,
    remove_fixed_constraint, Attachment,
    get_sample_fn, step_simulation,
    refine_path, get_joint_positions,
    wait_if_gui, flatten,
    expand_links, any_link_pair_collision,
    get_self_link_pairs, MAX_DISTANCE,
    get_moving_links, CollisionPair,
    parse_body, get_custom_limits,
    cached_fn, get_buffered_aabb,
    all_between, aabb_overlap,
    pairwise_link_collision, product,
    get_distance_fn, get_extend_fn,
    check_initial_end, birrt,
    interpolate_joint_waypoints,
    EPSILON, get_nonholonomic_distance_fn,
    get_nonholonomic_extend_fn, set_default_camera,
    load_model, is_placement,
    get_bodies, HideOutput,
    draw_pose, draw_global_system,
    pose_from_tform, interpolate_poses, multiply, invert
)

# , get_closest_points
import pybullet as p

# plan_direct_joint_motion
from collections import namedtuple

CLIENT = 0
CollisionInfo = namedtuple(
    "CollisionInfo",
    """
                           contactFlag
                           bodyUniqueIdA
                           bodyUniqueIdB
                           linkIndexA
                           linkIndexB
                           positionOnA
                           positionOnB
                           contactNormalOnB
                           contactDistance
                           normalForce
                           lateralFriction1
                           lateralFrictionDir1
                           lateralFriction2
                           lateralFrictionDir2
                           """.split(),
)


def load_world(task, demo_init_obj_poses, robot_pose):
    disable_body_links = None
    set_default_camera()
    draw_global_system()
    body_names = {}
    rise_robot_up = np.eye(4)
    if task.robot.robot_type == 'fixed':
        rise_robot_up[2, 3] += 0.01
    # robot_paths = {'panda': PANDA_ARM_URDF, 'ur5': UR5_URDF,
    # 'kmr_iiwa': KUKA_KMR_URDF}
    with HideOutput():
        robot_id = load_model(
            task.robot.urdfFilename.replace(".urdf", "_pddl.urdf"),
            fixed_base=True,
            pose=pose_from_tform(robot_pose.dot(rise_robot_up)),
        )
        for furniture in task.furniture:
            furniture_id = load_model(str(furniture.urdfFilename))
            body_names[furniture_id] = furniture.name
            # dict {'robot_body_id': [(tunnel_body_id,
            # ['tunnel_top_sideB_link_id', ...])]}

            if hasattr(furniture, "disabled_collision_links_for_robot"):
                if disable_body_links is None:
                    disable_body_links = {}
                if robot_id not in disable_body_links.keys():
                    disable_body_links[robot_id] = []
                numJoints = p.getNumJoints(furniture_id)
                links_p_ids = {}
                for jointIndex in range(numJoints):
                    jointInfo = p.getJointInfo(furniture_id, jointIndex)
                    linkName = jointInfo[12].decode(
                        "utf-8"
                    )  # Decode the name from bytes to string
                    linkId = jointInfo[0]
                    links_p_ids[linkName] = linkId

                disable_body_links[robot_id].append(
                    (
                        furniture_id,
                        [
                            links_p_ids[l_name]
                            for l_name in furniture.disabled_collision_links_for_robot
                        ],
                    )
                )

        object_ids = []
        for obj, obj_pose in zip(task.objects, demo_init_obj_poses):
            obj_id = load_model(obj.urdfFilename, fixed_base=False)
            set_pose(obj_id, pose_from_tform(obj_pose))
            body_names[obj_id] = obj.name
            object_ids.append(obj_id)
        print(disable_body_links)

    draw_pose(
        Pose(), parent=robot_id, parent_link=get_tool_link(robot_id)
    )  # TODO: not working
    movable_bodies = object_ids

    return robot_id, body_names, movable_bodies, disable_body_links


def pddlstream_from_problem(
    robot,
    final_poses,
    objects,
    robot_init_config,
    movable=[],
    teleport=False,
    grasp_name="top",
    disable_body_links=None,
    grasps="ours",
    robot_name="panda",
    allow_side_handles=False,
    verbose=False,
):
    # assert (not are_colliding(tree, kin_cache))

    domain_pddl = read(get_file_path(__file__, "domain.pddl"))
    stream_pddl = read(get_file_path(__file__, "stream.pddl"))
    constant_map = {}
    if verbose:
        print("Robot:", robot)

    conf = BodyConf(robot, robot_init_config)
    init = [("CanMove",), ("Conf", conf), ("AtConf", conf), ("HandEmpty",)]

    fixed = get_fixed(robot, movable)
    if verbose:
        print("Movable:", movable)
        print("Fixed:", fixed)
    for body in movable:
        pose = BodyPose(body, get_pose(body))
        init += [("Graspable", body), ("Pose", body, pose), ("AtPose", body, pose)]
        for surface in fixed:
            init += [("Stackable", body, surface)]
            if is_placement(body, surface):
                init += [("Supported", body, pose, surface)]

    goal = (
        "and",
        ("AtConf", conf),
    )

    for body, obj_pose in zip(movable, final_poses):
        # translate = Pose(point=[-0.1, 0., 0])
        # desired_pose = multiply(get_pose(body), translate)
        # desired_body_pose = BodyPose(body, desired_pose)
        desired_body_pose = BodyPose(body, pose_from_tform(obj_pose))
        init += [("Pose", body, desired_body_pose)]
        goal += (("AtPose", body, desired_body_pose),)

    stream_map = {
        "sample-pose": from_gen_fn(
            get_stable_gen(fixed, disable_body_links=disable_body_links)
        ),
        # 'sample-grasp': from_gen_fn(get_grasp_gen(robot, grasp_name)),
        "sample-grasp": from_gen_fn(
            get_hpp_grasps_gen(robot, movable, objects,
                               allow_side_handles=allow_side_handles)
            if grasps == "ours"
            else None
            # else get_grasp_gen(robot, grasp_name)
        ),
        "inverse-kinematics": from_fn(
            get_ik_fn(robot, fixed, teleport, disable_body_links=disable_body_links)
        ),
        "plan-free-motion": from_fn(
            get_free_motion_gen(
                robot, fixed, teleport, disable_body_links=disable_body_links
            )
        ),
        "plan-holding-motion": from_fn(
            get_holding_motion_gen(
                robot, fixed, teleport, disable_body_links=disable_body_links
            )
        ),
        "test-cfree-pose-pose": from_test(get_cfree_pose_pose_test(
            disable_body_links=disable_body_links)),
        "test-cfree-approach-pose": from_test(get_cfree_obj_approach_pose_test()),
        "test-cfree-traj-pose": from_test(
            negate_test(
                get_movable_collision_test(disable_body_links=disable_body_links)
            )
        ),
        # get_cfree_traj_pose_test()),
        "TrajCollision": get_movable_collision_test(
            disable_body_links=disable_body_links
        ),
    }

    return PDDLProblem(domain_pddl, constant_map, stream_pddl, stream_map, init, goal)


def postprocess_plan(plan):
    paths = []
    for name, args in plan:
        if name == "place":
            paths += args[-1].reverse().body_paths
        elif name in ["move", "move_free", "move_holding", "pick"]:
            paths += args[-1].body_paths
    return Command(paths)


def get_fixed(robot, movable):
    rigid = [body for body in get_bodies() if body != robot]
    fixed = [body for body in rigid if body not in movable]
    return fixed


# GRASP_INFO = {
#     "top": GraspInfo(
#         lambda body: get_side_grasps(
#             body, under=True, tool_pose=Pose(), max_width=0.1, grasp_length=0.0
#         ),
#         approach_pose=Pose(0.1 * Point(z=1)),
#     ),
    # 'top': GraspInfo(lambda body: get_top_grasps(body, under=True, tool_pose=Pose(),
    # max_width=INF, grasp_length=0.),
    #                  approach_pose=Pose(0.1 * Point(z=1))),
# }


# TOOL_FRAMES = {
#     # 'panda': 'panda_link8',
#     'panda': 'panda_hand',  # iiwa_link_ee | iiwa_link_ee_kuka
#     'kmr_iiwa': 'panda_hand',  # iiwa_link_ee | iiwa_link_ee_kuka
#     'ur5': 'panda_hand',  # iiwa_link_ee | iiwa_link_ee_kuka
# }

# DEBUG_FAILURE = False


##################################################


class BodyPose(object):
    num = count()

    def __init__(self, body, pose=None):
        if pose is None:
            pose = get_pose(body)
        self.body = body
        self.pose = pose
        self.index = next(self.num)

    @property
    def value(self):
        return self.pose

    def assign(self):
        set_pose(self.body, self.pose)
        return self.pose

    def __repr__(self):
        index = self.index
        # index = id(self) % 1000
        return "p{}".format(index)


class BodyGrasp(object):
    num = count()

    def __init__(self, body, grasp_pose, approach_pose, robot, link):
        self.body = body
        self.grasp_pose = grasp_pose
        self.approach_pose = approach_pose
        self.robot = robot
        self.link = link
        self.index = next(self.num)

    @property
    def value(self):
        return self.grasp_pose

    @property
    def approach(self):
        return self.approach_pose

    # def constraint(self):
    #    grasp_constraint()
    def attachment(self):
        return Attachment(self.robot, self.link, self.grasp_pose, self.body)

    def assign(self):
        return self.attachment().assign()

    def __repr__(self):
        index = self.index
        # index = id(self) % 1000
        return "g{}".format(index)


class BodyConf(object):
    num = count()

    def __init__(self, body, configuration=None, joints=None):
        if joints is None:
            joints = get_movable_joints(body)
        if configuration is None:
            configuration = get_joint_positions(body, joints)
        self.body = body
        self.joints = joints
        self.configuration = configuration
        if len(self.joints) != len(self.configuration):
            print("len mistmatch")
        self.index = next(self.num)

    @property
    def values(self):
        return self.configuration

    def assign(self):
        set_joint_positions(self.body, self.joints, self.configuration)
        return self.configuration

    def __repr__(self):
        index = self.index
        # index = id(self) % 1000
        return "q{}".format(index)


class BodyPath(object):
    def __init__(self, body, path, joints=None, attachments=[]):
        if joints is None:
            joints = get_movable_joints(body)
        self.body = body
        self.path = path
        self.joints = joints
        self.attachments = attachments

    def bodies(self):
        return set([self.body] + [attachment.body for attachment in self.attachments])

    def iterator(self):
        # TODO: compute and cache these
        # TODO: compute bounding boxes as well
        for i, configuration in enumerate(self.path):
            set_joint_positions(self.body, self.joints, configuration)
            for grasp in self.attachments:
                grasp.assign()
            yield i

    def control(self, real_time=False, dt=0):
        # TODO: just waypoints
        if real_time:
            enable_real_time()
        else:
            disable_real_time()
        for values in self.path:
            for _ in joint_controller(self.body, self.joints, values):
                enable_gravity()
                if not real_time:
                    step_simulation()
                time.sleep(dt)

    # def full_path(self, q0=None):
    #     # TODO: could produce sequence of savers
    def refine(self, num_steps=0):
        return self.__class__(
            self.body,
            refine_path(self.body, self.joints, self.path, num_steps),
            self.joints,
            self.attachments,
        )

    def reverse(self):
        return self.__class__(self.body, self.path[::-1], self.joints, self.attachments)

    def __repr__(self):
        return "{}({},{},{},{})".format(
            self.__class__.__name__,
            self.body,
            len(self.joints),
            len(self.path),
            len(self.attachments),
        )


##################################################


class ApplyForce(object):
    def __init__(self, body, robot, link):
        self.body = body
        self.robot = robot
        self.link = link

    def bodies(self):
        return {self.body, self.robot}

    def iterator(self, **kwargs):
        return []

    def refine(self, **kwargs):
        return self

    def __repr__(self):
        return "{}({},{})".format(self.__class__.__name__, self.robot, self.body)


class Attach(ApplyForce):
    def control(self, **kwargs):
        # TODO: store the constraint_id?
        add_fixed_constraint(self.body, self.robot, self.link)

    def reverse(self):
        return Detach(self.body, self.robot, self.link)


class Detach(ApplyForce):
    def control(self, **kwargs):
        remove_fixed_constraint(self.body, self.robot, self.link)

    def reverse(self):
        return Attach(self.body, self.robot, self.link)


class Command(object):
    num = count()

    def __init__(self, body_paths):
        self.body_paths = body_paths
        self.index = next(self.num)

    def bodies(self):
        return set(flatten(path.bodies() for path in self.body_paths))

    def step(self):
        for i, body_path in enumerate(self.body_paths):
            for j in body_path.iterator():
                msg = "{},{}) step?".format(i, j)
                wait_if_gui(msg)
                # print(msg)

    def execute(self, time_step=0.05):
        for i, body_path in enumerate(self.body_paths):
            for j in body_path.iterator():
                # time.sleep(time_step)
                wait_for_duration(time_step)

    def control(self, real_time=False, dt=0):  # TODO: real_time
        for body_path in self.body_paths:
            body_path.control(real_time=real_time, dt=dt)

    def refine(self, **kwargs):
        return self.__class__(
            [body_path.refine(**kwargs) for body_path in self.body_paths]
        )

    def reverse(self):
        return self.__class__(
            [body_path.reverse() for body_path in reversed(self.body_paths)]
        )

    def __repr__(self):
        index = self.index
        # index = id(self) % 1000
        return "c{}".format(index)


#######################################################


def get_tool_link(robot):
    print(f"Running get tool link: {get_body_name(robot)}")
    return link_from_name(robot, "panda_hand")


def pose_to_xyz_quat(handle_pose):
    gripper_pose = np.array(
        [
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0, 0.12],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    pddl_pose = gripper_pose.dot(np.linalg.inv(handle_pose))
    # pddl_pose = np.linalg.inv(handle_pose).dot(gripper_pose)
    xyz = pddl_pose[:3, 3]
    q = npq.as_float_array(npq.from_rotation_matrix(pddl_pose[:3, :3]))
    return (xyz, [q[1], q[2], q[3], q[0]])


# def get_hpp_grasps_gen(robot, obj_ids_real, obj_ids_pyphysx,
# allow_side_handles=False):

approach_points = {
    'Xp': Point(y=1),
    'Xm': Point(y=-1),
    'Yp': Point(x=1),
    'Ym': Point(x=-1),
    'Zp': Point(z=1),
    'Zm': Point(z=-1),
}
def get_hpp_grasps_gen(robot, obj_ids, objects, allow_side_handles=False):
    """
    How are grasps defined? Grasp in pddl is object center pose w.r.t. gripper
    """
    grasps_per_obj = {}

    for obj_id, obj in zip(obj_ids, objects):
        h_poses = obj.get_handles_poses()
        handles = list(h_poses.keys())
        if not allow_side_handles:
            handles = [h for h in handles if h[-2] != "S"]

        grasps_per_obj[obj_id] = [
            (pose_to_xyz_quat(h_poses[h]), Pose(0.1 * approach_points[h[6:8]])) for h in
            handles
        ]
    tool_link = get_tool_link(robot)

    def gen(body):
        for grasp_pose, approach_pose in grasps_per_obj[body]:
            body_grasp = BodyGrasp(body, grasp_pose, approach_pose, robot, tool_link)
            yield (body_grasp,)

    return gen


# def get_grasp_gen(robot, grasp_name="top"):
#     grasp_info = GRASP_INFO[grasp_name]
#     tool_link = get_tool_link(robot)
#
#     def gen(body):
#         grasp_poses = grasp_info.get_grasps(body)
#         # TODO: continuous set of grasps
#         for grasp_pose in grasp_poses:
#             body_grasp = BodyGrasp(
#                 body, grasp_pose, grasp_info.approach_pose, robot, tool_link
#             )
#             yield (body_grasp,)
#
#     return gen


def get_stable_gen(fixed=[], disable_body_links=None):
    def gen(body, surface):
        while True:
            pose = sample_placement(body, surface)
            if (pose is None) or any(
                pairwise_collision(body, b, disable_body_links=disable_body_links)
                for b in fixed
            ):
                continue
            body_pose = BodyPose(body, pose)
            yield (body_pose,)

    return gen


def get_ik_fn(
    robot, fixed=[], teleport=False, disable_body_links=None, num_attempts=10
):
    movable_joints = get_movable_joints(robot)
    sample_fn = get_sample_fn(robot, movable_joints)

    def fn(body, pose, grasp):
        set_pose(body, pose.pose)
        obstacles = [body] + fixed
        gripper_pose = end_effector_from_body(pose.pose, grasp.grasp_pose)
        approach_pose = approach_from_grasp(grasp.approach_pose, gripper_pose)
        for _ in range(num_attempts):
            set_joint_positions(robot, movable_joints, sample_fn())  # Random seed
            # TODO: multiple attempts?
            q_approach = inverse_kinematics(robot, grasp.link, approach_pose)
            if (q_approach is None) or any(
                pairwise_collision(robot, b, disable_body_links) for b in obstacles
            ):
                continue
            conf = BodyConf(robot, q_approach)
            q_grasp = inverse_kinematics(robot, grasp.link, gripper_pose)
            if (q_grasp is None) or any(
                pairwise_collision(robot, b, disable_body_links) for b in obstacles
            ):
                continue
            if teleport:
                path = [q_approach, q_grasp]
            else:
                conf.assign()
                # direction, _ = grasp.approach_pose
                # path = workspace_trajectory(robot, grasp.link,
                # point_from_pose(approach_pose), -direction,
                #                                   quat_from_pose(approach_pose))
                path = plan_direct_joint_motion(
                    robot,
                    conf.joints,
                    q_grasp,
                    obstacles=obstacles,
                    disable_body_links=disable_body_links,
                )
                if path is None:
                    continue
            command = Command(
                [
                    BodyPath(robot, path),
                    Attach(body, robot, grasp.link),
                    BodyPath(robot, path[::-1], attachments=[grasp]),
                ]
            )
            return (conf, command)
            # TODO: holding collisions
        return None

    return fn


##################################################


def assign_fluent_state(fluents):
    obstacles = []
    for fluent in fluents:
        name, args = fluent[0], fluent[1:]
        if name == "atpose":
            o, p = args
            obstacles.append(o)
            p.assign()
        else:
            raise ValueError(name)
    return obstacles


def get_free_motion_gen(
    robot, fixed=[], teleport=False, self_collisions=True, disable_body_links=None
):
    def fn(conf1, conf2, fluents=()):
        assert (conf1.body == conf2.body) and (conf1.joints == conf2.joints)
        if teleport:
            path = [conf1.configuration, conf2.configuration]
        else:
            conf1.assign()
            obstacles = list(set(fixed + assign_fluent_state(fluents)))
            if len(conf2.joints) != len(conf2.configuration):
                print("len mismatch")
            path = plan_joint_motion(
                robot,
                conf2.joints,
                conf2.configuration,
                obstacles=obstacles,
                self_collisions=self_collisions,
                disable_body_links=disable_body_links,
            )
            if path is None:
                return None
        command = Command([BodyPath(robot, path, joints=conf2.joints)])
        return (command,)

    return fn


def get_holding_motion_gen(
    robot, fixed=[], teleport=False, self_collisions=True, disable_body_links=None
):
    def fn(conf1, conf2, body, grasp, fluents=[]):
        assert (conf1.body == conf2.body) and (conf1.joints == conf2.joints)
        if teleport:
            path = [conf1.configuration, conf2.configuration]
        else:
            conf1.assign()
            # obstacles = fixed + assign_fluent_state(fluents)
            obstacles = list(set(fixed + assign_fluent_state(fluents)))
            path = plan_joint_motion(
                robot,
                conf2.joints,
                conf2.configuration,
                disable_body_links=disable_body_links,
                obstacles=obstacles,
                attachments=[grasp.attachment()],
                self_collisions=self_collisions,
            )
            if path is None:
                return None
        command = Command(
            [BodyPath(robot, path, joints=conf2.joints, attachments=[grasp])]
        )
        return (command,)

    return fn


##################################################


def get_movable_collision_test(disable_body_links=None):
    def test(command, body, pose):
        if body in command.bodies():
            return False
        pose.assign()
        # print(type(body))

        for path in command.body_paths:
            moving = path.bodies()
            # print(moving)
            if body in moving:
                # TODO: cannot collide with itself
                continue

            for _ in path.iterator():
                # TODO: could shuffle this
                if any(
                    pairwise_collision(mov, body, disable_body_links=disable_body_links)
                    for mov in moving
                ):
                    return True
        return False

    return test


def pairwise_collision(body1, body2, disable_body_links=None, **kwargs):
    if isinstance(body1, tuple) or isinstance(body2, tuple):
        body1, links1 = expand_links(body1)
        body2, links2 = expand_links(body2)
        if disable_body_links is not None:
            for k, v in disable_body_links.items():
                for body_id, links in v:
                    if body1 == k and body2 == body_id:
                        links2 = list(set(links2) - set(links))
                    elif body2 == k and body1 == body_id:
                        links1 = list(set(links1) - set(links))
        # TODO: exclude links here too?
        return any_link_pair_collision(body1, links1, body2, links2, **kwargs)
    return body_collision(body1, body2, disable_body_links=disable_body_links, **kwargs)

def get_cfree_pose_pose_test(collisions=True, disable_body_links=(), **kwargs):
    def test(b1, p1, b2, p2):
        if not collisions or (b1 == b2):
            return True
        p1.assign()
        p2.assign()
        return not pairwise_collision(b1, b2, disable_body_links=disable_body_links, **kwargs) #, max_distance=0.001)
    return test

def get_cfree_obj_approach_pose_test(collisions=True, disable_body_links=()):
    def test(b1, p1, g1, b2, p2):
        if not collisions or (b1 == b2):
            return True
        p2.assign()
        grasp_pose = multiply(p1.value, invert(g1.value))
        approach_pose = multiply(p1.value, invert(g1.approach), g1.value)
        for obj_pose in interpolate_poses(grasp_pose, approach_pose):
            set_pose(b1, obj_pose)
            if pairwise_collision(b1, b2, disable_body_links=disable_body_links):
                return False
        return True
    return test

def get_collision_fn(
    body,
    joints,
    obstacles,
    attachments,
    self_collisions,
    disabled_collisions,
    disable_body_links,
    custom_limits={},
    use_aabb=False,
    cache=False,
    max_distance=MAX_DISTANCE,
    **kwargs,
):
    # TODO: convert most of these to keyword arguments
    check_link_pairs = (
        get_self_link_pairs(body, joints, disabled_collisions)
        if self_collisions
        else []
    )
    moving_links = frozenset(get_moving_links(body, joints))
    attached_bodies = [attachment.child for attachment in attachments]
    moving_bodies = [CollisionPair(body, moving_links)] + list(
        map(parse_body, attached_bodies)
    )
    # moving_bodies = list(flatten(flatten_links(*pair) for pair in moving_bodies))
    # Introduces overhead
    # moving_bodies = [body] + [attachment.child for attachment in attachments]
    lower_limits, upper_limits = get_custom_limits(body, joints, custom_limits)
    get_obstacle_aabb = cached_fn(
        get_buffered_aabb, cache=cache, max_distance=max_distance / 2.0, **kwargs
    )

    # TODO: sort bodies by bounding box size

    def collision_fn(q, verbose=False):
        if not all_between(lower_limits, q, upper_limits):
            # print('Joint limits violated')
            if verbose:
                print("Joint limits violated")
                print(lower_limits, q, upper_limits)
            return True
        set_joint_positions(body, joints, q)
        for attachment in attachments:
            attachment.assign()
        # wait_for_duration(1e-2)
        get_moving_aabb = cached_fn(
            get_buffered_aabb, cache=True, max_distance=max_distance / 2.0, **kwargs
        )

        for link1, link2 in check_link_pairs:
            # Self-collisions should not have the max_distance parameter
            # TODO: self-collisions between body and attached_bodies
            #  (except for the link adjacent to the robot)
            if (
                not use_aabb
                or aabb_overlap(get_moving_aabb(body), get_moving_aabb(body))
            ) and pairwise_link_collision(
                body, link1, body, link2
            ):  # , **kwargs):
                # print(get_body_name(body), get_link_name(body, link1),
                # get_link_name(body, link2))
                if verbose:
                    print(body, link1, body, link2)
                return True
        # step_simulation()
        # for body1 in moving_bodies:
        #     for body2, _ in get_bodies_in_region(get_moving_aabb(body1)):
        #         if (body2 in obstacles) and
        #         pairwise_collision(body1, body2, **kwargs):
        #             #print(get_body_name(body1), get_body_name(body2))
        #             if verbose: print(body1, body2)
        #             return True
        for body1, body2 in product(moving_bodies, obstacles):
            # TODO: this will disable tunnel collision
            # if 0 in [body1.body, body2] and 3 in [body1.body, body2]:
            #     continue
            if (
                not use_aabb
                or aabb_overlap(get_moving_aabb(body1), get_obstacle_aabb(body2))
            ) and pairwise_collision(
                body1, body2, disable_body_links=disable_body_links, **kwargs
            ):
                # print(get_body_name(body1), get_body_name(body2))
                if verbose:
                    print(body1, body2)
                return True
        return False

    return collision_fn


def plan_joint_motion(
    body,
    joints,
    end_conf,
    obstacles=[],
    attachments=[],
    disable_body_links=None,
    self_collisions=True,
    disabled_collisions=set(),
    weights=None,
    resolutions=None,
    max_distance=MAX_DISTANCE,
    use_aabb=False,
    cache=True,
    custom_limits={},
    **kwargs,
):
    assert len(joints) == len(end_conf)
    if (weights is None) and (resolutions is not None):
        weights = np.reciprocal(resolutions)
    sample_fn = get_sample_fn(body, joints, custom_limits=custom_limits)
    distance_fn = get_distance_fn(body, joints, weights=weights)
    extend_fn = get_extend_fn(body, joints, resolutions=resolutions)
    collision_fn = get_collision_fn(
        body,
        joints,
        obstacles,
        attachments,
        self_collisions,
        disabled_collisions,
        custom_limits=custom_limits,
        max_distance=max_distance,
        disable_body_links=disable_body_links,
        use_aabb=use_aabb,
        cache=cache,
    )

    start_conf = get_joint_positions(body, joints)

    if not check_initial_end(start_conf, end_conf, collision_fn):
        return None
    return birrt(
        start_conf, end_conf, distance_fn, sample_fn, extend_fn, collision_fn, **kwargs
    )
    # return plan_lazy_prm(start_conf, end_conf, sample_fn, extend_fn, collision_fn)


plan_holonomic_motion = plan_joint_motion


def plan_waypoints_joint_motion(
    body,
    joints,
    waypoints,
    start_conf=None,
    obstacles=[],
    attachments=[],
    disable_body_links=None,
    self_collisions=True,
    disabled_collisions=set(),
    resolutions=None,
    custom_limits={},
    max_distance=MAX_DISTANCE,
    use_aabb=False,
    cache=True,
):
    if start_conf is None:
        start_conf = get_joint_positions(body, joints)
    assert len(start_conf) == len(joints)
    collision_fn = get_collision_fn(
        body,
        joints,
        obstacles,
        attachments,
        self_collisions,
        disabled_collisions,
        custom_limits=custom_limits,
        max_distance=max_distance,
        disable_body_links=disable_body_links,
        use_aabb=use_aabb,
        cache=cache,
    )
    waypoints = [start_conf] + list(waypoints)
    for i, waypoint in enumerate(waypoints):
        if collision_fn(waypoint):
            # print('Warning: waypoint configuration {}/{} is in
            # collision'.format(i, len(waypoints)))
            return None
    return interpolate_joint_waypoints(
        body, joints, waypoints, resolutions=resolutions, collision_fn=collision_fn
    )


def plan_direct_joint_motion(body, joints, end_conf, disable_body_links=None, **kwargs):
    return plan_waypoints_joint_motion(
        body, joints, [end_conf], disable_body_links=disable_body_links, **kwargs
    )


def plan_nonholonomic_motion(
    body,
    joints,
    end_conf,
    obstacles=[],
    attachments=[],
    self_collisions=True,
    disabled_collisions=set(),
    weights=None,
    resolutions=None,
    reversible=True,
    linear_tol=EPSILON,
    angular_tol=0.0,
    max_distance=MAX_DISTANCE,
    use_aabb=False,
    cache=True,
    custom_limits={},
    **kwargs,
):
    assert len(joints) == len(end_conf)
    sample_fn = get_sample_fn(body, joints, custom_limits=custom_limits)
    distance_fn = get_nonholonomic_distance_fn(
        body, joints, weights=weights, reversible=reversible, linear_tol=linear_tol
    )
    # , angular_tol=angular_tol)
    extend_fn = get_nonholonomic_extend_fn(
        body,
        joints,
        resolutions=resolutions,
        reversible=reversible,
        linear_tol=linear_tol,
        angular_tol=angular_tol,
    )
    collision_fn = get_collision_fn(
        body,
        joints,
        obstacles,
        attachments,
        self_collisions,
        disabled_collisions,
        custom_limits=custom_limits,
        max_distance=max_distance,
        use_aabb=use_aabb,
        cache=cache,
    )

    start_conf = get_joint_positions(body, joints)
    if not check_initial_end(start_conf, end_conf, collision_fn):
        return None
    return birrt(
        start_conf, end_conf, distance_fn, sample_fn, extend_fn, collision_fn, **kwargs
    )


plan_differential_motion = plan_nonholonomic_motion


def get_closest_points(
    body1,
    body2,
    link1=None,
    link2=None,
    max_distance=MAX_DISTANCE,
    use_aabb=False,
    disable_body_links=None,
):
    if use_aabb and not aabb_overlap(
        get_buffered_aabb(body1, link1, max_distance=max_distance / 2.0),
        get_buffered_aabb(body2, link2, max_distance=max_distance / 2.0),
    ):
        return []
    if (link1 is None) and (link2 is None):
        results = p.getClosestPoints(
            bodyA=body1, bodyB=body2, distance=max_distance, physicsClientId=CLIENT
        )
    elif link2 is None:
        results = p.getClosestPoints(
            bodyA=body1,
            bodyB=body2,
            linkIndexA=link1,
            distance=max_distance,
            physicsClientId=CLIENT,
        )
    elif link1 is None:
        results = p.getClosestPoints(
            bodyA=body1,
            bodyB=body2,
            linkIndexB=link2,
            distance=max_distance,
            physicsClientId=CLIENT,
        )
    else:
        results = p.getClosestPoints(
            bodyA=body1,
            bodyB=body2,
            linkIndexA=link1,
            linkIndexB=link2,
            distance=max_distance,
            physicsClientId=CLIENT,
        )
    res = []
    """
        0 contactFlag
        1 bodyUniqueIdA
        2 bodyUniqueIdB
        3 linkIndexA
        4 linkIndexB
        positionOnA
        positionOnB
        contactNormalOnB
        contactDistance
        normalForce
        lateralFriction1
        lateralFrictionDir1
        lateralFriction2
        lateralFrictionDir2
    """
    # disable_body_links - for a certain body (like robot) disable collision
    # with a certain link (like tunnel wall)
    # dict {'robot_body_id': [(tunnel_body_id, ['tunnel_top_sideB_link_id', ...])]}

    # TODO: rewrite this later
    for info in results:
        skip = False
        if disable_body_links is not None:
            for k, v in disable_body_links.items():
                if info[1] == k or info[2] == k:
                    second_body_id = info[2] if info[1] == k else info[1]
                    for body_id, links in v:
                        if body_id == second_body_id:
                            for link in links:
                                if info[4] == link:
                                    skip = True
        if not skip:
            res.append(CollisionInfo(*info))
    return res


def body_collision(body1, body2, disable_body_links=None, **kwargs):
    return (
        len(
            get_closest_points(
                body1, body2, disable_body_links=disable_body_links, **kwargs
            )
        )
        != 0
    )
