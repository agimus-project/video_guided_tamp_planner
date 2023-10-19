from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from tamp_guided_by_video.planners.hpp_planner import HppPlanner


def sample_state_on_transition_target(
    planner: "HppPlanner",
    q_from: list[float],
    transition: str,
    max_iter: int = 1,
    open_fingers: bool = True,
) -> Optional[list[float]]:
    """Sample a configuration, that lies on some transition from the q_from"""
    for i in range(max_iter):
        succ, q1, err = planner.cg.generateTargetConfig(
            transition, q_from, planner.robot.shootRandomConfig()
        )
        if succ:
            q = planner.robot.modify_open_gripper(q1.copy()) if open_fingers else q1
            res, msg = planner.robot.isConfigValid(q)
            if res:
                return q
    return None


def build_direct_path_any_direction(
    planner: "HppPlanner", q_from: list[float], q_to: list[float]
) -> (bool, int, str):
    """Tries to build a path between two configurations in both directions"""
    res, pid, msg = planner.ps.directPath(q_from, q_to, True)
    if not res:
        res, pid, msg = planner.ps.directPath(q_to, q_from, True)
    return res, pid, msg


def get_config_states(planner: "HppPlanner", config: list[float]) -> list[str]:
    """
    Return the states to which configuration belongs
    :return: list of state names
    """
    config_states = [
        k
        for k, v in planner.cg.nodes.items()
        if planner.cg.getConfigErrorForNode(k, config)[0]
    ]
    # config_states = [k for k in config_states if not 'free' in k]
    # if len(config_states) > 1:
    #     raise ValueError(f"How to deal with len = {len(config_states)}?
    #     Config: {config}, states: {config_states}")
    return config_states
