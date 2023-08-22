from hpp.corbaserver.manipulation.robot import Robot
from guided_tamp_benchmark.models.robots import BaseRobot


class HppRobot(Robot):
    def __init__(
        self, robot: BaseRobot, load: bool = True, root_joint_type: str = "anchor"
    ):
        self.robot = robot
        self.name = robot.name
        self.urdfFilename = robot.urdfFilename
        self.srdfFilename = robot.srdfFilename
        self.urdfSuffix = robot.urdfSuffix
        self.srdfSuffix = robot.srdfSuffix
        Robot.__init__(self, robot.name, robot.name, root_joint_type, load)

    def get_contact_surfaces(self) -> list[str]:
        """Returns names of robot contact surfaces"""
        return self.robot.get_contact_surfaces()

    def initial_configuration(self) -> list[float]:
        """Returns robot default configuration"""
        return self.robot.initial_configuration()

    def get_gripper_name(self) -> str:
        """Returns robot gripper name, assume one gripper per robot"""
        return self.robot.get_gripper_name()

    def modify_open_gripper(self, config: list[float]) -> list[float]:
        """Modifies panda hand fingers (last two joints) to maximum values"""
        ndof = len(self.initial_configuration())
        config[ndof - 2 : ndof] = [0.039, 0.039]
        return config
