import example_robot_data as erd
import numpy as np
from robomeshcat import Robot, Scene
from pathlib import Path
import pinocchio as pin
from guided_tamp_benchmark.models.objects import ObjectYCBV
from guided_tamp_benchmark.models.utils import get_models_data_directory
from guided_tamp_benchmark.models.robots import PandaRobot
"""
Visualize in robomeshcat panda robot with YCBV 02 object
Set robot pose to np.eye(4) and object pose to [0.6, 0., 0.11]
"""
robot = erd.load("panda")
rmodel1 = robot.model
rdata1 = rmodel1.createData()
q_init1 = pin.neutral(rmodel1)

pin.forwardKinematics(rmodel1, rdata1, q_init1)
pin.updateFramePlacements(rmodel1, rdata1)

# pring first link placement
# print(rdata1.oMi[0])

"Repeat same for guided_tamp_benchmark panda"
my_robot = PandaRobot()
rmodel2 = pin.buildModelFromUrdf(my_robot.urdfFilename)
rdata2 = rmodel2.createData()
q_init2 = pin.neutral(rmodel2)

pin.forwardKinematics(rmodel2, rdata2, q_init2)
pin.updateFramePlacements(rmodel2, rdata2)

for i in range(rmodel2.nframes):
    print(rmodel1.frames[i].name)
    print(rmodel2.frames[i].name)
    print(rdata1.oMf[i])
    print(rdata2.oMf[i])
    print(np.allclose(rdata2.oMf[i].homogeneous, rdata1.oMf[i].homogeneous))