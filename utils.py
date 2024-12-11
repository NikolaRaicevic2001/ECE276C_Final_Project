from typing import Tuple, List, Optional, Dict

import pybullet as p
import numpy as np

###################################
####### INFO FUNCTIONS ############
###################################
def print_robot_info(panda_id, num_joints: int):
    """Print information about the robot"""
    print(f"Loaded Panda robot with {num_joints} joints")
    for i in range(num_joints):
        joint_info = p.getJointInfo(panda_id, i)
        print(f"Joint {i}: {joint_info[1].decode('utf-8')} at joint angle {p.getJointState(panda_id, i)[0]}")

def get_robot_info(robot_id):
    num_joints = p.getNumJoints(robot_id)

    # Retrieve and print info for all links, including the base
    print("Base Info:")
    base_info = p.getBodyInfo(robot_id)
    print(f"Base Name: {base_info[0].decode('utf-8')}")

    for link_idx in range(num_joints):
        # Get link state
        link_name = p.getJointInfo(robot_id, link_idx)[12].decode('utf-8')
        print(f"Link {link_idx}: {link_name}")

        link_state = p.getLinkState(robot_id, link_idx)
        print(f"  World Position: {link_state[4]}")
        print(f"  World Orientation: {link_state[5]}")

###################################
####### STATE FUNCTIONS ###########
###################################
def get_position_from_id(object_id: int) -> Tuple[List[float], List[float]]:
    ''' Get the position and orientation of the object '''
    pos, ori = p.getBasePositionAndOrientation(object_id)
    return (pos, ori)

def get_end_effector_state(panda_id, end_effector_index) -> Tuple[List[float], List[float]]:
    """Get current end effector position and orientation"""
    state = p.getLinkState(panda_id, end_effector_index)
    return state[0], state[1]

###################################
##### COMPUTATION FUNCTIONS #######
###################################

def inverse_kinematics(robot_id, target_position, target_orientation=None):
    """Compute joint angles for a target gripper position"""
    # If no specific orientation is provided, use a default downward orientation
    if target_orientation is None:
        # Quaternion for pointing straight down
        target_orientation = p.getQuaternionFromEuler([np.pi, 0, 0])

    # Adjust target position to account for gripper offset
    # We need to move the target point back along the z-axis of the gripper
    link_state = p.getLinkState(robot_id, 6)
    current_orient = link_state[1]
    current_rot_matrix = np.array(p.getMatrixFromQuaternion(current_orient)).reshape(3, 3)

    # Adjust target position by subtracting the offset along the current z-axis
    adjusted_target = np.array(target_position) - current_rot_matrix[:, 2] * 0.1

    # Use PyBullet's inverse kinematics solver
    joint_angles = p.calculateInverseKinematics(
        robot_id,
        6,  # End-effector link index
        adjusted_target,
        targetOrientation=target_orientation
    )

    return joint_angles