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
def random_orientation():
    """Generate a random quaternion for target orientation."""
    u1, u2, u3 = np.random.uniform(0, 1, 3)
    w = np.sqrt(1 - u1) * np.sin(2 * np.pi * u2)
    x = np.sqrt(1 - u1) * np.cos(2 * np.pi * u2)
    y = np.sqrt(u1) * np.sin(2 * np.pi * u3)
    z = np.sqrt(u1) * np.cos(2 * np.pi * u3)
    return [x, y, z, w]

def inverse_kinematics(robot_id, target_position, target_orientation=None, num_attempts=5):
    """
    Compute joint angles for a target gripper position with more diverse solutions
    by randomizing orientations and initial joint configurations.
    """
    best_solution = None
    min_error = float('inf')

    for _ in range(num_attempts):
        # Randomize target orientation if not provided
        if target_orientation is None:
            random_target_orientation = random_orientation()
        else:
            random_target_orientation = target_orientation

        # Solve inverse kinematics
        joint_angles = p.calculateInverseKinematics(robot_id, 11, target_position, random_target_orientation)

        # Set joint angles to inverse kinematics positions
        for i in range(7):
            p.resetJointState(robot_id, i, joint_angles[i])

        # Evaluate solution quality based on distance/error to target
        end_effector_state = p.getLinkState(robot_id, 7)
        actual_position = np.array(end_effector_state[4])  # End effector world position
        error = np.linalg.norm(np.array(target_position) - actual_position)

        # Update best solution if the error is smaller
        if error < min_error:
            best_solution = joint_angles
            min_error = error

    return best_solution[:7]

