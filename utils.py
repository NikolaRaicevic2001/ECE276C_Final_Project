from typing import Tuple, List, Optional, Dict

import pybullet as p
import numpy as np

###################################
####### INFO FUNCTIONS ############
###################################
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
        print(f"  Limits: {p.getJointInfo(robot_id, link_idx)[8:10]}")

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

def wrap_joint_angles(joint_angles, joint_limits):
    """Wrap joint angles to stay within specified limits."""
    wrapped_angles = []
    for angle, (lower, upper) in zip(joint_angles, joint_limits):
        range = upper - lower
        while angle < lower:
            angle += 2*np.pi
        while angle > upper:
            angle -= 2*np.pi
        wrapped_angles.append(angle)
    return np.array(wrapped_angles)

def get_jacobian(robot_id, link_index, movable_joints):
    """Get the Jacobian matrix for the given link."""
    # Get current joint positions for movable joints
    joint_positions = [p.getJointState(robot_id, i)[0] for i in movable_joints]

    # Calculate Jacobian
    linear_jacobian, angular_jacobian = p.calculateJacobian(
        bodyUniqueId=robot_id,
        linkIndex=link_index,
        localPosition=[0.0, 0.0, 0.0],
        objPositions=joint_positions,
        objVelocities=[0.0] * len(joint_positions),
        objAccelerations=[0.0] * len(joint_positions)
    )
    
    # Combine linear and angular Jacobians
    return np.vstack((linear_jacobian, angular_jacobian))


def quaternion_difference(q1, q2):
    """Calculate the difference between two quaternions."""
    q1 = np.array(q1)
    q2 = np.array(q2)
    return np.array([
        q1[3]*q2[0] - q1[0]*q2[3] - q1[1]*q2[2] + q1[2]*q2[1],
        q1[3]*q2[1] + q1[0]*q2[2] - q1[1]*q2[3] - q1[2]*q2[0],
        q1[3]*q2[2] - q1[0]*q2[1] + q1[1]*q2[0] - q1[2]*q2[3],
        q1[3]*q2[3] + q1[0]*q2[0] + q1[1]*q2[1] + q1[2]*q2[2]
    ])

def damped_least_squares_ik(robot_id, target_position, target_orientation, joint_limits, movable_joints, max_iterations=100, step_size=0.1, damping=0.1):
    """
    Perform step-by-step IK calculation using damped least squares method.
    """
    num_joints = len(movable_joints)
    current_joint_angles = [p.getJointState(robot_id, i)[0] for i in movable_joints]
    
    for _ in range(max_iterations):
        # Get current end effector position and orientation
        current_pos, current_orn = p.getLinkState(robot_id, 11)[:2]
        
        # Calculate position and orientation error
        pos_error = np.array(target_position) - np.array(current_pos)
        orn_error = quaternion_difference(current_orn, target_orientation)[:3]  # Use only x, y, z components
        total_error = np.concatenate([pos_error, orn_error])
        
        if np.linalg.norm(total_error) < 1e-3:
            break  # Convergence achieved
        
        # Get Jacobian
        J = get_jacobian(robot_id, 7, movable_joints)
        
        # Compute damped least squares solution
        JTJ = np.dot(J.T, J)
        lambda_I = damping * np.eye(num_joints)
        inv_JTJ_plus_lambda = np.linalg.inv(JTJ + lambda_I)
        delta_theta = np.dot(np.dot(inv_JTJ_plus_lambda, J.T), total_error)
        
        # Apply joint limits
        for i in range(num_joints):
            lower, upper = joint_limits[i]
            current_joint_angles[i] += step_size * delta_theta[i]
            current_joint_angles[i] = np.clip(current_joint_angles[i], lower, upper)
        
        # Update robot configuration
        for i in range(num_joints):
            p.resetJointState(robot_id, i, current_joint_angles[i])
    
    return current_joint_angles

def inverse_kinematics(robot_id, target_position, target_orientation=None, num_attempts=5, joint_limits_scale = 0.9):
    """
    Compute joint angles for a target gripper position using step-by-step IK.
    """
    best_solution = None
    min_error = float('inf')

    # Identify movable joints
    num_joints = p.getNumJoints(robot_id)
    movable_joints = [i for i in range(num_joints) if p.getJointInfo(robot_id, i)[2] != p.JOINT_FIXED]

    # Get joint limits for the panda robot
    joint_limits = []
    for i in movable_joints:
        joint_info = p.getJointInfo(robot_id, i)
        lower_limit, upper_limit = joint_info[8:10]
        scaled_lower = lower_limit * joint_limits_scale
        scaled_upper = upper_limit * joint_limits_scale
        joint_limits.append((scaled_lower, scaled_upper))

    for _ in range(num_attempts):
        # Randomize target orientation if not provided
        if target_orientation is None:
            random_target_orientation = random_orientation()

        # # Solve inverse kinematics
        # joint_angles = p.calculateInverseKinematics(robot_id, 11, target_position, random_target_orientation)

        # Perform step-by-step IK
        joint_angles = damped_least_squares_ik(robot_id, target_position, random_target_orientation, joint_limits, movable_joints)

        # Evaluate solution quality
        end_effector_state = p.getLinkState(robot_id, 11)
        actual_position = np.array(end_effector_state[4])  # End effector world position
        error = np.linalg.norm(np.array(target_position) - actual_position)

        # Update best solution if the error is smaller
        if error < min_error:
            best_solution = joint_angles
            min_error = error

    return best_solution[:7]  # Return only the arm joint angles
