from typing import Tuple, List, Optional, Dict
import pybullet as p

###################################
############ FUNCTIONS ############
###################################
def get_position_from_id(object_id: int) -> Tuple[List[float], List[float]]:
    ''' Get the position and orientation of the object '''
    # for object_id in object_ids:
    pos, ori = p.getBasePositionAndOrientation(object_id)
    return (pos, ori)

def get_end_effector_state(panda_id, end_effector_index) -> Tuple[List[float], List[float]]:
    """Get current end effector position and orientation"""
    state = p.getLinkState(panda_id, end_effector_index)
    return state[0], state[1]

def print_robot_info(panda_id, num_joints: int):
    """Print information about the robot"""
    print(f"Loaded Panda robot with {num_joints} joints")
    for i in range(num_joints):
        joint_info = p.getJointInfo(panda_id, i)
        print(f"Joint {i}: {joint_info[1].decode('utf-8')}")

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
