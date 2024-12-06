from typing import Tuple, List, Optional, Dict
import pybullet as p

def get_position_from_id(object_id: int) -> Tuple[List[float], List[float]]:
    ''' Get the position and orientation of the object '''
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
