import numpy as np
import pybullet as p
import pybullet_data
import time
import os

from Planners.RRT import RRTManipulatorPlanner
from point_cloud import draw_camera_frame, camera_to_world, world_to_camera, get_point_cloud
from utils import *

def environment_setup(env_num = 1):
    ''' Setup the PyBullet environment '''
    # PyBullet setup
    physics_client = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)

    # Environment 01 - Kick a Soccer Ball
    collision_ids = []
    if env_num == 1:
        # Load Panda robot and its environment
        ground_id = p.loadURDF("plane.urdf")
        robot_id = p.loadURDF("franka_panda/panda.urdf", [0, 0.3, 0.6], useFixedBase=True)
        table_id = p.loadURDF(os.path.join(pybullet_data.getDataPath(), "table/table.urdf"), basePosition=[0, 0, 0], useFixedBase=True)
        # Add Collision Objects
        teddy_vhacd_id_1 = p.loadURDF("teddy_vhacd.urdf", [-0.5, 0.1, 0.62], baseOrientation=p.getQuaternionFromEuler([1.5, 0, -0.2]), globalScaling=2.0)
        teddy_vhacd_id_2 = p.loadURDF("teddy_vhacd.urdf", [0.3, -0.3, 0.62], baseOrientation=p.getQuaternionFromEuler([1.5, 0, -0.2]), globalScaling=2.0)
        teddy_vhacd_id_3 = p.loadURDF("teddy_vhacd.urdf", [-0.1, -0.1, 0.62], baseOrientation=p.getQuaternionFromEuler([1.5, 0, -0.2]), globalScaling=2.0)
        teddy_large_id_1 = p.loadURDF("teddy_large.urdf", basePosition = [-0.4, 0.1, 0.62], baseOrientation=p.getQuaternionFromEuler([1.5, 0, 0.2]), globalScaling=0.1)
        teddy_large_id_2 = p.loadURDF("teddy_large.urdf", basePosition = [-0.4, -0.2, 0.62], baseOrientation=p.getQuaternionFromEuler([1.5, 0, 0.2]), globalScaling=0.1)
        teddy_large_id_3 = p.loadURDF("teddy_large.urdf", basePosition = [0.3, 0.1, 0.62], baseOrientation=p.getQuaternionFromEuler([1.5, 0, 0.2]), globalScaling=0.1)
        duck_vhacd_id_1 = p.loadURDF("duck_vhacd.urdf", [-0.1, -0.2, 0.65], baseOrientation=p.getQuaternionFromEuler([0, 0, 0.5]), globalScaling=1.5)
        duck_vhacd_id_2 = p.loadURDF("duck_vhacd.urdf", [0.4, -0.3, 0.65], baseOrientation=p.getQuaternionFromEuler([0, 0, 0.5]), globalScaling=1.5)
        duck_vhacd_id_3 = p.loadURDF("duck_vhacd.urdf", [-0.5, -0.4, 0.65], baseOrientation=p.getQuaternionFromEuler([0, 0, 0.5]), globalScaling=1.5)
        cloth_z_up_id = p.loadURDF("cloth_z_up.urdf", [0, 1.4, 2.0], baseOrientation=p.getQuaternionFromEuler([-np.pi/2, 0, 0]), globalScaling=2.0, useFixedBase=True)
        goal_id = p.loadURDF("soccerball.urdf", [0.2, -0.3, 0.625], baseOrientation=p.getQuaternionFromEuler([0, 0, 0.5]), globalScaling=0.2)
        # Add Collision Objects
        collision_ids = [ground_id, table_id, teddy_vhacd_id_1, teddy_vhacd_id_2, teddy_vhacd_id_3, 
                         teddy_large_id_1, teddy_large_id_2, teddy_large_id_3, 
                         duck_vhacd_id_1, duck_vhacd_id_2, duck_vhacd_id_3, 
                         cloth_z_up_id]
        
    # Environment 02 - Spheres
    if env_num == 2:
        # Load Panda robot and its environment
        ground_id = p.loadURDF("plane.urdf")
        robot_id = p.loadURDF("franka_panda/panda.urdf", [0.0, 0.0, 0.0], useFixedBase=True)
        # Add Collision Objects
        sphere2red_id_1 = p.loadURDF("sphere2red.urdf", [-0.6, -0.5, 0.1], globalScaling=0.2)
        sphere2red_id_2 = p.loadURDF("sphere2red.urdf", [-0.3, -0.2, 0.1], globalScaling=0.2)
        sphere2red_id_3 = p.loadURDF("sphere2red.urdf", [0.3, -0.3, 0.1], globalScaling=0.2)
        sphere2_id_1 = p.loadURDF("sphere2.urdf", [-0.5, -0.4, 0.1], globalScaling=0.2)
        sphere2_id_2 = p.loadURDF("sphere2.urdf", [0.3, -0.2, 0.1], globalScaling=0.2)
        sphere2_id_3 = p.loadURDF("sphere2.urdf", [0.7, 0.1, 0.1], globalScaling=0.2)
        sphere_transparent_id_1 = p.loadURDF("sphere_transparent.urdf", [-0.6, 0.2, 0.1], globalScaling=0.2)
        sphere_transparent_id_2 = p.loadURDF("sphere_transparent.urdf", [0.1, -0.5, 0.1], globalScaling=0.2)
        sphere_transparent_id_3 = p.loadURDF("sphere_transparent.urdf", [0.4, -0.7, 0.1], globalScaling=0.2)
        goal_id = p.loadURDF("soccerball.urdf", [-0.2, -0.6, 0.1], baseOrientation=p.getQuaternionFromEuler([0, 0, 0.5]), globalScaling=0.2)
        # Add Collision Objects
        collision_ids = [ground_id, sphere2red_id_1, sphere2red_id_2, sphere2red_id_3,
                         sphere2_id_1, sphere2_id_2, sphere2_id_3,
                         sphere_transparent_id_1, sphere_transparent_id_2, sphere_transparent_id_3]

    # Environment 03 - Construction Site Static
    if env_num == 3:
        # Load Panda robot and its environment
        ground_id = p.loadURDF("plane.urdf")
        robot_id = p.loadURDF("franka_panda/panda.urdf", [0.0, 0.0, 0.0], useFixedBase=True)
        # Add Collision Objects
        r2d2_id_1 = p.loadURDF("r2d2.urdf", [-0.7, 0.4, 0.2], baseOrientation=p.getQuaternionFromEuler([0, 0, -0.5]), globalScaling=0.5)
        r2d2_id_2 = p.loadURDF("r2d2.urdf", [0.5, -0.8, 0.2], baseOrientation=p.getQuaternionFromEuler([0, 0, -0.8]), globalScaling=0.5)
        r2d2_id_3 = p.loadURDF("r2d2.urdf", [0.8, 0.1, 0.2], baseOrientation=p.getQuaternionFromEuler([0, 0, -0.2]), globalScaling=0.5)
        block_id = p.loadURDF("block.urdf", [0.0, -0.4, 0.5], baseOrientation=p.getQuaternionFromEuler([0, np.pi/2, 0]), globalScaling=10.0, useFixedBase=True)
        TwoJointRobot_id_1 = p.loadURDF("TwoJointRobot_wo_fixedJoints.urdf", [-1.5, 1.5, 0.6], baseOrientation=p.getQuaternionFromEuler([0, 0, 0]), globalScaling=1.5, useFixedBase=True)
        TwoJointRobot_id_2 = p.loadURDF("TwoJointRobot_wo_fixedJoints.urdf", [-1.5, 1.5, 0.8], baseOrientation=p.getQuaternionFromEuler([0, 0, 0]), globalScaling=1.5, useFixedBase=True)
        TwoJointRobot_id_3 = p.loadURDF("TwoJointRobot_wo_fixedJoints.urdf", [2.0, -2.2, 0.6], baseOrientation=p.getQuaternionFromEuler([0, 0, np.pi/2]), globalScaling=1.5, useFixedBase=True)
        TwoJointRobot_id_4 = p.loadURDF("TwoJointRobot_wo_fixedJoints.urdf", [2.0, -2.2, 0.8], baseOrientation=p.getQuaternionFromEuler([0, 0, np.pi/2]), globalScaling=1.5, useFixedBase=True)
        TwoJointRobot_id_5 = p.loadURDF("TwoJointRobot_wo_fixedJoints.urdf", [1.8, -2.5, 0.6], baseOrientation=p.getQuaternionFromEuler([0, 0, np.pi]), globalScaling=1.5, useFixedBase=True)
        TwoJointRobot_id_6 = p.loadURDF("TwoJointRobot_wo_fixedJoints.urdf", [1.8, -2.5, 0.8], baseOrientation=p.getQuaternionFromEuler([0, 0, np.pi]), globalScaling=1.5, useFixedBase=True)
        TwoJointRobot_id_7 = p.loadURDF("TwoJointRobot_wo_fixedJoints.urdf", [-2.0, 1.0, 0.6], baseOrientation=p.getQuaternionFromEuler([0, 0, 3*np.pi/2]), globalScaling=1.5, useFixedBase=True)
        TwoJointRobot_id_8 = p.loadURDF("TwoJointRobot_wo_fixedJoints.urdf", [-2.0, 1.0, 0.8], baseOrientation=p.getQuaternionFromEuler([0, 0, 3*np.pi/2]), globalScaling=1.5, useFixedBase=True)
        cube_id_1 = p.loadURDF("cube.urdf", [-1.5, -1.5, 0.5], baseOrientation=p.getQuaternionFromEuler([0, 0, np.pi/4]), globalScaling=0.5)
        cube_id_2 = p.loadURDF("cube.urdf", [1.0, 2.0, 0.1], baseOrientation=p.getQuaternionFromEuler([0, 0, np.pi/8]), globalScaling=0.5)
        cube_id_3 = p.loadURDF("cube.urdf", [1.5, -2.0, 0.1], baseOrientation=p.getQuaternionFromEuler([0, 0, np.pi/16]), globalScaling=0.5)
        cartpole_id = p.loadURDF("cartpole.urdf", [0.0, -2.8, 0.2], useFixedBase=True)
        spherical_joint_limit_id = p.loadURDF("spherical_joint_limit.urdf", [-0.3, -1.7, 0.2], baseOrientation=p.getQuaternionFromEuler([0, 0, np.pi/16]), globalScaling=0.5)
        goal_id = 0
        # Add Collision Objects
        collision_ids = [ground_id, r2d2_id_1, r2d2_id_2, r2d2_id_3, 
                         block_id, 
                         TwoJointRobot_id_1, TwoJointRobot_id_2, TwoJointRobot_id_3, TwoJointRobot_id_4, TwoJointRobot_id_5, TwoJointRobot_id_6, TwoJointRobot_id_7, TwoJointRobot_id_8, 
                         cube_id_1, cube_id_2, cube_id_3,
                         cartpole_id,
                         spherical_joint_limit_id]

    # Environment 04 - Construction Site Dynamic
    if env_num == 4:
        # Load Panda robot and its environment
        ground_id = p.loadURDF("plane.urdf")
        robot_id = p.loadURDF("franka_panda/panda.urdf", [0.0, 0.0, 0.0], useFixedBase=True)
        # Add Collision Objects
        r2d2_id_1 = p.loadURDF("r2d2.urdf", [-0.4, -0.4, 0.2], baseOrientation=p.getQuaternionFromEuler([0, 0, np.pi/2]), globalScaling=0.5)
        r2d2_id_2 = p.loadURDF("r2d2.urdf", [0.4, -0.4, 0.2], baseOrientation=p.getQuaternionFromEuler([0, 0, -np.pi/2]), globalScaling=0.5)
        r2d2_id_3 = p.loadURDF("r2d2.urdf", [0.0, -0.8, 0.2], baseOrientation=p.getQuaternionFromEuler([0, 0, np.pi]), globalScaling=0.5)
        block_id = p.loadURDF("block.urdf", [0.0, -0.4, 0.5], baseOrientation=p.getQuaternionFromEuler([0, np.pi/2, 0]), globalScaling=10.0, useFixedBase=True)
        goal_id = 0
        # Add Collision Objects
        collision_ids = [ground_id, r2d2_id_1, r2d2_id_2, r2d2_id_3, 
                         block_id]

        # Reset Panda's position to Home Position
        home_positions = [0.0, -0.485, 0.0, -1.056, 0.0, 0.7, 0.785]
        for i in range(7):
            p.resetJointState(robot_id, i, home_positions[i]),
    
        # Print Robot Information
        print_robot_info(robot_id, 7)

        # Print End Effector State
        print(f'End Effector State:{get_end_effector_state(robot_id, 7)}')

    return physics_client, collision_ids, goal_id, robot_id

def environment_update(collision_ids, dt = 1/240, velocity = 0.5, flag_01 = 1, flag_02 = 1, flag_03 = 1):
    ''' Update the environment to make R2D2s move back and forth'''
    # Moving R2D2 back and forth
    for i in range(3):
        pos, ori = p.getBasePositionAndOrientation(collision_ids[i+1]) 

        if i == 0:
            if flag_01 == 1:
                p.resetBasePositionAndOrientation(collision_ids[i+1], pos+np.array([dt*velocity,0, 0]), ori)
            else:
                p.resetBasePositionAndOrientation(collision_ids[i+1], pos+np.array([-dt*velocity,0,0]), ori)

            if pos[0] < -1.2:
                flag_01 = 1
            elif pos[0] > -0.4:
                flag_01 = 0

        if i == 1:
            if flag_02 == 1:
                p.resetBasePositionAndOrientation(collision_ids[i+1], pos+np.array([dt*velocity,0, 0]), ori)
            else:
                p.resetBasePositionAndOrientation(collision_ids[i+1], pos+np.array([-dt*velocity,0, 0]), ori)

            if pos[0] < 0.4:
                flag_02 = 1
            elif pos[0] > 1.2:
                flag_02 = 0

        if i == 2:
            if flag_03 == 1:
                p.resetBasePositionAndOrientation(collision_ids[i+1], pos+np.array([0,dt*velocity, 0]), ori)
            else:
                p.resetBasePositionAndOrientation(collision_ids[i+1], pos+np.array([0,-dt*velocity, 0]), ori)

            if pos[1] < -1.6:
                flag_03 = 1
            elif pos[1] > -0.8:
                flag_03 = 0

    return flag_01, flag_02, flag_03

if __name__ == "__main__":
    # Configuring parameters
    duration = 300; fps = 30
    time_steps = int(duration * fps); dt = 1.0 / fps
    
    point_cloud_count = 300
    env_num = 4

    point_cloud_debug_id = None
    point_cloud_debug_id_list = []

    # Set the environment
    physics_client, collision_ids, goal_id, robot_id = environment_setup(env_num=env_num)

    # Create and setup environment
    print(f"Get Position from ID: {get_position_from_id(goal_id)[0]}")
    planner = RRTManipulatorPlanner(robot_id= robot_id, collision_ids=collision_ids)
    planner.run(get_position_from_id(goal_id)[0])

    # Initializing Simulation
    p.setRealTimeSimulation(0)

    for step in range(time_steps):
        # Setting Flags for the first time
        if step == 0:
            flag_01 = 1
            flag_02 = 1
            flag_03 = 1

        # Update the environment
        if env_num == 4:
            flag_01, flag_02, flag_03 = environment_update(collision_ids, dt=dt, flag_01=flag_01, flag_02=flag_02, flag_03=flag_03)

        # Obtain Point Cloud
        # p.removeAllUserDebugItems()
        # point_cloud, wRc, wtc = get_point_cloud(camera_position=[0.0, -1.5, 2.0], camera_target=[0.0, -0.2, 0.5], img_flag=True)
        point_cloud_01, wRc_01, wtc_01 = get_point_cloud(camera_position=[-0.7, -1.2, 1.2], camera_target=[0.0, -0.2, 0.5], img_flag=False)
        point_cloud_02, wRc_02, wtc_02 = get_point_cloud(camera_position=[0.7, -1.2, 1.2], camera_target=[0.0, -0.2, 0.5], img_flag=False)
        point_cloud_03, wRc_03, wtc_03 = get_point_cloud(camera_position=[-0.7, 1.2, 1.2], camera_target=[0.0, -0.2, 0.5], img_flag=True)
        point_cloud = np.concatenate((point_cloud_01, point_cloud_02, point_cloud_03), axis=0)

        # If a previous debug item exists, remove it
        if point_cloud_debug_id is not None and step%5 == 0:
            for point_cloud_debug_id in point_cloud_debug_id_list:
                p.removeUserDebugItem(point_cloud_debug_id)
            point_cloud_debug_id_list = []

        # Display Downsampled Point Cloud
        if len(point_cloud) > 0:
            if len(point_cloud) > point_cloud_count:
                downsampled_cloud = point_cloud[np.random.choice(len(point_cloud), point_cloud_count, replace=False)]
                point_cloud_debug_id = p.addUserDebugPoints(downsampled_cloud, [[1, 0, 0]] * len(downsampled_cloud), pointSize=3)
                point_cloud_debug_id_list.append(point_cloud_debug_id)

        if step == 0:
            draw_camera_frame(wtc_01, wRc_01)
            draw_camera_frame(wtc_02, wRc_02)
            draw_camera_frame(wtc_03, wRc_03)

        p.stepSimulation()
        time.sleep(1.0 / 240.0)


    p.disconnect()


    # Outside the main loop, initialize the debug item ID
point_cloud_debug_id = None

for step in range(time_steps):
    # At the beginning of the loop, remove all debug items
    p.removeAllUserDebugItems()

    # ... (rest of your code)

    # Display Downsampled Point Cloud
    if len(point_cloud) > 0:
        if len(point_cloud) > point_cloud_count:
            downsampled_cloud = point_cloud[np.random.choice(len(point_cloud), point_cloud_count, replace=False)]
        
        # If a previous debug item exists, remove it
        if point_cloud_debug_id is not None:
            p.removeUserDebugItem(point_cloud_debug_id)
        
        # Add the new point cloud and store the debug item ID
        point_cloud_debug_id = p.addUserDebugPoints(
            downsampled_cloud, 
            [[1, 0, 0]] * len(downsampled_cloud), 
            pointSize=3
        )

    # ... (rest of your code)


    # # Create and setup environment
    # env = PandaEnvironment()
    # env.print_robot_info()

    # # Create RRT planner with Panda's joint limits
    # joint_limits = np.array([
    #     [-2.8973, 2.8973],    # Joint 1
    #     [-1.7628, 1.7628],    # Joint 2
    #     [-2.8973, 2.8973],    # Joint 3
    #     [-3.0718, -0.0698],   # Joint 4
    #     [-2.8973, 2.8973],    # Joint 5
    #     [-0.0175, 3.7525],    # Joint 6
    #     [-2.8973, 2.8973]     # Joint 7
    # ])

    # planner = RRTManipulatorPlanner(joint_limits)


    # def check_collision(config):
    #     """Collision checker for RRT planner"""
    #     # Set robot to configuration
    #     for i in range(7):
    #         p.resetJointState(env.panda_id, i, config[i])
    #     p.stepSimulation()
    #     return env.check_collisions()

    # # Set collision checker
    # planner.collision_checker = check_collision

    # # Custom forward kinematics using PyBullet
    # def pybullet_fk(configuration):
    #     """Forward kinematics using PyBullet"""
    #     # Set robot to configuration
    #     for i in range(7):
    #         p.resetJointState(env.panda_id, i, configuration[i])
    #     p.stepSimulation()
    #     return env.get_end_effector_state()[0]  # Return position only

    # # Replace planner's FK with PyBullet FK
    # planner.forward_kinematics = pybullet_fk

    # def execute_path(path):
    #     """Execute a path from RRT planner"""
    #     for config in path:
    #         success = env.execute_trajectory(config, duration=1.0)
    #         if not success:
    #             print("Path execution failed!")
    #             return False
    #     return True

    # def plan_and_execute(start_config, target_pos):
    #     """Plan and execute path to target"""
    #     print(f"Planning path to target position: {target_pos}")

    #     # Plan path using RRT
    #     path = planner.plan(start_config, np.array(target_pos), tolerance=0.05)

    #     if path is not None:
    #         print("Path found! Executing...")
    #         if execute_path(path):
    #             print("Path executed successfully!")
    #             return True, path[-1]  # Return last configuration
    #         else:
    #             print("Path execution failed!")
    #             return False, None
    #     else:
    #         print("No path found!")
    #         return False, None

    # # Main simulation loop
    # try:
    #     print("\nPress 'p' to start planning and execution to all target points")
    #     print("Press 'q' to quit")

    #     while True:
    #         env.step_simulation()

    #         # Handle keyboard input
    #         keys = p.getKeyboardEvents()
    #         for key, state in keys.items():
    #             if state & p.KEY_WAS_TRIGGERED:
    #                 if key == ord('p'):
    #                     # Get target points
    #                     targets = env.get_target_points()

    #                     # Start from home position
    #                     current_config = env.home_positions
    #                     env.execute_trajectory(current_config)

    #                     # Plan and execute to each target
    #                     for i, target in enumerate(targets):
    #                         print(f"\nMoving to target {i+1}/{len(targets)}")
    #                         success, new_config = plan_and_execute(current_config, target)

    #                         if success:
    #                             current_config = new_config
    #                             time.sleep(0.5)  # Pause between targets
    #                         else:
    #                             print(f"Failed to reach target {i+1}")
    #                             break

    #                     # Return to home position
    #                     print("\nReturning to home position...")
    #                     env.execute_trajectory(env.home_positions)

    #                 elif key == ord('q'):
    #                     raise KeyboardInterrupt

    #         time.sleep(1./240.)

    # except KeyboardInterrupt:
    #     print("\nSimulation stopped by user")
    #     env.close()



    # #######################
    # #### PROBLEM SETUP ####
    # #######################

    # # Initialize PyBullet
    # p.connect(p.GUI)
    # p.setAdditionalSearchPath(pybullet_data.getDataPath()) # For default URDFs
    # p.setGravity(0, 0, -9.8)

    # # Load the plane and robot arm
    # ground_id = p.loadURDF("plane.urdf")
    # arm_id = p.loadURDF("three_link_arm.urdf", [0, 0, 0], useFixedBase=True)

    # # Add Collision Objects
    # collision_ids = [ground_id] # add the ground to the collision list
    # collision_positions = [[0.3, 0.5, 0.251], [-0.3, 0.3, 0.101], [-1, -0.15, 0.251], [-1, -0.15, 0.752], [-0.5, -1, 0.251], [0.5, -0.35, 0.201], [0.5, -0.35, 0.602]]
    # collision_orientations =  [[0, 0, 0.5], [0, 0, 0.2], [0, 0, 0],[0, 0, 1], [0, 0, 0], [0, 0, .25], [0, 0, 0.5]]
    # collision_scales = [0.5, 0.25, 0.5, 0.5, 0.5, 0.4, 0.4]
    # for i in range(len(collision_scales)):
    #     collision_ids.append(p.loadURDF("cube.urdf",
    #         basePosition=collision_positions[i],  # Position of the cube
    #         baseOrientation=p.getQuaternionFromEuler(collision_orientations[i]),  # Orientation of the cube
    #         globalScaling=collision_scales[i]  # Scale the cube to half size
    #     ))

    # # Goal Joint Positions for the Robot
    # goal_positions = [[-2.54, 0.15, -0.15], [-1.82,0.15,-0.15],[0.5, 0.15,-0.15], [1.7,0.2,-0.15],[-2.54, 0.15, -0.15]]

    # # Joint Limits of the Robot
    # joint_limits = [[-np.pi, np.pi], [0, np.pi], [-np.pi, np.pi]]

    # # A3xN path array that will be filled with waypoints through all the goal positions
    # path_saved = np.array([[-2.54, 0.15, -0.15]]) # Start at the first goal position

    # ####################################################################################################
    # #### YOUR CODE HERE: RUN RRT MOTION PLANNER FOR ALL goal_positions (starting at goal position 1) ###
    # ####################################################################################################
   


    # ################################################################################
    # ####  RUN THE SIMULATION AND MOVE THE ROBOT ALONG PATH_SAVED ###################
    # ################################################################################

    # # Set the initial joint positions
    # for joint_index, joint_pos in enumerate(goal_positions[0]):
    #     p.resetJointState(arm_id, joint_index, joint_pos)

    # # Move through the waypoints
    # for waypoint in path_saved:
    #     # "move" to next waypoints
    #     for joint_index, joint_pos in enumerate(waypoint):
    #     # run velocity control until waypoint is reached
    #         while True:
    #             #get current joint positions
    #             goal_positions = [p.getJointState(arm_id, i)[0] for i in range(3)]
    #             # calculate the displacement to reach the next waypoint
    #             displacement_to_waypoint = waypoint-goal_positions
    #             # check if goal is reached
    #             max_speed = 0.05
    #             if(np.linalg.norm(displacement_to_waypoint) < max_speed):
    #                 break
    #             else:
    #                 # calculate the "velocity" to reach the next waypoint
    #                 velocities = np.min((np.linalg.norm(displacement_to_waypoint), max_speed))*displacement_to_waypoint/np.linalg.norm(displacement_to_waypoint)
    #                 for joint_index, joint_step in enumerate(velocities):
    #                     p.setJointMotorControl2(
    #                         bodyIndex=arm_id,
    #                         jointIndex=joint_index,
    #                         controlMode=p.VELOCITY_CONTROL,
    #                         targetVelocity=joint_step,
    #                     )
                        
    #             #Take a simulation step
    #             p.stepSimulation()            
    #     time.sleep(1.0 / 240.0)

    p.disconnect(physics_client)
