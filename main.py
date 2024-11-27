from PandaEnv import PandaEnvironment
import numpy as np
import pybullet as p
import pybullet_data
import time
from Planner import RRTManipulatorPlanner
from typing import Tuple, List, Optional, Dict
import math


def main():
    # Create and setup environment
    env = PandaEnvironment()
    env.print_robot_info()

    # Create RRT planner with Panda's joint limits
    joint_limits = np.array([
        [-2.8973, 2.8973],    # Joint 1
        [-1.7628, 1.7628],    # Joint 2
        [-2.8973, 2.8973],    # Joint 3
        [-3.0718, -0.0698],   # Joint 4
        [-2.8973, 2.8973],    # Joint 5
        [-0.0175, 3.7525],    # Joint 6
        [-2.8973, 2.8973]     # Joint 7
    ])

    planner = RRTManipulatorPlanner(joint_limits)

    def check_collision(config):
        """Collision checker for RRT planner"""
        # Set robot to configuration
        for i in range(7):
            p.resetJointState(env.panda_id, i, config[i])
        p.stepSimulation()
        return env.check_collisions()

    # Set collision checker
    planner.collision_checker = check_collision

    # Custom forward kinematics using PyBullet
    def pybullet_fk(configuration):
        """Forward kinematics using PyBullet"""
        # Set robot to configuration
        for i in range(7):
            p.resetJointState(env.panda_id, i, configuration[i])
        p.stepSimulation()
        return env.get_end_effector_state()[0]  # Return position only

    # Replace planner's FK with PyBullet FK
    planner.forward_kinematics = pybullet_fk

    def execute_path(path):
        """Execute a path from RRT planner"""
        for config in path:
            success = env.execute_trajectory(config, duration=1.0)
            if not success:
                print("Path execution failed!")
                return False
        return True

    def plan_and_execute(start_config, target_pos):
        """Plan and execute path to target"""
        print(f"Planning path to target position: {target_pos}")

        # Plan path using RRT
        path = planner.plan(start_config, np.array(target_pos), tolerance=0.05)
        planner.visualize_path(path)

        if path is not None:
            print("Path found! Executing...")
            print(path)
            if execute_path(path):
                print("Path executed successfully!")
                return True, path[-1]  # Return last configuration
            else:
                print("Path execution failed!")
                return False, None
        else:
            print("No path found!")
            return False, None

    # Main simulation loop
    try:
        print("\nPress 'p' to start planning and execution to all target points")
        print("Press 'q' to quit")

        while True:
            env.step_simulation()

            # Handle keyboard input
            keys = p.getKeyboardEvents()
            for key, state in keys.items():
                if state & p.KEY_WAS_TRIGGERED:
                    if key == ord('p'):
                        # Get target points
                        targets = env.get_target_points()
                        print(targets)

                        # Start from home position
                        current_config = env.home_positions
                        env.execute_trajectory(current_config)

                        # Plan and execute to each target
                        for i, target in enumerate(targets):
                            print(f"\nMoving to target {i+1}/{len(targets)}")
                            success, new_config = plan_and_execute(current_config, target)
                            print(success)
                            if success:
                                current_config = new_config
                                time.sleep(0.5)  # Pause between targets
                            else:
                                print(f"Failed to reach target {i+1}")
                                break

                        # Return to home position
                        print("\nReturning to home position...")
                        env.execute_trajectory(env.home_positions)

                    elif key == ord('q'):
                        raise KeyboardInterrupt

            time.sleep(1./240.)

    except KeyboardInterrupt:
        print("\nSimulation stopped by user")
        env.close()



if __name__ == "__main__":
    main()
