import pybullet as p
import pybullet_data
import time
import numpy as np
from Planner import RRTManipulatorPlanner
from typing import Tuple, List, Optional, Dict
import math

class PandaEnvironment:
    def __init__(self):
        # PyBullet setup
        self.physics_client = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)

        # Environment objects
        self.plane_id = None
        self.panda_id = None
        self.cube_id = None
        self.target_points = []
        self.target_visual_ids = []

        # Robot parameters
        self.num_joints = 0
        self.end_effector_index = 6
        self.home_positions = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]
        self.joint_ranges = []
        self.rest_poses = []

        # Planning parameters
        self.max_force = 250
        self.position_control_gain = 0.03
        self.velocity_control_gain = 1

        # Setup
        self._setup_environment()

    def _setup_environment(self):
        """Initialize all objects in the environment"""
        self._load_plane()
        self._load_panda()
        self._load_target_cube()
        self._create_target_points()
        self._setup_collision_detection()
        self._setup_joint_parameters()

    def _setup_joint_parameters(self):
        """Setup joint parameters for motion planning"""
        self.joint_ranges = []
        self.rest_poses = []

        for i in range(7):  # Only for the 7 main joints
            joint_info = p.getJointInfo(self.panda_id, i)
            self.joint_ranges.append((joint_info[8], joint_info[9]))  # lower and upper limits
            self.rest_poses.append(self.home_positions[i])

    def _load_plane(self):
        """Load ground plane"""
        self.plane_id = p.loadURDF("plane.urdf")

    def _load_panda(self):
        """Load Panda robot and set initial position"""
        start_pos = [0, 0, 0]
        start_orientation = p.getQuaternionFromEuler([0, 0, 0])
        self.panda_id = p.loadURDF("franka_panda/panda.urdf", start_pos, start_orientation, useFixedBase=True)
        self.num_joints = p.getNumJoints(self.panda_id)

        # Set home position
        for i in range(7):
            p.resetJointState(self.panda_id, i, self.home_positions[i])

    def _load_target_cube(self):
        """Load target cube with specific size and position"""
        cube_size = 0.1
        cube_mass = 1.0
        cube_position = [0.5, 0, cube_size/2]
        cube_orientation = p.getQuaternionFromEuler([0, 0, 0])

        collision_shape_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=[cube_size/2]*3)
        visual_shape_id = p.createVisualShape(p.GEOM_BOX, halfExtents=[cube_size/2]*3, rgbaColor=[0.8, 0.2, 0.2, 1])

        self.cube_id = p.createMultiBody(
            baseMass=cube_mass,
            baseCollisionShapeIndex=collision_shape_id,
            baseVisualShapeIndex=visual_shape_id,
            basePosition=cube_position,
            baseOrientation=cube_orientation
        )

        self.cube_size = cube_size
        self.cube_position = cube_position

    def _create_target_points(self):
        """Create target points on vertical faces of the cube"""
        cube_pos = np.array(self.cube_position)
        half_size = self.cube_size / 2

        face_centers = [
            [cube_pos[0] + half_size, cube_pos[1], cube_pos[2]],
            [cube_pos[0], cube_pos[1] + half_size, cube_pos[2]],
            [cube_pos[0] - half_size, cube_pos[1], cube_pos[2]],
            [cube_pos[0], cube_pos[1] - half_size, cube_pos[2]]
        ]

        for point in face_centers:
            self.target_points.append(point)

            visual_id = p.createVisualShape(
                p.GEOM_SPHERE,
                radius=0.005,
                rgbaColor=[0, 1, 0, 1]
            )
            marker_id = p.createMultiBody(
                baseMass=0,
                baseVisualShapeIndex=visual_id,
                basePosition=point
            )
            self.target_visual_ids.append(marker_id)

    def _setup_collision_detection(self):
        """Setup collision detection parameters"""
        for joint in range(self.num_joints):
            p.setCollisionFilterPair(self.panda_id, self.cube_id, joint, -1, 1)

    def calculate_ik(self, target_pos: List[float], target_ori: List[float]) -> Optional[List[float]]:
        """
        Calculate inverse kinematics for a target position and orientation
        Returns joint angles if solution is found, None otherwise
        """
        target_joint_positions = p.calculateInverseKinematics(
            self.panda_id,
            self.end_effector_index,
            target_pos,
            target_ori,
            lowerLimits=[joint_range[0] for joint_range in self.joint_ranges],
            upperLimits=[joint_range[1] for joint_range in self.joint_ranges],
            jointRanges=[joint_range[1] - joint_range[0] for joint_range in self.joint_ranges],
            restPoses=self.rest_poses,
            maxNumIterations=100,
            residualThreshold=1e-5
        )
        return list(target_joint_positions)[:7]  # Return only the 7 main joint angles

    def execute_trajectory(self, joint_angles: List[float], duration: float = 2.0) -> bool:
        """
        Execute a trajectory to reach target joint angles
        Returns True if execution successful, False otherwise
        """
        steps = int(duration * 240)  # 240 Hz control
        current_joint_angles = [p.getJointState(self.panda_id, i)[0] for i in range(7)]

        for step in range(steps):
            # Interpolate joint angles
            t = float(step) / steps
            target_angles = [
                current_joint_angles[i] + t * (joint_angles[i] - current_joint_angles[i])
                for i in range(7)
            ]

            # Apply position control
            for i in range(7):
                p.setJointMotorControl2(
                    bodyIndex=self.panda_id,
                    jointIndex=i,
                    controlMode=p.POSITION_CONTROL,
                    targetPosition=target_angles[i],
                    force=self.max_force
                )

            # Check for collisions
            if self.check_collisions():
                print("Collision detected during trajectory execution!")
                return False

            p.stepSimulation()
            time.sleep(1./240.)

        return True

    def plan_to_target(self, target_point: List[float], approach_distance: float = 0.1) -> bool:
        """
        Plan and execute trajectory to a target point
        Returns True if successful, False otherwise
        """
        # Calculate approach orientation (facing the target)
        current_pos, _ = self.get_end_effector_state()
        direction = np.array(target_point) - np.array(current_pos)
        direction = direction / np.linalg.norm(direction)

        # Convert direction to orientation (assuming we want the end effector to point towards the target)
        rotation_matrix = np.zeros((3, 3))
        rotation_matrix[:, 2] = direction
        rotation_matrix[:, 1] = np.cross(direction, [0, 0, 1])
        rotation_matrix[:, 0] = np.cross(rotation_matrix[:, 1], direction)

        # Convert rotation matrix to quaternion
        target_ori = p.getQuaternionFromEuler([0, math.pi/2, 0])  # Simplified orientation

        # Calculate IK solution
        joint_solution = self.calculate_ik(target_point, target_ori)

        if joint_solution is None:
            print("No IK solution found!")
            return False

        # Execute trajectory
        return self.execute_trajectory(joint_solution)

    def move_to_targets(self):
        """Move to each target point in sequence"""
        # First move to home position
        self.execute_trajectory(self.home_positions)

        for i, target in enumerate(self.target_points):
            print(f"Moving to target point {i+1}")
            if self.plan_to_target(target):
                print(f"Successfully reached target {i+1}")
            else:
                print(f"Failed to reach target {i+1}")
            time.sleep(0.5)  # Pause between targets

    def check_collisions(self) -> bool:
        """Check for collisions between robot and cube"""
        contact_points = p.getContactPoints(self.panda_id, self.cube_id)
        return len(contact_points) > 0

    def get_end_effector_state(self) -> Tuple[List[float], List[float]]:
        """Get current end effector position and orientation"""
        state = p.getLinkState(self.panda_id, self.end_effector_index)
        return state[0], state[1]

    def get_target_points(self) -> List[List[float]]:
        """Return list of target points"""
        return self.target_points

    def step_simulation(self):
        """Step the simulation"""
        p.stepSimulation()

    def close(self):
        """Clean up and close the environment"""
        p.disconnect(self.physics_client)

    def print_robot_info(self):
        """Print information about the robot"""
        print(f"Loaded Panda robot with {self.num_joints} joints")
        for i in range(self.num_joints):
            joint_info = p.getJointInfo(self.panda_id, i)
            print(f"Joint {i}: {joint_info[1].decode('utf-8')}")


<<<<<<< HEAD


=======
>>>>>>> 1681d17f76814bad7ae0bb968fca9cf7202ccef2
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

        if path is not None:
            print("Path found! Executing...")
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

                        # Start from home position
                        current_config = env.home_positions
                        env.execute_trajectory(current_config)

                        # Plan and execute to each target
                        for i, target in enumerate(targets):
                            print(f"\nMoving to target {i+1}/{len(targets)}")
                            success, new_config = plan_and_execute(current_config, target)

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


    # # Disconnect from PyBullet
    # time.sleep(100) # Remove this line -- it is just to keep the GUI open when you first run this starter code
    # p.disconnect()

if __name__ == "__main__":
    main()
