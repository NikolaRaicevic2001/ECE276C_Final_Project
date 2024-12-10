import pybullet as p
import pybullet_data
import time
import numpy as np
import random

import pybullet as p
import numpy as np
import time
import random

#########################
##### RealTimeRRT #######
#########################
class RealTimeRRT:
    def __init__(self, robot_id=None, collision_ids=None, goal_position=None, max_planning_time=0.1):
        """
        Initialize Real-Time RRT* Planner

        :param robot_id: ID of the robot in PyBullet simulation
        :param collision_ids: List of object IDs to check for collisions
        :param goal_position: Initial goal position
        :param max_planning_time: Maximum time allowed for tree expansion in each iteration
        """
        # Core planner components
        self.robot_id = robot_id
        self.collision_ids = collision_ids

        # Planning parameters
        self.current_goal = goal_position
        self.max_planning_time = max_planning_time

        # Tree representation
        self.nodes = []
        self.edges = {}
        self.costs = {}

        # Motion planning parameters
        self.step_size = 0.1
        self.goal_bias = 0.1
        self.max_iterations = 1000

        # Agent state tracking
        self.current_position = None
        self.current_path = []

    def forward_kinematics(self, joint_angles):
        """Compute end-effector position for given joint angles"""
        # Similar to RRTManipulatorPlanner's method
        for i, angle in enumerate(joint_angles):
            p.resetJointState(self.robot_id, i, angle)

        end_effector_state = p.getLinkState(self.robot_id, 6)
        end_effector_pos = end_effector_state[0]
        return end_effector_pos

    def inverse_kinematics(self, target_position):
        """Compute joint angles for a target gripper position"""
        # Similar to RRTManipulatorPlanner's method
        target_orientation = p.getQuaternionFromEuler([np.pi, 0, 0])

        joint_angles = p.calculateInverseKinematics(
            self.robot_id,
            6,  # End-effector link index
            target_position,
            targetOrientation=target_orientation
        )

        return joint_angles[:7]

    def is_state_valid(self, joint_angles):
        """Check if the given joint configuration is collision-free"""
        # Reset joint states
        for i, angle in enumerate(joint_angles):
            p.resetJointState(self.robot_id, i, angle)

        # Check collisions
        for collision_id in self.collision_ids:
            if p.getContactPoints(self.robot_id, collision_id):
                return False
        return True

    def distance(self, config1, config2):
        """Compute distance between two joint configurations"""
        return np.linalg.norm(np.array(config1) - np.array(config2))

    def random_configuration(self):
        """Generate a random valid joint configuration"""
        joint_angles = []
        for i in range(7):
            joint_info = p.getJointInfo(self.robot_id, i)
            joint_angles.append(random.uniform(joint_info[8], joint_info[9]))
        return joint_angles

    def interpolate_path(self, start, end, num_steps=10):
        """Interpolate between two configurations"""
        interpolated_configs = []
        for i in range(num_steps + 1):
            t = i / num_steps
            interpolated_config = [
                start[j] + t * (end[j] - start[j])
                for j in range(len(start))
            ]
            interpolated_configs.append(interpolated_config)
        return interpolated_configs

    def expand_and_rewire(self, start_config, goal_config):
        """
        Expand and rewire the RRT* tree

        This is equivalent to Algorithm 2 in the RT-RRT* paper
        """
        start_time = time.time()

        while time.time() - start_time < self.max_planning_time:
            # Goal biasing
            if random.random() < self.goal_bias:
                rand_config = goal_config
            else:
                rand_config = self.random_configuration()

            # Find nearest node
            nearest_node = min(self.nodes, key=lambda n: self.distance(n, rand_config))

            # Steer towards random configuration
            direction = np.array(rand_config) - np.array(nearest_node)
            direction_norm = np.linalg.norm(direction)

            if direction_norm > self.step_size:
                direction = direction / direction_norm * self.step_size

            new_node = (np.array(nearest_node) + direction).tolist()

            # Collision and path checking
            if not self.is_state_valid(new_node):
                continue

            interpolated_path = self.interpolate_path(nearest_node, new_node)
            if not all(self.is_state_valid(config) for config in interpolated_path):
                continue

            # Add node to tree
            self.nodes.append(new_node)

            # Cost and parent selection
            best_parent = nearest_node
            best_cost = self.costs.get(tuple(nearest_node), 0) + self.distance(nearest_node, new_node)

            # Rewiring
            near_nodes = [n for n in self.nodes if self.distance(n, new_node) < self.step_size * 2]

            for near_node in near_nodes:
                interpolated_path = self.interpolate_path(near_node, new_node)

                if not all(self.is_state_valid(config) for config in interpolated_path):
                    continue

                tentative_cost = self.costs.get(tuple(near_node), 0) + self.distance(near_node, new_node)

                if tentative_cost < best_cost:
                    best_parent = near_node
                    best_cost = tentative_cost

            # Update tree
            self.edges[tuple(new_node)] = best_parent
            self.costs[tuple(new_node)] = best_cost

            # Check goal proximity
            if self.distance(new_node, goal_config) < self.step_size:
                break

    def extract_path(self, goal_config):
        """Extract path from start to goal"""
        path = [goal_config]
        current = goal_config

        while tuple(current) in self.edges:
            current = self.edges[tuple(current)]
            path.append(current)

        path.reverse()
        return path

    def real_time_rrt_star(self, start_config, goal_config):
        """
        Main Real-Time RRT* algorithm

        Implements the main loop from Algorithm 1
        """
        # Initialize tree with start configuration
        self.nodes = [start_config]
        self.edges = {}
        self.costs = {tuple(start_config): 0}


        next_waypoint = np.array([0, 0, 0])
        # Main planning loop
        while True:
            # Update goal and free space (in a real system, this would use sensor data)
            # Here, we'll simulate by potentially changing goal slightly
            self.current_goal = goal_config  # In a real system, this would be dynamically updated

            # Expand and rewire tree
            self.expand_and_rewire(start_config, self.current_goal)

            # Plan path to current goal
            self.current_path = self.extract_path(self.current_goal)

            # In a real system, check agent proximity to path start
            # For simulation, we'll use the first waypoint
            if len(self.current_path) > 1:
                next_waypoint = self.current_path[1]

                # Move towards next waypoint
                for i, angle in enumerate(next_waypoint):
                    p.setJointMotorControl2(self.robot_id, i, p.POSITION_CONTROL, angle)

                p.stepSimulation()
                time.sleep(0.1)  # Visualization delay

            # In a real system, this would continue until goal is reached
            # Here, we'll add a break condition for demonstration
            # print(self.inverse_kinematics(next_waypoint))
            if self.distance(next_waypoint, self.forward_kinematics(goal_config)) < 0.1:
                break

    def run(self, start_position, goal_position):
        """Execute the Real-Time RRT* planner"""
        # Convert positions to joint configurations using inverse kinematics
        start_config = self.inverse_kinematics(start_position)
        goal_config = self.inverse_kinematics(goal_position)

        # Run the real-time path planning
        self.real_time_rrt_star(start_config, goal_config)

        print("Path planning completed!")

###################################
##### RRTManipulatorPlanner #######
###################################
class RRTManipulatorPlanner:
    def __init__(self, robot_id=None, collision_ids=None): 
        # Load environment and robot
        self.robot_id = robot_id
        self.collision_ids = collision_ids

        # Gripper offset to account for gripper length
        # This value should be adjusted based on the actual gripper length
        self.gripper_offset = 0.1  # 10 cm offset from end-effector

    def forward_kinematics(self, joint_angles):
        """Compute end-effector position for given joint angles"""
        for i, angle in enumerate(joint_angles):
            p.resetJointState(self.robot_id, i, angle)

        # Get the end-effector link state
        end_effector_state = p.getLinkState(self.robot_id, 6)
        end_effector_pos = end_effector_state[0]
        end_effector_orient = end_effector_state[1]

        # Calculate gripper tip position considering orientation
        rot_matrix = p.getMatrixFromQuaternion(end_effector_orient)
        rot_matrix = np.array(rot_matrix).reshape(3, 3)

        # Apply offset along the z-axis of the end-effector
        gripper_tip = end_effector_pos + rot_matrix[:, 2] * self.gripper_offset

        return gripper_tip

    def inverse_kinematics(self, target_position, target_orientation=None):
        """Compute joint angles for a target gripper position"""
        # If no specific orientation is provided, use a default downward orientation
        if target_orientation is None:
            # Quaternion for pointing straight down
            target_orientation = p.getQuaternionFromEuler([np.pi, 0, 0])

        # Adjust target position to account for gripper offset
        # We need to move the target point back along the z-axis of the gripper
        link_state = p.getLinkState(self.robot_id, 6)
        current_orient = link_state[1]
        current_rot_matrix = np.array(p.getMatrixFromQuaternion(current_orient)).reshape(3, 3)

        # Adjust target position by subtracting the offset along the current z-axis
        adjusted_target = np.array(target_position) - current_rot_matrix[:, 2] * self.gripper_offset

        # Use PyBullet's inverse kinematics solver
        joint_angles = p.calculateInverseKinematics(
            self.robot_id,
            6,  # End-effector link index
            adjusted_target,
            targetOrientation=target_orientation
        )

        return joint_angles

    def is_state_valid(self, joint_angles):
        """Check if the given joint configuration is collision-free"""
        for i, angle in enumerate(joint_angles):
            p.resetJointState(self.robot_id, i, angle)

        collisions = []
        for collision_id in self.collision_ids:
            collisions = p.getContactPoints(self.robot_id, collision_id)

        return len(collisions) == 0

    def _setup_collision_detection(self):
        """Setup collision detection parameters"""
        for joint in range(self.num_joints):
            p.setCollisionFilterPair(self.panda_id, self.cube_id, joint, -1, 1)

    def check_collisions(self) -> bool:
        """Check for collisions between robot and cube"""
        contact_points = p.getContactPoints(self.panda_id, self.cube_id)
        return len(contact_points) > 0
    
    def random_configuration(self):
        """Generate a random valid joint configuration"""
        joint_angles = []
        for i in range(7):
            joint_info = p.getJointInfo(self.robot_id, i)
            joint_angles.append(random.uniform(joint_info[8], joint_info[9]))
        return joint_angles

    def distance(self, config1, config2):
        """Compute distance between two joint configurations"""
        return np.linalg.norm(np.array(config1) - np.array(config2))

    def plan_rrt_star(self, start_angles, goal_angles, max_iterations=1000, step_size=0.1, goal_bias=0.1):
        """RRT* path planning algorithm with comprehensive collision checking"""
        nodes = [start_angles]
        edges = {}
        costs = {tuple(start_angles): 0}

        for _ in range(max_iterations):
            # Goal bias for directed exploration
            if random.random() < goal_bias:
                rand_config = goal_angles[:7]
            else:
                rand_config = self.random_configuration()

            # Find nearest node
            nearest_node = min(nodes, key=lambda n: self.distance(n, rand_config))

            # Steer towards the random configuration
            direction = np.array(rand_config) - np.array(nearest_node)
            direction_norm = np.linalg.norm(direction)

            # Limit step size
            if direction_norm > step_size:
                direction = direction / direction_norm * step_size

            new_node = (np.array(nearest_node) + direction).tolist()

            # Comprehensive collision checking
            # 1. Check if the new node itself is collision-free
            if not self.is_state_valid(new_node):
                continue

            # 2. Check if the path (edge) between nearest node and new node is collision-free
            def interpolate_path(start, end, num_steps=10):
                """Interpolate between two configurations to check intermediate states"""
                interpolated_configs = []
                for i in range(num_steps + 1):
                    t = i / num_steps
                    interpolated_config = [
                        start[j] + t * (end[j] - start[j])
                        for j in range(len(start))
                    ]
                    interpolated_configs.append(interpolated_config)
                return interpolated_configs

            # Check intermediate configurations along the path
            interpolated_path = interpolate_path(nearest_node, new_node)
            if not all(self.is_state_valid(config) for config in interpolated_path):
                continue

            # If collision checks pass, add the node
            nodes.append(new_node)

            # Find near nodes for potential rewiring
            near_nodes = [n for n in nodes if self.distance(n, new_node) < step_size * 2]

            # Choose parent with minimum cost
            best_parent = nearest_node
            best_cost = costs.get(tuple(nearest_node), float('inf')) + self.distance(nearest_node, new_node)

            # Check alternative parents
            for near_node in near_nodes:
                # Interpolate and check path to this potential parent
                interpolated_path = interpolate_path(near_node, new_node)

                # Skip if path is not collision-free
                if not all(self.is_state_valid(config) for config in interpolated_path):
                    continue

                # Compute potential cost
                tentative_cost = costs.get(tuple(near_node), float('inf')) + self.distance(near_node, new_node)

                if tentative_cost < best_cost:
                    best_parent = near_node
                    best_cost = tentative_cost

            # Update edges and costs
            edges[tuple(new_node)] = best_parent
            costs[tuple(new_node)] = best_cost

            # Rewire nearby nodes
            for near_node in near_nodes:
                current_near_cost = costs.get(tuple(near_node), float('inf'))
                potential_new_cost = best_cost + self.distance(new_node, near_node)

                # Interpolate and check path for rewiring
                interpolated_path = interpolate_path(new_node, near_node)

                # Only rewire if path is collision-free and cost is lower
                if (potential_new_cost < current_near_cost and
                    all(self.is_state_valid(config) for config in interpolated_path)):
                    edges[tuple(near_node)] = new_node
                    costs[tuple(near_node)] = potential_new_cost

            # Check if goal is reached
            if self.distance(new_node, goal_angles[:7]) < step_size:
                # Check if path to goal is collision-free
                goal_path = interpolate_path(new_node, goal_angles[:7])
                if all(self.is_state_valid(config) for config in goal_path):
                    nodes.append(goal_angles[:7])
                    edges[tuple(goal_angles[:7])] = new_node
                    costs[tuple(goal_angles[:7])] = best_cost
                    break

        # Extract path
        path = []
        current = goal_angles[:7]
        while tuple(current) in edges:
            path.append(current)
            current = edges[tuple(current)]
        path.append(start_angles)
        path.reverse()

        return path

    def execute_path(self, path):
        """Execute a planned path in simulation"""
        for joint_angles in path:
            for i, angle in enumerate(joint_angles):
                p.setJointMotorControl2(self.robot_id, i, p.POSITION_CONTROL, angle)
            p.stepSimulation()
            time.sleep(0.1)  # Visualization delay   

    def plan_and_execute_trajectory(self, goal_positions):
        """Plan and execute a trajectory through multiple goal positions"""
        # Start from initial zero configuration
        current_angles = [0, 0, 0, 0, 0, 0, 0]
        total_path = []
        verified_goal_angles = []

        # Downward orientation for gripper (pointing straight down)
        goal_orientation = p.getQuaternionFromEuler([np.pi, 0, 0])

        # First, plan paths and verify tip positions for all goal positions
        for goal_position in goal_positions:
            # Calculate goal joint angles using IK
            goal_angles = self.inverse_kinematics(goal_position, goal_orientation)
            print(f"Moving to position: {goal_position}")
            print("Goal Joint Angles:", goal_angles[:7])

            # Verify gripper tip position
            for i, angle in enumerate(goal_angles[:7]):
                p.resetJointState(self.robot_id, i, angle)

            gripper_pos = self.forward_kinematics(goal_angles[:7])
            print("Actual Gripper Position:", gripper_pos)
            print("Target Position:", goal_position)

            distance_to_target = np.linalg.norm(np.array(gripper_pos) - np.array(goal_position))
            print("Distance to Target:", distance_to_target)

            # Plan path using RRT*
            path = self.plan_rrt_star(current_angles, goal_angles)
            print("Planned Path Length:", len(path))

            total_path.append(path)
            verified_goal_angles.append(goal_angles[:7])


            if path:
                self.execute_path(path)
                time.sleep(1)

            # Update current angles to the last configuration of the path
            current_angles = path[-1]

        # Now execute the entire planned trajectory
        # for path in total_path:
        #     if path:
        #         self.execute_path(path)
        #         time.sleep(1)




    def run(self, goal_positions):
        """Main execution method"""
        try:
            # Plan and execute trajectory
            self.plan_and_execute_trajectory(goal_positions)

            # Keep simulation running
            while True:
                p.stepSimulation()
                time.sleep(0.01)
        except KeyboardInterrupt:
            self.cleanup()

    def cleanup(self):
        """Disconnect from PyBullet simulation"""
        p.disconnect()

def main():
    # Define goal positions
    goal_positions = [
        [0.5, -0.0, 0.5],
        [-0.5, 0.0, 0.5],
        [0.5, 0.5, 0.5],
        [0.5, -0.5, 0.5]
    ]

    # Create planner and run
    planner = RRTManipulatorPlanner()
    planner.run(goal_positions)

if __name__ == "__main__":
    main()


# # A3xN path array that will be filled with waypoints through all the goal positions
# path_saved = np.array([[-2.54, 0.15, -0.15]]) # Start at the first goal position

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

# Cost changing
# Scheduler
