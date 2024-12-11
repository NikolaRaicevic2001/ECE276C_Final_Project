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
    def __init__(self, robot_id=None, collision_ids=None):
        # Load environment and robot
        self.robot_id = robot_id
        self.collision_ids = collision_ids

        # Gripper offset to account for gripper length
        self.gripper_offset = 0.1  # 10 cm offset from end-effector

        # Endpoint link ID
        self.ee_id = 8

    def forward_kinematics(self, joint_angles):
        """Compute end-effector position for given joint angles"""
        for i, angle in enumerate(joint_angles):
            p.resetJointState(self.robot_id, i, angle)

        end_effector_state = p.getLinkState(self.robot_id, self.ee_id)
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
        link_state = p.getLinkState(self.robot_id, self.ee_id)
        current_orient = link_state[1]
        current_rot_matrix = np.array(p.getMatrixFromQuaternion(current_orient)).reshape(3, 3)

        # Adjust target position by subtracting the offset along the current z-axis
        adjusted_target = np.array(target_position) - current_rot_matrix[:, 2] * self.gripper_offset

        # Use PyBullet's inverse kinematics solver
        joint_angles = p.calculateInverseKinematics(
            self.robot_id,
            self.ee_id,  # End-effector link index
            adjusted_target,
            targetOrientation=target_orientation,
            lowerLimits=[-2*np.pi]*7,  # Adjust based on your robot's joint limits
            upperLimits=[2*np.pi]*7,
            jointRanges=[4*np.pi]*7,
            restPoses=[0]*7
        )

        return list(joint_angles[:7])

    def is_state_valid(self, joint_angles):
        """Check if the given joint configuration is collision-free"""
        for i, angle in enumerate(joint_angles):
            p.resetJointState(self.robot_id, i, angle)

        collisions = []
        for collision_id in self.collision_ids:
            collisions.extend(p.getContactPoints(self.robot_id, collision_id))

        return len(collisions) == 0

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

    def plan_rrt_star(self, start_angles, goal_angles, max_iterations=1000, step_size=1, goal_bias=0.1):
        """RRT* path planning algorithm with comprehensive collision checking"""
        nodes = [start_angles]
        edges = {}
        costs = {tuple(start_angles): 0}

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
            if not self.is_state_valid(new_node):
                continue

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
                time.sleep(0.1)

            # Update current angles to the last configuration of the path
            current_angles = path[-1]

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




class RealTimeRRTObstacleAvoidance(RealTimeRRT):
    def __init__(self, robot_id=None, collision_ids=None, safety_distance=0.2):
        super().__init__(robot_id = robot_id, collision_ids = collision_ids)
        # Load environment and robot
        # self.robot_id = robot_id
        # self.collision_ids = collision_ids

        # Safety parameters
        self.safety_distance = safety_distance  # 20 cm safety zone
        # self.gripper_offset = 0.1  # 10 cm offset from end-effector
        # self.ee_id = 8  # Endpoint link ID

    def get_obstacle_positions(self):
        """
        Retrieve current positions of all collision objects

        Returns:
        list: List of obstacle positions
        """
        obstacle_positions = []
        for collision_id in self.collision_ids:
            # Get the base position of the obstacle
            obstacle_pos, _ = p.getBasePositionAndOrientation(collision_id)
            obstacle_positions.append(obstacle_pos)
        return obstacle_positions

    def check_obstacle_proximity(self, gripper_pos):
        """
        Check if any obstacles are too close to the gripper

        Args:
        gripper_pos (list/np.array): Current gripper position

        Returns:
        bool: True if an obstacle is too close, False otherwise
        list: List of nearby obstacles
        """
        nearby_obstacles = []
        obstacle_positions = self.get_obstacle_positions()

        for obstacle_pos in obstacle_positions:
            distance = np.linalg.norm(np.array(gripper_pos) - np.array(obstacle_pos))

            if distance <= self.safety_distance:
                nearby_obstacles.append((obstacle_pos, distance))

        return len(nearby_obstacles) > 0, nearby_obstacles

    def generate_avoidance_goal(self, current_pos, nearby_obstacles):
        """
        Generate a new goal position to avoid nearby obstacles

        Args:
        current_pos (list/np.array): Current gripper position
        nearby_obstacles (list): List of nearby obstacles

        Returns:
        list: New goal position to avoid obstacles
        """
        # Convert to numpy array for easier manipulation
        current_pos = np.array(current_pos)

        # Calculate avoidance vector
        avoidance_vector = np.zeros(3)
        for obstacle_pos, distance in nearby_obstacles:
            # Compute repulsive force inversely proportional to distance
            direction = current_pos - np.array(obstacle_pos)
            direction_norm = np.linalg.norm(direction)

            # Normalize and scale the avoidance vector
            repulsive_force = direction / (direction_norm ** 2)
            avoidance_vector += repulsive_force

        # Normalize avoidance vector and scale
        avoidance_scale = min(self.safety_distance, np.linalg.norm(avoidance_vector))
        if np.linalg.norm(avoidance_vector) > 0:
            avoidance_vector = (avoidance_vector / np.linalg.norm(avoidance_vector)) * avoidance_scale

        # Generate new goal slightly away from obstacles
        new_goal = current_pos + avoidance_vector

        return new_goal.tolist()

    def dynamic_trajectory_planning(self, original_path, goal_positions):
        """
        Dynamically replan trajectory considering real-time obstacle positions

        Args:
        original_path (list): Initially planned path
        goal_positions (list): Original goal positions

        Returns:
        list: Updated path with obstacle avoidance
        """
        updated_path = []
        current_angles = original_path[0]

        for segment_index, segment in enumerate(original_path[1:], 1):
            # Forward kinematics to get current gripper position
            for i, angle in enumerate(segment):
                p.resetJointState(self.robot_id, i, angle)

            gripper_pos = self.forward_kinematics(segment)

            # Check obstacle proximity
            is_obstacle_near, nearby_obstacles = self.check_obstacle_proximity(gripper_pos)

            if is_obstacle_near:
                print(f"Obstacle detected near segment {segment_index}")

                # Generate avoidance goal
                avoidance_goal = self.generate_avoidance_goal(gripper_pos, nearby_obstacles)
                print(f"Avoidance goal: {avoidance_goal}")

                # Replan path to avoidance goal
                avoidance_angles = self.inverse_kinematics(avoidance_goal)
                avoidance_path = self.plan_rrt_star(current_angles, avoidance_angles)

                # Extend updated path
                updated_path.extend(avoidance_path[:-1])

                # Update current angles
                current_angles = avoidance_path[-1]
            else:
                # If no obstacles, continue with original path
                updated_path.append(segment)
                current_angles = segment

        return updated_path

    def plan_and_execute_trajectory(self, goal_positions):
        """Enhanced trajectory planning with dynamic obstacle avoidance"""
        current_angles = [0, 0, 0, 0, 0, 0, 0]
        goal_orientation = p.getQuaternionFromEuler([np.pi, 0, 0])

        # Original RRT* path planning
        full_path = []

        for goal_position in goal_positions:
            goal_angles = self.inverse_kinematics(goal_position, goal_orientation)
            path = self.plan_rrt_star(current_angles, goal_angles)

            full_path.extend(path[:-1])
            current_angles = path[-1]

        # Dynamic obstacle avoidance replan
        updated_path = self.dynamic_trajectory_planning(full_path, goal_positions)

        # Execute updated path
        self.execute_path(updated_path)




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

    def plan_rrt_star(self, start_angles, goal_angles, max_iterations=1000, step_size=1, goal_bias=0.1):
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
            # print("lmao")
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
        # print("lmao", goal_positions)

        # Downward orientation for gripper (pointing straight down)
        goal_orientation = p.getQuaternionFromEuler([np.pi, 0, 0])

        # First, plan paths and verify tip positions for all goal positions
        for goal_position in goal_positions:
            # print(goal_position)
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
                time.sleep(0.1)

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

