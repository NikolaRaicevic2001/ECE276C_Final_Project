#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
import pybullet as p
import random
import time

class RealTimeRRT_ObstacleAvoidance:
    def __init__(self, q_start, q_goal, robot_id, obstacle_ids, max_iter=500, step_size=0.5, safety_distance=0.2):
        """ RRT* Initialization with Obstacle Avoidance """
        self.q_start = self.Node(q_start)
        self.q_goal = self.Node(q_goal)
        self.obstacle_ids = obstacle_ids
        self.robot_id = robot_id
        self.q_limits = []
        self.max_iter = max_iter
        self.step_size = step_size
        self.safety_distance = safety_distance
        self.node_list = [self.q_start]
        self.path = []

        # Fetch joint limits
        for i in range(7):
            joint_info = p.getJointInfo(robot_id, i)
            self.q_limits.append(joint_info[8:10])

    class Node:
        def __init__(self, joint_angles):
            self.joint_angles = np.array(joint_angles)
            self.parent = None
            self.cost = 0

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
        for obstacle_id in self.obstacle_ids:
            # Get the base position of the obstacle
            obstacle_pos, _ = p.getBasePositionAndOrientation(obstacle_id)
            distance = np.linalg.norm(np.array(gripper_pos) - np.array(obstacle_pos))

            if distance <= self.safety_distance:
                nearby_obstacles.append((obstacle_pos, distance))

        return len(nearby_obstacles) > 0, nearby_obstacles

    def forward_kinematics(self, joint_angles):
        """Compute end-effector position for given joint angles"""
        gripper_offset = 0.1  # 10 cm offset from end-effector
        ee_id = 8  # Endpoint link ID

        for i, angle in enumerate(joint_angles):
            p.resetJointState(self.robot_id, i, angle)

        end_effector_state = p.getLinkState(self.robot_id, ee_id)
        end_effector_pos = end_effector_state[0]
        end_effector_orient = end_effector_state[1]

        # Calculate gripper tip position considering orientation
        rot_matrix = p.getMatrixFromQuaternion(end_effector_orient)
        rot_matrix = np.array(rot_matrix).reshape(3, 3)

        # Apply offset along the z-axis of the end-effector
        gripper_tip = end_effector_pos + rot_matrix[:, 2] * gripper_offset

        return gripper_tip

    def generate_avoidance_goal(self, current_pos, nearby_obstacles):
        """
        Generate a new goal position to avoid nearby obstacles

        Args:
        current_pos (list/np.array): Current gripper position
        nearby_obstacles (list): List of nearby obstacles

        Returns:
        list: New goal position to avoid obstacles
        """
        current_pos = np.array(current_pos)
        avoidance_vector = np.zeros(3)

        for obstacle_pos, distance in nearby_obstacles:
            direction = current_pos - np.array(obstacle_pos)
            direction_norm = np.linalg.norm(direction)

            # Compute repulsive force inversely proportional to distance
            repulsive_force = direction / (direction_norm ** 2)
            avoidance_vector += repulsive_force

        # Normalize and scale avoidance vector
        avoidance_scale = min(self.safety_distance, np.linalg.norm(avoidance_vector))
        if np.linalg.norm(avoidance_vector) > 0:
            avoidance_vector = (avoidance_vector / np.linalg.norm(avoidance_vector)) * avoidance_scale

        # Generate new goal slightly away from obstacles
        new_goal = current_pos + avoidance_vector
        return new_goal.tolist()

    def check_node_collision(self, joint_position):
        """
        Checks for collisions between a robot and obstacles in PyBullet.

        Args:
            joint_position (list): List of joint positions.

        Returns:
            bool: True if a collision is detected, False otherwise.
        """
        # Set joint positions
        for joint_index, joint_pos in enumerate(joint_position):
            p.resetJointState(self.robot_id, joint_index, joint_pos)

        # Perform collision check for all links
        for object_id in self.obstacle_ids:
            for link_index in range(0, p.getNumJoints(self.robot_id)):
                contact_points = p.getClosestPoints(bodyA=self.robot_id, bodyB=object_id,
                                                    distance=0.01, linkIndexA=link_index)
                if contact_points:  # If any contact points exist, a collision is detected
                    return True
        return False

    def check_edge_collision(self, joint_position_start, joint_position_end, discretization_step=0.01):
        """
        Checks for collision between two joint positions of a robot in PyBullet.

        Args:
            joint_position_start (list): List of joint positions to start from.
            joint_position_end (list): List of joint positions to get to.
            discretization_step (float): maximum interpolation distance before a new collision check is performed.

        Returns:
            bool: True if a collision is detected, False otherwise.
        """
        # Interpolate joint positions between start and end
        interpolated_positions = np.linspace(joint_position_start, joint_position_end,
                                             int(np.linalg.norm(np.array(joint_position_end) -
                                                               np.array(joint_position_start))/discretization_step))

        # Check for collision at each interpolated joint position
        for joint_position in interpolated_positions:
            if self.check_node_collision(joint_position):
                return True
        return False

    def step(self, from_node, to_joint_angles):
        """Step from "from_node" to "to_joint_angles", that should
         (a) return the to_joint_angles if it is within the self.step_size or
         (b) only step so far as self.step_size, returning the new node within that distance"""

        # Calculate the distance between the two nodes
        distance = np.linalg.norm(to_joint_angles - from_node.joint_angles)
        if distance <= self.step_size:
            return self.Node(to_joint_angles)
        else:
            # Calculate the step size
            step = (to_joint_angles - from_node.joint_angles) / distance * self.step_size
            return self.Node(from_node.joint_angles + step)

    def get_nearest_node(self, random_point):
        """Find the nearest node in the tree to a given point."""

        # Find the nearest node to the random point
        distances = [np.linalg.norm(node.joint_angles - random_point) for node in self.node_list]
        nearest_node = self.node_list[np.argmin(distances)]
        return nearest_node

    def neighbors(self, node, gamma=0.005, etta=0.005):
        """Find the neighbors of a node within a certain radius."""
        # Neighborhood radius
        radius = min(gamma*(np.log(len(self.node_list))/len(self.node_list))**(1/len(self.q_limits)), etta)

        # Find the neighbors of a node
        neighbors = [n for n in self.node_list if np.linalg.norm(n.joint_angles - node.joint_angles) < radius]
        return neighbors

    def cost_optimal(self, node, neighbors, goal_weight=1.0):
        """Update the cost of a node using a heuristic that biases towards the goal."""
        for neighbor in neighbors:
            if not self.check_edge_collision(node.joint_angles, neighbor.joint_angles):
                cost = neighbor.cost + np.linalg.norm(node.joint_angles - neighbor.joint_angles) + goal_weight * np.linalg.norm(node.joint_angles - self.q_goal.joint_angles)
                if cost < node.cost:
                    node.cost = cost
                    node.parent = neighbor
        return node

    def rewiring(self, node, neighbors):
        """Rewire the neighbors of a node based on the new node."""
        # Rewire the neighbors of the node
        for neighbor in neighbors:
            if neighbor.cost > node.cost + np.linalg.norm(node.joint_angles - neighbor.joint_angles) and not self.check_edge_collision(node.joint_angles, neighbor.joint_angles):
                neighbor.parent = node
        return neighbors

    def variable_goal_bias(self, current_iter, max_iter, initial_bias, final_bias):
        alpha = 2.0  # Form factor
        P = initial_bias + (final_bias - initial_bias) * (1 - np.exp(-alpha * current_iter / max_iter))
        return min(P, 0.75)  # Cap at 0.75 to ensure some exploration

    def informed_sample(self, c_best):
        if c_best < float('inf'):
            c_min = np.linalg.norm(self.q_start.joint_angles - self.q_goal.joint_angles)
            x_center = (self.q_start.joint_angles + self.q_goal.joint_angles) / 2
            C = self.get_rotation_to_world(self.q_start.joint_angles, self.q_goal.joint_angles)
            r = np.array([(c_best**2 - c_min**2) / 4] + [0] * (len(self.q_limits) - 1))
            L = np.diag(r)
            x_ball = self.sample_unit_ball()
            return C @ L @ x_ball + x_center
        else:
            return np.random.uniform(low=[limit[0] for limit in self.q_limits], high=[limit[1] for limit in self.q_limits]) + np.random.normal(0, 0.1, size=len(self.q_limits))

    def get_rotation_to_world(self, start, goal):
        a1 = np.array(goal) - np.array(start)
        a1_unit = a1 / np.linalg.norm(a1)
        M = np.eye(len(self.q_limits))
        M[:, 0] = a1_unit
        return M

    def sample_unit_ball(self):
        while True:
            x = np.random.uniform(-1, 1, size=len(self.q_limits))
            if np.linalg.norm(x) <= 1:
                return x

    def plan(self, initial_goal_bias=0.1, final_goal_bias=1.0, c_best=float('inf')):
        """Run the RRT* algorithm with dynamically increasing goal bias and obstacle avoidance."""
        for i in range(self.max_iter):
            # Variable increase goal bias over iterations
            goal_bias = self.variable_goal_bias(i, self.max_iter, initial_goal_bias, final_goal_bias)

            # Use goal bias to select the goal as the target with some probability
            if np.random.rand() < goal_bias:
                random_point = self.q_goal.joint_angles
            else:
                random_point = self.informed_sample(c_best)

            nearest_node = self.get_nearest_node(random_point)
            new_node = self.step(nearest_node, random_point)

            # Check for collisions and obstacle proximity
            gripper_pos = self.forward_kinematics(new_node.joint_angles)
            is_obstacle_near, nearby_obstacles = self.check_obstacle_proximity(gripper_pos)

            if is_obstacle_near:
                # Generate avoidance goal
                avoidance_goal = self.generate_avoidance_goal(gripper_pos, nearby_obstacles)
                # Adjust new_node's joint angles to avoid obstacles
                # You might want to implement inverse kinematics here to convert avoidance_goal to joint angles
                continue

            # Check for collisions
            if not self.check_edge_collision(nearest_node.joint_angles, new_node.joint_angles):
                new_node.parent = nearest_node
                new_node.cost = nearest_node.cost + np.linalg.norm(new_node.joint_angles - nearest_node.joint_angles)
                neighbors = self.neighbors(new_node)
                new_node = self.cost_optimal(new_node, neighbors)
                self.node_list.append(new_node)

                # Rewire the neighbors of the new node
                neighbors = self.rewiring(new_node, neighbors)

                if np.linalg.norm(new_node.joint_angles - self.q_goal.joint_angles) < self.step_size:
                    if not self.check_edge_collision(new_node.joint_angles, self.q_goal.joint_angles):
                        self.q_goal.parent = new_node
                        self.q_goal.cost = new_node.cost + np.linalg.norm(self.q_goal.joint_angles - new_node.joint_angles)
                        self.node_list.append(self.q_goal)
                        break

        return self.node_list

    def get_path(self):
        """Return the path from the start node to the goal node."""

        self.plan()
        # print(self.node_list)
        some_path = []
        for no in self.node_list:
            some_path.append(no.joint_angles)

        node = self.q_goal
        while node is not None:
            self.path.append(node.joint_angles)
            node = node.parent

        if not np.array_equal(self.path[-1], self.q_start.joint_angles):
            print("Path does not connect back to the start! Debug required.")

        return self.path[::-1], some_path[::-1]

    def visualize(self, goal_index=None):
        """Visualize the RRT* tree and path."""
        plt.figure(figsize=(10, 10))

        # Plot all nodes and connections
        for node in self.node_list:
            if node.parent:
                plt.plot([node.joint_angles[0], node.parent.joint_angles[0]],
                         [node.joint_angles[1], node.parent.joint_angles[1]],
                         'b-', alpha=0.3)  # Blue lines for connections

        # Plot all nodes
        node_x = [node.joint_angles[0] for node in self.node_list]
        node_y = [node.joint_angles[1] for node in self.node_list]
        plt.scatter(node_x, node_y, c='g', s=20)  # Green dots for nodes

        # Plot start and goal
        plt.scatter(self.q_start.joint_angles[0], self.q_start.joint_angles[1], c='b', s=150, marker='*')  # Blue star for start
        plt.scatter(self.q_goal.joint_angles[0], self.q_goal.joint_angles[1], c='r', s=150, marker='*')  # Red star for goal

        # Plot the final path
        if self.path:
            path_x = [node[0] for node in self.path]
            path_y = [node[1] for node in self.path]
            plt.plot(path_x, path_y, 'r-', linewidth=2)  # Red line for the final path

        plt.title("RRT* Obstacle Avoidance Tree and Path")
        plt.xlabel("Joint Angle 1")
        plt.ylabel("Joint Angle 2")
        plt.grid(True)
        plt.savefig(f"Visualizations/RRT_Star_Obstacle_09_{goal_index:02d}.png")
