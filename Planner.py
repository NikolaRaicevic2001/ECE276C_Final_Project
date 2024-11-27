import numpy as np
from typing import List, Tuple, Optional
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class Node:
    def __init__(self, configuration: np.ndarray):
        self.configuration = configuration  # 7-DOF joint angles
        self.parent = None
        self.path_from_parent = []

class RRTManipulatorPlanner:
    def __init__(self, joint_limits: np.ndarray, collision_checker=None):
        """
        Initialize RRT planner for 7-DOF manipulator

        Args:
            joint_limits: Array of shape (7, 2) containing min/max limits for each joint
            collision_checker: Optional function to check for collisions
        """
        self.joint_limits = joint_limits
        self.collision_checker = collision_checker
        self.nodes = []
        self.step_size = 0.2  # Increased step size
        self.max_iterations = 10000

    def forward_kinematics(self, configuration: np.ndarray) -> np.ndarray:
        """
        Improved forward kinematics calculation.
        """
        # DH parameters (example values - adjust for your robot)
        d = [0.333, 0, 0.316, 0, 0.384, 0, 0]  # Link offsets
        a = [0, 0, 0, 0.0825, -0.0825, 0, 0.088]  # Link lengths
        alpha = [0, -np.pi/2, np.pi/2, np.pi/2, -np.pi/2, np.pi/2, np.pi/2]  # Link twists

        # Initialize transformation matrix
        T = np.eye(4)

        # Compute forward kinematics
        for i in range(7):
            theta = configuration[i]

            # DH transformation matrix
            ct = np.cos(theta)
            st = np.sin(theta)
            ca = np.cos(alpha[i])
            sa = np.sin(alpha[i])

            T_i = np.array([
                [ct, -st*ca, st*sa, a[i]*ct],
                [st, ct*ca, -ct*sa, a[i]*st],
                [0, sa, ca, d[i]],
                [0, 0, 0, 1]
            ])

            T = T @ T_i

        return T[:3, 3]  # Return position component

    def random_configuration(self) -> np.ndarray:
        """Generate random joint configuration within limits"""
        return np.random.uniform(self.joint_limits[:, 0], self.joint_limits[:, 1])

    def sample_biased_configuration(self, target_position: np.ndarray) -> np.ndarray:
        """
        Sample configuration with bias towards configurations that might reach the target
        """
        # Generate multiple random configurations and select the one closest to target
        best_dist = float('inf')
        best_config = None

        for _ in range(10):  # Try 10 random configurations
            config = self.random_configuration()
            ee_pos = self.forward_kinematics(config)
            dist = np.linalg.norm(ee_pos - target_position)

            if dist < best_dist:
                best_dist = dist
                best_config = config

        return best_config

    def nearest_neighbor(self, target_config: np.ndarray) -> Node:
        """Find nearest node in tree to target configuration"""
        min_dist = float('inf')
        nearest_node = None

        for node in self.nodes:
            # Use weighted joint distance for better 7-DOF matching
            weights = np.array([1.0, 1.0, 1.0, 0.8, 0.8, 0.5, 0.3])  # Prioritize base joints
            dist = np.linalg.norm(weights * (node.configuration - target_config))
            if dist < min_dist:
                min_dist = dist
                nearest_node = node

        return nearest_node


    def extend(self, from_config: np.ndarray, to_config: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        Extend tree from one configuration toward another
        Returns new configuration and discretized path
        """
        diff = to_config - from_config
        dist = np.linalg.norm(diff)

        if dist < self.step_size:
            return to_config, [to_config]

        # Take a step in the direction of the target
        direction = diff / dist
        new_config = from_config + direction * self.step_size

        # Ensure within joint limits
        new_config = np.clip(new_config, self.joint_limits[:, 0], self.joint_limits[:, 1])

        # Create discretized path
        num_steps = int(np.ceil(dist / self.step_size))
        path = []
        for i in range(num_steps):
            t = min(1.0, (i + 1) * self.step_size / dist)
            path_config = from_config + t * diff
            path_config = np.clip(path_config, self.joint_limits[:, 0], self.joint_limits[:, 1])
            path.append(path_config)

        return new_config, path

    def is_valid_path(self, path: List[np.ndarray]) -> bool:
        """Check if path is collision-free"""
        if self.collision_checker is None:
            return True

        return all(not self.collision_checker(config) for config in path)

    def plan(self, start_config: np.ndarray, target_position: np.ndarray,
            tolerance: float = 0.1) -> Optional[List[np.ndarray]]:  # Increased tolerance
        """
        Plan path from start configuration to reach target end-effector position
        """
        # Initialize tree with start configuration
        start_node = Node(start_config)
        self.nodes = [start_node]

        best_distance = np.linalg.norm(self.forward_kinematics(start_config) - target_position)
        best_node = None

        for i in range(self.max_iterations):
            # Adaptive sampling strategy
            if np.random.random() < 0.3:  # Increased bias toward target
                random_config = self.sample_biased_configuration(target_position)
            else:
                random_config = self.random_configuration()

            # Find nearest node in tree
            nearest_node = self.nearest_neighbor(random_config)

            # Extend tree toward random configuration
            new_config, path = self.extend(nearest_node.configuration, random_config)

            # Check if path is valid
            if not self.is_valid_path(path):
                continue

            # Create and add new node
            new_node = Node(new_config)
            new_node.parent = nearest_node
            new_node.path_from_parent = path
            self.nodes.append(new_node)

            # Check distance to target
            ee_pos = self.forward_kinematics(new_config)
            distance = np.linalg.norm(ee_pos - target_position)

            # Update best distance
            if distance < best_distance:
                best_distance = distance
                best_node = new_node
                print(f"New best distance: {distance:.3f}")

            # Check if we've reached target
            if distance < tolerance:
                return self.extract_path(new_node)

            # Early stopping if we're not making progress
            if i % 1000 == 0 and i > 0:
                print(f"Iteration {i}, Best distance: {best_distance:.3f}")

        # If we didn't reach the exact target, return the best path we found
        if best_node is not None and best_distance < tolerance * 2:
            print(f"Returning best found path with distance: {best_distance:.3f}")
            return self.extract_path(best_node)

        return None

    def extract_path(self, node: Node) -> List[np.ndarray]:
        """Extract path from start to given node"""
        path = []
        current = node

        while current.parent is not None:
            path = current.path_from_parent + path
            current = current.parent

        return [current.configuration] + path

    def visualize_path(self, path: List[np.ndarray]):
        """Visualize the path in task space"""
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Plot path
        ee_positions = [self.forward_kinematics(config) for config in path]
        ee_positions = np.array(ee_positions)

        # Plot robot workspace (approximate)
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        radius = 1.0  # Approximate robot reach
        x = radius * np.outer(np.cos(u), np.sin(v))
        y = radius * np.outer(np.sin(u), np.sin(v))
        z = radius * np.outer(np.ones(np.size(u)), np.cos(v))
        ax.plot_surface(x, y, z, alpha=0.1, color='gray')

        # Plot path
        ax.plot(ee_positions[:, 0], ee_positions[:, 1], ee_positions[:, 2], 'b-', linewidth=2, label='Path')
        ax.scatter(ee_positions[0, 0], ee_positions[0, 1], ee_positions[0, 2],
                  c='g', marker='o', s=100, label='Start')
        ax.scatter(ee_positions[-1, 0], ee_positions[-1, 1], ee_positions[-1, 2],
                  c='r', marker='o', s=100, label='End')

        # Plot intermediate points
        ax.scatter(ee_positions[1:-1, 0], ee_positions[1:-1, 1], ee_positions[1:-1, 2],
                  c='blue', marker='.', s=30, alpha=0.5)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend()

        # Set equal aspect ratio
        ax.set_box_aspect([1,1,1])

        plt.show()

# Example usage:
if __name__ == "__main__":
    # Define joint limits (in radians)
    joint_limits = np.array([
        [-np.pi, np.pi],      # Joint 1
        [-np.pi/2, np.pi/2],  # Joint 2
        [-np.pi, np.pi],      # Joint 3
        [-np.pi, np.pi],      # Joint 4
        [-np.pi, np.pi],      # Joint 5
        [-np.pi, np.pi],      # Joint 6
        [-np.pi, np.pi]       # Joint 7
    ])

    # Create planner
    planner = RRTManipulatorPlanner(joint_limits)

    # Define start configuration and target position
    start_config = np.zeros(7)  # All joints at zero position
    target_position = np.array([0.5, 0.3, 0.7])  # Target end-effector position

    # Plan path
    path = planner.plan(start_config, target_position)

    if path is not None:
        print("Path found!")
        planner.visualize_path(path)
    else:
        print("No path found.")
