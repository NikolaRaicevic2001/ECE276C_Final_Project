import pybullet as p
import pybullet_data
import time
import numpy as np
import random

class RRTManipulatorPlanner:
    def __init__(self, robot_id=None):
        # Load environment and robot
        self.robot_id = robot_id

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
        collisions = p.getContactPoints()
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
        """RRT* path planning algorithm"""
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

            # Check collision and validity
            if self.is_state_valid(new_node):
                nodes.append(new_node)

                # Find near nodes for potential rewiring
                near_nodes = [n for n in nodes if self.distance(n, new_node) < step_size * 2]

                # Choose parent with minimum cost
                best_parent = nearest_node
                best_cost = costs.get(tuple(nearest_node), float('inf')) + self.distance(nearest_node, new_node)

                for near_node in near_nodes:
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

                    if potential_new_cost < current_near_cost:
                        edges[tuple(near_node)] = new_node
                        costs[tuple(near_node)] = potential_new_cost

                # Check if goal is reached
                if self.distance(new_node, goal_angles[:7]) < step_size:
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
