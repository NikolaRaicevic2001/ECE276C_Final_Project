from datetime import datetime

import pybullet as p
import numpy as np

import pybullet_data
import imageio
import time

def check_node_collision(robot_id, object_ids, joint_position):
    """
    Checks for collisions between a robot and an object in PyBullet. 

    Args:
        robot_id (int): The ID of the robot in PyBullet.
        object_id (int): The ID of the object in PyBullet.
        joint_position (list): List of joint positions. 

    Returns:
        bool: True if a collision is detected, False otherwise.
    """
    # Set joint positions
    for joint_index, joint_pos in enumerate(joint_position):
        p.resetJointState(robot_id, joint_index, joint_pos)

    # Perform collision check for all links
    for object_id in object_ids: # Check for each object
        for link_index in range(0, p.getNumJoints(robot_id)): # Check for each link of the robot
            contact_points = p.getClosestPoints(bodyA=robot_id, bodyB=object_id, distance=0.01, linkIndexA=link_index)
            # print(f"robot_id: {robot_id}, object_id: {object_id}, link_index: {link_index}, contact_points: {contact_points}")
            if contact_points:  # If any contact points exist, a collision is detected
                return True
    return False

#################################################
#### YOUR CODE HERE: COLLISION EDGE CHECKING ####
#################################################
def check_edge_collision(robot_id, object_ids, joint_position_start, joint_position_end, discretization_step=0.01):
    """ 
    Checks for collision between two joint positions of a robot in PyBullet.
    Args:
        robot_id (int): The ID of the robot in PyBullet.
        object_ids (list): List of IDs of the objects in PyBullet.
        joint_position_start (list): List of joint positions to start from.
        joint_position_end (list): List of joint positions to get to.
        discretization_step (float): maximum interpolation distance before a new collision check is performed.
    Returns:
        bool: True if a collision is detected, False otherwise.
    """

    # Interpolate joint positions between start and end
    interpolated_positions = np.linspace(joint_position_start, joint_position_end, int(np.linalg.norm(np.array(joint_position_end) - np.array(joint_position_start))/discretization_step))

    # Check for collision at each interpolated joint position
    for joint_position in interpolated_positions:
        if check_node_collision(robot_id, object_ids, joint_position):
            return True
        # time.sleep(0.2)
    return False
    
# Provided 
class Node:
    def __init__(self, joint_angles):
        self.joint_angles = np.array(joint_angles)  # joint angles of the node in n-dimensional space
        self.parent = None
        self.cost = 0

######################################################################
##################### YOUR CODE HERE: RRT CLASS ######################
######################################################################
class RRT:
    def __init__(self, q_start, q_goal, robot_id, obstacle_ids, q_limits, max_iter=10000, step_size=0.5):
        """
        RRT Initialization.

        Parameters:
        - q_start: List of starting joint angles [x1, x2, ..., xn].
        - q_goal: List of goal joint angles [x1, x2, ..., xn].
        - obstacle_ids: List of obstacles, each as a tuple ([center1, center2, ..., centern], radius).
        - q_limits: List of tuples [(min_x1, max_x1), ..., (min_xn, max_xn)] representing the limits in each dimension.
        - max_iter: Maximum number of iterations.
        - step_size: Maximum step size to expand the tree.
        """
        self.q_start = Node(q_start)
        self.q_goal = Node(q_goal)
        self.obstacle_ids = obstacle_ids
        self.robot_id = robot_id
        self.q_limits = q_limits
        self.max_iter = max_iter
        self.step_size = step_size
        self.node_list = [self.q_start]

    def step(self, from_node, to_joint_angles):
        """Step from "from_node" to "to_joint_angles", that should
         (a) return the to_joint_angles if it is within the self.step_size or
         (b) only step so far as self.step_size, returning the new node within that distance"""
        
        # Calculate the distance between the two nodes
        distance = np.linalg.norm(to_joint_angles - from_node.joint_angles)
        if distance <= self.step_size:
            return Node(to_joint_angles)
        else:
            # Calculate the step size
            step = (to_joint_angles - from_node.joint_angles) / distance * self.step_size
            return Node(from_node.joint_angles + step)
        
    def get_nearest_node(self, random_point):
        """Find the nearest node in the tree to a given point."""
        
        # Find the nearest node to the random point
        distances = [np.linalg.norm(node.joint_angles - random_point) for node in self.node_list]
        nearest_node = self.node_list[np.argmin(distances)]
        return nearest_node

    def plan(self):
        """Run the RRT algorithm to find a path of dimension Nx3. Limit the search to only max_iter iterations."""
        
        # Iterate through the max number of iterations
        for _ in range(self.max_iter):
            # Generate a random node
            random_point = np.random.uniform(low=[limit[0] for limit in self.q_limits], high=[limit[1] for limit in self.q_limits])
            nearest_node = self.get_nearest_node(random_point)
            new_node = self.step(nearest_node, random_point)
            
            # Check for collisions
            if not check_edge_collision(self.robot_id, self.obstacle_ids, nearest_node.joint_angles, new_node.joint_angles):
                new_node.parent = nearest_node
                self.node_list.append(new_node)

                # Check if the goal is reached
                if np.linalg.norm(new_node.joint_angles - self.q_goal.joint_angles) < self.step_size:
                    self.q_goal.parent = new_node
                    self.node_list.append(self.q_goal)
                    break

        return self.node_list
    
    def neighbors(self, node, gamma = 0.005, etta = 0.005):
        """Find the neighbors of a node within a certain radius."""
        # Neighborhood radius
        # print(f"""min({gamma*(np.log(len(self.node_list))/len(self.node_list))**(1/len(self.q_limits))}, {etta})""")
        radius = min(gamma*(np.log(len(self.node_list))/len(self.node_list))**(1/len(self.q_limits)), etta)

        # Find the neighbors of a node
        neighbors = [n for n in self.node_list if np.linalg.norm(n.joint_angles - node.joint_angles) < radius]
        return neighbors
    
    def cost_optimal(self, node, neighbors):
        """Update the cost of a node based on its neighbors."""
        # Update the cost of the node
        for neighbor in neighbors:
            if not check_edge_collision(self.robot_id, self.obstacle_ids, node.joint_angles, neighbor.joint_angles):
                cost = neighbor.cost + np.linalg.norm(node.joint_angles - neighbor.joint_angles)
                if cost < node.cost:
                    node.cost = cost
                    node.parent = neighbor
        return node

    def rewiring(self, node, neighbors):
        """Rewire the neighbors of a node based on the new node."""
        # Rewire the neighbors of the node
        for neighbor in neighbors:
            if neighbor.cost > node.cost + np.linalg.norm(node.joint_angles - neighbor.joint_angles) and not check_edge_collision(self.robot_id, self.obstacle_ids, node.joint_angles, neighbor.joint_angles):
                neighbor.parent = node
        return neighbors

    def plan_2(self):
        """Run the RRT* algorithm to find a path of dimension Nx3. Limit the search to only max_iter iterations."""
            
        # Iterate through the max number of iterations
        for _ in range(self.max_iter):
            # Generate a random node
            random_point = np.random.uniform(low=[limit[0] for limit in self.q_limits], high=[limit[1] for limit in self.q_limits])
            nearest_node = self.get_nearest_node(random_point)
            new_node = self.step(nearest_node, random_point)
            
            # Check for collisions
            if not check_edge_collision(self.robot_id, self.obstacle_ids, nearest_node.joint_angles, new_node.joint_angles):
                # Add the new node to the tree
                new_node.parent = nearest_node # Set the parent of the new node
                new_node.cost = nearest_node.cost + np.linalg.norm(new_node.joint_angles - nearest_node.joint_angles) # Update the cost of the new node
                neighbors = self.neighbors(new_node)  # Identify neighboring Nodes
                new_node = self.cost_optimal(new_node, neighbors) # Choosing Optimal Parameters
                self.node_list.append(new_node) 

                # Rewire the neighbors of the new node
                neighbors = self.rewiring(new_node, neighbors)

                # Check if the goal is reached
                if np.linalg.norm(new_node.joint_angles - self.q_goal.joint_angles) < self.step_size:
                    self.q_goal.parent = new_node
                    self.node_list.append(self.q_goal)
                    break

        return self.node_list

    def get_path(self, plan_id=1):
        """Return the path from the start node to the goal node."""
        
        # Run the RRT/RRT* algorithm
        if plan_id == 1:
            self.plan()
        elif plan_id == 2:
            self.plan_2()

        # Get the path from the start node to the goal node
        path = []
        node = self.q_goal
        while node is not None:
            path.append(node.joint_angles)
            node = node.parent
        return path[::-1]
    
if __name__ == "__main__":
      # Goal Joint Positions for the Robot
    goal_positions = [[-2.54, 0.15, -0.15], [-1.82,0.15,-0.15],[0.5, 0.15,-0.15], [1.7,0.2,-0.15],[-2.54, 0.15, -0.15]]

    # Joint Limits of the Robot
    joint_limits = [[-np.pi, np.pi], [0, np.pi], [-np.pi, np.pi]]

    # A3xN path array that will be filled with waypoints through all the goal positions
    path_saved = np.array([[-2.54, 0.15, -0.15]]) # Start at the first goal position

     #######################
    #### PROBLEM SETUP ####
    #######################

    # Initialize PyBullet
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath()) # For default URDFs
    p.setGravity(0, 0, -9.8)

    # Load the plane and robot arm
    ground_id = p.loadURDF("plane.urdf")
    arm_id = p.loadURDF("three_link_arm.urdf", [0, 0, 0], useFixedBase=True)
    
    # Add Collision Objects
    collision_ids = [ground_id] # add the ground to the collision list
    collision_positions = [[0.3, 0.5, 0.251], [-0.3, 0.3, 0.101], [-1, -0.15, 0.251], [-1, -0.15, 0.752], [-0.5, -1, 0.251], [0.5, -0.35, 0.201], [0.5, -0.35, 0.602]]
    collision_orientations =  [[0, 0, 0.5], [0, 0, 0.2], [0, 0, 0],[0, 0, 1], [0, 0, 0], [0, 0, .25], [0, 0, 0.5]]
    collision_scales = [0.5, 0.25, 0.5, 0.5, 0.5, 0.4, 0.4]
    for i in range(len(collision_scales)):
        collision_ids.append(p.loadURDF("cube.urdf", basePosition=collision_positions[i], baseOrientation=p.getQuaternionFromEuler(collision_orientations[i]), globalScaling=collision_scales[i]))

    # Goal Joint Positions for the Robot
    goal_positions = [[-2.54, 0.15, -0.15], [-1.82,0.15,-0.15],[0.5, 0.15,-0.15], [1.7,0.2,-0.15],[-2.54, 0.15, -0.15]]

    # Joint Limits of the Robot
    joint_limits = [[-np.pi, np.pi], [0, np.pi], [-np.pi, np.pi]]

    # A3xN path array that will be filled with waypoints through all the goal positions
    path_saved = np.array([[-2.54, 0.15, -0.15]]) # Start at the first goal position

    ####################################################################################################
    #### YOUR CODE HERE: RUN RRT MOTION PLANNER FOR ALL goal_positions (starting at goal position 1) ###
    ####################################################################################################

    ##############################################################
    ############# Test check_edge_collision function #############
    ##############################################################
    # # Get the joint indices of the robot
    # for idx in range(p.getNumJoints(arm_id)): 
    #     info = p.getJointInfo(arm_id, idx)
    #     print('Joint {}: {}'.format(info[1], info[14]))

    # for _ in range(1200): 
    #     p.stepSimulation()
    #     time.sleep(1./240.)

    # # Test check_edge_collision function
    # print('Testing check_edge_collision function:')
    # print(f'COllision_ids = {collision_ids}')
    # print(check_edge_collision(arm_id, collision_ids, [-2.54, 0.15, -0.15], [-1.82,0.15,-0.15], discretization_step=0.01))  # Expected output: True
    # print(check_edge_collision(arm_id, collision_ids, [-1.82,0.15,-0.15], [0.5, 0.15,-0.15], discretization_step=0.01)) # Expected output: True
    # print(check_edge_collision(arm_id, collision_ids, [-1.82, 0.15, -0.15 ], [0.39628076, 0.47177292, -0.03040937], discretization_step=0.01)) # Expected output: True
    # print(check_edge_collision(arm_id, collision_ids, [0.39628076, 0.47177292, -0.03040937], [0.5, 0.15,-0.15], discretization_step=0.01)) # Expected output: True


    ##############################################################
    ########### RUN RRT MOTION PLANNING IMPLEMENTATION ###########
    ##############################################################
    
    # Set the motion planning algorithm to run
    motion_planning_alogirthm = 2

    if motion_planning_alogirthm == 1:
        # Run RRT Motion Planner for all goal_positions
        for i in range(1, len(goal_positions)):
            print(f"Running RRT Motion Planner for goal position {i}")
            print(f"Goal positions: {goal_positions[i-1]}, {goal_positions[i]}")

            # Initialize the RRT planner
            rrt = RRT(q_start=goal_positions[i-1], q_goal=goal_positions[i], robot_id=arm_id, obstacle_ids=collision_ids, q_limits=joint_limits, max_iter=10000, step_size=0.5)

            # Run the RRT planner
            path_saved = np.concatenate((path_saved, rrt.get_path(plan_id = motion_planning_alogirthm)), axis=0)

        print(f"Path saved: {path_saved}")

    if motion_planning_alogirthm == 2:
        # Run RRT* Motion Planner for all goal_positions
        for i in range(1, len(goal_positions)):
            print(f"Running RRT Star Motion Planner for goal position {i}")
            print(f"Goal positions: {goal_positions[i-1]}, {goal_positions[i]}")

            # Initialize the RRT planner
            rrt = RRT(q_start=goal_positions[i-1], q_goal=goal_positions[i], robot_id=arm_id, obstacle_ids=collision_ids, q_limits=joint_limits, max_iter=10000, step_size=0.5)

            # Run the RRT planner
            path_saved = np.concatenate((path_saved, rrt.get_path(plan_id = motion_planning_alogirthm)), axis=0)

        print(f"Path saved: {path_saved}")

    ################################################################################
    ####  RUN THE SIMULATION AND MOVE THE ROBOT ALONG PATH_SAVED ###################
    ################################################################################

    # Move to all goal positions to check for collisions
    for i,goal_position in enumerate(goal_positions):
        print(f"Goal position {i}: {goal_position}")
        for joint_index, joint_pos in enumerate(goal_positions[i]):
            p.resetJointState(arm_id, joint_index, joint_pos)
        time.sleep(2.0)

    # Set the initial joint positions
    for joint_index, joint_pos in enumerate(goal_positions[0]):
        p.resetJointState(arm_id, joint_index, joint_pos)

    # Move through the waypoints
    frames_list = []
    duration = 10
    previous_time = datetime.now()
    for waypoint in path_saved:
        counter = 0
        for joint_index, joint_pos in enumerate(waypoint):
        # run velocity control until waypoint is reached
            while True:
                #get current joint positions
                goal_positions_waypoint = [p.getJointState(arm_id, i)[0] for i in range(3)]
                # calculate the displacement to reach the next waypoint
                displacement_to_waypoint = waypoint-goal_positions_waypoint
                # check if goal is reached
                max_speed = 0.05
                if(np.linalg.norm(displacement_to_waypoint) < max_speed):
                    frame = p.getCameraImage(1280, 960)
                    frames_list.append(frame[2])
                    break
                else:
                    # calculate the "velocity" to reach the next waypoint
                    velocities = np.min((np.linalg.norm(displacement_to_waypoint), max_speed))*displacement_to_waypoint/np.linalg.norm(displacement_to_waypoint)
                    for joint_index, joint_step in enumerate(velocities):
                        p.setJointMotorControl2(bodyIndex=arm_id,jointIndex=joint_index,controlMode=p.VELOCITY_CONTROL,targetVelocity=joint_step)

                # Capture frames for animation every 2 seconds
                current_time = datetime.now()
                if (current_time - previous_time).total_seconds() > 1.0:
                    frame = p.getCameraImage(1280, 960)
                    frames_list.append(frame[2])
                    previous_time = current_time

                # Check if the waypoint is a goal position and taken a snapshot
                if any(np.allclose(waypoint, np.array(goal)) for goal in goal_positions) and counter < 3:
                    frame = p.getCameraImage(1280, 960)
                    frames_list.append(frame[2])
                    counter += 1

                #Take a simulation step
                p.stepSimulation()            
                time.sleep(1.0 / 2400.0)

    if motion_planning_alogirthm == 1:
        # Save each set of frames as an animated GIF
        with imageio.get_writer('ECE276C_HW04_RRT.gif', mode='I', duration=duration/len(frames_list)) as writer:
            for frame in frames_list:
                writer.append_data(frame)
    if motion_planning_alogirthm == 2:
        # Save each set of frames as an animated GIF
        with imageio.get_writer('ECE276C_HW04_RRT_star.gif', mode='I', duration=duration/len(frames_list)) as writer:
            for frame in frames_list:
                writer.append_data(frame)

    # Disconnect from PyBullet
    p.disconnect()