import pybullet as p
import pybullet_data

import matplotlib.pyplot as plt
import numpy as np
import time

def get_point_cloud(width=640, height=480, fov=60, near_plane=0.01, far_plane=1000):
    # Adjust camera position to better view the Panda arm
    view_matrix = p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=[0, 0, 0.5], distance=1.5, yaw=45, pitch=-30, roll=0, upAxisIndex=2)
    projection_matrix = p.computeProjectionMatrixFOV(fov=fov, aspect=width/height, nearVal=near_plane, farVal=far_plane)

    # Get camera image
    _, _, rgb_image, depth_buffer, _ = p.getCameraImage(width, height, viewMatrix=view_matrix, projectionMatrix=projection_matrix, renderer=p.ER_BULLET_HARDWARE_OPENGL)

    # Convert depth buffer to a numpy array
    depth = np.array(depth_buffer)

    # Calculate focal length
    focal_length = 0.5 * height / (np.tan(0.5*np.pi/180*fov))

    # Create a grid of coordinates
    x, y = np.meshgrid(np.arange(width), np.arange(height))

    # Calculate 3D coordinates
    z = far_plane * near_plane / (far_plane - (far_plane - near_plane) * depth)
    x = (x - width/2) * z / focal_length
    y = (y - height/2) * z / focal_length

    # Reshape to point cloud format
    point_cloud = np.stack((x, y, z), axis=-1).reshape(-1, 3)

    # Convert view matrix to numpy array and extract rotation and translation
    view_matrix = np.array(view_matrix).reshape(4, 4).T
    R = view_matrix[:3, :3]
    t = view_matrix[:3, 3]

    # Transform points to world coordinates
    # point_cloud_world = np.dot(point_cloud, R.T) + t

    return point_cloud

if __name__ == "__main__":
    physics_client = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)

    # Load Panda robot and fix its base
    ground_id = p.loadURDF("plane.urdf")
    robot_id = p.loadURDF("franka_panda/panda.urdf", [0, 0, 0], useFixedBase=True)

    # Add Collision Objects
    collision_ids = [ground_id] # add the ground to the collision list
    collision_positions = [[0.3, 0.5, 0.251], [-0.3, 0.3, 0.101], [-1, -0.15, 0.251], [-1, -0.15, 0.752], [-0.5, -1, 0.251], [0.5, -0.35, 0.201], [0.5, -0.35, 0.602]]
    collision_orientations =  [[0, 0, 0.5], [0, 0, 0.2], [0, 0, 0],[0, 0, 1], [0, 0, 0], [0, 0, .25], [0, 0, 0.5]]
    collision_scales = [0.5, 0.25, 0.5, 0.5, 0.5, 0.4, 0.4]
    for i in range(len(collision_scales)):
        collision_ids.append(p.loadURDF("cube.urdf",
            basePosition=collision_positions[i],  # Position of the cube
            baseOrientation=p.getQuaternionFromEuler(collision_orientations[i]),  # Orientation of the cube
            globalScaling=collision_scales[i]  # Scale the cube to half size
        ))

    # Initializing Simulation
    p.setRealTimeSimulation(0)
    duration = 10
    fps = 30
    time_steps = int(duration * fps)
    dt = 1.0 / fps

    # Display point cloud
    for step in range(time_steps):
        # Get point cloud
        point_cloud = get_point_cloud(width=640, height=480, fov=60, near_plane=0.01, far_plane=10)

        # Downsample and display point cloud
        p.removeAllUserDebugItems()
        downsampled_cloud = point_cloud[::5]  # Take every xth point
        p.addUserDebugPoints(downsampled_cloud, [[1, 0, 0]] * len(downsampled_cloud), pointSize=3)

        p.stepSimulation()
        time.sleep(1/fps)
  

    p.disconnect()
    
  
