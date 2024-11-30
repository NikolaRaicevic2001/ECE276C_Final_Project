import pybullet as p
import pybullet_data

import matplotlib.pyplot as plt
import numpy as np
import time
import os

def draw_camera_frame(Translation, Rotation_Matrix, size=0.2):
    ''' Draw the camera frame in the PyBullet simulation '''
    # Extract camera position and orientation from view matrix
    cam_pos = Translation
    cam_rot = Rotation_Matrix
    
    # Define coordinate axes
    axes = np.array([[size, 0, 0], [0, size, 0], [0, 0, size]])
    
    # Transform axes to camera frame
    transformed_axes = cam_rot.dot(axes.T).T + cam_pos
    
    # Draw coordinate axes
    colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]  # Red, Green, Blue for X, Y, Z
    labels = ['X', 'Y', 'Z']
    for i in range(3):
        p.addUserDebugLine(cam_pos, transformed_axes[i], colors[i], lineWidth=2)
        p.addUserDebugText(labels[i], transformed_axes[i], colors[i], textSize=1.5)
    
    # Add "Camera" label at the camera position
    p.addUserDebugText("Camera", cam_pos, [1, 1, 1], textSize=1.5)

def compute_view_matrix(camera_target_position, distance, yaw, pitch, roll):
    """
    Compute the view matrix for a camera.
    
    Args:
    camera_target_position (list): The target position [x, y, z] the camera is looking at.
    distance (float): The distance from the camera to the target.
    yaw (float): Rotation around the y-axis in degrees (in camera frame, world z-axis).
    pitch (float): Rotation around the z-axis in degrees (in camera frame, world -x-axis).
    roll (float): Rotation around the x-axis in degrees (in camera frame, world x-axis).
    
    Returns:
    numpy.ndarray: The 4x4 view matrix.
    """
    # Convert angles to radians
    y = np.radians(yaw)
    p = np.radians(pitch)
    r = np.radians(roll)
    
    # Compute view matrix elements (rotation part)
    view_matrix = np.array([
        [np.cos(y)*np.cos(p), np.cos(y)*np.sin(p)*np.sin(r) - np.sin(y)*np.cos(r), np.cos(y)*np.sin(p)*np.cos(r) + np.sin(y)*np.sin(r), 0],
        [np.sin(y)*np.cos(p), np.sin(y)*np.sin(p)*np.sin(r) + np.cos(y)*np.cos(r), np.sin(y)*np.sin(p)*np.cos(r) - np.cos(y)*np.sin(r), 0],
        [-np.sin(p), np.cos(p)*np.sin(r), np.cos(p)*np.cos(r), 0],
        [0, 0, 0, 1]
    ])
    
    # Compute camera position
    camera_offset = np.array([
        np.sin(y) * distance,
        -np.sin(p) * np.cos(y) * distance,
        -np.cos(p) * np.cos(y) * distance
    ])
    
    camera_pos = np.array(camera_target_position) - camera_offset
    
    # Add translation to view matrix
    view_matrix[0, 3] = -np.dot(view_matrix[0, :3], camera_pos)
    view_matrix[1, 3] = -np.dot(view_matrix[1, :3], camera_pos)
    view_matrix[2, 3] = -np.dot(view_matrix[2, :3], camera_pos)
    
    return view_matrix

def get_point_cloud(width=640, height=480, fov=60, near_plane=0.01, far_plane=1000, cameraTargetPosition = [0, 0, 0.9],distance=1.3, yaw=0, pitch=0, roll=0):
    ''' Get the point cloud from the camera view '''
    # Adjust camera position to better view the Panda arm
    view_matrix = p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=cameraTargetPosition, distance=distance, yaw=yaw, pitch=pitch, roll=roll, upAxisIndex=2)
    projection_matrix = p.computeProjectionMatrixFOV(fov=fov, aspect=width/height, nearVal=near_plane, farVal=far_plane)

    # Get camera image
    _, _, rgb_image, depth_buffer, _ = p.getCameraImage(width, height, viewMatrix=view_matrix, projectionMatrix=projection_matrix, renderer=p.ER_BULLET_HARDWARE_OPENGL)

    # Convert depth buffer to a numpy array
    depth = np.reshape(depth_buffer, (height, width))

    # Calculate focal length
    focal_length = 0.5 * height / (np.tan(0.5*np.pi/180*fov))

    # Create a grid of coordinates
    x, y = np.meshgrid(np.arange(width), np.arange(height))

    # Calculate 3D coordinates
    z = far_plane * near_plane / (far_plane -(far_plane - near_plane) * depth)
    x = (x - width/2) * z / focal_length
    y = (y - height/2) * z / focal_length

    # Reshape to point cloud format
    point_cloud = np.stack((x, y, z), axis=-1).reshape(-1, 3)

    # Convert view matrix to numpy array and extract rotation and translation    
    # view_matrix = compute_view_matrix(cameraTargetPosition, distance, yaw, pitch, roll)
    view_matrix = np.array(view_matrix).reshape(4, 4).T
    R = view_matrix[:3, :3] 
    t = view_matrix[:3, 3]

    print(f'view_matrix = {view_matrix}')
    # print(f'R = {R}')
    # print(f't = {t}')

    yaw = 0
    Rotation_Yaw = np.array([[np.cos(np.radians(yaw)), 0, np.sin(np.radians(yaw))], [0, 1, 0], [-np.sin(np.radians(yaw)), 0, np.cos(np.radians(yaw))]])
    pitch = 0
    Rotation_Pitch = np.array([[1, 0, 0], [0, np.cos(np.radians(pitch)), -np.sin(np.radians(pitch))], [0, np.sin(np.radians(pitch)), np.cos(np.radians(pitch))]])
    roll = 0
    Rotation_Roll = np.array([[np.cos(np.radians(roll)), -np.sin(np.radians(roll)), 0], [np.sin(np.radians(roll)), np.cos(np.radians(roll)), 0], [0, 0, 1]])

    # Transform points to world coordinates
    Translation = np.array([[-1, 0, 0], [0, 0, 1], [0, -1, 0]]) @ t 
    Rotation_Matrix = Rotation_Yaw @ Rotation_Pitch @ Rotation_Roll @ R
    point_cloud_world = np.dot(point_cloud,Rotation_Matrix.T) + Translation

    return point_cloud_world, Translation, Rotation_Matrix

if __name__ == "__main__":
    physics_client = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)

    # Load Panda robot and fix its base
    ground_id = p.loadURDF("plane.urdf")
    robot_id = p.loadURDF("franka_panda/panda.urdf", [0, 0.3, 0.6], useFixedBase=True)
    tableUid = p.loadURDF(os.path.join(pybullet_data.getDataPath(), "table/table.urdf"), basePosition=[0, 0, 0])

    # Add Collision Objects
    collision_ids = [ground_id] # add the ground to the collision list
    collision_positions = [[0.5, -0.3, 0.72],[-0.3, -0.3, 0.72]]
    collision_orientations =  [[0, 0, 0.5],[0, 0, 0.5]]
    collision_scales = [0.2, 0.2]
    for i in range(len(collision_scales)):
        collision_ids.append(p.loadURDF("cube.urdf",
            basePosition=collision_positions[i],  # Position of the cube
            baseOrientation=p.getQuaternionFromEuler(collision_orientations[i]),  # Orientation of the cube
            globalScaling=collision_scales[i]  # Scale the cube to half size
        ))

    # Initializing Simulation
    p.setRealTimeSimulation(0)
    duration = 30
    fps = 30
    time_steps = int(duration * fps)
    dt = 1.0 / fps

    # Display point cloud
    for step in range(time_steps):
        # Get point cloud
        point_cloud_world, Translation, Rotation_Matrix = get_point_cloud(width=640, height=480, fov=60, near_plane=0.01, far_plane=10)

        # Downsample and display point cloud
        p.removeAllUserDebugItems()
        downsampled_cloud = point_cloud_world[::5]  # Take every xth point
        p.addUserDebugPoints(downsampled_cloud, [[1, 0, 0]] * len(downsampled_cloud), pointSize=3)

        # Draw camera coordinate frame
        draw_camera_frame(Translation, Rotation_Matrix)

        p.stepSimulation()
        time.sleep(1/fps)
  

    p.disconnect()
    