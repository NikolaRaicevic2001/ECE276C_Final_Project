import pybullet as p
import pybullet_data
import numpy as np
import time
import os

def draw_camera_frame(t, R, size=0.2):
    ''' Draw the camera frame in the PyBullet simulation '''
    # Define coordinate axes
    axes = np.array([[size, 0, 0], [0, 0, -size], [0, size, 0]])
    
    # Transform axes to camera frame
    transformed_axes = (R @ axes) + t

    # Draw coordinate axes
    colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]  # Red, Green, Blue for X, Y, Z
    labels = ['X', 'Y', 'Z']
    for i in range(3):
        p.addUserDebugLine(t, transformed_axes[i], colors[i], lineWidth=2)
        p.addUserDebugText(labels[i], transformed_axes[i], colors[i], textSize=1.5)
    
    p.addUserDebugText("Camera", t, [1, 1, 1], textSize=1.5)

def camera_to_world(point_camera, view_matrix):
    """
    Transform point from camera frame to world frame
    point_camera: [x, y, z] in camera frame
    view_matrix: 4x4 view matrix (world to camera)
    """
    # Extract rotation (upper 3x3) and translation from view matrix
    R = view_matrix[:3, :3]  # 3x3 rotation
    t = view_matrix[:3, 3]   # translation
    
    # To go from camera to world:
    # 1. Inverse rotation: R.T (since R is orthogonal)
    # 2. Inverse translation: -R.T @ t
    point_world = (R.T @ point_camera.T).T - R.T @ t

    return point_world

def world_to_camera(point_world, view_matrix):
    """Transform point from world to camera frame"""
    R = view_matrix[:3, :3]
    t = view_matrix[:3, 3]
    point_camera = R @ point_world + t
    return point_camera

def get_point_cloud(width=640, height=480):
    # Camera setup
    fov = 75
    aspect = width / height
    near = 0.01
    far = 10
    
    camera_position = [0.0, -1.2, 1.8]
    camera_target = [0.0, 1.0, 0.0]
    up_vector = [0.0, 0.0, 1.0]
    
    view_matrix = np.array(p.computeViewMatrix(camera_position, camera_target, up_vector)).reshape(4,4).T
    proj_matrix = p.computeProjectionMatrixFOV(fov, aspect, near, far)

    # Get camera image with segmentation mask
    _, _, _, depth_buffer, seg_mask = p.getCameraImage(
        width, height,
        viewMatrix=view_matrix.T.ravel(),
        projectionMatrix=proj_matrix,
        renderer=p.ER_BULLET_HARDWARE_OPENGL,
        flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX,
        shadow=0
    )
    
    depth_buffer = np.array(depth_buffer).reshape(height, width)
    seg_mask = np.array(seg_mask).reshape(height, width)

    # Actual point cloud filtering code
    depth_mask = depth_buffer < 1.0  # True for valid depth points (not at infinity)

    # Extract object IDs from segmentation mask
    object_ids = seg_mask & ((1 << 24) - 1)  # Get lower 24 bits containing object ID

    # Create mask that excluds ground (ID 0) and robot base (ID 1)
    object_mask = ~np.isin(object_ids, [0, 1])
    
    # Combine all masks
    mask = depth_mask & object_mask 

    if not np.any(mask):
        print("No valid depth points found!")
        return np.array([])

    # Convert normalized depth to metric depth
    z_eye = near * far / (far - depth_buffer * (far - near))
    
    # Generate pixel coordinates
    x_ndc, y_ndc = np.meshgrid(np.linspace(-aspect, aspect, width), np.linspace(1, -1, height))
    
    # Convert to camera space
    fov_factor = np.tan(np.radians(fov/2))
    x_cam = x_ndc * z_eye * fov_factor
    y_cam = y_ndc * z_eye * fov_factor
    z_cam = -z_eye

    # Stack and reshape
    points_cam = np.stack([x_cam, y_cam, z_cam], axis=-1)
    points_cam = points_cam.reshape(-1, 3)
    
    # Apply mask
    points_cam = points_cam[mask.reshape(-1)]
    
    # Transform to world space
    R = view_matrix[:3, :3].T
    t = view_matrix[:3, 3]
    points_world = camera_to_world(points_cam, view_matrix)

    return points_world, R, -R @ t

if __name__ == "__main__":
    physics_client = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)

    # Load Panda robot and fix its base
    ground_id = p.loadURDF("plane.urdf")
    robot_id = p.loadURDF("franka_panda/panda.urdf", [0, 0.3, 0.6], useFixedBase=True)
    tableUid = p.loadURDF(os.path.join(pybullet_data.getDataPath(), "table/table.urdf"), basePosition=[0, 0, 0])

    # Add Collision Objects
    collision_ids = [ground_id] 
    collision_positions = [[0.5, -0.3, 0.72],[-0.3, -0.3, 0.72]]
    collision_orientations =  [[0, 0, 0.5],[0, 0, 0.5]]
    collision_scales = [0.2, 0.2]
    for i in range(len(collision_scales)):
        collision_ids.append(p.loadURDF("cube.urdf",
            basePosition=collision_positions[i],  
            baseOrientation=p.getQuaternionFromEuler(collision_orientations[i]),  
            globalScaling=collision_scales[i] 
        ))

    # Initializing Simulation
    p.setRealTimeSimulation(0)
    duration = 30
    fps = 30
    time_steps = int(duration * fps)
    dt = 1.0 / fps
    point_cloud_count = 1000

    # Display point cloud
    for step in range(time_steps):
        # Step simulation to let objects settle
        p.stepSimulation()

        # Obtain Point Cloud
        point_cloud, wRc, wtc = get_point_cloud()
        p.removeAllUserDebugItems()

        # Display Downsampled Point Cloud
        if len(point_cloud) > 0:
            if len(point_cloud) > point_cloud_count:
                downsampled_cloud = point_cloud[np.random.choice(len(point_cloud), point_cloud_count, replace=False)]
        p.addUserDebugPoints(downsampled_cloud, [[1, 0, 0]] * len(downsampled_cloud), pointSize=3)

        # Draw Camera Coordinate Frame
        if step%10 == 0:
            draw_camera_frame(wtc, wRc)

        time.sleep(1/fps)

    # Disconnect PyBullet
    p.disconnect()

