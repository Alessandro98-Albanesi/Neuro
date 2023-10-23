#------------------------------------------------------------------------------
# This script receives video from the HoloLens depth camera in long throw mode
# and plays it. The resolution is 320x288 @ 5 FPS. The stream supports three
# operating modes: 0) video, 1) video + rig pose, 2) query calibration (single 
# transfer). Depth and AB data are scaled for visibility. The ahat and long 
# throw streams cannot be used simultaneously.
# Press esc to stop. 
#------------------------------------------------------------------------------

from pynput import keyboard

import numpy as np
import cv2
import hl2ss_imshow
import hl2ss
import hl2ss_lnm
import pykinect_azure as pykinect
import open3d as o3d
import math
import hl2ss_3dcv

# Settings --------------------------------------------------------------------
FX_DEPTH = 175.33813
FY_DEPTH = 181.17589
CX_DEPTH = 165.09985
CY_DEPTH = 169.58142

# HoloLens address
host = "192.168.0.101"

# Operating mode
# 0: video
# 1: video + rig pose
# 2: query calibration (single transfer)
mode = hl2ss.StreamMode.MODE_1

# Framerate denominator (must be > 0)
# Effective framerate is framerate / divisor
divisor = 1 

# Depth encoding profile
profile_z = hl2ss.DepthProfile.SAME

# Video encoding profile
profile_ab = hl2ss.VideoProfile.H265_MAIN

#------------------------------------------------------------------------------

if (mode == hl2ss.StreamMode.MODE_2):
    data = hl2ss_lnm.download_calibration_rm_depth_longthrow(host, hl2ss.StreamPort.RM_DEPTH_LONGTHROW)
    print('Calibration data')
    print('Image point to unit plane')
    print(data.uv2xy)
    print('Extrinsics')
    print(data.extrinsics)
    print(f'Scale: {data.scale}')
    print('Undistort map')
    print(data.undistort_map)
    print('Intrinsics (undistorted only)')
    print(data.intrinsics)
    quit()

calibration = hl2ss_lnm.download_calibration_rm_depth_longthrow(host, hl2ss.StreamPort.RM_DEPTH_LONGTHROW)
enable = True

def on_press(key):
    global enable
    enable = key != keyboard.Key.esc
    return enable

listener = keyboard.Listener(on_press=on_press)
listener.start()

client = hl2ss_lnm.rx_rm_depth_longthrow(host, hl2ss.StreamPort.RM_DEPTH_LONGTHROW, mode=mode, divisor=divisor)
client.open()

def Compute_3D_blob_coord(keys,depth):

    key_3D = []
    depth_image_meters = depth/1
    for blob in keys:
       u = round(blob.pt[0])
       v = round(blob.pt[1])
       z = depth_image_meters[v][u] + 0.0054
       #print("{:.3f}".format(z)) 
       x = (u - CX_DEPTH) * z  / FX_DEPTH
       y = (v - CY_DEPTH) * z / FY_DEPTH
       #key_3D.append([x,y,z])
    
    #cleared_points = Clear_blobs(key_3D, 0.14)
    #print(cleared_points)
    #pcd = o3d.geometry.PointCloud()  # create point cloud object
    #pcd.points = o3d.utility.Vector3dVector(key_3D)  # set pcd_np as the point cloud points   
    #vis.clear_geometries()
    #vis.add_geometry(pcd)
    #vis.update_geometry(pcd)
    #vis.poll_events()
    #vis.update_renderer()   
    return key_3D

def Kabsch_Algorithm (A,B):

    
    N = A.shape[1]
    T = np.array([[0.0,0.0,0.0,0.0],
                [0.0,0.0,0.0,0.0],
                [0.0,0.0,0.0,0.0],
                [0.0,0.0,0.0,1]])
   
    # calculate centroids
    A_centroid = np.reshape(1/N * (np.sum(A, axis=1)), (3,1))
    B_centroid = np.reshape(1/N * (np.sum(B, axis=1)), (3,1))
    
    # calculate the vectors from centroids
    A_prime = A - A_centroid
    B_prime = B - B_centroid
    
    # rotation estimation
    H = np.zeros([3, 3])
    for i in range(N):
        ai = A_prime[:, i]
        bi = B_prime[:, i]
        H = H + np.outer(ai, bi)
    U, S, V_transpose = np.linalg.svd(H)
    V = np.transpose(V_transpose)
    U_transpose = np.transpose(U)
    R = V @ np.diag([1, 1, np.linalg.det(V) * np.linalg.det(U_transpose)]) @ U_transpose
    
    # translation estimation
    t = B_centroid - R @ A_centroid
    T[0:3,0:3] = R
    T[0][-1] = t[0]
    T[1][-1] = t[1]
    T[2][-1] = t[2]

    return T



def Blob_detector(im,depth,detector):
    
    # Detect blobs.
    im_conv = hl2ss_3dcv.rm_depth_to_uint8(im)
    im_conv = hl2ss_3dcv.rm_depth_undistort(im_conv,calibration.undistort_map)
    depth_normal = hl2ss_3dcv.rm_depth_normalize(depth, calibration.scale)
    #depth_undistort = hl2ss_3dcv.rm_depth_undistort(depth_normal,calibration.undistort_map)
    depth_conv = hl2ss_3dcv.rm_depth_to_uint8(depth_normal)   
    ret,im_conv_treshold = cv2.threshold(im_conv, 40, 255, cv2.THRESH_BINARY)

    
    keypoints = detector.detect(im_conv_treshold)
    
    for i, keypoint in enumerate(keypoints):
        x, y = int(keypoint.pt[0]), int(keypoint.pt[1])
        cv2.putText(im_conv_treshold, str(i+1), (x, y ), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 2)  
    
    im_with_keypoints = cv2.drawKeypoints(im_conv_treshold, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    depth_with_keypoints = cv2.drawKeypoints(depth_conv, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    
    cv2.imshow("Keypoints", im_with_keypoints)
    cv2.imshow("Keypoints_depth", depth_with_keypoints)
    
    Camera_coord = Compute_3D_blob_coord(keypoints,depth_normal)

    return Camera_coord

   

# Set up the SimpleBlobDetector with default parameters
params = cv2.SimpleBlobDetector_Params()

# Set the threshold
params.minThreshold = 5
params.maxThreshold = 200

# Set the area filter
params.filterByArea = True
params.minArea = 0.5
params.maxArea = 500
# Set the circularity filter
params.filterByCircularity = True
params.minCircularity = 0.1
params.maxCircularity = 1

# Set the convexity filter
params.filterByConvexity = False
params.minConvexity = 0.6
params.maxConvexity = 1

# Set the inertia filter
params.filterByInertia = False
params.minInertiaRatio = 0.01
params.maxInertiaRatio = 1

# Set the color filter
params.filterByColor = True
params.blobColor = 255


# Create a detector with the parameters
detector = cv2.SimpleBlobDetector_create(params)    
#vis = o3d.visualization.VisualizerWithKeyCallback()
#vis.create_window('viewer', 512, 512)

while (enable):
    data = client.get_next_packet()
    
    #print(f'Pose at time {data.timestamp}')
    #print(data.pose)
    #cv2.imshow('Depth', data.payload.depth / np.max(data.payload.depth)) # Scaled for visibility
    #cv2.imshow('AB', data.payload.ab / np.max(data.payload.ab)) # Scaled for visibility

    Blob_detector(data.payload.ab,data.payload.depth,detector)
    Camera_frame_tool = Blob_detector(data.payload.ab,data.payload.depth,detector)
    print(Camera_frame_tool)
    cv2.waitKey(1)

client.close()
listener.join()