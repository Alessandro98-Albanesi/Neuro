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

# Settings --------------------------------------------------------------------
FX_DEPTH = 113.92193
FY_DEPTH = 114.5772
CX_DEPTH = 258.27924
CY_DEPTH = 257.61118

# HoloLens address
host = "192.168.0.101"

# Operating mode
# 0: video
# 1: video + rig pose
# 2: query calibration (single transfer)
mode = hl2ss.StreamMode.MODE_0

# Framerate denominator (must be > 0)
# Effective framerate is framerate / divisor
divisor = 1 

# Depth encoding profile
profile_z = hl2ss.DepthProfile.SAME

# Video encoding profile
profile_ab = hl2ss.VideoProfile.H265_MAIN

#------------------------------------------------------------------------------

if (mode == hl2ss.StreamMode.MODE_2):
    data = hl2ss_lnm.download_calibration_rm_depth_ahat(host, hl2ss.StreamPort.RM_DEPTH_AHAT)
    print('Calibration data')
    print('Image point to unit plane')
    print(data.uv2xy)
    print('Extrinsics')
    print(data.extrinsics)
    print(f'Scale: {data.scale}')
    print(f'Alias: {data.alias}')
    print('Undistort map')
    print(data.undistort_map)
    print('Intrinsics (undistorted only)')
    print(data.intrinsics)
    quit()

enable = True

def on_press(key):
    global enable
    enable = key != keyboard.Key.esc
    return enable

listener = keyboard.Listener(on_press=on_press)
listener.start()

client = hl2ss_lnm.rx_rm_depth_longthrow(host, hl2ss.StreamPort.RM_DEPTH_LONGTHROW, mode=mode, divisor=divisor)
client.open()

# Define a function to calculate the Euclidean distance between two points
def euclidean_distance(point1, point2):
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

# Define a function to remove points with distances greater than a threshold
def Clear_blobs(keys, threshold):
    remaining_points = []
    for i, point1 in enumerate(keys):
        far = True
        for j, point2 in enumerate(keys):
            if i != j:  # Avoid comparing the point to itself
                distance = euclidean_distance(point1, point2)
                if distance <= threshold:
                    far = False
                    break
        if not far:
            remaining_points.append(point1)
    return remaining_points
    


def Compute_3D_blob_coord(keys,im,depth):
    key_3D = []
    depth_image_meters = depth / 1000
    for blob in keys:
       j = round(blob.pt[0])
       i = round(blob.pt[1])
       z = depth_image_meters[i][j]
       x = (j - CX_DEPTH) * z / FX_DEPTH
       y = (i - CY_DEPTH) * z / FY_DEPTH
       key_3D.append([x, -y, z])
    
    cleared_points = Clear_blobs(key_3D, 0.14)
    print(cleared_points)
    pcd = o3d.geometry.PointCloud()  # create point cloud object
    pcd.points = o3d.utility.Vector3dVector(cleared_points)  # set pcd_np as the point cloud points   
    vis.clear_geometries()
    vis.add_geometry(pcd)
    vis.update_geometry(pcd)
    vis.poll_events()
    vis.update_renderer()   
    return key_3D


def Blob_detector(im,depth,detector):
    
    # Detect blobs.
    im_conv = (im/256).astype('uint8')
    depth_conv = (depth/256).astype('uint8')
    
    print(depth.shape)
    print(im_conv.shape)
    #cv2.imshow("Keypoints", im)
    ret,im_conv_treshold = cv2.threshold(im_conv,41,255,cv2.THRESH_BINARY)
    
    cv2.imshow("Tresh", im_conv_treshold)
    keypoints = detector.detect(im_conv_treshold)
   
    #cv2.imshow("Keypoints", im_conv)
    
    
    # Draw detected blobs as red circles.
    # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
    im_with_keypoints = cv2.drawKeypoints(im_conv_treshold, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    depth_with_keypoints = cv2.drawKeypoints(depth_conv, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    
    cv2.imshow("Keypoints", im_with_keypoints)
    cv2.imshow("Keypoints_depth", depth_with_keypoints)
    
    #Compute_3D_blob_coord(keypoints,im,depth)

# Set up the SimpleBlobDetector with default parameters
params = cv2.SimpleBlobDetector_Params()

# Set the threshold
params.minThreshold = 10
params.maxThreshold = 300

# Set the area filter
params.filterByArea = True
params.minArea = 0.2
params.maxArea = 80

# Set the circularity filter
params.filterByCircularity = True
params.minCircularity = 0.2
params.maxCircularity = 1

# Set the convexity filter
params.filterByConvexity = True
params.minConvexity = 0.1
params.maxConvexity = 1

# Set the inertia filter
params.filterByInertia = True
params.minInertiaRatio = 0.1
params.maxInertiaRatio = 1

# Set the color filter
params.filterByColor = True
params.blobColor = 255


# Create a detector with the parameters
detector = cv2.SimpleBlobDetector_create(params)    
vis = o3d.visualization.VisualizerWithKeyCallback()
vis.create_window('viewer', 512, 512)

while (enable):
    data = client.get_next_packet()
    
    print(f'Pose at time {data.timestamp}')
    print(data.pose)
    cv2.imshow('Depth', data.payload.depth / np.max(data.payload.depth)) # Scaled for visibility
    cv2.imshow('AB', data.payload.ab / np.max(data.payload.ab)) # Scaled for visibility
    
    Blob_detector(data.payload.ab,data.payload.depth,detector)
    cv2.waitKey(1)

client.close()
listener.join()