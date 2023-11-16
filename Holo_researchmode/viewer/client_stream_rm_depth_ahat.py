#------------------------------------------------------------------------------
# This script receives video from the HoloLens depth camera in ahat mode and 
# plays it. The resolution is 512x512 @ 45 FPS. The stream supports three 
# operating modes: 0) video, 1) video + rig pose, 2) query calibration (single 
# transfer). Depth and AB data are scaled for visibility. The ahat and long 
# throw streams cannot be used simultaneously.
# Press esc to stop.
# See https://github.com/jdibenes/hl2ss/tree/main/extensions before setting
# profile_z to hl2ss.DepthProfile.ZDEPTH (lossless* compression).
#------------------------------------------------------------------------------

from pynput import keyboard
import numpy as np
import cv2
import hl2ss_imshow
import hl2ss
import hl2ss_lnm
import hl2ss_rus
import hl2ss_3dcv
import pykinect_azure as pykinect
import open3d as o3d
import math
import socket
import pickle
import json
import time
import imutils
import itertools
import matplotlib.pyplot as plt
from tracker import Tracker


# Settings --------------------------------------------------------------------
FX_DEPTH = 113.92193
FY_DEPTH = 114.5772
CX_DEPTH = 258.27924
CY_DEPTH = 257.61118

Obj_frame_tool = [[0,0,0], [0,-0.045,-0.05],[0,-0.09,-0.025], [0,-0.14,-0.025]]
Obj_frame_verification = [[0,0,0,1], [0,-0.045,-0.05,1],[0,-0.09,-0.025,1], [0,-0.14,-0.025,1]]
T_world_object = np.identity(4)
temporal_array = []

# HoloLens address
host = "192.168.227.235"

# Create a socket server
#server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
host2 = "127.0.0.1"  # Server IP address
port = 3816  # Port number





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

calibration = hl2ss_lnm.download_calibration_rm_depth_ahat(host, hl2ss.StreamPort.RM_DEPTH_AHAT)
T_camera_rig = calibration.extrinsics
T_camera_rig[:3,-1] = T_camera_rig[-1,:3]
T_camera_rig[-1,:3] = 0

print("T_cam_rig", T_camera_rig)

lookup_table = hl2ss_3dcv.compute_uv2xy(calibration.intrinsics, 512, 512)

enable = True

def on_press(key):
    global enable
    enable = key != keyboard.Key.esc
    return enable

listener = keyboard.Listener(on_press=on_press)
listener.start()

client = hl2ss_lnm.rx_rm_depth_ahat(host, hl2ss.StreamPort.RM_DEPTH_AHAT, mode=mode, divisor=divisor, profile_z=profile_z, profile_ab=profile_ab)
client.open()

#sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#sock.connect((host2, port))
#print(f"Connection from {sock}")







def KF_marker(coordinate,image,KF):

    # Predict
    (x, y) = KF.predict()
    # Draw a rectangle as the predicted object position
    cv2.rectangle(image, (int(x- 15), int(y- 15)), (int(x+ 15), int(y+ 15)), (0,0,255), 2)

    # Update
    (x1, y1) = KF.update(coordinate[0])
    print(x1, y1)
    # Draw a rectangle as the estimated object position
    cv2.rectangle(image, (int(x1- 15), int(y1- 15)), (int(x1+ 15), int(y1+ 15)), (0,255,0), 2)

    cv2.putText(image, "Estimated Position", (int(x1 + 15), int(y1 + 10)), 0, 0.5, (0,255,0), 2)
    cv2.putText(image, "Predicted Position", (int(x + 15), int(y + 10 )), 0, 0.5, (0,0,255), 2)
    cv2.putText(image, "Measured Position", (int(coordinate[0][0] + 15), int(coordinate[1][0] - 15)), 0, 0.5, (0,0,255), 2)

    return x1, y1

def _fitzpatricks_X(svd):
    """This is from Fitzpatrick, chapter 8, page 470.
       it's used in preference to Arun's equation 13,
       X = np.matmul(svd[2].transpose(), svd[0].transpose())
       to avoid reflections.
    """
    VU = np.matmul(svd[2].transpose(), svd[0])
    detVU = np.linalg.det(VU)

    diag = np.eye(3, 3)
    diag[2][2] = detVU

    X = np.matmul(svd[2].transpose(), np.matmul(diag, svd[0].transpose()))
    return X


def Compute_3D_blob_coord(keys,depth):
    key_3D = []
    
    for blob in keys:
       u = int(blob.pt[0])
       v = int(blob.pt[1])
       z = depth[v][u]
       #x = (u - CX_DEPTH) * z  / FX_DEPTH
       #y = (v - CY_DEPTH) * z / FY_DEPTH
       xy = lookup_table[u][v]
       XYZ = [xy[0]*z,xy[1]*z,z]/np.linalg.norm([xy[0],xy[1],1])
       XYZ_sphere = (XYZ *(1 + 0.0054/z)).tolist()
       #print("point from function", [x,y,z])
       key_3D.append(XYZ_sphere)
       #print(key_3D)
    
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
    H = np.matmul(A_prime, B_prime.transpose())
    svd = np.linalg.svd(H)

    # Replace Arun Equation 13 with Fitzpatrick, chapter 8, page 470,
    # to avoid reflections, see issue #19
    X = _fitzpatricks_X(svd)

    # Arun step 5, after equation 13.
    det_X = np.linalg.det(X)

    if det_X < 0 and np.all(np.flip(np.isclose(svd[1], np.zeros((3, 1))))):

        # Don't yet know how to generate test data.
        # If you hit this line, please report it, and save your data.
        raise ValueError("Registration fails as determinant < 0"
                         " and no singular values are close enough to zero")

    if det_X < 0 and np.any(np.isclose(svd[1], np.zeros((3, 1)))):
        # Implement 2a in section VI in Arun paper.
        v_prime = svd[2].transpose()
        v_prime[0][2] *= -1
        v_prime[1][2] *= -1
        v_prime[2][2] *= -1
        X = np.matmul(v_prime, svd[0].transpose())

    # Compute output
    R = X
    t = B_centroid - R @ A_centroid
    T[0:3,0:3] = R
    T[0][-1] = t[0]
    T[1][-1] = t[1]
    T[2][-1] = t[2]
    return R, t, T


def Brute_force_matching(Measured_points):
        
    permuted_list = list(itertools.permutations(Measured_points))
    Y = np.transpose(np.array(Obj_frame_tool))
    min_err = 1000
    for iter in range(len(permuted_list)):
     
        P = np.transpose(np.array(permuted_list[iter]))
        #print(P)
        Rot, Transl, T = Kabsch_Algorithm (P,Y)
        error = np.linalg.norm(Rot @ P + Transl - Y, 'fro')
        
        if error < min_err:
            min_err = error
            match = P
            match_R = Rot
            match_t = Transl
            T_final = T
    
    #print(min_err)
    return T_final
        


def Blob_detector(im,depth,detector):
    
    # Detect blobs.
    
    im_conv = hl2ss_3dcv.rm_depth_undistort(im,calibration.undistort_map)
    im_conv = hl2ss_3dcv.rm_depth_to_uint8(im_conv)
    im_conv = cv2.normalize(im_conv, 100 ,200, cv2.NORM_MINMAX)
    im_conv = cv2.GaussianBlur(im_conv,(5,5),0)
    ret,im_conv_treshold = cv2.threshold(im_conv, 4, 255, cv2.THRESH_BINARY)
    #kernel = np.ones((3,3),np.uint8)
    #im_conv = cv2.dilate(im_conv_treshold, kernel, iterations = 1)
    

    depth_normal = hl2ss_3dcv.rm_depth_normalize(depth, calibration.scale)
    depth_undistort = hl2ss_3dcv.rm_depth_undistort(depth_normal,calibration.undistort_map)
    depth_conv = hl2ss_3dcv.rm_depth_to_uint8(depth_undistort)   
    
   
    
    keypoints = detector.detect(im_conv_treshold)

    # Draw the detected circle
    im_with_keypoints = cv2.drawKeypoints(im_conv_treshold, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    centers = []
    corrected_keypoint = []
    
    for i, keypoint in enumerate(keypoints):
        centers.append(np.array([[keypoint.pt[0]], [keypoint.pt[1]]]))
        

    if (len(centers) > 0):
        # Track object using Kalman Filter
        tracker.Update(centers)
        
        # For identified object tracks draw tracking line
        # Use various colors to indicate different track_id
        for i in range(len(tracker.tracks)):
            if (len(tracker.tracks[i].trace) > 1):
                for j in range(len(tracker.tracks[i].trace)-1):

                    #print("measured",(centers[i]))
                    # Draw trace line
                    print("updated" + str(j),(tracker.tracks[i].trace[0][0][0],tracker.tracks[i].trace[0][1][0]))
                    x1 = tracker.tracks[i].trace[j][0][0]
                    y1 = tracker.tracks[i].trace[j][1][0]
                    x2 = tracker.tracks[i].trace[j+1][0][0]
                    y2 = tracker.tracks[i].trace[j+1][1][0]
                    clr = tracker.tracks[i].track_id % 9
                    cv2.line(im_with_keypoints, (int(x1), int(y1)), (int(x2), int(y2)),5)
                    cv2.rectangle(im_with_keypoints, (int(x1 - 15), int(y1 - 15)), (int(x2 + 15), int(y2 + 15)), (255, 0, 0), 2)



    cv2.imshow('image', im_with_keypoints)

        #im_with_keypoints = cv2.drawKeypoints(im_conv_treshold, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        #depth_with_keypoints = cv2.drawKeypoints(depth_conv, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        #cv2.imshow("Keypoints", im_with_keypoints)
        #cv2.imshow("Keypoints_depth", depth_with_keypoints)

    Camera_coord = Compute_3D_blob_coord(keypoints,depth_undistort)

    return Camera_coord
    
# Set up the SimpleBlobDetector with default parameters
params = cv2.SimpleBlobDetector_Params()

# Set the threshold
params.minThreshold = 100
params.maxThreshold = 500

# Set the area filter
params.filterByArea = True
params.minArea = 0.1
params.maxArea = 200
# Set the circularity filter
params.filterByCircularity = True
params.minCircularity = 0.1
params.maxCircularity = 100

# Set the convexity filter
params.filterByConvexity = True
params.minConvexity = 0.1
params.maxConvexity = 30

# Set the inertia filter
params.filterByInertia = False
params.minInertiaRatio = 0.1
params.maxInertiaRatio = 1

# Set the color filter
params.filterByColor = False
params.blobColor = 255


# Create a detector with the parameters
detector = cv2.SimpleBlobDetector_create(params)

tracker = Tracker(160, 30, 5, 100)

while (enable):

    data = client.get_next_packet()

    T_world_rig = data.pose
    T_world_rig[:3,-1] = T_world_rig[-1,:3]
    T_world_rig[-1,:3] = 0
    #\print("T_world_rig", T_world_rig)
    #cv2.imshow('Depth', data.payload.depth / np.max(data.payload.depth)) # Scaled for visibility
    cv2.imshow('AB', data.payload.ab / np.max(data.payload.ab)) # Scaled for visibility
    
    Camera_frame_tool = Blob_detector(data.payload.ab,data.payload.depth,detector)
    #temporal_array.append(Camera_frame_tool)
    #print(Camera_frame_tool)

    if(len(Camera_frame_tool) == 4):

        T_obj_camera = Brute_force_matching(np.array(Camera_frame_tool))
        T_world_object = T_world_rig @ T_camera_rig @ np.linalg.inv(T_obj_camera)
        T_world_object = np.round(T_world_object, decimals = 2)
        #print(T_world_object)
    # Convert to a left-handed frame matrix by negating the Z-component (third column)
    

    matrixString = '\n'.join([','.join(map(str, row)) for row in T_world_object])

    #print(matrixString)

    #sock.sendall(matrixString.encode("UTF-8"))


    cv2.waitKey(1)

#print(temporal_array)
client.close()
listener.join()
