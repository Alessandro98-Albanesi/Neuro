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
from open3d import *
import math
import socket
import pickle
import json
import time
import imutils
import itertools
import matplotlib.pyplot as plt
from tracker import Tracker
import pyvista as pv
import keyboard



'''
# Settings ahat
FX_DEPTH = 115.40451
FY_DEPTH = 116.04211
CX_DEPTH = 255.91125
CY_DEPTH = 259.48816
'''

# Settings depth
FX_DEPTH = 173.25438 
FY_DEPTH = 181.43912
CX_DEPTH = 161.41142
CY_DEPTH = 174.2087 



# HoloLens address
host = "192.168.0.102"

# Create a socket server

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
    #print(f'Alias: {data.alias}')
    print('Undistort map')
    print(data.undistort_map)
    print('Intrinsics (undistorted only)')
    print(data.intrinsics)
    quit()

calibration = hl2ss_lnm.download_calibration_rm_depth_longthrow(host, hl2ss.StreamPort.RM_DEPTH_LONGTHROW)

lookup_table = hl2ss_3dcv.compute_uv2xy(calibration.intrinsics, 320, 288)

enable = True

T_camera_rig = calibration.extrinsics
T_camera_rig[:3,-1] = T_camera_rig[-1,:3]
T_camera_rig[-1,:3] = 0


'''
def on_press(key):
    global enable
    enable = key != keyboard.Key.esc
    return enable

listener = keyboard.Listener(on_press=on_press)
listener.start()
'''

client = hl2ss_lnm.rx_rm_depth_longthrow(host, hl2ss.StreamPort.RM_DEPTH_LONGTHROW, mode=mode, divisor=divisor)
client.open()

#sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#sock.connect((host2, port))
#print(f"Connection from {sock}")



def Compute_3D_coord(keys,depth):
    key_3D = []
    
    for blob in keys:
        
        u = int(blob.pt[0])
        v = int(blob.pt[1])
        depth = depth/1
        z = depth[v][u]
        xy = lookup_table[u][v]
        XYZ = [xy[0]*z,xy[1]*z,z]
        #x = (u - CX_DEPTH) * z  / FX_DEPTH
        #y = (v - CY_DEPTH) * z / FY_DEPTH
        #XYZ_sphere = (XYZ *(1 + 0.0054/z)).tolist()
        #print("point from function", [x,y,z])
        key_3D.append([XYZ])
    print(key_3D)
    
    return key_3D

def clusterObjManich(pcd, epsValue):
    with o3d.utility.VerbosityContextManager(
            o3d.utility.VerbosityLevel.Debug) as mm:
        labels = np.array(
            pcd.cluster_dbscan(eps=epsValue, min_points=20, print_progress=True))

    max_label = labels.max()
    print(f"point cloud has {max_label + 1} clusters")
    colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
    colors[labels < 0] = 0
    pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
    o3d.visualization.draw_geometries([pcd])
    points = np.asarray(pcd.points)
    listCluster=[]
    nPt_cluster_i=np.zeros(max_label+1)
    for i in range (0, max_label+1):
        cluster_i=points[labels == i]
        listCluster.append(cluster_i)
        nPt_cluster_i[i]=cluster_i.shape[0]

    clusterMaxind= np.argmax(nPt_cluster_i)

    return listCluster[clusterMaxind]


def Blob_detector(im,depth,detector):
    
    # Detect blobs.
    
    im_conv = hl2ss_3dcv.rm_depth_to_uint8(im)
    depth_conv = hl2ss_3dcv.rm_depth_to_uint8(depth)
    ret,im_conv_treshold = cv2.threshold(im_conv,40,255,cv2.THRESH_BINARY)
    cv2.imshow("Tresh", im_conv_treshold)
    
    keypoints = detector.detect(im_conv_treshold)
    
    cv2.imshow("Keypoints", im_conv)
    
    
    # Draw detected blobs as red circles.
    # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
    im_with_keypoints = cv2.drawKeypoints(im_conv_treshold, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    depth_with_keypoints = cv2.drawKeypoints(depth_conv, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    for blob in keypoints:
        cv2.circle(im_with_keypoints, (int(blob.pt[0]),int(blob.pt[1])), 1, (0,255,0), -1)

    cv2.imshow("Keypoints", im_with_keypoints)
    cv2.imshow("Keypoints_depth", depth_with_keypoints)


    #
    print(Compute_3D_coord(keypoints,depth))  


# Set up the SimpleBlobDetector with default parameters
params = cv2.SimpleBlobDetector_Params()

# Set the threshold
params.minThreshold = 40
params.maxThreshold = 500

# Set the area filter
params.filterByArea = True
params.minArea = 0.1
params.maxArea = 50
# Set the circularity filter
params.filterByCircularity = True
params.minCircularity = 0.1
params.maxCircularity = 1

# Set the convexity filter
params.filterByConvexity = False
params.minConvexity = 0.1
params.maxConvexity = 30

# Set the inertia filter
params.filterByInertia = True
params.minInertiaRatio = 0.1
params.maxInertiaRatio = 1

# Set the color filter
params.filterByColor = True
params.blobColor = 255


# Create a detector with the parameters
detector = cv2.SimpleBlobDetector_create(params)

while(enable):

    
    
    keyboard.wait("ENTER")
    '''
    data = client.get_next_packet()
    data_depth = hl2ss_3dcv.rm_depth_undistort(data.payload.depth,calibration.undistort_map)
    data_IR = hl2ss_3dcv.rm_depth_undistort(data.payload.ab,calibration.undistort_map)
    
    #cv2.imshow("AB" , data.payload.depth/ np.max(data.payload.depth)) # Scaled for visibility
    #depth_normal = hl2ss_3dcv.rm_depth_normalize(data.payload.ab, calibration.scale)
    #depth_undistort = hl2ss_3dcv.rm_depth_undistort(depth_normal,calibration.undistort_map)
    #points = Compute_3D_coord(data)
    cv2.imshow('AB', data.payload.ab / np.max(data.payload.ab))
    
    Blob_detector(data_IR,data_depth,detector)
    
    cv2.waitKey(1)
    '''
    data = client.get_next_packet()
    data = data.payload.depth
    #data = hl2ss_3dcv.rm_depth_undistort(data,calibration.undistort_map)
    o3d_depth_image = o3d.geometry.Image(data)
    
    intrinsic = o3d.camera.PinholeCameraIntrinsic()
    intrinsic.set_intrinsics(320, 288, FX_DEPTH, FY_DEPTH, CX_DEPTH, CY_DEPTH)

    # Create a point cloud from the depth image
    point_cloud = o3d.geometry.PointCloud.create_from_depth_image(
        o3d_depth_image,
        intrinsic,
        T_camera_rig,
        depth_scale=1000.0,  # Adjust based on your specific depth values
        #depth_trunc=0.5,  # Adjust based on your specific depth values
    )
    
    print(np.asarray(point_cloud.points))
    min_bound = np.array([0.0,-math.inf,-math.inf])
    max_bound = np.array([math.inf, math.inf, 1])
    
    inlier_indices = np.all((min_bound <= point_cloud.points) & (point_cloud.points <= max_bound), axis=1)
    cropped_point_cloud = point_cloud.select_by_index(np.where(inlier_indices)[0].tolist())
    
    # Remove statistical outliers
    cl, ind = cropped_point_cloud.remove_statistical_outlier(nb_neighbors=20, std_ratio=1) 
    filtered_pc = cropped_point_cloud.select_by_index(ind)
    #filtered_pc = filtered_pc.voxel_down_sample(voxel_size=0.001)
    clust = clusterObjManich(filtered_pc, 0.01)
    #cropped_point_cloud.estimate_normals()
    #mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(cropped_point_cloud)

    #o3d.io.write_triangle_mesh("C:/Users/Alessandro/Desktop/Neuro/pcl_holo.obj", mesh)
    
    clustered_cloud = o3d.geometry.PointCloud()
    clustered_cloud.points = o3d.utility.Vector3dVector(clust)
    o3d.io.write_point_cloud("C:/Users/Alessandro/Desktop/Neuro/pcl_holo.ply",clustered_cloud)
    
    pyvista_cloud = pv.PolyData(np.asarray(clustered_cloud.points))
    pyvista_cloud.plot(point_size=1, color="red")

    
    

    #points = np.asarray(point_cloud.points)
    #pyvista_cloud = pv.PolyData(clust)
    #pyvista_cloud.plot(point_size=1, color="red")

    cv2.waitKey(1)
    #time.sleep(0.1)
   
#print(temporal_array)
client.close()
listener.join()
