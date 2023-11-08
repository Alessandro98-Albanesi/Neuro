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
import pyvista as pv


# Settings --------------------------------------------------------------------
FX_DEPTH = 113.92193
FY_DEPTH = 114.5772
CX_DEPTH = 258.27924
CY_DEPTH = 257.61118


# HoloLens address
host = "192.168.0.101"

# Create a socket server

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
    #print(f'Alias: {data.alias}')
    print('Undistort map')
    print(data.undistort_map)
    print('Intrinsics (undistorted only)')
    print(data.intrinsics)
    quit()

calibration = hl2ss_lnm.download_calibration_rm_depth_ahat(host, hl2ss.StreamPort.RM_DEPTH_AHAT)

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



def Compute_3D_coord(depth):
    key_3D = []
    
    for point_u in range(320):
       for point_v in range(288):
       
        z = depth[point_v][point_u]
        #x = (u - CX_DEPTH) * z  / FX_DEPTH
        #y = (v - CY_DEPTH) * z / FY_DEPTH
        xy = lookup_table[point_u][point_v]
        XYZ = [xy[0]*z,xy[1]*z,z]
        #XYZ_sphere = (XYZ *(1 + 0.0054/z)).tolist()
        #print("point from function", [x,y,z])
        key_3D.append(XYZ)
    print(key_3D)
    
    return key_3D

def clusterObjManich(pcd, epsValue):
    with o3d.utility.VerbosityContextManager(
            o3d.utility.VerbosityLevel.Debug) as cm:
        labels = np.array(
            pcd.cluster_dbscan(eps=epsValue, min_points=10, print_progress=True))

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

while(True):

    data = client.get_next_packet()
    data = hl2ss_3dcv.rm_depth_normalize(data.payload.depth, calibration.scale)
    data = hl2ss_3dcv.rm_depth_undistort(data,calibration.undistort_map)
    #cv2.imshow("AB" , data.payload.depth/ np.max(data.payload.depth)) # Scaled for visibility
    #depth_normal = hl2ss_3dcv.rm_depth_normalize(data.payload.ab, calibration.scale)
    #depth_undistort = hl2ss_3dcv.rm_depth_undistort(depth_normal,calibration.undistort_map)
    #points = Compute_3D_coord(data)


    o3d_depth_image = o3d.geometry.Image(data)
    intrinsic = o3d.camera.PinholeCameraIntrinsic()
    intrinsic.set_intrinsics(512, 512, FX_DEPTH, FY_DEPTH, CX_DEPTH, CY_DEPTH)

    # Create a point cloud from the depth image
    point_cloud = o3d.geometry.PointCloud.create_from_depth_image(
        o3d_depth_image,
        intrinsic,
        depth_scale=1.0,  # Adjust based on your specific depth values
        depth_trunc=10,  # Adjust based on your specific depth values
    )
    
    clust = clusterObjManich(point_cloud, 0.01)
    print(clust)

    #points = np.asarray(point_cloud.points)
    pyvista_cloud = pv.PolyData(clust)
    pyvista_cloud.plot(point_size=1, color="red")

    cv2.waitKey(1)

#print(temporal_array)
client.close()
listener.join()