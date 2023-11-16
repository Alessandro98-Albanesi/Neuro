import socket
import time

import numpy as np
import array
import pickle
import open3d as o3d
import copy
import struct
import o3d_utils as o3dUtils
import math
from scipy.spatial.transform import Rotation as Rot
import pyvista as pv

#objPatient="C:/Users/palum/OneDrive/Desktop/Chiara/PhD/Hololens/Progetto_Idrocefalo/ValutazioneRegistrazione/1modello_112216_besta_circoV/skin_reduced_Metri.obj"
objPatient="C:/Users/Alessandro/Desktop/Neuro/face_3t_mWtextr.obj"
#Atlas used for the hidden point removal
objFile = "C:/Users/Alessandro/Desktop/Neuro/facciaTesta_metri2.obj"


def saveptRcv(ptRcvd, nameFile):
    with open(nameFile, "wb") as fp:  # Pickling
        file=copy.deepcopy(ptRcvd)
        pickle.dump(file, fp)
    return

def OpenptRcv(nameFile):
    with open(nameFile, "rb") as fp:  # Unpickling
       b = pickle.load(fp)
    return b


def hiddenPointRemoval(pcd):
    # Convert mesh to a point cloud and estimate dimensions.
    #pcd = o3d.io.read_triangle_mesh(TriangleMeshFile)
    pcd = pcd.sample_points_poisson_disk(20000)
    diameter = np.linalg.norm(np.asarray(pcd.get_max_bound()) - np.asarray(pcd.get_min_bound()))
    print("Displaying input point cloud ...")
    # o3d.visualization.draw([pcd], point_size=5)

    # Define parameters used for hidden_point_removal.
    originCamera = np.asarray([0, 0, 0])
    camera = [0, -diameter, 0]  # l'asse y è quello che punta verso la faccia, verificare che sia sempre così
    radius = diameter * 100

    # Get all points that are visible from given view point.
    _, pt_map = pcd.hidden_point_removal(camera, radius)

    print("Displaying point cloud after hidden point removal ...")
    pcd_withoutHidden = copy.deepcopy(pcd)
    pcd_withoutHidden = pcd_withoutHidden.select_by_index(pt_map)
    pcd_withoutHidden.paint_uniform_color([0, 0.706, 0])
    return pcd_withoutHidden

def workflow2RemoveHiddenPoint(objPatient,objFile):
    objPatient_PC = o3dUtils.loadObjfile_asPC(objPatient)
    objAtlas_PC = o3dUtils.loadObjfile_asPC(objFile)

    target = objAtlas_PC
    source = objPatient_PC
    voxel_size = 10  # means 10 cm for this dataset

    source_down, source_fpfh = o3dUtils.preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = o3dUtils.preprocess_point_cloud(target, voxel_size)


    result_ransac = o3dUtils.execute_global_registration(source_down, target_down,
                                                source_fpfh, target_fpfh,
                                                voxel_size)
    print(result_ransac)
    #draw_registration_result(source_down, target_down, result_ransac.transformation)

    result_icp = o3dUtils.refine_registration(source, target, source_fpfh, target_fpfh, voxel_size,
                                              result_ransac.transformation)
    print(result_icp)
    draw_registration_result(source, target, result_icp.transformation)
    source_Reg = copy.deepcopy(source)
    source_Reg.paint_uniform_color([1, 192 / 255, 203/255]) #Giallo
    source_Reg.transform(result_icp.transformation)
    source_Reg.estimate_normals()
    #o3d.visualization.draw_geometries([source_Reg])
    source_RegMesh=o3d.io.read_triangle_mesh(objPatient)
    source_RegMesh.transform(result_icp.transformation)
    pcd_withoutHiddenRotated = hiddenPointRemoval(source_RegMesh)
    # applico la trasformazione inversa in modo tale che l obj patient ritroni al suo orientamento iniziale
    pcd_withoutHiddenRotated.paint_uniform_color([1, 192 / 255, 203/255])  # viola
    #o3d.visualization.draw_geometries([pcd_withoutHiddenRotated])
    pcd_withoutHiddenCorrect = copy.deepcopy(pcd_withoutHiddenRotated)
    pcd_withoutHiddenCorrect = pcd_withoutHiddenCorrect.transform(np.linalg.inv(result_icp.transformation))
    pcd_withoutHiddenCorrect.paint_uniform_color([0.5, 0.5, 0])  # verde

    o3d.visualization.draw_geometries([pcd_withoutHiddenRotated, pcd_withoutHiddenCorrect, source, target])
    o3d.visualization.draw_geometries([pcd_withoutHiddenCorrect])
    return  pcd_withoutHiddenCorrect

def pcSmoothing(pcd):
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

    o3d.geometry.PointCloud.orient_normals_towards_camera_location(pcd, [0, 0, 0])
    radii = [0.005, 0.01, 0.02, 0.04]
    rec_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        pcd, o3d.utility.DoubleVector(radii))
    mesh_out = rec_mesh.filter_smooth_simple(number_of_iterations=7)
    mesh_out.compute_vertex_normals()
    mesh_out.scale(1.05, center=mesh_out.get_center())

    pcd_smooth=o3d.geometry.PointCloud()
    pcd_smooth.points=o3d.utility.Vector3dVector(np.asarray(mesh_out.vertices))

    return pcd_smooth

def saveData(ptRcv):
    numPt = int(ptRcv[0])
    pickle.dump(numPt, open("numPt.p", "wb"))

    posRot = []
    row = [ptRcv[1], ptRcv[2], ptRcv[3]]
    posRot.append(row)
    row2 = [ptRcv[4], ptRcv[5], ptRcv[6], ptRcv[7]]
    posRot.append(row2)
    pickle.dump(posRot, open("posRot.p", "wb"))

    pointCl = []
    for i in range(8, numPt + 6, 3):
        row = [ptRcv[i], ptRcv[i + 1], ptRcv[i + 2]]
        pointCl.append(row)
    pickle.dump(pointCl, open("pointCl.p", "wb"))
    return

def computeCluster(pcd):
    with o3d.utility.VerbosityContextManager(
            o3d.utility.VerbosityLevel.Debug) as cm:
        labels = np.array(
            pcd.cluster_dbscan(eps=0.0035, min_points=10, print_progress=True))

    max_label = labels.max()
    labelList = []
    for i in range(0, max_label + 1, 1):
        occurrences = np.count_nonzero(labels == i)
        print(i, "has :", occurrences)
        labelList.append(occurrences);

    max_index_col = np.argmax(np.asarray(labelList), axis=0)
    indici = np.in1d(labels, max_index_col)
    indici = np.nonzero(indici)
    indici = np.asarray(indici)

    cou = np.asarray(indici).size
    pcNew = [];
    for pt in indici[0, :]:
        pt1 = pcd.points[pt]
        pcNew.append(pt1)

    cluster1 = o3d.geometry.PointCloud()
    pcNew1 = np.asarray(pcNew)
    cluster1.points = o3d.utility.Vector3dVector(pcNew1)
    cluster1.paint_uniform_color([1, 0, 0])
    pcd.paint_uniform_color([0, 0.651, 0.929])
    #o3d.visualization.draw_geometries([cluster1, pcd])
    return cluster1


def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 192 / 255, 203/255]) #rosa
    target_temp.paint_uniform_color([0, 128/255, 1])
    #source.paint_uniform_color([1, 0,0])
    source_temp.transform(transformation)
    sdr = o3d.geometry.TriangleMesh.create_coordinate_frame()
    o3d.visualization.draw_geometries([source_temp])
    o3d.visualization.draw_geometries([target_temp])
    o3d.visualization.draw_geometries([source_temp, target_temp])

def preprocess_point_cloud(pcd, voxel_size):
    print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh

def prepare_dataset(voxel_size, source,target):
    print(":: Load two point clouds and disturb initial pose.")
    '''trans_init = np.asarray([[0.0, 0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 0.0],
                             [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])'''
    trans_init = np.identity(4)
    source.transform(trans_init)
    #draw_registration_result(source, target, np.identity(4))

    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)
    return source, target, source_down, target_down, source_fpfh, target_fpfh

def execute_fast_global_registration(source_down, target_down, source_fpfh,
                                     target_fpfh, voxel_size):
    distance_threshold = voxel_size * 0.5
    print(":: Apply fast global registration with distance threshold %.3f" \
            % distance_threshold)
    result = o3d.pipelines.registration.registration_fast_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh,
        o3d.pipelines.registration.FastGlobalRegistrationOption(
            maximum_correspondence_distance=distance_threshold))
    return result

def execute_global_registration(source_down, target_down, source_fpfh,
                                target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5 #1.5
    print(":: RANSAC registration on downsampled point clouds.")
    print("   Since the downsampling voxel size is %.3f," % voxel_size)
    print("   we use a liberal distance threshold %.3f." % distance_threshold)
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                0.1),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(10000, 0.999))
    return result

def refine_registration(source, target, source_fpfh, target_fpfh, voxel_size,trasf_init2):
    distance_threshold = voxel_size * 0.05  #0.01*0.05=0.0005m =0.5mm
    treshold= 0.01 #1cm
    print(":: Point-to-plane ICP registration is applied on original point")
    print("   clouds to refine the alignment. This time we use a strict")
    print("   distance threshold %.3f." % distance_threshold)
    result = o3d.pipelines.registration.registration_icp(
        source, target, treshold, trasf_init2,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    return result

def writeRtMat(Rt_Mat, send):
    # R = np.asarray(Rt_Mat[0:3, 0:3])  # convert the rototrnaslation matrix into position and rotation Euler angles
    # Rscip = Rot.from_matrix(R)  #NON VIENE USATO ALLA FINE si può togliere anche la libreria
    # Euler = Rscip.as_euler('xyz', degrees=True)
    # print("Transf Matrix=", np.asarray(Rt_Mat))
    # t = np.asarray(Rt_Mat[0:3, 3])
    # tnew = t.copy()
    # print("t first is: ", t)
    # print("Euler first is: ", Euler)
    #
    # tnew[0] = t[0] * (-1)  # invert the x and the rotation around y and z to pass into a LH sdr
    #
    # Rnew = np.asarray(Euler)
    # Rnew = Rnew.copy()
    #
    # #Rnew[0] = Rnew[0] * (-1)
    # Rnew[2] = Rnew[2] * (-1)
    # Rnew[1] = Rnew[1] * (-1)
    # print("t is: ", tnew)
    # print("Euler is: ", Rnew)
    #send = []

    '''#se voglio mandare t e Euler ###############################################
    send.extend(tnew)  # send position and rotation back to hololens
    send.extend(Rnew)
    ##########################################################################'''

    #Sevoglio mandare l intera matrice di rototraslazione
    send.extend(np.asarray(Rt_Mat[0,:]))
    send.extend(np.asarray(Rt_Mat[1, :]))
    send.extend(np.asarray(Rt_Mat[2, :]))
    send.extend(np.asarray(Rt_Mat[3, :]))

    send_ = np.asarray(send, dtype=float)
    print("determinante=", np.linalg.det(Rt_Mat))

    #send_[0:6]=0
    data = struct.pack('f' * len(send_), *send_)
    conn.sendall(data)
    return data

def comp_Rt_mat(ptRcv, mesh):
    startTime=time.time()
    #numPt = int(ptRcv[0])
    numPt = int(ptRcv.shape[0])
    posRot = []
    row = [ptRcv[1], ptRcv[2], ptRcv[3]]
    posRot.append(row)
    row2 = [ptRcv[4], ptRcv[5], ptRcv[6], ptRcv[7]]
    posRot.append(row2)

    pointCl = []
    for i in range(8, numPt + 6, 3):
        row = [ptRcv[i], ptRcv[i + 1], ptRcv[i + 2]]
        pointCl.append(row)

    #"opening mesh obj")
    #mesh = o3d.io.read_triangle_mesh("segmnet1_metri_reduced2.obj")
    #mesh = o3d.io.read_triangle_mesh("pt1Ametri_cut_085_3red.obj")
    #questa in realtà è gia un point cloud
    #mesh = o3d.io.read_triangle_mesh("C:/Users/palum/OneDrive/Desktop/Chiara/PhD/Hololens/Progetto_Idrocefalo/ValutazioneRegistrazione/1modello_112216_besta_circoV/skin_reduced_Metri.obj")
    #"Computing normal and rendering it.")
    #mesh.compute_vertex_normals()

    posRot_ar = np.array(posRot)

    ''''#Rotate and translate the obj mesh with the position and rotation given from unity
    #pay attention: x  rotY rotZ multiply by -1
    mesh_r = copy.deepcopy(mesh)
    R = mesh.get_rotation_matrix_from_xyz(
        (np.pi * posRot_ar[1, 0] / 180, np.pi * posRot_ar[1, 1] / 180 * -1, np.pi * posRot_ar[1, 2] / 180 * -1))
    mesh_r.rotate(R, center=(0, 0, 0))
    mesh_r = copy.deepcopy(mesh_r).translate((posRot_ar[0, 0] * -1, posRot_ar[0, 1], posRot_ar[0, 2]))'''


    # Rotate and translate the obj mesh with the position and rotation given from unity
    # pay attention: x  rotY rotZ  multiply by -1
    mesh_r = copy.deepcopy(mesh)
    #R = mesh.get_rotation_matrix_from_xyz((np.pi * posRot_ar[1, 0] / 180, np.pi * posRot_ar[1, 1] / 180 , np.pi * posRot_ar[1, 2] / 180 ))
    #R = mesh.get_rotation_matrix_from_xyz(
    #    (np.pi * (posRot_ar[1, 0]) / 180, np.pi * (posRot_ar[1, 1]) / -180, np.pi * (posRot_ar[1, 2]) / -180))
    #da unity dovrei ricevere w x y z quaternion

    #R = mesh.get_rotation_matrix_from_quaternion(np.asarray(posRot_ar[1][0:4]))
    quaternion=np.asarray(posRot_ar[1][0:4])
    quaternion2=[quaternion[0],quaternion[1],quaternion[2]*-1,quaternion[3]*-1]
    #quaternion2=[quaternion[1]*-1,quaternion[2],quaternion[3],quaternion[0]*1]

    R = mesh.get_rotation_matrix_from_quaternion(quaternion2)

    mesh_r.rotate(R, center=[0,0,0])
    #mesh_r.rotate(R)
    mesh_r = copy.deepcopy(mesh_r).translate((posRot_ar[0][0] *-1 , posRot_ar[0][ 1], posRot_ar[0][ 2]))
    #xyzMesh = np.asarray(mesh_r.vertices);
    #xyzMesh[:, 0] = xyzMesh[:, 0] * -1;  # mirroring the mesh of the obj w/ respect to X
    #xyzMesh[:, 2] = xyzMesh[:, 2] * -1;  # mirroring the mesh of the obj w/ respect to X

    # '''From obj to pointcloud for the mesh'''
    #mesh_pc = o3d.geometry.PointCloud()
    #mesh_pc.points = o3d.utility.Vector3dVector(np.asarray(mesh_r.vertices))
    mesh_pc=mesh_r
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pointCl)
    x_var = np.asarray(pcd.points)
    print("x first ", x_var[0, 0])
    x_var[:, 0] = x_var[:, 0] * -1   #mirroring the pc of the scene w/ respect to X
    #x_var[:, 2] = x_var[:, 2]*-1   #mirroring the pc of the scene  w/ respect to Z
    print("x then", x_var[0, 0])
    print(posRot_ar)
    pcd.points = o3d.utility.Vector3dVector(x_var)
    sdr = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    pcd.paint_uniform_color([0, 128/255, 1])  #blu
    pcd.estimate_normals();
    # mesh_pc.paint_uniform_color([1, 192 / 255, 203/255])   #rosa
    # o3d.visualization.draw_geometries([pcd, sdr])
    # o3d.visualization.draw_geometries([pcd, mesh_pc, sdr])
    pcd=computeCluster(pcd)
    pcd.paint_uniform_color([0, 128 / 255, 1])
    mesh_pc.paint_uniform_color([0, 239 / 255, 1])
    #o3d.visualization.draw_geometries([pcd, mesh_pc, sdr])
    #pcd=pcSmoothing(pcd)

    GlobalRegistration=1

    print("Recompute the normal of the downsampled point cloud")
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

    o3d.geometry.PointCloud.orient_normals_towards_camera_location(pcd, [0, 0, 0])

    source = mesh_pc
    target = pcd
    sourceOriginal=copy.deepcopy(mesh_pc)

    if (GlobalRegistration==1):
        voxel_size = 0.01  # means 5cm for this dataset
        source, target, source_down, target_down, source_fpfh, target_fpfh = prepare_dataset(voxel_size, source, target)

        #result_ransac = execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size)
        startGR = time.time()
        result_ransac =execute_fast_global_registration(source_down, target_down,source_fpfh, target_fpfh,voxel_size)
        print(result_ransac)


        result_icp = refine_registration(source, target, source_fpfh, target_fpfh,
                                         voxel_size, result_ransac.transformation)
        print("Global registration took %.3f sec.\n" % (time.time() - startGR))
        print(result_icp)
        #assign in the position R30 (the one for scaling) the fitness value
        fitness=np.float64(result_icp.fitness)
        transformToSend=result_icp.transformation.copy()
        transformToSend[3,0]=fitness
        print("fitness is =", fitness)
        print("transformazion with fitness on R30 is:", transformToSend)
        target_down.paint_uniform_color([0, 128 / 255, 1])  # blu
        source_down.paint_uniform_color([0, 239 / 255, 1])  # celeste
        #draw_registration_result(source_down, target_down, result_ransac.transformation)
        target.paint_uniform_color([0, 128 / 255, 1])  # blu
        source.paint_uniform_color([0, 239 / 255, 1])  # celeste
        #draw_registration_result(source, target, result_icp.transformation)
        send = []

        #data2=writeRtMat(result_icp.transformation, send)
        data2 = writeRtMat(transformToSend, send)
        conn.sendall(data2)
        print("Global Time %.3f sec.\n" % (time.time() - startTime))

        return
    else:

        threshold = 0.025

        '''trans_init = np.asarray([[1, 0, 0, 0],
                                 [0, 1, 0, 0],
                                 [0, 0, 1, 0], [0.0, 0.0, 0.0, 1.0]])'''

        trans_init = np.identity(4)
        '''print("Initial alignment")
        evaluation = o3d.pipelines.registration.evaluate_registration(
            source, target, threshold, trans_init)
        print(evaluation)'''

        '''print("Apply point-to-point ICP")
        reg_p2p = o3d.pipelines.registration.registration_icp(
            source, target, threshold, trans_init,
            o3d.pipelines.registration.TransformationEstimationPointToPoint())
        print(reg_p2p)
        print("Transformation is:")
        print(reg_p2p.transformation)
        draw_registration_result(source, target, reg_p2p.transformation)'''
        send = []
        print("Apply point-to-point ICP")
        reg_p2p = o3d.pipelines.registration.registration_icp(
            source, target, threshold, trans_init, o3d.pipelines.registration.TransformationEstimationPointToPoint())
        print(reg_p2p)
        print("Transformation is:")
        print(reg_p2p.transformation)

        #draw_registration_result(source, target, reg_p2p.transformation)
        data2 = writeRtMat(reg_p2p.transformation, send)
        conn.sendall(data2)
        return
    
nameFile="test1"
computeHidden=1

if computeHidden==1:
    mesh = workflow2RemoveHiddenPoint(objPatient, objFile)
    #mesh.points= o3d.utility.Vector3dVector(computeCluster(mesh))
    o3d.io.write_point_cloud("C:/Users/Alessandro/Desktop/Neuro/FacewtRemovedHiddenptManich3t2.ply", mesh)
    print("done")

else:
    mesh=o3d.io.read_point_cloud("C:/Users/Alessandro/Desktop/Neuro/FacewtRemovedHiddenptManich3t.ply")

while True:

    try:

        while True:
            HOST = '192.168.0.112'  # Standard loopback interface address (localhost)
            PORT = 2000  # Port to listen on (non-privileged ports are > 1023)
            first = 1

            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind((HOST, PORT))
                s.listen()
                print("waiting for a client")
                conn, addr = s.accept()
                with conn:
                    print('Connected by', addr)
                    while True:
                        data = conn.recv(400000)
                        arr = array.array('f', data)
                        if data:
                            ptRcv = arr.tolist()                    #in ptRcv: #punti, position Rotation of the object, point cloud point x y z in coloumn
                            numPt = int(ptRcv[0])

                            print(numPt)
                            #saveData(ptRcv)
                            #saveptRcv(ptRcv, nameFile)
                            #ptRcv2=OpenptRcv(nameFile)
                            comp_Rt_mat(ptRcv, mesh)             #compute the rototranslation matrix
                        if not data:
                            break



    except:
        continue
