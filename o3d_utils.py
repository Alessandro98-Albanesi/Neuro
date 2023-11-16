import pickle
import open3d as o3d
import numpy as np
import copy

def loadPfile_asPC(filename):
    xyz = pickle.load( open( filename, "rb" ) )
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    print("Recompute the normal of the downsampled point cloud")
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

    o3d.geometry.PointCloud.orient_normals_towards_camera_location(pcd, [0, 0, 0])
    return pcd

def loadObjfile_asPC(objFile):
    print("opening mesh obj")
    mesh = o3d.io.read_triangle_mesh(objFile)
    print("Try to render a mesh with normals (exist: " + str(mesh.has_vertex_normals()) + ") and colors (exist: " +
          str(mesh.has_vertex_colors()) + ")")
    # o3d.visualization.draw_geometries([mesh])
    print("A mesh with no normals and no colors does not seem good.")

    print("Computing normal and rendering it.")
    mesh.compute_vertex_normals()
    print(np.asarray(mesh.triangle_normals))
    mesh_pc = o3d.geometry.PointCloud()
    mesh_pc.points = o3d.utility.Vector3dVector(np.asarray(mesh.vertices))
    return mesh_pc

def display_inlier_outlier(cloud, ind):
    inlier_cloud = cloud.select_by_index(ind)
    outlier_cloud = cloud.select_by_index(ind, invert=True)

    print("Showing outliers (red) and inliers (gray): ")
    outlier_cloud.paint_uniform_color([1, 0, 0])
    inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])

def radiusRemoveOutlr(min_Npts, radius, pcd):
    cl, ind = pcd.remove_radius_outlier(min_Npts, radius)
    display_inlier_outlier(pcd, ind)
    inlier_cloud = pcd.select_by_index(ind)
    return inlier_cloud

def writePLY(pcd, name):
    o3d.io.write_point_cloud(name, pcd)
    return

def writeObj(mesh, name):
    o3d.io.write_triangle_mesh(name, mesh)
    return

def renderPC(pcd):
    pcd.paint_uniform_color([1, 0.706, 0])
    o3d.visualization.draw_geometries([pcd])

def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0]) #Giallo
    target_temp.paint_uniform_color([0, 0.651, 0.929]) #celeste

    source_temp.transform(transformation)
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
    o3d.visualization.draw_geometries([source_down,target_down])
    return source, target, source_down, target_down, source_fpfh, target_fpfh

def execute_global_registration(source_down, target_down, source_fpfh,
                                target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5
    print(":: RANSAC registration on downsampled point clouds.")
    print("   Since the downsampling voxel size is %.3f," % voxel_size)
    print("   we use a liberal distance threshold %.3f." % distance_threshold)
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
    return result

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

def refine_registration(source, target, source_fpfh, target_fpfh, voxel_size,trasf_init2):
    distance_threshold = voxel_size * 0.05
    treshold= 0.01
    print(":: Point-to-plane ICP registration is applied on original point")
    print("   clouds to refine the alignment. This time we use a strict")
    print("   distance threshold %.3f." % distance_threshold)
    result = o3d.pipelines.registration.registration_icp(
        source, target, treshold, trasf_init2,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000))
    return result



sdr=sdr = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)


# pc="PointCl.p"
# pcd= loadPfile_asPC(pc)
# # #pcd= loadObjfile_asPC(pc)
# writePLY(pcd)
# o3d.visualization.draw_geometries([pcd,sdr])
#
# diameter=1
# #diameter = np.linalg.norm(np.asarray(pcd.get_max_bound()) - np.asarray(pcd.get_min_bound()))
# print("Define parameters used for hidden_point_removal")
# camera = [0, 0 , diameter]
# radius = diameter * 100
# pcd_nonHidden=copy.deepcopy(pcd)
# pcd.paint_uniform_color([1, 0.706, 0])
#
# print("Get all points that are visible from given view point")
#
# o3d.visualization.draw_geometries([pcd])
# _, pt_map = pcd_nonHidden.hidden_point_removal(camera, radius)
#
# print("Visualize result")
# pcd_nonHidden = pcd_nonHidden.select_by_index(pt_map)
# pcd_nonHidden.paint_uniform_color([1, 0.0, 0])
# o3d.visualization.draw_geometries([pcd_nonHidden,pcd,sdr])
#
#
#
# pcd_inlier= radiusRemoveOutlr(20, 0.005)
# renderPC(pcd_inlier)