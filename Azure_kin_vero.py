import cv2
import numpy as np
import pykinect_azure as pykinect
import open3d as o3d

# Depth camera parameters:
FX_DEPTH = 503.988861
FY_DEPTH = 503.946228
CX_DEPTH = 320.462463
CY_DEPTH = 329.527344


def Compute_3D_blob_coord(keys,im,depth):
    key_3D = []
    depth = depth/1000

    for blob in keys:
       u = int(blob.pt[0])
       v = int(blob.pt[1])

       z = depth[v][u]
       x = (u - CX_DEPTH) * z  / FX_DEPTH
       y = (v - CY_DEPTH) * z / FY_DEPTH
       
       #XYZ = [xy[0]*z,xy[1]*z,z]/np.linalg.norm([xy[0],xy[1],1])
       #XYZ_sphere = (XYZ *(1 + 0.0054/z)).tolist()
       #print("point from function", [x,y,z])
       key_3D.append([x,y,z])
       #print(key_3D)
    
    return key_3D
    


def Blob_detector(im,depth,detector):
    
    # Detect blobs.
    im_conv = (im/256).astype('uint8')
    depth_conv = (depth/256).astype('uint8')
    
    print(depth.shape)
    print(im_conv.shape)
    #cv2.imshow("Keypoints", im)
    ret,im_conv_treshold = cv2.threshold(im_conv,250,255,cv2.THRESH_BINARY)
    cv2.imshow("Tresh", im_conv_treshold)
    keypoints = detector.detect(im_conv_treshold)
    
    #cv2.imshow("Keypoints", im_conv)
    
    
    # Draw detected blobs as red circles.
    # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
    im_with_keypoints = cv2.drawKeypoints(im_conv_treshold, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    depth_with_keypoints = cv2.drawKeypoints(depth_conv, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    #for blob in keypoints:
        #cv2.circle(im_with_keypoints, (int(blob.pt[0]),int(blob.pt[1])), 1, (0,255,0), -1)

    cv2.imshow("Keypoints", im_with_keypoints)
    cv2.imshow("Keypoints_depth", depth_with_keypoints)


    print(Compute_3D_blob_coord(keypoints,im_conv,depth))  



if __name__ == "__main__":

    # Initialize the library, if the library is not found, add the library path as argument
    pykinect.initialize_libraries()

    # Modify camera configuration
    device_config = pykinect.default_configuration
    #device_config.color_format = pykinect.K4A_COLOR_FORMAT_COLOR_BGRA32
    device_config.color_resolution = pykinect.K4A_COLOR_RESOLUTION_1440P
    device_config.depth_mode = pykinect.K4A_DEPTH_MODE_NFOV_UNBINNED
    print(device_config)

    # Initialize Open3D Visualizer
    #vis = o3d.visualization.Visualizer()
    
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window('viewer', 640, 576)

    vis2 = o3d.visualization.Visualizer()
    vis2.create_window('viewer2', 640, 576)
    # Start device
    device = pykinect.start_device(config=device_config)
    
    #cv2.namedWindow('Infrared Image',cv2.WINDOW_NORMAL)
    #cv2.namedWindow('tracked',cv2.WINDOW_NORMAL)
    
    # Set up the SimpleBlobDetector with default parameters
    params = cv2.SimpleBlobDetector_Params()

    # Set the threshold
    params.minThreshold = 100
    params.maxThreshold = 300

    # Set the area filter
    params.filterByArea = True
    params.minArea = 10
    params.maxArea = 1000

    # Set the circularity filter
    params.filterByCircularity = True
    params.minCircularity = 0.8
    params.maxCircularity = 1

    # Set the convexity filter
    params.filterByConvexity = False
    params.minConvexity = 0.2
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
    
    
    while True:
   
        # Get capture
        capture = device.update()

        # Get the infrared image
        ret, ir_image = capture.get_ir_image()
        ret2, depth_image = capture.get_depth_image()
        
        if not ret:
            continue
        if not ret2:
            continue

        Blob_detector(ir_image,depth_image,detector)
        # Plot imageq
        #cv2.imshow('Infrared Image',ir_image)

        # Press q key to stop
        if cv2.waitKey(1) == ord('q'):  
            break