# Neuro

1) Azure_kinect_script: stream the data from the Kinect Azure and track the Instrument with retroreflective markers:
   - Get the IR and DEPTH data
   - Perform a first binary treshoilding
   - Perform a blob detection
   - reconstruct the 3D coordinates of the detected keypoints knowing the 2D coordinates and the corresponding depth info

2) hlss: Manage the stream of all the info from the Hololens (cameras, intrinsic/extrinsic metrics, pose ecc...)
3) Holo_server: Unity scene to stream everything
