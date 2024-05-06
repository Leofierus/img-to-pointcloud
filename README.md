# Creating pointcloud from a set of images
This repository contains the code to create a pointcloud from a set of images. The code is written in Python and uses OpenCV and Open3D libraries.

## Code Structure
Files in the repository:
- `Final.py`: The main file to run the code. Contains the final logic to create the pointcloud.
- `load_plys.py`: Contains the code to load the pointclouds from the PLY files and visualize them.
- `sift_det.py`: Contains the code to detect the keypoints and descriptors using SIFT and visualize them.
- `combination(x).py`, `Recalibrated(x).py`, `test(x).py` & `Optimized_K.py`: Contains intermediate code created during testing and optimization.
- `vid2gif.py`: Contains the code to convert a video to a GIF file.

Folders in the repository:
- `Gifs`: Contains the GIF files created using the code for various videos.
- `Gun`: Contains images of a gun taken from different angles.
- `Manual`: Contains the plys created using manual stitching in either Blender or MeshLab.
- `Treasure_Chest(x)`: Contains images of a treasure chest taken from different angles.
- `Videos`: Contains the videos used to create the GIF files.
- `Well(x)`: Contains images of a well taken from different angles.

## Pointclouds
### Treasure Chest
- ![Treasure Chest Capture in Blender](Videos/tc_cap.mp4)
- ![Intermediate Treasure Chest pointcloud](Videos/int_tcs.mp4)
- ![Intermediate Treasure Chest pointcloud](Videos/int_tcs%20(online-video-cutter.com).mp4)
- ![Final Treasure Chest pointcloud](Videos/fin_tcs.mp4)

### Well
- ![Well Capture in Blender](Videos/well_cap.mp4)
- ![Intermediate Well pointcloud](Videos/int_well.mp4)
- ![Final Well pointcloud](Videos/fin_well.mp4)