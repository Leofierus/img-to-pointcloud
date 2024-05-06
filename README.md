# Creating pointcloud from a set of images
This repository contains the code to create a pointcloud from a set of images. The code is written in Python and uses OpenCV and Open3D libraries.

## Code Structure
Files in the repository:
- `Final.py`: The main file to run the code. Contains the final logic to create the pointcloud. Will require the libraries mentioned in the imports.
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
<details>
<summary>Treasure Chest</summary>

https://github.com/Leofierus/img-to-pointcloud/assets/51908556/50cfb51e-9ee3-4d77-8e2f-e4965f426ca6

https://github.com/Leofierus/img-to-pointcloud/assets/51908556/1dd3eb53-f6bf-4616-a9f6-96c85e65dbdd

https://github.com/Leofierus/img-to-pointcloud/assets/51908556/889bd799-969b-4a90-aa1c-6de8210caa51

https://github.com/Leofierus/img-to-pointcloud/assets/51908556/aa4a86e0-2af6-40c0-b4ab-881a8ee45376

</details>

<details>
<summary>Well</summary>

https://github.com/Leofierus/img-to-pointcloud/assets/51908556/24c841ec-58da-4212-b0e5-9982d368842a

https://github.com/Leofierus/img-to-pointcloud/assets/51908556/2e847800-0b56-43bd-a889-df58e6d4c755

https://github.com/Leofierus/img-to-pointcloud/assets/51908556/e6eb2454-a75a-4a1b-8de6-f0e290b82906

</details>
