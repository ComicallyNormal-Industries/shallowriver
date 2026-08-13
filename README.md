
# SHALLOWRIVER

**Shallow River is a multi-camera body-pose pipline optimized for latency on the Jetson Orin.** It provides an alternitive approach to NVIDIA deepstream, where realtime frame latency is prioritized over total fps. This is acomplished using ARM unified memory to pipline camera frames with a zero copy queue architecture and NVIDIA CUDA running the bounding box and body pose models.

## Purpose

**Shallow River is a neccessary optimization for Phyiscal AI where latency is the main bottleneck.** Industries such as VR and robotics are highly sensitive to frame latency where current approaches fall short.  

## Orin Setup

Follow this installation guide, this project assumes you are using JetPack 7.2 paired with Jetson Linux 39.2 [https://docs.nvidia.com/jetson/orin-nano-devkit/user-guide/latest/quick_start.html](https://docs.nvidia.com/jetson/orin-nano-devkit/user-guide/latest/quick_start.html)

## Camera Requirements

This project is setup by default to support up to two USB webcams that run 1920x1080 at 30 fps using v4l2src. If your camera does not support this format then you will have to update the camera definitions in `inc/camera_defs.hpp` to match what your camera supports.

## Install

To install using the provided apt .deb package use this command, or follow the build instructions to compile from source. Download the shallowriver.deb to /tmp and run the command there so that it does not get blocked by usr read and write protections.
```
sudo apt install ./shallowriver.deb
```
## Build

### Dependencies

Run the install script to install dependencies, create the build directory and compile

```
chmod +x install.sh
sudo ./install.sh
```

### Building the project
Create the build directory, then use CMake and make to build the project. The build directory is expected to be under `/shallowriver/` for the `/res` folder to be copied correctly.

```
mkdir build
cd build
cmake ..
make
```


## Setup and Calibration

**There are a few requirements to run Shallowriver.**

**Calibration**

*All cameras used must be calibrated for for the output to work correctly.*

1. Generate the ChAruCo board.

* run the --generate argument explained in the **Running the Project** section. This will out put the `charuco_board.png` file used to calculate the camera intrensics. The file will be saved in the `build` directory. print `charuco_board.png` on white paper and tape it to a flat surface. The dimensions of the squares should be 40mm and the markers should be 30mm. 

* for more information read here [https://docs.opencv.org/4.13.0/da/d13/tutorial_aruco_calibration.html](https://docs.opencv.org/4.13.0/da/d13/tutorial_aruco_calibration.html)


2. Calibrate the Cameras.

* To calibrate a camera use -calibrate or --c followed by 1 for the first camera or 2 for the second camera. Use the provided ChArUco board to capture 15 frames by pressing the space bar. Make sure to capture frames with the ChArUco board at the edges at different angles, and one up close covering the frame. Make sure at least four id's are showing in red over the ArUco makers before capturing a frame. When 15 frames are collected, press enter to run the calibration. Each calibration file is saved under the `/build/res` directory. Do this seperately for each camera used.

**Building the engine file**

The onnx engine file must be built before running. It will take 10-15 minutes. This can be done with the -engine or --e argument, or it is done on the first run of the project.

## Running the project 

**To view the command line arguments run `./shallowriver --h`** The options are explained here

* `--calibrate --c [1|2]` 
  * must be run for each camera 
  * example: `./shallowriver --c 1`
* `--run --r [1|2|both]` 
  * runs the inference model on the cameras specified. Camera 1 maps to `/etc/video0` and camera 2 maps to `/etc/video2`. If the cameras are not using the default ports, then you can modify the OpenCV camera settings in `/inc/camera_defs.hpp` and switch `v4l2src device=` to the correct port for each camera. 
  * example: `./shallowriver --r both`chmod +x install.sh
* `--engine --e` 
  * recompiles the onnx engine 
  * example:  `./shallowriver --e`
* `--generate --g` 
  * generates the ChArUco board used for calibration. the file is put into `/build`
  * example: `./shallowriver --g`
* `--log --l`
  * enable per-frame fps/latency CSV logging (with --run/--engine)
  * example: `./shallowriver --r both --l`
* `--print --p` 
  * print pipeline benchmark stats to the terminal (with --run/--engine)
  * example: `./shallowriver --r both --p`

## Outputs

### frame view

One or two Opencv windows will appear based on the number of cameras used. For best accuracy make sure the full body is in view of the camera.

### 3d keypoints
There are two output locations, differing based on how the project is run. When run through the apt package, the base path will be `~/.local/share/shallowriver`. When running in the build directory it will be local, in `/build/res`. If logging is enabled, the logs will be generated in the `logging` directory, the 3d outpus will be put in the `output` directory. For example: `/logging/perf_1.csv` or `/output/3d_key_points_1.txt`.

## Visualization

There are two python scripts provided to visualize the 3d output and performance of shallowriver. 

### 3d visualizer

in the `/helper/` directory, run
```
python3 dual_visualizer.py
```

When this program is run, a plot will appear that you can view the keypoint frames. The frames are normalized to the camera for ease of viewing.

### performance visualizer

first, run shallowriver with the --logging option `./shallowriver --r both --l` to generate the performance log files. Then, in the `/helper/` directory, run
```
python3 stat_plotter.py
```

This program generates the .png files used in the paper, which shows the fps and latency of each frame.

## Contributors

Baylor McElroy

Baiden McElroy

Alex Diviney
