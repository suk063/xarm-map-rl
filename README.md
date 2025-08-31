# erl_xArm
This repository contains the code to run xArm6 robot in both real and simulation.

## Dependencies
- [xArm Python SDK](https://github.com/xArm-Developer/xArm-Python-SDK)
- [robosuite fork](https://github.com/tianyudwang/robosuite)
- scipy
- opencv-python
- pyrealsense2

## Usage
1. Mandatory Setup 
    - Set up python path
        ```bash
        export PYTHONPATH=$PYTHONPATH:/path/to/erl_xArm
        ```

2. (Optional) Setup for real robot experiments
    - Ensure fsr port is open
        ```bash
        sudo chmod a+rw /dev/ttyACM0
        ```
    - Run camera extrinsics calibration
        ```bash
        cd scripts
        python3 camera_extrinsics_calibration.py
        ```
        Note: extrinsics.npz is saved to the directory from where you execute camera_extrinsics_calibration.py.

3. Run experiments
    ```bash
    cd scripts
    python3 gui_run_experiment.py
    ```
    Note: If you wish to run tasks on xarm that use camera input, then extrinsics.npz must be present in the scripts directory# xarm-map-rl
