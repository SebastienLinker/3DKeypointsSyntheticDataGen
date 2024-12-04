# Simple synthetic data generation for 3D keypoints reconstruction

This repository proposes a simple approach to generate synthetic data for 2D and 3D keypoints.
It allows one to generate a basic keypoints dataset to go from 2D to 3D without the need to learn complex libraries.

## Data generation

An example is provided in file `augment_h36m.py`. This loads a copy of H36m dataset and update some keypoints.
In the example, the foot keypoints are generated artificially, based on standard movement of the human knees.

TODO ideas (I'll probably never do all of that on my free time): 

  - [x] Simple synthetic data generation
  - [ ] Describe joints movements using quaternions 
  - [x] Synthetic data generation of the feet following rotative movement around the knees
  - [ ] Synthetic data generation of other keypoints
  - [ ] Add extra keypoints (toes, fingers, facial features ...) 
  - [ ] Update camera location
  - [ ] Update camera matrix (fish-eye camera ...)

## Application

H36m and other datasets containing 3D keypoints only contains standard gaits in their training set, making them unusable for actual applications related to sports, when we try reconstructing uncommon poses.
This is where synthetic dataset comes handy. Synthetic dataset can simulate any arbitrary kind of posture.

Here we want the 3D pose keypoints dataset to contain the full range of motion of the knees, so we can apply this to real-life sports monitoring, but feel free to use that approach for any application.

## How to use

### Synthetic dataset generation

Run the script `augment_h36m.py`, tranformations are described in `h36m_helper/h36m_motions`.

### Train model

Training script is provided by [MMPose](https://github.com/open-mmlab/mmpose/tree/main).

After creating a synthetic dataset, use one of their models to go from 2D to 3D. For demo, we use [SimpleBaseline](https://github.com/open-mmlab/mmpose/blob/v1.3.2/configs/body_3d_keypoint/image_pose_lift/h36m/simplebaseline3d_h36m.md) models (see the config file provided).

Sample config file is provided under `3D/cfg.py`.

### Demo

TODO:

  - [x] Simple python demo script (`apply_2d_and_3d.py`), that processes 2D keypoints detection and upgrade to 3D with SimpleBaseline. It then calculates the distances hip/knee and hip/ankle on the resulting 3D, with two models. I recommend using the original checkpoint from MMPose and a newly trained checkpoint, with synthetic data, for a fair comparison and validation of the approach
  - [ ] Notebook with animated 3D views