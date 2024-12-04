# Simple synthetic data generation for 3D keypoints reconstruction

This repository proposes a simple approach to generate synthetic data for 2D and 3D keypoints.
It allows one to generate a basic keypoints dataset to go from 2D to 3D without the need to learn complex libraries.

![image](https://editor.analyticsvidhya.com/uploads/65603fig.png)

Can we get 3D reconstructions from those keypoints?

## Data generation

An example is provided in file `augment_h36m.py`. This loads a copy of H36m dataset and update some keypoints.
In the example, the foot keypoints are generated artificially, based on standard movement of the human knees.

TODO ideas (I'll probably never do all of that on my free time, feel free to make PRs): 

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

The 2D model in this example is [HRNet dark](https://github.com/open-mmlab/mmpose/blob/v1.3.2/configs/body_2d_keypoint/topdown_heatmap/mpii/hrnet_dark_mpii.md), trained on MPII dataset.
The demo code measures the distance between hips and knees, and then the distance between hips and ankle. On the test image, the latter is expected to be short for the left leg. Below is a sample output:

``` console
1st model results
left leg
[[-293.85285907 -525.18202608 5461.63515928]
 [-293.74601971 -524.84906592 5461.91864272]
 [-293.65775186 -525.1680771  5462.11168426]]
Distance between hip and knees
0.45015550510704466
Distance between hip and foot
0.515109165200248
Right leg
[[-293.90355043 -525.15212697 5461.64464776]
 [-293.86642318 -524.78511029 5461.81986042]
 [-293.75353483 -524.45447778 5461.88815796]]
Distance between hip and knees
0.4083860339809041
Distance between hip and foot
0.7540001893661517
Loads checkpoint by local backend from path: work_dirs/cfg/best_MPJPE_epoch_70_bak.pth

2nd model results
left leg
[[-293.82325543 -525.16779201 5461.67113275]
 [-293.74142388 -524.78317978 5461.81832662]
 [-293.6266207  -525.07278836 5461.57193941]]
Distance between hip and knees
0.4198678396147984
Distance between hip and foot
0.23985459710494678
Right leg
[[-293.93315508 -525.16618675 5461.60873135]
 [-293.868663   -524.75359278 5461.74127185]
 [-293.81253732 -524.36663746 5461.63542099]]
Distance between hip and knees
0.4381323936884442
Distance between hip and foot
0.8090364921942099
```

Even if the backprojection looks correct, we can see from Euclidean distances that the second model (retrained with synthetic data) gives a more realistic result.