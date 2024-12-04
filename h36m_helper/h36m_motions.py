import os

import numpy as np
import tqdm
from loguru import logger

from .h36m_viewer import ImageViewer, KptsProjector


def estimate_yaw_angle(kpts, kpts3d):
    # Evaluate rotation angles between the person and the camera
    # Assumes the person stands still and that Z-axis represents the depth
    # Assumes coordinates to be given in a human referential
    # This mesures the ratio between the back and the shoulders width in both 2D and 3D
    # If the person faces the camera, the ratio should be similar in both 2D and 3D
    # First we get an angle between 0 and 90 degrees, and then we check the orientation
    #
    # Angles defined as follow
    # Faces the camera: 0degrees
    # Looking to the right side: 90 degrees
    # Turning his back: 180 degrees
    # Looking to the left: 270 degrees
    HIP_LEFT_IDX = 1
    HIP_MIDDLE_IDX = 0
    HIP_RIGHT_IDX = 4
    SHOULDER_LEFT_IDX = 11
    SHOULDER_MIDDLE_IDX = 8
    SHOULDER_RIGHT_IDX = 14
    SHOULDER_WIDTH = 40
    BACK_HEIGHT = 48.8

    back_height3d = kpts3d[SHOULDER_MIDDLE_IDX, :] - kpts3d[HIP_MIDDLE_IDX, :]
    back_height3d = np.sqrt(np.sum(back_height3d**2))
    shoulder_width3d = kpts3d[SHOULDER_RIGHT_IDX, :] - kpts3d[SHOULDER_LEFT_IDX, :]
    shoulder_width3d = np.sqrt(np.sum(shoulder_width3d**2))
    expected_ratio = shoulder_width3d / back_height3d

    back_height = kpts[SHOULDER_MIDDLE_IDX, :2] - kpts[HIP_MIDDLE_IDX, :2]
    back_height = np.sqrt(np.sum(back_height**2))
    shoulder_width = kpts[SHOULDER_RIGHT_IDX, :2] - kpts[SHOULDER_LEFT_IDX, :2]
    shoulder_width = np.sqrt(np.sum(shoulder_width**2))
    exp_width = back_height * expected_ratio
    if shoulder_width > exp_width:  # Rounding errors
        logger.debug(
            f'Shoulder width wider than maximal expected, found {shoulder_width}, max_expected" {exp_width}'
        )
        shoulder_width = exp_width
    angle = np.arccos(shoulder_width / exp_width)
    angle = angle * 180 / np.pi

    # Define orientation
    faces_camera = kpts[SHOULDER_RIGHT_IDX, 0] <= kpts[SHOULDER_LEFT_IDX, 0]
    rs = kpts3d[SHOULDER_RIGHT_IDX, :]
    ls = kpts3d[SHOULDER_LEFT_IDX, :]
    rh = kpts3d[HIP_RIGHT_IDX, :]
    lh = kpts3d[HIP_LEFT_IDX, :]
    left_side_front = (
        rh[2] > lh[2]
    )  # Left side of the body is actually closer to the camera
    if faces_camera:  # Angle between -90 and + 90
        if left_side_front:
            angle = 360 - angle
    else:  # Angle between 90 and 270 degrees
        if left_side_front:
            angle = 180 + angle
        else:
            angle = 180 - angle
    return angle


class PositionUpdater:
    def __init__(self, kpts3d: np.ndarray, camera_yaw: float):
        self.kpts3d = kpts3d
        self.camera_yaw = np.deg2rad(camera_yaw)
        self.facing_camera_kpts = self.rotate_kpts(self.kpts3d, [self.camera_yaw])

    def rotate_kpts(self, kpts3d, angles: list[float]):
        # TODO: Uses quaternions

        # Here: rotates on XZ plan, assumes human to be around Y axis, yaw angle only
        HIP_MIDDLE_IDX = 0
        pivot = kpts3d[HIP_MIDDLE_IDX, :]
        kpts3d = kpts3d - pivot

        x2 = (kpts3d[:, 0] * np.cos(angles[0])) - (kpts3d[:, 2] * np.sin(angles[0]))
        y2 = (kpts3d[:, 0] * np.sin(angles[0])) + (kpts3d[:, 2] * np.cos(angles[0]))
        kpts3d[:, 0] = x2
        kpts3d[:, 2] = y2

        kpts3d = kpts3d + pivot
        return kpts3d

    def _add_ankle(self, rotated_kpts3d, idxs: list[int], rotation_angle: float):
        hip_idx, knee_idx, ankle_idx = idxs
        v_hip_knee = rotated_kpts3d[hip_idx, :] - rotated_kpts3d[knee_idx, :]
        v_knee_ank = v_hip_knee  # # Assumes the length of the leg to be twice the distancefrom hip to knee
        pt1 = (v_knee_ank[1] * np.cos(rotation_angle)) - (
            v_knee_ank[2] * np.sin(rotation_angle)
        )
        pt2 = (v_knee_ank[1] * np.sin(rotation_angle)) + (
            v_knee_ank[2] * np.cos(rotation_angle)
        )
        v_knee_ank[1] = pt1
        v_knee_ank[2] = pt2
        rotated_kpts3d[ankle_idx, :] = rotated_kpts3d[knee_idx, :] + v_knee_ank

    def simulate_knee_movement(self, left_ang=90, right_ang=90):
        HIP_LEFT_IDX = 1
        HIP_RIGHT_IDX = 4
        ANKLE_LEFT_IDX = 3
        ANKLE_RIGHT_IDX = 6
        KNEE_LEFT_IDX = 2
        KNEE_RIGHT_IDX = 5

        # Get new points according to the angle
        # Here we only consider a simple movement from the knee
        # After rotation, it will be on axis 1 and 2
        rot_left_leg = np.deg2rad(left_ang)
        left_leg_idxs = [HIP_LEFT_IDX, KNEE_LEFT_IDX, ANKLE_LEFT_IDX]
        self._add_ankle(self.facing_camera_kpts, left_leg_idxs, rot_left_leg)
        rot_right_leg = np.deg2rad(right_ang)
        r_leg_idxs = [HIP_RIGHT_IDX, KNEE_RIGHT_IDX, ANKLE_RIGHT_IDX]
        self._add_ankle(self.facing_camera_kpts, r_leg_idxs, rot_right_leg)

        # Rotate back
        self.kpts3d = self.rotate_kpts(self.facing_camera_kpts, [-self.camera_yaw])


def simulate_pts(ds, im_folder, display=False):
    idx: int = 0
    n_images = len(ds["imgname"])
    n_kpts = 17
    new_kpts2d = np.ones((n_images, n_kpts, 3), dtype=float)
    new_kpts3d = np.ones((n_images, n_kpts, 4), dtype=float)
    for kpts, kpts3d, im_name, center, scale, K, zind in tqdm.tqdm(
        zip(
            ds["part"],
            ds["S"],
            ds["imgname"],
            ds["center"],
            ds["scale"],
            ds["K"],
            ds["zind"],
        )
    ):
        im_path = os.path.join(im_folder, im_name[0][:2], im_name[0]) + ".jpg"
        kpts3d = kpts3d.T
        angle = estimate_yaw_angle(kpts, kpts3d)

        updater = PositionUpdater(kpts3d, angle)
        angles = np.random.random((2)) * 180
        updater.simulate_knee_movement(*angles)
        kpts3d /= 1000

        # Update camera location

        if display:
            viewer = ImageViewer(im_path, kpts3d, K)
            viewer.add_text(f"Angle: {int(angle)} degrees")
            viewer.view()

        projector = KptsProjector(kpts3d, K)
        annots2d = projector.kpts2d.reshape((n_kpts, 2)).astype(float)
        new_kpts2d[idx, :, :2] = annots2d

        annots3d = kpts3d.reshape((17, 3)).astype(float)
        new_kpts3d[idx, :, :3] = annots3d
        idx += 1

    out = {}
    out["part"] = new_kpts2d
    out["S"] = new_kpts3d
    for key in ["imgname", "center", "scale", "K", "zind"]:
        out[key] = ds[key]
    imgnames = np.array([str(_[0]) for _ in ds["imgname"]])
    out["imgname"] = np.array(imgnames, dtype=str)
    return out
