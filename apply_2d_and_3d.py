import cv2
import fire
import numpy as np
import torch
from mmpose.apis import inference_pose_lifter_model, init_model, visualize
from mmpose.apis.inference import inference_topdown
from mmpose.visualization import PoseLocalVisualizer
from mmpose.structures import PoseDataSample
from mmengine.structures import InstanceData
from h36m_helper.h36m_viewer import ImageViewer


def apply_2d(img, cfg, ckpt):
    mdl = init_model(cfg, ckpt)
    preds = inference_topdown(mdl, img)
    return preds


def apply_3d(kpts, cfg, ckpt):
    mdl = init_model(cfg, ckpt)
    gt_instances = InstanceData()
    gt_instances.bboxes = torch.rand((1, 4))
    gt_instances.keypoints = torch.from_numpy(kpts.astype(np.float32))
    gt_instances.keypoints_visible = torch.ones((1,17))
    data_sample = PoseDataSample(gt_instances=gt_instances.clone(), pred_instances=gt_instances.clone(), track_id=1)
    out = inference_pose_lifter_model(mdl, [[data_sample]], with_track_id=False, image_size=None)
    out[0]._pred_instances.keypoints /= 1.
    return out


def results_3d(viewer: ImageViewer, preds: np.ndarray):
    viewer.img = np.ones((1000,1000,3), np.uint8)
    viewer.view()
    print('left leg')
    print(preds[0,[1,2,3],:])
    print('Distance between hip and knees')
    print(np.sqrt(np.sum((preds[0,1,:] - preds[0,2,:]) ** 2)))
    print('Distance between hip and foot')
    print(np.sqrt(np.sum((preds[0,1,:] - preds[0,3,:]) ** 2)))
    print('Right leg')
    print(preds[0,[4,5,6],:])
    print('Distance between hip and knees')
    print(np.sqrt(np.sum((preds[0,4,:] - preds[0,5,:]) ** 2)))
    print('Distance between hip and foot')
    print(np.sqrt(np.sum((preds[0,4,:] - preds[0,6,:]) ** 2)))


def main(im_path, cfg2d, ckpt2d, cfg3d, ckpt3d1, ckpt3d2):
    img = cv2.imread(im_path)
    datasample2d = apply_2d(img, cfg2d, ckpt2d)
    # Switch to H36m dataset
    reorder = [6, 3,4,5, 2,1,0, -1, 7,8,9, 13,14,15, 12,11,10]

    kpts2d = datasample2d[0]._pred_instances.keypoints.astype(int)
    kpts2d = kpts2d[:, reorder, :]
    kpts2d[:, 7, :] = (kpts2d[:, 0, :] + kpts2d[:, 8, :]) / 2
    kpts2d_scores = datasample2d[0]._pred_instances.keypoint_scores
    viewer = PoseLocalVisualizer(name='2D detection')
    viewer.kpt_color = np.array([(0,0,255)] * 17)
    kpts_img = visualize(img, kpts2d, visualizer=viewer, show_kpt_idx=True, show=True)

    out = apply_3d(kpts2d, cfg3d, ckpt3d1)
    shift = np.array([-293.87820511, -525.1670764 , 5461.63990357])
    K = np.array([[1.14504940e+03, 0.00000000e+00, 5.12541505e+02],
                  [0.00000000e+00, 1.14378110e+03, 5.15451487e+02],
                  [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

    preds = out[0]._pred_instances.keypoints * 1000 + shift
    viewer = ImageViewer(im_path, preds, K)
    print('1st model results')
    results_3d(viewer, preds)
    cv2.imshow('1st model', viewer.img)

    out = apply_3d(kpts2d, cfg3d, ckpt3d2)
    preds = out[0]._pred_instances.keypoints * 1000 + shift
    print()
    print('2nd model results')
    results_3d(viewer, preds)



if __name__ == '__main__':
    fire.Fire(main)
    '''
    Example 
    main('2d_tests/QuadStretch_annotated-88a3172391cc4daa85fd643d14f73a15.webp',
         '2d_tests/cfg.py', '2d_tests/hrnet_w48_mpii_256x256_dark-0decd39f_20200927.pth',
         '3D/cfg.py', 
         'simple3Dbaseline_h36m-f0ad73a4_20210419.pth',
         'work_dirs/cfg/best_MPJPE_epoch_200.pth'
         )
    '''