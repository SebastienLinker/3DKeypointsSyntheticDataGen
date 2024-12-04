import numpy as np
import cv2


class ImageViewer():
    def __init__(self, im_path, kpts3d, K):
        self.img = cv2.imread(im_path)
        self.kpts3d = kpts3d
        self.n_lines = 0
        self.K = K

    def add_text(self, txt: str):
        y = 100 + self.n_lines * 50
        cv2.putText(self.img, txt, (100, y), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 1, cv2.LINE_AA)
        self.n_lines += 1

    def kpts_3d_to_2d(self, center=None, scale=None, K=1, zind=None):
        kpts3d = self.kpts3d.reshape((1,-1,3))
        # Define the distortion coefficients 
        dist_coeffs = np.zeros((5, 1), np.float32) 
        # Define the rotation and translation vectors 
        rvec = np.zeros((3, 1), np.float32) 
        tvec = np.zeros((3, 1), np.float32) 

        # Map the 3D point to 2D point 
        points_2d, _ = cv2.projectPoints(kpts3d, 
                                        rvec, tvec, 
                                        K, 
                                        dist_coeffs) 
        return points_2d.reshape((-1, 2))

    def add_kpts(self, with_idx=True, color=(0,0,255)):
        kpts = self.kpts_3d_to_2d(K=self.K).astype(int).tolist()
        for idx, kpt in enumerate(kpts):
            cv2.circle(self.img, tuple(kpt), 3, color, -1)
            if with_idx:
                cv2.putText(self.img, str(idx), tuple(kpt), cv2.FONT_HERSHEY_COMPLEX,  
                    1, (255, 0, 0), 1, cv2.LINE_AA)

    def view(self, with_keypoints: bool=True) -> None:
        if with_keypoints:
            self.add_kpts()
        cv2.imshow('Rendered_image', self.img)
        cv2.waitKey()


class KptsProjector():
    def __init__(self, kpts3d, K):
        self.kpts3d = kpts3d
        self.K = K
        self.kpts2d = self.kpts_3d_to_2d()

    def kpts_3d_to_2d(self,):
        kpts3d = self.kpts3d.reshape((1,-1,3))
        # Define the distortion coefficients 
        dist_coeffs = np.zeros((5, 1), np.float32) 
        # Define the rotation and translation vectors 
        rvec = np.zeros((3, 1), np.float32) 
        tvec = np.zeros((3, 1), np.float32) 

        # Map the 3D point to 2D point 
        points_2d, _ = cv2.projectPoints(kpts3d, 
                                        rvec, tvec, 
                                        self.K, 
                                        dist_coeffs) 
        return points_2d.reshape((-1, 2))