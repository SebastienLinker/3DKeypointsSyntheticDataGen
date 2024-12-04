import os
import numpy as np
import h5py
from scipy.io import loadmat


def _extract_mat(mat_file: str) -> dict:
    annots = loadmat(mat_file)
    out = {}
    for key in ('S', 'imgname', 'center', 'scale', 'K'):
        out[key] = annots["annot"][0, 0][key]
    return out


def _read_h5(path: str) -> dict:
    f1 = h5py.File(path, "r+")
    out = {}
    assert all(x in f1.keys() for x in {"S", "center", "imgname", "part", "scale", "zind"})
    for key in {"S", "center", "part", "scale"}:
        print(key)
        print(type(f1[key]))
        print(f1[key])
        out[key] = np.array(f1[key]).astype(float)
    kpts = np.ones((*out["part"].shape[:2], 3), dtype=out["part"].dtype)
    kpts[:, :, :2] = out["part"]
    out["part"] = kpts
    # Downscaling
    # for idx in range(3):
    #     out["S"][:, :, idx] = out["S"][:, :, idx] - np.mean(out["S"][:, :, idx])
    # out["S"] = out["S"] / 1000
    kpts3d = np.ones((*out["S"].shape[:2], 4), dtype=out["S"].dtype)
    kpts3d[:, :, :3] = out["S"]
    out["S"] = kpts3d
    out["imgname"] = np.array(f1["imgname"], dtype=str)
    out["zind"] = np.array(f1["zind"], dtype=np.uint8)
    return out


def _merge_mat_h5(mat_data: dict, h5_data: dict) -> dict:
    S = np.swapaxes(h5_data['S'][:,:,:3], 1, 2)
    if mat_data is not None:
        # Get image names and 3D keypoints together
        assert mat_data['S'].shape[0] == h5_data['S'].shape[0]
        assert np.all(mat_data['center'] == mat_data['center'])
        assert np.all(h5_data['scale'] == mat_data['scale'].reshape(-1).astype(float))

        assert np.all(S == mat_data['S'].astype(float))
        out = dict(S=S, center=h5_data['center'], scale=h5_data['scale'], 
                part=h5_data['part'], zind=h5_data['zind'],
                K=np.stack(mat_data['K'][:,0]), imgname=mat_data['imgname'].reshape(-1))
    else:
        k = [np.array([[1.14551134e+03, 0.00000000e+00, 5.14968197e+02],
                        [0.00000000e+00, 1.14477393e+03, 5.01882019e+02],
                        [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])] * h5_data['S'].shape[0]
        imgname = np.array([["S1_Directions_1.54138969_000001.jpg"]] * h5_data["imgname"].shape[0])
        out = dict(S=S, center=h5_data['center'], scale=h5_data['scale'], 
                part=h5_data['part'], zind=h5_data['zind'],
                K=np.array(k), imgname=imgname.reshape(-1,1))
    return out


def h36m_reader(data_folder: str, file: str='valid') -> dict:
    # Reads H36M dataset from several sources
    mat_file = os.path.join(data_folder, f'{file}.mat')
    h5_file = os.path.join(data_folder, f'{file}.h5')

    # read_files(data_folder)
    mat_content = _extract_mat(mat_file) if os.path.exists(mat_file) else None
    h5_content = _read_h5(h5_file)
    out_content = _merge_mat_h5(mat_content, h5_content)
    return out_content
