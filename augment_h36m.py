from typing import List

import fire
import os
import numpy as np

from h36m_helper.h36m_reader import h36m_reader
from h36m_helper.h36m_motions import simulate_pts



def main(h36m_path: str, out_path: str):
    images_folder = os.path.join(h36m_path, 'images')
    data_folder = os.path.join(h36m_path, 'data/h36m/annot/')

    out_content = h36m_reader(data_folder, 'train')
    new_data = simulate_pts(out_content, images_folder)

    if out_path is not None:
        with open(out_path, "wb") as f:
            np.savez(f, **new_data)



if __name__ == '__main__':
    fire.Fire(main)