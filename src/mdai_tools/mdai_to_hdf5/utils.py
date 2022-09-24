import matplotlib.pyplot as plt
import tqdm
import numpy as np
import os


def str2bool(s):
    if s.lower() in ['1', 'true', 't']:
        return True
    if s.lower() in ['0', 'false', 'f']:
        return False
    raise ValueError("Invalid entry. Must be 1/0, True/False, T/F")


def vis_freeform_annotations(ct, annotations, vis_path):
    num_creators = annotations['annotations'].shape[0]
    depths = annotations['annotations'].shape[1]

    for depth in range(depths):
        plt.close('all')
        if ct is not None:
            fig, axs = plt.subplots(1, num_creators + 1, figsize=(10, 4))
            axs[0].imshow(ct[depth])
        else:
            fig, axs = plt.subplots(1, num_creators, figsize=(5, 2))
            if num_creators == 1:
                axs = [axs]
        pixel_count = 0
        for creator in range(num_creators):
            index = creator + 1 if ct is not None else creator
            axs[index].imshow(annotations['annotations'][creator, depth])
            pixel_count += np.sum(annotations['annotations'][creator, depth] > 0)

        if pixel_count > 0:
            plt.tight_layout()
            plt.savefig(os.path.join(vis_path, 'z{}.png'.format(depth)), dpi=150)