from PIL import Image
import numpy as np
import os
import torch
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw 
import nibabel as nib


def image_orientation(dicom):
    orientation = list(np.round(dicom[0x20, 0x37].value).astype(int))
    if orientation == [1, 0, 0, 0, 1, 0]:
        return 'transverse'
    if orientation == [1, 0, 0, 0, 0, -1]:
        return 'coronal'
    else:
        return None


def save_tiff(save_dir, filename, data):
    data_scaled = (data * 255).astype(np.uint8)
    im = Image.fromarray(data_scaled)
    im.save(
        os.path.join(save_dir, filename)
    )


def save_rgba(save_dir, filename, x):
    alpha = np.sum(x, axis=-1, keepdims=True)
    inds = np.where(alpha > 0)
    x[inds[0], inds[1], :] = x[inds[0], inds[1], :] / alpha[inds[0], inds[1], :]

    x_bar_scaled = (x * 255).astype(np.uint8)
    im = Image.fromarray(x_bar_scaled)

    alpha_scaled = (alpha * 255).astype(np.uint8)
    im_alpha = Image.fromarray(np.squeeze(alpha_scaled))

    im.putalpha(im_alpha)
    im.save(
        os.path.join(save_dir, filename)
    )


def construct_nii(ct, outputs=None, color=None, text=None):
    # scale the ct using min max, then to 0 255
    nii = torch.clamp(ct, min=-0.6, max=1.5)
    nii = torch.unsqueeze(torch.clamp((nii + 0.6)/(1.5 + 0.6) * 255, 0, 255), -1)
    nii = nii.cpu().numpy().astype(np.uint8)

    # add labeled
    if text is not None:
        for idx in range(0, nii.shape[0]):
            lbl = Image.fromarray(nii[idx, 0, ..., 0])
            draw = ImageDraw.Draw(lbl)
            font = ImageFont.truetype("src/resources/OpenSans-Regular.ttf", 36)
            draw.text((9, 0), text, (255), font=font)
            nii[idx, 0, ..., 0] = np.array(lbl)

    # resize/format ct/nifti to accomodate RGB embedded-overlay
    if outputs is not None:
        nii = np.tile(nii, (1, 1, 1, 3))
        classes = torch.argmax(outputs, dim=1, keepdim=True).cpu().numpy()

        # modify classes argmax to hide vessels
        classes[classes == 1] = 0

        mask = np.expand_dims(classes, -1)
        classes = np.tile(mask, (1,1,1,1,3))

        # add color if color lookup array is provided
        if color is not None:
            for class_id in range(2,5):
                colorgrid = np.expand_dims(color[class_id - 2], axis=(0,1,2,3))
                colorgrid = np.tile(colorgrid, np.divide(nii.shape, colorgrid.shape).astype(int))
                nii = np.where(classes == class_id, colorgrid, nii)
            
    # transpose volume for appropriate rendering in NIFTI viewer
    nii = np.transpose(nii[:,0,...], (2, 1, 0, 3))
    nii = np.flip(nii, (0, 1, 2)) if outputs is not None else np.flip(nii)
    if outputs is not None:
        mask = np.transpose(mask[:,0,...], (2, 1, 0, 3))
        mask = np.squeeze(mask)
        mask = np.flip(mask)
        shape_3d = nii.shape[0:3]
        rgb_dtype = np.dtype([('R', 'u1'), ('G', 'u1'), ('B', 'u1')])
        nii = nii.copy().view(dtype=rgb_dtype).reshape(shape_3d)  # copy used 
        mask = mask.copy()
    else:
        nii = nii[0:-2].copy()
        mask = None
    #to force fresh internal structure
    return nii, mask

