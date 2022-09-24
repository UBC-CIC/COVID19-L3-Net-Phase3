import itk 
from scipy.ndimage.morphology import binary_erosion
from src.lungmask import mask as mask_lungs
import os
from src.utilities.image_utils import image_orientation
import pydicom as dcm
import numpy as np


def load_dicom(series_directory):
    dcm_files = [file for file in os.listdir(series_directory) if os.path.splitext(file)[-1] == '.dcm']
    dcm_paths = [os.path.join(series_directory, file) for file in dcm_files]
    pixel_data = []
    sopinstanceuids = []
    instance_numbers = []
    z_positions = []
    orientation = None

    bad_series = False
    for dcm_file, dcm_path in zip(dcm_files, dcm_paths):
        dicom = dcm.read_file(dcm_path)
        try:
            intercept = float(dicom["RescaleIntercept"].value)
            slope = float(dicom["RescaleSlope"].value)
            pixel_array = (dicom.pixel_array * slope + intercept).astype(np.int)
            slice_thickness = dicom[0x18, 0x50].value
            z_positions.append(int(dicom[0x20, 0x32].value[-1]))
            pixel_data.append(np.expand_dims(pixel_array, 0))
            sopinstanceuids.append(dicom["SOPInstanceUID"].value)
            instance_numbers.append(int(dicom.InstanceNumber))
            slice_thickness = float(slice_thickness) if slice_thickness is not None else -1.
            orientation = image_orientation(dicom)
        except Exception as e:
            bad_series = True

    if not bad_series :
        sort_order = np.argsort(instance_numbers)
        instance_numbers = np.array(instance_numbers)[sort_order]
        sopinstanceuids = np.array(sopinstanceuids)[sort_order]
        z_positions = np.array(z_positions)[sort_order]
        pixel_data = [pixel_data[idx] for idx in sort_order]
        slice_increment = np.abs(np.diff(z_positions).mean())
        if slice_increment == 0:
            orientation = None
    
    if orientation == 'transverse':
        pixel_data = np.concatenate(pixel_data, axis=0)

    return pixel_data, sopinstanceuids


def get_vesselness_mask(pixel_data, lung_mask):
    sigma = 2.0
    alpha1 = 0.5
    alpha2 = 1.0

    pixel_data[pixel_data == pixel_data.min()] = -1024  # set background to air
    pixel_data = (pixel_data - pixel_data.min())/(pixel_data.max()-pixel_data.min())
    pixel_data = pixel_data.astype(float)

    input_image = itk.GetImageFromArray(pixel_data)
    hessian_image = itk.hessian_recursive_gaussian_image_filter(input_image, sigma=sigma)

    vesselness_filter = itk.Hessian3DToVesselnessMeasureImageFilter[itk.ctype("float")].New()
    vesselness_filter.SetInput(hessian_image)
    vesselness_filter.SetAlpha1(alpha1)
    vesselness_filter.SetAlpha2(alpha2)

    vesselness = itk.GetArrayFromImage(vesselness_filter) * binary_erosion(lung_mask, iterations=3)
    vesselness = (vesselness - vesselness.min())/(vesselness.max() - vesselness.min())
    vesselness = vesselness

    return vesselness


def get_lung_mask(pixel_data, lung_model_path):
    model = mask_lungs.get_model('unet', 'R231CovidWeb', modelpath=lung_model_path)
    mask = mask_lungs.apply(pixel_data, model)
    mask[mask > 0] = 1
    return mask