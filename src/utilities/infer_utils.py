import pydicom as dcm
from torch.utils.data import Dataset
import os
import numpy as np
from multiprocessing import Pool
from src.utilities.mask_utils import *
from src.utilities.image_utils import image_orientation, construct_nii
import nibabel as nib


def save_nii(output_dir, ct, outputs, slice_thickness):
    color = np.array([[0, 150, 255],
            [255, 189, 51],
            [255, 51, 51]]).astype(np.uint8)

    nii, mask = construct_nii(ct, outputs, color)

    nii = nib.Nifti1Image(nii, np.eye(4))
    nii.header['pixdim'][1:4] = [1, 1, slice_thickness]
    if mask is not None:
        mask = nib.Nifti1Image(mask, np.eye(4)) 
        mask.header['pixdim'][1:4] = [1, 1, slice_thickness]
        mask.header['cal_min'] = 0
        mask.header['cal_max'] = 4
    else:
        mask = None

    os.makedirs(output_dir, exist_ok=True)
    nib.save(nii, os.path.join(output_dir,"inference.nii.gz"))
    nib.save(mask, os.path.join(output_dir,"mask.nii.gz"))


    # Save plain CT
    nii, _ = construct_nii(ct)
    nii = nib.Nifti1Image(nii, np.eye(4))
    nii.header['pixdim'][1:4] = [1, 1, slice_thickness]
    os.makedirs(output_dir, exist_ok=True)
    nib.save(nii, os.path.join(output_dir,"CT.nii.gz"))


class DICOMSeries(Dataset):

    def __init__(self, dataset_path=None, study_uid=None, series_uid=None, image_transform=None):
        self.bad_series = False
        self.dataset_path = dataset_path
        self.study_uid = study_uid
        self.series_uid = series_uid
        self.patient_id = None
        self.image_transform = image_transform
        self.slice_thickness = None
        self.slice_increment = None
        self.sopinstanceuids, self.instance_numbers, self.pixel_data, self.sliceID, self.orientation = [None] * 5
        self.inference = None

        self._load_dicom(self.series_uid) if self.series_uid is not None else None

    def __len__(self):
        return np.shape(self.pixel_data)[0]

    def __getitem__(self, item):
        image = self.pixel_data[item].astype(np.float32)
        image = self.image_transform(image)
        out = {
            'image': image,
            'scanID': self.study_uid,
            'seriesID': self.series_uid,
            'sliceID': self.sliceID[item],
        }
        return out

    def _load_dicom(self, series_uid):
        series_directory = os.path.join(self.dataset_path, self.study_uid, series_uid)
        if os.path.isdir(series_directory):
            try:
                dcm_files = [file for file in os.listdir(series_directory) if os.path.splitext(file)[-1] == '.dcm']
                dcm_paths = [os.path.join(series_directory, file) for file in dcm_files]
            except Exception as e:
                print(e)
                return None

            pixel_data = []
            sopinstanceuids = []
            instance_numbers = []
            z_positions = []
            orientation = None

            for dcm_file, dcm_path in zip(dcm_files, dcm_paths):
                dicom = dcm.read_file(dcm_path)
                try:
                    intercept = float(dicom["RescaleIntercept"].value)
                    slope = float(dicom["RescaleSlope"].value)
                    pixel_array = (dicom.pixel_array * slope + intercept).astype(np.int)
                    z_positions.append(int(dicom[0x20, 0x32].value[-1]))
                    pixel_data.append(np.expand_dims(pixel_array, 0))
                    sopinstanceuids.append(dicom["SOPInstanceUID"].value)
                    instance_numbers.append(int(dicom.InstanceNumber))
                    if dicom.get((0x18, 0x50)) is not None:
                        if dicom[(0x18, 0x50)].value is not None:
                            self.slice_thickness = float(dicom[(0x18, 0x50)].value)
                    self.patient_id = dicom["PatientID"].value
                    orientation = image_orientation(dicom)
                except Exception as e:
                    self.bad_series = True
                    error_msg = e

            if not self.bad_series :
                sort_order = np.argsort(instance_numbers)
                instance_numbers = np.array(instance_numbers)[sort_order]
                sopinstanceuids = np.array(sopinstanceuids)[sort_order]
                z_positions = np.array(z_positions)[sort_order]
                pixel_data = [pixel_data[idx] for idx in sort_order]
                self.slice_increment = np.abs(np.diff(z_positions).mean())
                if self.slice_increment == 0:
                    orientation = None
            
            if orientation == 'transverse':
                try:
                    self.pixel_data = np.concatenate(pixel_data, axis=0)
                    self.instance_numbers = instance_numbers
                    self.sopinstanceuids = sopinstanceuids
                    self.sliceID = z_positions
                    self.orientation = orientation
 
                except Exception as e:
                    self.bad_series = True
                    error_msg = e

            if self.bad_series:    
                print("Error encountered. Message: {}".format(error_msg))
                return None

            if orientation != 'transverse':
                print("Skipping. Incorrect Orientation", flush=True)
                self.bad_series = True
                return None

            # NOTE: If there is a need to extract any DICOM header data, access the 'dicom' object. See a list of
            # header information by print(dicom). Modify the function return.
        else:
            print(series_directory)
            with open(os.path.join('missing_CT_series.log'), 'a') as f:
                print("{}".format(self.study_uid), file=f)
                print("\t{}".format(series_uid), file=f)
