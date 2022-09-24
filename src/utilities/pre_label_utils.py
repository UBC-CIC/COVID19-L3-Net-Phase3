from matplotlib import pyplot as plt
import pydicom as dcm
from torch.utils.data import Dataset
import os
import numpy as np
from PIL import Image
from skimage import measure as skmeasure
from tqdm import tqdm
from src.mdai_tools.upload_to_mdai import extract_max_prob_points
import h5py
from multiprocessing import Pool
from src.utilities.mask_utils import *
from src.utilities.image_utils import image_orientation


def save_visuals(idx, save_path, ct, inference, point_mode):
    # print("start save visual {}".format(idx), flush=True)
    color_lookup = {0: (235, 200, 61),
                    1: (186, 223, 85),
                    2: (53, 177, 201),
                    3: (176, 109, 173),
                    4: (233, 96, 96)}

    points = extract_max_prob_points(inference, mode=point_mode)

    # threshold and scale the image for saving
    ct[ct > 350] = 350
    ct[ct < -1000] = -1000
    ct = (ct - ct.min()) / (ct.max() - ct.min())

    # convert from 0-1 into 0-255
    ct = np.tile(np.expand_dims(ct * 255, -1), (1, 1, 3)).astype(np.uint8)
    img = np.copy(ct)

    # overlay probability mask
    if inference is not None:
        for cidx in range(1, inference.shape[0]):
            inferred_class = inference[cidx]
            for channel in range(0, 2):
                img[:, :, channel][inferred_class > 0] = color_lookup[cidx][channel]

    # plot points of maximum probability
    if points is not None:
        for label_idx in range(1, 5):
            labels = [labels[1:] for labels in points if labels[0] == label_idx]
            for point in labels:
                img[point[0], point[1], :] = [255, 255, 255]

    img = Image.fromarray(img)
    img.save(os.path.join(save_path, 'slice{:03d}_label.tiff'.format(idx)))

    ct = Image.fromarray(ct)
    ct.save(os.path.join(save_path, 'slice{:03d}_ct.tiff'.format(idx)))


class DICOMSeries(Dataset):

    def __init__(self, dataset_path=None, study_uid=None, series_uid=None, image_transform=None, debug=False,
                 infer_every_mm=1, lung_model_path=None, point_mode='abs', normal_lung_sz_thresh=0, normal_lung_hu_thresh=0):
        self.bad_series = False
        self.dataset_path = dataset_path
        self.study_uid = study_uid
        self.series_uid = series_uid
        self.patient_id = None
        self.debug = debug
        self.infer_every_mm = infer_every_mm
        self.image_transform = image_transform
        self.lung_model_path = lung_model_path
        self.point_mode = point_mode
        self.slice_thickness = None
        self.slice_increment = None
        self.sopinstanceuids, self.instance_numbers, self.pixel_data, self.sliceID, self.orientation = [None] * 5
        self.inference = None
        self.lung_mask = None
        self.vesselness = None
        self.normal_lung_sz_thresh = normal_lung_sz_thresh
        self.normal_lung_hu_thresh = normal_lung_hu_thresh
        self.normal_lung_by_hu = None
        self.max_probs_loc = None

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

    def _get_normal_lung_by_hu(self):
        normal_lung = self.pixel_data <= self.normal_lung_hu_thresh
        for sidx in range(0, self.pixel_data.shape[0]):
            select_slice = normal_lung[sidx, :, :] 
            labeled, n_objects = skmeasure.label(select_slice, return_num=True)
            
            for obj_idx in range(1, n_objects + 1):
                selected_obj = np.isin(labeled, obj_idx)
                if np.sum(selected_obj) < self.normal_lung_sz_thresh:
                    select_slice[selected_obj] = 0  # silence objects if the object is smaller than minimum size

            normal_lung[sidx, :, :] = select_slice
        return normal_lung

    def attach_inference(self, inference):
        self.inference = inference

    def save_inference(self, save_path):
        # save the inference with select header info
        os.remove(save_path) if os.path.isfile(save_path) else None
        with h5py.File(save_path, 'a') as file:
            # create headers
            header = file.create_group('header')
            header.attrs.create('bad_series', data=self.bad_series)
            header.attrs.create('dataset_path', data=self.dataset_path)
            header.attrs.create('patient_id', data=self.patient_id)
            header.attrs.create('study_uid', data=self.study_uid)
            header.attrs.create('series_uid', data=self.series_uid)
            header.attrs.create('infer_every_mm', data=self.infer_every_mm)
            header.attrs.create('slice_thickness', data=self.slice_thickness)
            header.attrs.create('slice_increment', data=self.slice_increment)
            header.attrs.create('sopinstanceuids', data=np.array(self.sopinstanceuids, dtype='S'))
            header.attrs.create('instance_numbers', data=self.instance_numbers)
            header.attrs.create('sliceID', data=self.sliceID)
            header.attrs.create('orientation', data=self.orientation)

            # attach data
            file.create_dataset('inference', data=self.inference, compression="gzip", compression_opts=9)
            file.create_dataset('lung_mask', data=self.lung_mask, compression="gzip", compression_opts=9)
            file.create_dataset('vesselness', data=self.vesselness, compression="gzip", compression_opts=9)

            if self.normal_lung_by_hu is not None:
                file.create_dataset('normal_lung_by_hu', data=self.normal_lung_by_hu, compression="gzip", compression_opts=9)
                header.attrs.create('normal_lung_sz_thresh', data=self.normal_lung_sz_thresh)
                header.attrs.create('normal_lung_hu_thresh', data=self.normal_lung_hu_thresh)
                
            file.close()

    def load_inference(self, load_path):
        file = h5py.File(load_path, 'r')

        # read full header data
        self.bad_series = file['header'].attrs.get('bad_series')
        self.dataset_path = file['header'].attrs.get('dataset_path')
        self.patient_id = file['header'].attrs.get('patient_id')
        self.study_uid = file['header'].attrs.get('study_uid')
        self.series_uid = file['header'].attrs.get('series_uid')
        self.debug = file['header'].attrs.get('debug')
        self.infer_every_mm = file['header'].attrs.get('infer_every_mm')
        self.slice_thickness = file['header'].attrs.get('slice_thickness')
        self.slice_increment = file['header'].attrs.get('slice_increment')
        self.sopinstanceuids = file['header'].attrs.get('sopinstanceuids')
        self.instance_numbers = file['header'].attrs.get('instance_numbers')
        self.pixel_data = file['header'].attrs.get('pixel_data')
        self.sliceID = file['header'].attrs.get('sliceID')
        self.orientation = file['header'].attrs.get('orientation')

        # return the pointer to open h5 object for as-needed access
        self.inference = file['inference']
        self.lung_mask = file['lung_mask']

        if 'normal_lung_by_hu' in file:
            self.normal_lung_by_hu = file['normal_lung_by_hu']
            self.normal_lung_sz_thresh = file['header'].attrs.get('normal_lung_sz_thresh')
            self.normal_lung_hu_thresh = file['header'].attrs.get('normal_lung_hu_thresh')

    def find_max_probability(self):
        if self.inference is not None:
            self.max_probs_loc = extract_max_prob_points(self.inference, min_size=10, mode=self.point_mode)
        else:
            print("Unable to extract points of maximum probability since inference has not been attached")

    def _load_dicom(self, series_uid):
        series_directory = os.path.join(self.dataset_path, self.study_uid, series_uid)
        if not self.debug and os.path.isdir(series_directory):
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
                    slice_thickness = dicom[0x18, 0x50].value
                    z_positions.append(int(dicom[0x20, 0x32].value[-1]))
                    pixel_data.append(np.expand_dims(pixel_array, 0))
                    sopinstanceuids.append(dicom["SOPInstanceUID"].value)
                    instance_numbers.append(int(dicom.InstanceNumber))
                    self.slice_thickness = float(slice_thickness) if slice_thickness is not None else -1.
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
                    pixel_data = np.concatenate(pixel_data, axis=0)

                    # Read slice thickness and reduce the number of slices to keep if requested
                    print("{}/{}: ".format(self.patient_id, self.series_uid), end='')
                    if self.infer_every_mm is not None:
                        if self.infer_every_mm > self.slice_increment:
                            steps = np.int(np.round(self.infer_every_mm / self.slice_increment, 0))
                            self.slice_increment = np.round(self.slice_increment * steps, 1)
                        else:
                            steps = 1  # use all slices
                            self.slice_increment = np.round(self.slice_increment, 1)
                        print("q {} slice (~{} mm) - ".format(steps, self.slice_increment), end='')
                        keep_idx = np.arange(0, len(sopinstanceuids), steps)
                        self.instance_numbers = instance_numbers[keep_idx]
                        self.sopinstanceuids = sopinstanceuids[keep_idx]
                        self.pixel_data = pixel_data[keep_idx, :, :]
                        self.sliceID = z_positions[keep_idx]  # only makes sense in axial/transverse slices.
                        self.orientation = orientation

                        lung_mask = get_lung_mask(pixel_data, self.lung_model_path)                
                        vesselness = get_vesselness_mask(pixel_data, lung_mask) 
                        self.lung_mask = lung_mask[keep_idx]
                        self.vesselness = vesselness[keep_idx]  
                        if self.normal_lung_sz_thresh > 0:
                            self.normal_lung_by_hu = self._get_normal_lung_by_hu() * self.lung_mask

                        # visualize vesselness for debug
                        test_dir = 'vessel_test/{}'.format(self.series_uid)
                        os.makedirs(test_dir, exist_ok=True)
                        for idx in tqdm(range(self.vesselness.shape[0]),leave=False): 
                            # plt.imsave(os.path.join(test_dir,'{}_vessel.png'.format(idx)), vesselness[idx], cmap='gray')
                            plt.imsave(os.path.join(test_dir,'{}_image.png'.format(idx)), self.pixel_data[idx], cmap='gray') 
                            for thresh in tqdm(np.arange(0.05, .30, 0.05), leave=False):
                                pixel_3 = np.stack([self.pixel_data[idx], self.pixel_data[idx], self.pixel_data[idx]], axis=-1)
                                pixel_3[self.vesselness[idx] > thresh, 0] = 1
                                plt.imsave(os.path.join(test_dir,'{}_thresh{:.2f}.png'.format(idx, thresh)), pixel_3, cmap='gray')
 
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

    def visualize(self, save_path, post_process=True):
        if self.pixel_data is not None:
            # code for multiprocess saving visualizations
            mp = True  # to multiprocess or not multiprocess
            pool = Pool(os.cpu_count() - 2 if os.cpu_count() >= 3 else 1) if mp else None

            # attach an inference output to the object
            if post_process:
                inference = self.inference
                for midx in range(0, inference.shape[1]):
                    inference[:, midx, :, :] = inference[:, midx, :, :] * self.lung_mask
                for cidx in range(0, inference.shape[0]):
                    # threshold the inferences
                    threshold = 0.5  # if cidx == 0 else 0.8
                    inference[cidx, inference[cidx] < threshold] = 0
            else:
                inference = self.inference

            # tweak inference if normal_lung_by_hu
            if self.normal_lung_by_hu is not None:
                viz_data = self.inference
                viz_data[:, 1, :, :] = self.normal_lung_by_hu

            for idx in range(0, self.pixel_data.shape[0]):
                if mp:
                    pool.apply_async(save_visuals,
                                        (self.instance_numbers[idx], save_path, self.pixel_data[idx, :, :],
                                        inference[idx], self.point_mode))
                else:
                    save_visuals(self.instance_numbers[idx], save_path, self.pixel_data[idx, :, :],
                                 inference[idx], self.point_mode)

            if mp:
                pool.close()
                pool.join()
