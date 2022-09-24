import os
import h5py
import numpy as np
from PIL import Image
import pydicom as dcm
import pandas as pd
from medpy.io import load
import cv2

from src.utilities.mask_utils import *
from .utils import vis_freeform_annotations
from src.utilities.directory_utils import UIDMapper
import matplotlib.pyplot as plt
from collections import OrderedDict


def load_mask_instance(row):
    """Load instance masks for the given annotation row. Masks can be different types,
    mask is a binary true/false map of the same size as the image.
    """

    mask = np.zeros((int(row.height), int(row.width)), dtype=np.uint8)

    annotation_mode = row.annotationMode

    if annotation_mode == "bbox":
        # Bounding Box
        x = int(row["data"]["x"])
        y = int(row["data"]["y"])
        w = int(row["data"]["width"])
        h = int(row["data"]["height"])
        mask_instance = mask[:,:].copy()
        cv2.rectangle(mask_instance, (x, y), (x + w, y + h), 255, -1)
        mask[:,:] = mask_instance

    # FreeForm or Polygon
    elif annotation_mode == "freeform" or annotation_mode == "polygon":
        vertices = np.array(row["data"]["vertices"])
        vertices = vertices.reshape((-1, 2))
        mask_instance = mask[:,:].copy()
        cv2.fillPoly(mask_instance, np.int32([vertices]), (255, 255, 255))
        mask[:,:] = mask_instance

    # Line
    elif annotation_mode == "line":
        vertices = np.array(row["data"]["vertices"])
        vertices = vertices.reshape((-1, 2))
        mask_instance = mask[:,:].copy()
        cv2.polylines(mask_instance, np.int32([vertices]), False, (255, 255, 255), 12)
        mask[:,:] = mask_instance

    elif annotation_mode == "location":
        # Bounding Box
        x = int(row["data"]["x"])
        y = int(row["data"]["y"])
        mask_instance = mask[:,:].copy()
        cv2.circle(mask_instance, (x, y), 7, (255, 255, 255), -1)
        mask[:,:] = mask_instance

    elif annotation_mode == "mask":
        mask_instance = mask[:, :].copy()
        if row.data["foreground"]:
            for i in row.data["foreground"]:
                mask_instance = cv2.fillPoly(mask_instance, [np.array(i, dtype=np.int32)], (255, 255, 255))
        if row.data["background"]:
            for i in row.data["background"]:
                mask_instance = cv2.fillPoly(mask_instance, [np.array(i, dtype=np.int32)], (0,0,0))
        mask[:, :] = mask_instance

    elif annotation_mode is None:
        print("Not a local instance")


    return mask.astype(np.bool)


def visualize_hdf5(ct, label, viz_path, sop, idx):
    color_lookup = {0: (186, 223, 85),
                    1: (53, 177, 201),
                    2: (176, 109, 173)}

    # threshold and scale the image for saving
    
    ct[ct > 350] = 350
    ct[ct < -1000] = -1000
    ct = (ct - ct.min()) / (ct.max() - ct.min())

    # convert from 0-1 into 0-255
    ct = np.tile(np.expand_dims(ct * 255, -1), (1, 1, 3)).astype(np.uint8)
    img = np.copy(ct)

    # overlay probability mask
    for cidx in range(0, 3):
        for channel in range(0, 3):
            img[:, :, channel][label[cidx] == 1] = color_lookup[cidx][channel]

    img = Image.fromarray(img)
    img.save(os.path.join(viz_path, '{}-{}_label.tiff'.format(idx, sop[0:5])))

    ct = Image.fromarray(ct)
    ct.save(os.path.join(viz_path, '{}-{}_ct.tiff'.format(idx, sop[0:5])))


class StudyInstanceAnnotations:

    def __init__(self, dataset_path, lung_model_path, save_path, studyinstanceuid, study_df, data_use, visualize, error_path, debug=True):
        self.dataset_path = dataset_path
        self.lung_model_path = lung_model_path
        self.save_path = save_path
        self.study_uid = studyinstanceuid
        self.study_df = study_df
        self.data_use = data_use
        self.visualize = visualize
        self.error_path = error_path
        self.debug = debug

        self.StudyUIDMapper = UIDMapper(dataset_path)
        os.makedirs(self.save_path, exist_ok=True)
        self.exists = True if os.path.isdir(os.path.join(self.dataset_path, self.study_uid)) else False

    def build_study(self):
        study_summary = []
        
        hdf5_path = os.path.join(self.save_path, 'hdf5')
        csv_path = os.path.join(self.save_path, 'csv')
        os.makedirs(hdf5_path, exist_ok=True)
        os.makedirs(csv_path, exist_ok=True)
        study_path = os.path.join(hdf5_path, self.study_uid + '.hdf5')
        if not os.path.isfile(study_path):
            study_hdf5 = h5py.File(study_path, 'w') if not self.debug else None
            series_uids = self.study_df['SeriesInstanceUID'].unique()
            for series_uid in series_uids:
                series = self._build_series(series_uid)
                if series is None:
                    continue
                patient_id = series['PatientID']
                series_hdf5 = study_hdf5.create_group(series_uid) if not self.debug else None
                series_hdf5.create_dataset('CT', data=series['CT'], compression="gzip", compression_opts=9)                
                series_hdf5.create_dataset('Parenchyma', data=series['Parenchyma'], compression="gzip", compression_opts=9)
                series_hdf5.create_dataset('Bone', data=series['Bone'], compression="gzip", compression_opts=9)
                series_hdf5.create_dataset('Effusion', data=series['Effusion'], compression="gzip", compression_opts=9)
                series_hdf5.create_dataset('SOPInstanceUIDs', data=np.array(series['SOPUIDs'], dtype='S75'))
                series_hdf5.create_dataset('SOPLabels', data=np.array(series['SOPLabels'], dtype='S75'))
                series_hdf5.create_dataset('DataUse', data=np.array(['Test' if self.data_use == 'test' else 'Train'], dtype='S75'))

                # Vesselness processing
                series_directory = os.path.join(self.dataset_path, self.study_uid, series_uid)
                pixel_data, sop_instance_uids = load_dicom(series_directory)
                lung_mask = get_lung_mask(pixel_data, self.lung_model_path)
                vesselness = get_vesselness_mask(pixel_data, lung_mask)
                keep_idx = []
                for idx, sop in enumerate(sop_instance_uids):
                    if sop in series['SOPUIDs']:
                        keep_idx.append(idx)
                vesselness = vesselness[keep_idx]
                series_hdf5.create_dataset('Vesselness', data=vesselness, compression="gzip", compression_opts=9)

                # Generate a summary of everything made in this series
                for item in self._process_series_summary(series):
                    study_summary.append(item)

            study_hdf5.close()
            pid_lookup_file = os.path.join(csv_path, 'patientID_lookup.csv')
            with open(pid_lookup_file, 'a') as f:
                if not os.path.isfile(pid_lookup_file):
                    print("StudyInstanceUID,PatientID", file=f)
                print("{},{}".format(self.study_uid, patient_id), file=f)

            summary_df = pd.DataFrame(study_summary, columns=study_summary[0].keys())
            summary_csv = os.path.join(csv_path, 'data_summary.csv')
            summary_df.to_csv(summary_csv, header=True) if not os.path.isfile(summary_csv) else summary_df.to_csv(summary_csv, header=False, mode='a')
            
            return summary_csv

    def _process_series_summary(self, series):
        # Process summary 
        series_summary = []
        for ct_idx in range(0,series['CT'].shape[0]):
            series_summary.append(OrderedDict())
            series_summary[-1]['Dataset'] = self.study_df['dataset'].values[0]
            series_summary[-1]['DatasetID'] = self.study_df['datasetId'].values[0]
            series_summary[-1]['StudyInstanceUID'] = self.study_uid
            series_summary[-1]['SeriesInstanceUID'] = series['SeriesUID']
            series_summary[-1]['SOPInstanceUID'] =  series['SOPUIDs'][ct_idx]
            series_summary[-1]['Dim0Index'] = ct_idx
            series_summary[-1]['ParenchymaData'] = 'False' if np.all(series['Parenchyma'][ct_idx] == -1) else 'True'
            series_summary[-1]['BoneData'] = 'False' if np.all(series['Bone'][ct_idx] == -1) else 'True'
            series_summary[-1]['EffusionData'] ='False' if np.all(series['Effusion'][ct_idx] == -1) else 'True'
            series_summary[-1]['DataUseAlloc'] = 'Test' if self.data_use == 'test' else 'Train'
        
        return series_summary

    def _build_series(self, series_uid):
        series_df = self.study_df[self.study_df['SeriesInstanceUID'] == series_uid]
        sop_uids = series_df['SOPInstanceUID'].unique()
        
        ct_data = []
        parenchyma_data = []
        bone_data = []
        effusion_data = []
        sop_label = []
        order = []

        for sop_uid in sop_uids:
            sop_df = self.study_df[self.study_df['SOPInstanceUID'] == sop_uid]
            if isinstance(sop_uid, str):
                sop = self._build_sop(series_uid, sop_uid, sop_df)
                if sop is None:
                    continue
                patient_id = sop['PatientID']
                ct_data.append(sop['CT'])
                parenchyma_data.append(sop['Parenchyma'])
                bone_data.append(sop['Bone'])
                effusion_data.append(sop['Effusion'])
                sop_label.append(sop['SOPLabels'])
                order.append(sop['Order'])

        if len(order) == 0:
            return None
        
        sort_order = np.argsort(order)
        ct = np.vstack([np.expand_dims(ct_data[idx], 0) for idx in sort_order])
        parenchyma = np.vstack([np.expand_dims(parenchyma_data[idx], 0) for idx in sort_order])
        bone = np.vstack([np.expand_dims(bone_data[idx], 0) for idx in sort_order])
        effusion = np.vstack([np.expand_dims(effusion_data[idx], 0) for idx in sort_order])
        sop_label = [sop_label[idx] for idx in sort_order]
        sop_uids = sop_uids[sort_order]

        viz_path = os.path.join(self.save_path, 'viz')
        if self.visualize:
            for idx in range(0,ct.shape[0]):
                img_path = os.path.join(viz_path,self.study_uid,series_uid)
                os.makedirs(img_path,exist_ok=True)
                visualize_hdf5(ct[idx], parenchyma[idx], img_path, sop_uids[idx], idx)
        
        return {'PatientID': patient_id, 'SeriesUID': series_uid, 'CT': ct, 'Parenchyma': parenchyma, 'Bone': bone, 'Effusion': effusion, 'SOPUIDs': sop_uids, 'SOPLabels': sop_label}

    def _build_sop(self, series_uid, sop_uid, sop_df):
        dicom = dcm.read_file(os.path.join(self.dataset_path, self.StudyUIDMapper.lookup(self.study_uid), series_uid, sop_uid + '.dcm'))
        intercept = float(dicom["RescaleIntercept"].value)
        slope = float(dicom["RescaleSlope"].value)
        patient_id = dicom["PatientID"].value
        ct = (dicom.pixel_array * slope + intercept).astype(np.int)
        order = dicom.InstanceNumber

        labelNames = sop_df['labelName'].unique()
        labelMasks = {}
        SOPLabels = []

        # Skip SOP/Slice if missing AI Lung Mask
        if 'AI Lung Mask' not in labelNames: 
            return None

        for labelName in labelNames:
            if labelName in ['AI Normal Lung', 'Normal']: # Normal lung will be created by flexible THRESHOLDING. Ignore these labels
                continue  
            elif labelName in ['Artifact']: # <- SOP/Image-level labels
                SOPLabels.append(labelName)
            else:
                label_df = sop_df[sop_df['labelName'] == labelName]
                mask = np.zeros(ct.shape).astype(np.bool)
                
                for _, row in label_df.iterrows():
                    mask = np.bitwise_or(mask, load_mask_instance(row)) if row.data is not None else mask

            labelMasks[labelName] = mask 
        
        parenchyma = self._process_parenchyma_labels(ct, labelMasks)
        bone = labelMasks['Bone'] if 'Bone' in labelMasks.keys() else np.ones(ct.shape) * -1
        effusion = labelMasks['Effusion'] if 'Effusion' in labelMasks.keys() else np.ones(ct.shape) * -1
        SOPLabels.append("") if len(SOPLabels) == 0 else None
        return {'PatientID': patient_id, 'CT': ct, 'Parenchyma': parenchyma, 'Bone': bone, 'Effusion': effusion, 'SOPLabels': SOPLabels, 'Order': order}

    def _process_parenchyma_labels(self, ct, labelMasks):
        parenchyma = np.tile(np.ones(ct.shape) * -1, [3, 1, 1]) # default mask to no data

        # create lung mask by combining AI Lung Mask and substracting Background, if it exists
        lung = labelMasks['AI Lung Mask'] if 'AI Lung Mask' in labelMasks.keys() else None
        if  lung is not None:
            if 'Background' in labelMasks.keys(): # Manually removed lung mask
               lung[labelMasks['Background']] = 0
            if 'Lung' in labelMasks.keys(): # Manually added lung mask
                lung[labelMasks['Lung']] = 1
        lung = lung.astype(np.bool)

        # calculate GGO from mask and upper/lower HU boundaries
        ggo = labelMasks['GGO'].astype(np.bool) if 'GGO' in labelMasks.keys() else np.zeros(lung.shape).astype(np.bool)

        # build consolidation mask
        consolidation = labelMasks['Consolidation'] if 'Consolidation' in labelMasks.keys() else np.zeros(ct.shape).astype(np.bool)

        # build parenchyma labels based on order of lowest priority to highest priority
        parenchyma[0] = lung  # fully labeled [0, 1] ONLY
        parenchyma[1][ggo] = 1  # incompletely labeled [-1, 1] ONLY
        parenchyma[2] = consolidation # fully labeled [0, 1] ONLY

        return parenchyma