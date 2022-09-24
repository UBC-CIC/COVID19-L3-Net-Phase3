from typing import OrderedDict
import torch
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
import os
import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

# Statistics preserved from Phase 1 
MEAN = [-653.2204]
STD = [628.5188]
PHASE3_CLASSES = {-1: 'Unlabelled', 0: 'Background (Inverse Lung)', 1:'Normal Lung', 2:'GGO', 3:'Consolidation'}


class Threshold:
    def __init__(self, min=None, max=None):
        self.min = min
        self.max = max

    def __call__(self, x):
        assert isinstance(x, np.ndarray), 'Input to threshold must be a np.ndarray'
        x = np.clip(x, self.min, self.max)

        return x


class Squeeze:
    def __call__(self, x):
        return x.squeeze()


def build_data_df(search_path, data_use_filter=None, holdout_size=None, slice_thickness=None, verbose=False):
    # slice thickness can be THICK (>2) or THIN (<=2)
    data_df = []

    # Find all data_summary.csv
    for root, _, files in os.walk(search_path):
        for file in files:
            if file == 'data_summary.csv':
                temp_df = pd.read_csv(os.path.join(root, file), header=0, index_col=0)
                temp_df['AbsFilePath'] = ''
                hdf5_path = os.path.dirname(os.path.dirname(root))
                
                # For each data_summary.csv corresponding to one dataset folder, assign every hdf5 filepath within data_summary.csv as AbsFilePath
                for root, _, files in os.walk(hdf5_path):
                    for file in files:
                        if file.endswith('.hdf5'):
                            file_uid = os.path.split(file)[-1].replace('.hdf5','')
                            temp_df.loc[temp_df['StudyInstanceUID'] == file_uid, 'AbsFilePath'] = os.path.join(root, file)

                data_df.append(temp_df)

    data_df = pd.concat(data_df)
    data_df = data_df[data_df['DataUseAlloc'].str.lower() == data_use_filter.lower()] if data_use_filter is not None else data_df
    assert (data_df['AbsFilePath'].str.len() > 0).all(), 'Certain file paths were not found!' 

    # hold out data for validation if given a value 0 < split < 1 based on STUDY_UID and not entry number
    if holdout_size is not None:
        assert holdout_size >= 0 and holdout_size <= 1
        study_uids = data_df['StudyInstanceUID'].unique()
        _, uid_holdout = train_test_split(study_uids, test_size=holdout_size, random_state=940521)
        data_df.loc[data_df['StudyInstanceUID'].isin(uid_holdout),'DataUseAlloc'] = 'Holdout'
    
    # process for slice thickness
    if slice_thickness is not None:
        assert 'SliceThickness' in data_df
        init_series_count = len(data_df['SeriesInstanceUID'].unique())
        if slice_thickness.lower() == 'thin':
            data_df = data_df[data_df['SliceThickness'] <= 2.0] 
            drops = True
        elif slice_thickness.lower() == 'thick':
            data_df = data_df[data_df['SliceThickness'] > 2.0] 
            drops = True
        elif slice_thickness.lower() == 'all':
            drops = False
        else:
            raise ValueError('Slice Thickness argument must be THICK or THIN')
        if drops and verbose:
            print("Keeping {:d} series meeting {} slice criteria in {}".format(len(data_df['SeriesInstanceUID'].unique()), slice_thickness, data_use_filter))
            print("\tTotal removed series from dataset: {:d}".format(init_series_count-len(data_df['SeriesInstanceUID'].unique())))

    return data_df.reset_index(drop=True)


class Phase3HDF5(torch.utils.data.Dataset):

    def __init__(
            self,
            data_df,
            thresh_hu_normal_lung=None,
            thresh_hu_consolidation=None,
            thresh_vesselness=None,
            unlabeled_data_mode='a',
            verbose=True,
    ):
        self.verbose = verbose
        self.data_df = data_df
        self.thresh_hu_normal_lung = thresh_hu_normal_lung
        self.thresh_hu_consolidation = thresh_hu_consolidation
        self.unlabeled_data_mode = unlabeled_data_mode
        self.thresh_vesselness = thresh_vesselness

    def calculate_class_pixel_stats(self, exclude_unlabelled=False, return_counter=False, return_pos_weights=False, return_percentages=False):
        # FOR WBCE USE AND PIXEL STATS
        # Initialize counter
        counter = OrderedDict()
        start = 0 if exclude_unlabelled else -1
        for idx in range(start ,4):
            counter[PHASE3_CLASSES[idx]] = 0

        # Iterate over every entry and get class counts
        for didx in tqdm(range(0,len(self.data_df)), leave=False, desc='Getting pixel class counts'):
            row = self.data_df.iloc[didx, :]
            hdf5 =  h5py.File(row['AbsFilePath'], 'r')
            unique, unique_counts = np.unique(hdf5[row['SeriesInstanceUID']]['Parenchyma'][row['Dim0Index']], return_counts=True)
            hdf5.close()
            for idx, uniq in enumerate(unique):
                if exclude_unlabelled:
                    if uniq >= 0:
                        counter[PHASE3_CLASSES[uniq]] += unique_counts[idx] 
                else:
                    counter[PHASE3_CLASSES[uniq]] += unique_counts[idx] 
        
        # Calculate relative percentage of occurence ignoring unlabelled data
        weight = counter.copy()
        percentage = counter.copy()
        for key, value in counter.items():
            total_pixel_samples = np.sum(list(counter.values()))
            weight[key] = total_pixel_samples / (len(counter) * value)
            percentage[key] = value / total_pixel_samples * 100

        ret = []
        if return_counter:
            ret.append(counter)
        if return_percentages:
            ret.append(percentage)
        if return_pos_weights:
            ret.append(weight)

        return ret if len(ret) > 0 else None

    def _refactor_parenchyma(self, ct, parenchyma, vesselness):
        # annotation rules: 
        # - if a label is outside the lung mask, it's omitted
        # - The lung masks have been adjusted significantly and a retrain of the open source model is needed
        # - If multiple labels involve the same pixel, this is the order of hierarchy for the label: Normal lung > consolidation > GGO
        # - Please post process the GGO labels to only be allowed between -350 and -775 HU. If a pixel falls outside that within a GGO label, please omit/delete that label.
        # - Please use -775 HU as the normal lung threshold
        
        # HU Limits
        NORMAL_LUNG_HU = self.thresh_hu_normal_lung if self.thresh_hu_normal_lung is not None else -775    # LESS THAN (+/- 25 around -775)
        GGO_LOWER_LIMIT = NORMAL_LUNG_HU   # GREATER THAN
        GGO_UPPER_LIMIT = self.thresh_hu_consolidation if self.thresh_hu_consolidation is not None else -375  # LESS THAN (+/- 25 around -375)
        
        refactor = np.ones(ct.shape) * -1

        # Prepare parenchyma labels 
        # - 0 = Lung Mask [0, 1]
        # - 1 = GGO [-1, 1]
        # - 2 = Consolidation [0, 1]
        lung_mask = parenchyma[0].astype(np.bool)
        normal_lung = (ct < NORMAL_LUNG_HU) & lung_mask
        ggo = (parenchyma[1] == 1).astype(np.bool) & (ct > GGO_LOWER_LIMIT) & (ct < GGO_UPPER_LIMIT) & lung_mask
        consolidation = parenchyma[2].astype(np.bool) & lung_mask
        if self.thresh_vesselness < 0:
            pulm_vasc = (ct >= GGO_UPPER_LIMIT) & ~consolidation & lung_mask
        elif 0 <= self.thresh_vesselness <= 1:
            pulm_vasc = vesselness > self.thresh_vesselness
        
        refactor[ggo] = 3
        refactor[consolidation] = 4
        refactor[normal_lung] = 2
        refactor[pulm_vasc] = 1 
        refactor[~lung_mask] = 0
        
        # experimental 2022/08/07
        if self.unlabeled_data_mode == 'b':
            # set all unlabeled as normal lung
            refactor[refactor == -1] = 2
        if self.unlabeled_data_mode == 'c':
            # only set unlabeled within GGO limits to normal lung
            refactor[(refactor == -1) & (ct > GGO_LOWER_LIMIT) & (ct < GGO_UPPER_LIMIT)] = 2

        return refactor

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        row = self.data_df.iloc[idx, :]
        hdf5 =  h5py.File(row['AbsFilePath'], 'r')
        ct = hdf5[row['SeriesInstanceUID']]['CT'][row['Dim0Index']]
        parenchyma = hdf5[row['SeriesInstanceUID']]['Parenchyma'][row['Dim0Index']]
        vesselness = hdf5[row['SeriesInstanceUID']]['Vesselness'][row['Dim0Index']]
        hdf5.close()

        
        parenchyma = self._refactor_parenchyma(ct, parenchyma, vesselness)

        # add channel dimension
        ct = np.expand_dims(ct, -1).astype(np.float32)      
        label = np.tile(np.expand_dims(np.zeros(parenchyma.shape), -1), (1, 1, 5)).astype(np.float32)
        for idx in range(0, label.shape[-1]):
            label[:, :, idx][parenchyma == -1] = -1
            label[:, :, idx][parenchyma == idx] = 1

        transform_raw = transforms.Compose([
                           transforms.ToPILImage(),         
                           transforms.ToTensor(),
                           transforms.CenterCrop(512)])

        transform_image = transforms.Compose([
                            Threshold(min=-1000, max=350),
                            transforms.ToPILImage(),
                            transforms.ToTensor(),
                            transforms.CenterCrop(512),
                            transforms.Normalize(
                                mean = torch.tensor(MEAN),
                                std = torch.tensor(STD))])
        
        transform_label = transforms.Compose([
                            transforms.ToTensor(),
                            transforms.CenterCrop(512),
                            Squeeze()])
            
        out = {'raw_ct': transform_raw(ct),
               'parenchyma': transform_label(parenchyma),
               'image': transform_image(ct),
               'dataset': row['Dataset'],
               'label': transform_label(label).float(),
               'StudyInstanceUID': row['StudyInstanceUID'],
               'SeriesInstanceUID': row['SeriesInstanceUID'],
               'SOPInstanceUID': row['SOPInstanceUID'],
               'Dim0Index': row['Dim0Index']}

        return out

    def __len__(self):
        return len(self.data_df)


if __name__ == '__main__':
    df = build_data_df('E:\cic_covid19_phase3/data_out/download', 'train', holdout_size=0.2)
    trainset = Phase3HDF5(df)    
    for idx in range(0,len(trainset)):
        if (trainset[idx]['image'].shape[-2]) != 512 or (trainset[idx]['image'].shape[-1]) != 512:
            print(trainset[idx]['StudyInstanceUID'])
            print(trainset[idx]['SeriesInstanceUID'])
            print(trainset[idx]['SOPInstanceUID'])
