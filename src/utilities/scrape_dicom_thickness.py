from turtle import st
import pydicom as dcm
import argparse
import pandas as pd
import os
from tqdm import tqdm
from general_utils import str2bool


def splitall(path):
    allparts = []
    while 1:
        parts = os.path.split(path)
        if parts[0] == path:  # sentinel for absolute paths
            allparts.insert(0, parts[0])
            break
        elif parts[1] == path: # sentinel for relative paths
            allparts.insert(0, parts[1])
            break
        else:
            path = parts[0]
            allparts.insert(0, parts[1])
    return allparts


def get_dicom_paths(path):
    dcm_paths = []
    series_list = []
    studies_list = []
    dataset_list = []
    for root, _, files in tqdm(os.walk(path), desc='Getting file paths...', leave=True):
        for file in files:
            if os.path.splitext(file)[-1] == '.dcm':
                parts = splitall(root)
                dataset = parts[-3]
                study = parts[-2]
                series = parts[-1]
                if series not in series_list:
                    dcm_paths.append(os.path.join(root, file))
                    series_list.append(series)
                    studies_list.append(study)
                    dataset_list.append(dataset)

    return dcm_paths, series_list, studies_list, dataset_list


def get_dicom_thicknesses(data_search_path, output_path):
    dcm_paths, series_list, studies_list, dataset_list = get_dicom_paths(data_search_path)
    thicknesses = []
    for filepath in tqdm(dcm_paths, desc='Running the list...', leave=True):
        dicom = dcm.read_file(filepath)
        thicknesses.append(-1.)
        if dicom.get((0x18, 0x50)) is not None:
            if dicom[(0x18, 0x50)].value is not None:
                thicknesses[-1] = (float(dicom[(0x18, 0x50)].value))

    df = pd.DataFrame(list(zip(dataset_list, studies_list, series_list, thicknesses)),columns=['Dataset', 'StudyInstanceUID', 'SeriesInstanceUID', 'SliceThickness'])
    df.sort_values(by=['Dataset','StudyInstanceUID','SeriesInstanceUID'], inplace=True)
    df.set_index(keys=['Dataset','StudyInstanceUID','SeriesInstanceUID'], inplace=True)
    df.to_csv(os.path.join(output_path,'thicknesses.csv'))
    return df


def append_to_download_csv(df_thickness, hdf5_data_dir):
    roots = [os.path.join(hdf5_data_dir, dir) for dir in os.listdir(hdf5_data_dir) if os.path.isdir(os.path.join(hdf5_data_dir, dir))] 
    roots.sort()
    for root in roots:
        data_summary = pd.read_csv(os.path.join(root,'csv','data_summary.csv'),index_col=0,header=0)
        data_summary.reset_index(drop=True, inplace=True)
        for idx in tqdm(range(len(data_summary)), desc='Appending...'):
            ds = data_summary.loc[idx,'Dataset']
            st = data_summary.loc[idx,'StudyInstanceUID']
            sr = data_summary.loc[idx,'SeriesInstanceUID']
            data_summary.loc[idx,'SliceThickness'] =  df_thickness.loc[(ds,st,sr)]['SliceThickness']
        data_summary.to_csv(os.path.join(root,'csv','data_summary.csv'))
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_search_path', type=str)
    parser.add_argument('--output_path', type=str)
    # Settings for updating download.py csv outputs with the slice thicknesses
    parser.add_argument('--append_to_download_csv', type=str2bool)
    parser.add_argument('--hdf5_data_dir',type=str)

    args = parser.parse_args()
    df_thickness = get_dicom_thicknesses(args.data_search_path, output_path=args.output_path)
    if args.append_to_download_csv:
        append_to_download_csv(df_thickness, args.hdf5_data_dir)