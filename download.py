from src.mdai_tools.mdai_to_hdf5.mdai import MDAIAnnotations
from src.mdai_tools.mdai_to_hdf5.annotations import StudyInstanceAnnotations
from src.utilities.general_utils import str2bool
from src.mdai_tools.mdai_lookups import *
from sklearn.model_selection import train_test_split
import os
import argparse
import multiprocessing as mp
from tqdm import tqdm
from src.utilities import run_tools
import atexit
import pandas as pd


def process_study(dataset_path, lung_model_path, save_path, studyinstanceuid, study_df, data_use, visualize, debug, error_path):
    study = StudyInstanceAnnotations(dataset_path, lung_model_path, save_path, studyinstanceuid, study_df, data_use, visualize, error_path, debug)
    summary_csv = study.build_study()
    return summary_csv


if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=False,
                                     description='Generate HDF5s for Phase 2 Data of the UBCCICxVGH CoVID-19 AI Project')
    required = parser.add_argument_group('required arguments')
    optional = parser.add_argument_group('optional arguments')
    optional.add_argument(
        '-h',
        '--help',
        action='help',
        default=argparse.SUPPRESS,
        help='show this help message and exit'
    )

    required.add_argument('-d', '--dataset', type=str,
                          help='Dataset to process. Not case sensitive', required=True)
    required.add_argument('-i', '--input_path', type=str,
                          help='Path to dataset containing DICOMs stored in the following format: '
                               '\t[--input_path]/[StudyInstanceUIDs]/[SeriesInstanceUIDs]/[SOPInstanceUIDs].dcm',
                          required=True)
    required.add_argument('-o', '--output_path', type=str,
                          help='Path to store outputs. A dataset-specific folder will be created in this path. '
                               'If the path does not exist, it will be created.', required=True,
                          default='C://Users/Marco/PycharmProjects/outputs/hdf5s/')
    required.add_argument('--lung_model', type=str, default=r'model_in\unet_r231covid-0de78a7e.pth')

    optional.add_argument('-m', '--multiprocess', type=str2bool, default=True, help='Whether to use multiple processes')
    optional.add_argument('-v', '--visualize', type=str2bool, default=True, help='Whether to visualize')
    optional.add_argument('-f', '--force_download', type=str2bool, default=False, help='Whether to force download JSON annotations or use existing file.')
    optional.add_argument('--scheduled_start', type=str, default="")
    optional.add_argument('--debug', type=str2bool, default=False,
                          help='Set debugging mode. Will not create HDF5s. For development only.')
    args = parser.parse_args()

    if len(args.scheduled_start) > 0:
        scheduler = run_tools.RunScheduler(args.scheduled_start)

    timer = run_tools.RunTimer(args.output_path, name=__file__ + ' - {}'.format(args.dataset))
    timer.add_marker('Start')
    atexit.register(timer.exit_handler)

    debug = args.debug
    datasetid = dataset_ids[args.dataset]
    save_path = os.path.join(args.output_path, args.dataset.upper())
    error_path = os.path.join(save_path, 'errors')
    input_path = args.input_path
    visualize = args.visualize

    os.makedirs(save_path, exist_ok=True)
    package = MDAIAnnotations(save_path, project_ids['phase3'], datasetid, access_token, args.force_download)

    csv_path = os.path.join(save_path, 'csv')

    # Modify Train/Test studies if MG dataset
    if args.dataset[0:2] == 'mg':
        # As instructed by Dr. William Parker on 2022-05-08, MG dataset should have randomly picked test studies and to ignore MD.ai labels for TEST set holdout
        print("Ignoring MD.ai TEST Label... Regenerating Holdout")
        study_uids = list(package.studies['StudyInstanceUID'].unique())
        _, package.test_studies = train_test_split(study_uids, test_size=0.2)
        package.train_studies = [study for study in study_uids if study not in package.test_studies]
        package.dump_pkl(package.pkl_file)

    # build hdf5s
    for index, row in tqdm(package.studies.iterrows(),total=len(package.studies)):
        study_uid = row['StudyInstanceUID']
        data_use = 'test' if study_uid in package.test_studies else 'train'
        summary_csv = process_study(input_path, args.lung_model, save_path, study_uid, package.annotations[package.annotations['StudyInstanceUID'] == study_uid], data_use, visualize, debug, error_path)

    # create csv with list of train and test studies
    with open(os.path.join(csv_path, 'test_studies.csv'),'w') as f:
        print('StudyInstanceUID',file=f)
        for study_uid in package.test_studies:
            print(study_uid, file=f)

    with open(os.path.join(csv_path,'train_studies.csv'),'w') as f:
        print('StudyInstanceUID',file=f)
        for study_uid in package.train_studies:
            print(study_uid, file=f)
