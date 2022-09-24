import os
import mdai
from utilities.pre_label_utils import DICOMSeries
from src.datasets.phase3_dataset import Phase3HDF5, build_data_df
from src.mdai_tools.upload_to_mdai import *
from src.mdai_tools.mdai_lookups import *
import argparse
from src.utilities import general_utils
from src.utilities import run_tools
import atexit
from tqdm import tqdm

from src.mdai_tools.mdai_to_hdf5.annotations import load_mask_instance


def upload_phase2_model_inferences(args, mdai_client):
    search_path = args.search_path
    force_upload = args.force_upload

    # search for inferences
    inference_paths = [os.path.join(dp, f) for dp, dn, filenames in os.walk(search_path) for f in filenames if
                       os.path.splitext(f)[1] == '.hdf5']

    log_file = os.path.join(args.search_path, 'uploads.log')
    if not os.path.exists(log_file) or force_upload:
        upload_log = open(log_file, 'w')
        uploaded = []
    else:
        upload_log = open(log_file, 'r+')
        uploaded = [line.rstrip() for line in upload_log]

    # process each inference file
    for idx, inference_path in enumerate(inference_paths):
        series = DICOMSeries()
        series.load_inference(inference_path)
        if series.series_uid not in uploaded:
            mdai_annotations = []

            print("{}/{}; PID {}; SeriesUID {}; ".format(idx+1, len(inference_paths), series.patient_id, series.series_uid), flush=True, end='')
            print("approx. every ~{} mm\n\t".format(series.slice_increment), end='')

            for idx in range(0, len(series.sopinstanceuids)):
                sopinstanceuid = series.sopinstanceuids[idx].decode("utf-8")
                print("{}".format(series.instance_numbers[idx]), end='|')

                if series.normal_lung_by_hu is not None and args.process_normal_lung_by_hu:
                    normal_lung_by_hu = series.normal_lung_by_hu[idx, :]
                else:
                    normal_lung_by_hu = None
                 
                mdai_annotations += process_sopinstance(sopinstanceuid=sopinstanceuid, 
                                                        inference=series.inference[idx, :],
                                                        lung_mask=series.lung_mask[idx, :],
                                                        normal_lung_by_hu=normal_lung_by_hu,
                                                        process_lung_mask=args.process_lung_mask, 
                                                        process_opacity=args.process_opacity)

            print("done.")
            print("\tTotal annotations to upload: {}".format(len(mdai_annotations)), flush=True)
            mdai_client.import_annotations(mdai_annotations, project_ids[args.project_key], dataset_ids[args.dataset_key])
            print(series.series_uid, file=upload_log, flush=True)
        else:
            print("SeriesUID - {} has already been uploaded".format(series.series_uid), flush=True)


def upload_phase3_groundtruth(args, mdai_client):
    summary_df = build_data_df(args.search_path)

    for ds_idx, ds in enumerate(summary_df['Dataset'].unique()):
        print("Processing {} Dataset".format(ds))
        ds_df = summary_df[summary_df['Dataset'] == ds]
        for se_idx, se in enumerate(ds_df['SeriesInstanceUID'].unique()):
            annotations = []
            print("Dataset {} of {} - Exam {} of {} | SeriesInstanceUID: {}".format(ds_idx+1,len(summary_df['Dataset'].unique()),se_idx+1,len(ds_df['SeriesInstanceUID'].unique()), se))
            se_df = ds_df[ds_df['SeriesInstanceUID'] == se]
            for sop_idx in tqdm(range(len(se_df)), leave=False):
                dataset = Phase3HDF5(se_df)
                data = dataset[sop_idx]
                labels = ['background', 'normal', 'ggo', 'consolidation']
                for idx, label in enumerate(labels):
                    sublabel = data['label'][idx].cpu() == 1
                    if sublabel.any():
                        mdai_data = mdai.common_utils.convert_mask_data(sublabel)                  
                        annotations.append(build_annotation_dict(data['SOPInstanceUID'], gtlabel_ids[label], idx, mdai_data))

            # mdai_client.import_annotations(annotations, project_ids[args.project_key], dataset_ids[ds.lower()])
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data Settings
    parser.add_argument('--search_path', type=str, default='data_out')
    parser.add_argument('--project_key', type=str, default='phase3')  # <-- upload_phase2_model_inferences
    parser.add_argument('--dataset_key', type=str, default='vgh') # <-- upload_phase2_model_inferences

    # Task
    parser.add_argument('--task', type=str)

    # Upload Parameters for Phase 2 Model Inferences
    parser.add_argument('--force_upload', type=str, default='true') 
    parser.add_argument('--process_opacity', type=str, default='false')
    parser.add_argument('--process_lung_mask', type=str, default='true')
    parser.add_argument('--process_normal_lung_by_hu', type=str, default='true')
    args = parser.parse_args()
    
    args.process_opacity = general_utils.str2bool(args.process_opacity)
    args.process_lung_mask = general_utils.str2bool(args.process_lung_mask)
    args.process_normal_lung_by_hu = general_utils.str2bool(args.process_normal_lung_by_hu)
    args.force_upload = general_utils.str2bool(args.force_upload)

    timer = run_tools.RunTimer(args.search_path, name=__file__)
    timer.add_marker('Start')
    atexit.register(timer.exit_handler)

    mdai_client = mdai.Client(domain='vgh.md.ai', access_token=access_token)


    if args.task == 'phase2_inference':
        upload_phase2_model_inferences(args, mdai_client)
    if args.task == 'phase3_groundtruth':
        upload_phase3_groundtruth(args, mdai_client)