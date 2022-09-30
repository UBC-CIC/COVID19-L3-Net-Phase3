import torch
import torchvision.transforms as transforms
import argparse
import os
import numpy as np
from src.utilities import run_tools

from src.utilities import general_utils
from src.utilities.infer_utils import DICOMSeries, save_nii
from src.architecture.segmentor import SegmentorUNet2D
from src.datasets.phase3_dataset import Threshold

import atexit


Label_names = {0: 'bg',
               1: 'vessels',
               2: 'normal',
               3: 'ggo/crazy-paving', 
               4: 'consolidation'}


def infer_dicom(model_weights, dicom_folder, output_directory, save_nifti=False, skip_existing=True):
    # build the model and prepare functions
    # Load model
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    net = SegmentorUNet2D(num_channels=1, num_classes=5, model_type='phase3', checkpoint=model_weights).to(device)

    # Create transform
    MEAN = [-653.2204]
    STD = [628.5188]
    image_transform = transforms.Compose([
        Threshold(min=-1000, max=350),
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=torch.tensor(MEAN),
            std=torch.tensor(STD)
        )
    ])

    # extract tree structure of the input data path if dcm
    dir_tree = [[root, dirs, files] for root, dirs, files in
                os.walk(dicom_folder) if len(files) > 0]

    # walk the directory tree to pull a project_dir, study_dir, and series_dir (synonymous with uid). ignore lookup tables.
    series_paths = np.unique([dir_path[0] for dir_path in dir_tree if os.path.split(dir_path[0])[-1] != 'lookup.csv'])

    # iterate through the series paths to create inferences for each
    for sidx, series_path in enumerate(series_paths):
        study_path, series_uid = os.path.split(series_path)
        data_path, study_uid = os.path.split(study_path)
        print("[{}] Series {} of {}: Study {} - ".format(run_tools.datetime.now().strftime("%H:%M:%S"), sidx+1,len(series_paths), study_uid, series_uid), flush=True, end='')
        
        save_path = output_directory
        check_path = os.path.join(save_path, 'inference', study_uid, series_uid)

        # make sure series has more than one .dcm (otherwise, not a CT volume, and can skip)
        if len(os.listdir(os.path.join(data_path, study_uid, series_uid))) <= 1:
            print("Skipping. Unlikely to be CT Volume.", flush=True)
            continue

        if os.path.exists(check_path) and skip_existing:
            print("Skipping. Inference Folder Exists.", flush=True)
            continue

        try:
            dcm_series = DICOMSeries(dataset_path=data_path,
                                    study_uid=study_uid,
                                    series_uid=series_uid,
                                    image_transform=image_transform)
        except Exception as e:
            print("\tSkipping. Error Encountered - {}".format(e), flush=True)
            continue

        if not dcm_series.bad_series:
            # generate inferences
            for i, batch in enumerate(torch.utils.data.DataLoader(dcm_series, batch_size=args.batch_size)):
                with torch.no_grad():
                    if i == 0:
                        probs = net.predict_on_batch(batch['image'].to(device))
                        ct = batch['image']
                    else:
                        probs = torch.concat((probs, net.predict_on_batch(batch['image'].to(device))), dim=0)
                        ct = torch.concat((ct, batch['image']), dim=0)

            inference_path = os.path.join(save_path, study_uid, series_uid)
            os.makedirs(inference_path, exist_ok=True)

            # Save into numpy file
            np.savez_compressed(os.path.join(inference_path, "inference.npz"), 
                                in_=dcm_series.pixel_data, 
                                out_=np.argmax(probs.cpu().detach().numpy(),axis=1), 
                                probs=probs.cpu().detach().numpy(),
                                sops=dcm_series.sopinstanceuids, 
                                labels=np.array(['bg','vessels','normal','ggo','consolidation']),
                                slice_thickness=np.array(dcm_series.slice_thickness))

            # Generate NIFTI
            if save_nifti:
                save_nii(inference_path, ct, probs, dcm_series.slice_thickness)

            series_msg = 'Success!\n'
            print(series_msg, flush=True, end='')
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # HyperParams
    parser.add_argument('--model', type=str, default='unet2d')
    parser.add_argument('--num_channels', type=int, default=1)
    parser.add_argument('--num_classes', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--verbose', '-v', type=general_utils.str2bool, default='false')
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--save_nii', type=general_utils.str2bool, default='false')  # if true, will save nifti files in addition to .npz
    parser.add_argument('--skip_existing', type=general_utils.str2bool, default='true')
    parser.add_argument('--scheduled_start', type=str, default="")

    # Container environment
    parser.add_argument('--data_path', '-d', type=str,
                        default=None)
    parser.add_argument('--model_dir', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--log_dir', type=str, default=None)

    args = parser.parse_args()

    if len(args.scheduled_start) > 0:
        scheduler = run_tools.RunScheduler(args.scheduled_start)
    os.makedirs(args.output_dir, exist_ok=True)

    timer = run_tools.RunTimer(log_path=args.output_dir, name=__file__)
    timer.add_marker('Start')
    atexit.register(timer.exit_handler)

    infer_dicom(model_weights=args.checkpoint, dicom_folder=args.data_path, output_directory=args.output_dir, save_nifti=args.save_nii, skip_existing=args.skip_existing)