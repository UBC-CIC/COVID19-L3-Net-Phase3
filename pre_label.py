import torch
import torchvision.transforms as transforms
import argparse
import os
import numpy as np
from src.utilities import run_tools

from src.utilities.image_utils import save_tiff, save_rgba
from src.utilities import general_utils
from src.utilities.pre_label_utils import DICOMSeries
from src.architecture.segmentor import SegmentorUNet2D
from src.datasets.phase3_dataset import Threshold

import shutil
import atexit

MEAN = [-653.2204]
STD = [628.5188]

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

Label_names = {0: 'bg',
               1: 'normal',
               2: 'ggo',
               3: 'crazy paving',  # in phase 2 ONLY. grouped with GGO in phase 3
               4: 'consolidation'}


def infer_dcm(args, image_transform):
    # extract tree structure of the input data path if dcm
    dir_tree = [[root, dirs, files] for root, dirs, files in
                os.walk(args.data_path) if len(files) > 0]

    # walk the directory tree to pull a project_dir, study_dir, and series_dir (synonymous with uid). ignore lookup tables.
    series_paths = np.unique([dir_path[0] for dir_path in dir_tree if os.path.split(dir_path[0])[-1] != 'lookup.csv'])

    # iterate through the series paths to create inferences for each
    for sidx, series_path in enumerate(series_paths):
        study_path, series_uid = os.path.split(series_path)
        data_path, study_uid = os.path.split(study_path)
        print("[{}] Run {} of {}: {} - ".format(run_tools.datetime.now().strftime("%H:%M:%S"), sidx+1,len(series_paths), study_uid, series_uid), flush=True, end='')
        
        save_path = args.output_dir
        check_path = os.path.join(save_path, 'inference', study_uid, series_uid)

        # make sure series has more than one .dcm (otherwise, not a CT volume, and can skip)
        if len(os.listdir(os.path.join(data_path, study_uid, series_uid))) <= 1:
            print("Skipping. Unlikely to be CT Volume.", flush=True)
            continue

        if os.path.exists(check_path) and args.skip_existing:
            print("Skipping. Inference Folder Exists.", flush=True)
            continue

        try:
            dcm_series = DICOMSeries(dataset_path=data_path,
                                    study_uid=study_uid,
                                    series_uid=series_uid,
                                    infer_every_mm=args.infer_every_mm,
                                    image_transform=image_transform,
                                    lung_model_path=args.lung_model,
                                    point_mode=args.point_mode,
                                    normal_lung_sz_thresh=args.normal_lung_sz_thresh,
                                    normal_lung_hu_thresh=args.normal_lung_hu_thresh)
        except Exception as e:
            print("\tSkipping. Error Encountered - {}".format(e), flush=True)
            continue

        if not dcm_series.bad_series:
            # generate inferences
            probs = []
            for i, batch in enumerate(torch.utils.data.DataLoader(dcm_series, batch_size=1)):
                image = batch['image'].to(device)
                with torch.no_grad():
                    outputs = net.predict_on_batch(image)
                    probs.append(torch.softmax(outputs, dim=1).detach().cpu().numpy())

            probs = np.concatenate(probs, axis=0)
            dcm_series.attach_inference(probs, )
            dcm_series.find_max_probability()

            # save inference
            if args.save_inference:
                inference_path = os.path.join(save_path, 'inference', study_uid, series_uid)
                os.makedirs(inference_path) if not os.path.isdir(inference_path) else None
                dcm_series.save_inference(os.path.join(inference_path, 'inference.hdf5')) 
            
            # visualize
            if args.make_plots:
                viz_path = os.path.join(save_path, 'visualizations', study_uid, series_uid)
                shutil.rmtree(viz_path, ignore_errors=True)
                os.makedirs(viz_path) if not os.path.isdir(viz_path) else None
                dcm_series.visualize(viz_path)

            series_msg = 'Success!\n'
            print(series_msg, flush=True, end='')
            



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # HyperParams
    parser.add_argument('--model', type=str, default='unet2d')
    parser.add_argument('--loss_fn', type=str, default='kld')
    parser.add_argument('--num_channels', type=int, default=1)
    parser.add_argument('--num_classes', type=int, default=5)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--iter_per_update', type=int, default=100)
    parser.add_argument('--test_batch_size', type=int, default=1)
    parser.add_argument('--verbose', '-v', type=str, default='false')
    parser.add_argument('--make_plots', type=str, default='true')
    parser.add_argument('--multi_gpu', type=str, default='true')
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--lung_model', type=str, default=None)
    parser.add_argument('--point_mode', type=str, default=r'local')
    parser.add_argument('--infer_every_mm', type=int, default=10)
    parser.add_argument('--normal_lung_sz_thresh', type=int, default=0, help='<0 to disable replacing normal lung inference')
    parser.add_argument('--normal_lung_hu_thresh', type=int, default=0)
    parser.add_argument('--skip_existing', type=str, default='true')
    parser.add_argument('--save_inference', type=str, default='true')
    parser.add_argument('--scheduled_start', type=str, default="")

    # Container environment
    parser.add_argument('--data_path', '-d', type=str, default=None)
    parser.add_argument('--model_dir', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--log_dir', type=str, default=None)

    args = parser.parse_args()
    args.skip_existing = general_utils.str2bool(args.skip_existing)
    args.verbose = general_utils.str2bool(args.verbose)
    args.make_plots = general_utils.str2bool(args.make_plots)
    args.multi_gpu = general_utils.str2bool(args.multi_gpu) and (torch.cuda.device_count() > 1)
    args.save_inference = general_utils.str2bool(args.save_inference)

    if len(args.scheduled_start) > 0:
        scheduler = run_tools.RunScheduler(args.scheduled_start)
    os.makedirs(args.output_dir, exist_ok=True)

    timer = run_tools.RunTimer(log_path=args.output_dir, name=__file__)
    timer.add_marker('Start')
    atexit.register(timer.exit_handler)

    # Load pre-trained model
    net = SegmentorUNet2D(num_channels=1, num_classes=5, model_type='phasex', checkpoint=args.checkpoint).to(device)

    # Create transform
    image_transform = transforms.Compose([
        Threshold(min=-1000, max=350),
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=torch.tensor(MEAN),
            std=torch.tensor(STD)
        )
    ])

    infer_fn = infer_dcm
    infer_fn(args, image_transform)