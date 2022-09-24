import argparse
import json
from collections import OrderedDict

from src.utilities.general_utils import str2bool
from src.utilities import run_tools
from src.datasets.phase3_dataset import Phase3HDF5, build_data_df
from src.utilities.image_utils import construct_nii
import atexit
import os
import sys
from tqdm import tqdm
import torch
from src.architecture.segmentor import SegmentorUNet2D
import torchnet.meter as meter
import re 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib




device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


def construct_datasets(args,thresh_hu_normal_lung,thresh_hu_consolidation,thresh_vesselness,slice_thickness='all',unlabeled_data_mode='a', verbose=True):
    train_df, val_df, test_df = [None] * 3

    # Force validation if debugging using validation set as a test set for performance demo
    if args.val_as_test:
        print("Validation forced to TRUE for debugging due to VAL_AS_TEST=TRUE flag. A validation hold-out will be used as the TEST set.") if verbose else None

    # Search data path for data summaries and build dataframes
    data_df = build_data_df(search_path=args.data_search_path, data_use_filter='train', holdout_size=0.2 if args.val_as_test else None, slice_thickness=slice_thickness)
    train_df = data_df.loc[data_df['DataUseAlloc'] == 'Train']
    val_df = data_df.loc[data_df['DataUseAlloc'] == 'Holdout']

    test_df = val_df if args.val_as_test else build_data_df(search_path=args.data_search_path, data_use_filter='test', slice_thickness=slice_thickness, verbose=verbose) 

    # Construct Pytorch Custom Datasets
    datasets = [Phase3HDF5(data_df=train_df, thresh_hu_normal_lung=thresh_hu_normal_lung, 
                           thresh_hu_consolidation=thresh_hu_consolidation, thresh_vesselness=thresh_vesselness,
                           unlabeled_data_mode=unlabeled_data_mode) if train_df is not None else None,
                Phase3HDF5(data_df=val_df, thresh_hu_normal_lung=thresh_hu_normal_lung, 
                           thresh_hu_consolidation=thresh_hu_consolidation, thresh_vesselness=thresh_vesselness,
                           unlabeled_data_mode=unlabeled_data_mode) if val_df is not None else None,
                Phase3HDF5(data_df=test_df, thresh_hu_normal_lung=thresh_hu_normal_lung, 
                           thresh_hu_consolidation=thresh_hu_consolidation, thresh_vesselness=thresh_vesselness,
                           unlabeled_data_mode=unlabeled_data_mode) if test_df is not None else None]
    return datasets


def get_model_paths(path, filter=''):
    models = []
    configs = []
    # Find all models in the model search path
    for root, _, files in os.walk(path):
        for file in files:
            if os.path.splitext(file)[-1] == '.ckpt' and filter in root:
                models.append([root, file])
                configs.append([root, '{}.json'.format(os.path.splitext(file)[0])]) if os.path.exists(os.path.join(root,'{}.json'.format(os.path.splitext(file)[0]))) else configs.append([])

    return models, configs


def test_on_loader(net, dataloader, t0, t1, t2):
    # n.b. if test_dl batch size > 1, the torchnet meters are average-over-batches, and not average of all samples.
    meters_metrics = {}
    meters_loss = meter.AverageValueMeter()
    net.eval()

    for batch_idx, batch in enumerate(tqdm(dataloader, desc='Testing {}/{}/{}'.format(t0, t1, t2), leave=False)):

        with torch.no_grad():
            loss, _, metrics = net.process_batch(batch, device)
            meters_loss.add(loss.item())

            if metrics is not None:
                if batch_idx == 0:
                    for key1 in metrics.keys():
                        if key1[0:5] == 'class':
                            for idx in range(len(metrics[key1])):  # ignore vessel metrics
                                if idx != 1:
                                    meters_metrics['c{}_{}'.format(key1[-3:], idx)] = meter.AverageValueMeter()    
                        else:
                            meters_metrics[key1] = meter.AverageValueMeter()

                for key1, values1 in metrics.items():
                    if key1[0:5] == 'class': # separate the class-wise accuracies
                        for idx in range(len(metrics[key1])): 
                            value = values1.detach().cpu()[idx]
                            if not torch.isnan(value) and idx != 1:  # ignore vessel metrics
                                meters_metrics['c{}_{}'.format(key1[-3:], idx)].add(value)
                                if key1[-3] == 'sen' :
                                    print(meters_metrics['c{}_{}'.format(key1[-3:], idx)].value()[0], flush=True)
                    else:
                        meters_metrics[key1].add(values1.detach().cpu())
    
    for key1, values1 in meters_metrics.items():
        meters_metrics[key1] = np.array(values1.value()[0])

    return meters_metrics
    
    
def save_results(idx, model_path, parameters, metrics, output_dir):
    entry = OrderedDict()
    entry['model_path'] = '/'.join(re.split('/|\\\\',model_path)[-2:])
    for key, value in parameters.items():
        entry[key] = value
    for key1, values1 in metrics.items():
        if key1 != 'confusion':
            entry[key1] = values1
    df = pd.DataFrame(entry, columns=entry.keys(), index=[idx])
    if not os.path.exists(os.path.join(output_dir,'results.csv')):
        df.to_csv(os.path.join(output_dir,'results.csv'), header=True, mode='w')
    else:
        df.to_csv(os.path.join(output_dir,'results.csv'), header=False, mode='a')


def performance(args, models_phase3, phase3_configs, models_other):
    # Load phase 3 models
    if len(models_phase3) > 0:
        for idx, data in tqdm(enumerate(zip(models_phase3, phase3_configs)),leave=True,desc='Evaluating Phase 3 Models', total=len(models_phase3)):
            model_path = os.path.join(*data[0])
            config_path = os.path.join(*data[1])
            with open(config_path, 'r') as f:
                config = json.load(f)

            # Get model-specific dataset
            thresh_params = OrderedDict()
            thresh_params['nl'] = config['thresh_hu_normal_lung']
            thresh_params['cn'] = config['thresh_hu_consolidation']
            thresh_params['vs'] = config['thresh_vesselness']
            thresh_params['ul'] = os.path.splitext(os.path.split(model_path)[-1])[0][-1:].lower()
            datasets = construct_datasets(args, thresh_params['nl'], thresh_params['cn'], thresh_params['vs'], 
                                          slice_thickness=args.slice_thickness, 
                                          unlabeled_data_mode=thresh_params['ul'], verbose=False)  

            # Construct Pytorch Dataloaders
            test_dl = torch.utils.data.DataLoader(datasets[2], batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False, drop_last=False)

            # Load model and test
            net = SegmentorUNet2D(num_channels=1, num_classes=5, reset_fc=False, 
                                checkpoint=model_path, multi_gpu=args.multi_gpu, model_type='phase3', verbose=False) 
            net.to(device)

            metrics = test_on_loader(net, test_dl, thresh_params['nl'], thresh_params['cn'], thresh_params['vs'])
            save_results(idx, model_path, thresh_params, metrics, args.output_dir)

    # Load other models
    if len(models_other) > 0:
        for idx, data in tqdm(enumerate(models_other),leave=True,desc='Evaluating Other Models', total=len(models_other)):
            model_path = os.path.join(*data)

            # Get model-specific dataset
            thresh_params = OrderedDict()
            thresh_params['nl'] = -750
            thresh_params['cn'] = -350
            thresh_params['vs'] = 0.25
            thresh_params['ul'] = 'a'
            datasets = construct_datasets(args, thresh_params['nl'], thresh_params['cn'], thresh_params['vs'], 
                                          slice_thickness=args.slice_thickness, 
                                          unlabeled_data_mode=thresh_params['ul'], verbose=False)

            # Construct Pytorch Dataloaders
            test_dl = torch.utils.data.DataLoader(datasets[2], batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False, drop_last=False)

            # Load model and test
            net = SegmentorUNet2D(num_channels=1, num_classes=5, reset_fc=False, 
                                checkpoint=model_path, multi_gpu=args.multi_gpu, model_type='phasex', verbose=False) 
            net.to(device)

            metrics = test_on_loader(net, test_dl, thresh_params['nl'], thresh_params['cn'], thresh_params['vs'])
            save_results(idx, model_path, thresh_params, metrics, args.output_dir)


def visualize_png(args, models_phase3, phase3_configs, models_other):
    datasets = construct_datasets(args, -750, -350, 0.25, slice_thickness=args.slice_thickness, verbose=False)
    test_ds = datasets[2]

    with torch.no_grad():
        for idx in tqdm(range(len(test_ds))):
            outputs = []
            names = []
            for model_phase3, phase3_config in zip(models_phase3, phase3_configs):
                model_path = os.path.join(*model_phase3)
                config_path = os.path.join(*phase3_config)

                with open(config_path, 'r') as f:
                    config = json.load(f)

                # Get model-specific dataset
                test_ds.thresh_hu_normal_lung = config['thresh_hu_normal_lung']
                test_ds.thresh_hu_consolidation = config['thresh_hu_consolidation']
                test_ds.thresh_vesselness = config['thresh_vesselness']           
                net = SegmentorUNet2D(num_channels=1, num_classes=5, reset_fc=False, 
                                      checkpoint=model_path, multi_gpu=args.multi_gpu, model_type='phase3', verbose=False) 
                
                data = test_ds[idx]
                outputs.append(net.predict_on_batch(torch.unsqueeze(data['image'], 0)))
                names.append(os.path.splitext(model_phase3[-1])[0])
                
            for model_other in models_other:
                model_path = os.path.join(*model_other)

                net = SegmentorUNet2D(num_channels=1, num_classes=5, reset_fc=False, 
                        checkpoint=model_path, multi_gpu=args.multi_gpu, model_type='phasex', verbose=False) 
                outputs.append(net.predict_on_batch(torch.unsqueeze(data['image'], 0)))
                names.append(os.path.splitext(model_other[-1])[0])

            # Sort the outputs and names for generating an ORDERED figure
            sort_args = np.argsort(names)
            image = data['image']
            series_uid = data['SeriesInstanceUID']
            study_uid = data['StudyInstanceUID']
            filepath = os.path.join(args.output_dir, study_uid, series_uid)
            os.makedirs(filepath, exist_ok=True)

            # Build Original CT
            npy_image = np.squeeze(image.detach().cpu().numpy())
            color = np.array([[0, 150, 255, 0],
                            [255, 189, 51, 0],
                            [255, 51, 51, 0]])/255

            if args.separate_figs:
                plt.figure(constrained_layout=True, dpi=100, figsize=(4, 4))
                plt.imshow(npy_image, cmap='gray', vmin = -0.6, vmax = 1.5)
                plt.axis('off')
                plt.savefig(os.path.join(filepath,'{}-0.png'.format(data['Dim0Index'])))
                plt.close()
            else:
                # fig = plt.figure(constrained_layout=True, dpi=100, figsize=((1 + len(outputs)) * len(outputs), 4)) 
                plot_cols = np.ceil((len(sort_args)+1)/2).astype(int)
                fig = plt.figure(constrained_layout=True, dpi=150, figsize=(4*plot_cols , 4*2 )) 
                gs = fig.add_gridspec(2, plot_cols)
                ax = fig.add_subplot(gs[0, 0])
                ax.imshow(npy_image, cmap='gray', vmin = -0.6, vmax = 1.5)

                # # experimental - include parenchyma/label where -1 == normal lung if within lung mask
                # parenchyma = np.squeeze(parenchyma.detach().cpu().numpy())
                # parenchyma[parenchyma == -1] = 2
                # alpha_scale_factor = 0.25
                # for midx in range(2, 5):
                #     overlay = np.zeros([*npy_image.shape, 4])
                #     overlay[:,:,:] = color[midx-2] # get color
                #     idx_alpha = np.ones([*npy_image.shape]) # get probabilities
                #     idx_alpha[parenchyma != midx] = 0 # set 
                #     idx_alpha = idx_alpha * alpha_scale_factor
                #     overlay[..., -1] = idx_alpha 
                #     ax.imshow(overlay)

                ax.set_ylim([0, 512])
                ax.set_xlim([0, 512])
                ax.invert_yaxis()
                ax.axis('off')
                ax.text(5, 5, 'CT', color='white', fontweight='bold', ha='left', va='top')

            # Loop to add model outputs
            for axidx, sidx in enumerate(sort_args):
                axidxs = axidx + 1
                name = names[sidx]
                output = outputs[sidx]
                npy_output = np.squeeze(output.detach().cpu().numpy())
                output = np.argmax(npy_output,axis=0)
                if args.separate_figs:
                    plt.figure(constrained_layout=True, dpi=100, figsize=(4, 4))
                    plt.imshow(npy_image, cmap='gray', vmin = -0.6, vmax = 1.5)
                    plt.axis('off')
                    alpha_scale_factor = 0.7
                    for midx in range(2, npy_output.shape[0]):
                        overlay = np.zeros([*npy_image.shape, 4])
                        overlay[:,:,:] = color[midx-2] # get color
                        idx_alpha = 1 # npy_output[midx] # get probabilities
                        idx_alpha[output != midx] = 0 # set 
                        idx_alpha = idx_alpha * alpha_scale_factor
                        overlay[..., -1] = idx_alpha 
                        plt.imshow(overlay)
                    plt.savefig(os.path.join(filepath,'{}-{}.png'.format(data['Dim0Index'], axidx+1)))
                    plt.close()
                else:
                    if axidxs < plot_cols:
                        x = 0  
                        y = axidxs
                    else:
                        x = 1
                        y = axidxs - plot_cols
                    ax = fig.add_subplot(gs[x, y])
                    ax.imshow(npy_image, cmap='gray', vmin = -0.6, vmax = 1.5)   
                    ax.axis('off')   
                    alpha_scale_factor = 0.75
                    for midx in range(2, npy_output.shape[0]):
                        overlay = np.zeros([*npy_image.shape, 4])
                        overlay[:,:,:] = color[midx-2] # get color
                        idx_alpha = npy_output[midx] # get probabilities
                        idx_alpha[output != midx] = 0 # set 
                        idx_alpha = idx_alpha * alpha_scale_factor
                        overlay[..., -1] = idx_alpha 
                        ax.imshow(overlay)
                    ax.text(5, 5, name, color='white', fontweight='bold', ha='left', va='top')

            if not args.separate_figs:
                plt.savefig(os.path.join(filepath,'{}.png'.format(data['Dim0Index'])))
                plt.close()


def visualize_nii(args, models_phase3, phase3_configs, models_other):
    with torch.no_grad():
        datasets = construct_datasets(args, -750, -350, 0.25, slice_thickness=args.slice_thickness, verbose=False)
        test_ds = datasets[2]
        data_df = test_ds.data_df
        series_unique = test_ds.data_df['SeriesInstanceUID'].unique()

        color = np.array([[0, 150, 255],
                [255, 189, 51],
                [255, 51, 51]]).astype(np.uint8)

        for series in tqdm(series_unique,desc='Series', total=len(series_unique)):

            test_ds.data_df = data_df[data_df['SeriesInstanceUID'] == series]
            test_ds.data_df.sort_values(by=['Dim0Index'])

            outs_phase3 = []
            for model_phase3, phase3_config in tqdm(zip(models_phase3, phase3_configs),leave=False,desc='Phase 3',total=len(models_phase3)):
                model_path = os.path.join(*model_phase3)
                config_path = os.path.join(*phase3_config)

                with open(config_path, 'r') as f:
                    config = json.load(f)

                # Get model-specific dataset
                test_ds.thresh_hu_normal_lung = config['thresh_hu_normal_lung']
                test_ds.thresh_hu_consolidation = config['thresh_hu_consolidation']
                test_ds.thresh_vesselness = config['thresh_vesselness']           
                net = SegmentorUNet2D(num_channels=1, num_classes=5, reset_fc=False, 
                                      checkpoint=model_path, multi_gpu=args.multi_gpu, model_type='phase3', verbose=False) 
                
                test_dl = torch.utils.data.DataLoader(test_ds, batch_size=args.batch_size if args.batch_size > 0 else len(test_ds),
                                                      num_workers=args.num_workers, shuffle=False, drop_last=False)

                for batch_idx, batch in enumerate(tqdm(test_dl, leave=False,desc='Batches')):
                    if batch_idx == 0:
                        outputs = net.predict_on_batch(batch['image'])
                        ct = batch['image']
                    else:
                        outputs = torch.concat((outputs, net.predict_on_batch(batch['image'])), dim=0)
                        ct = torch.concat((ct, batch['image']), dim=0)

                outs_phase3.append(outputs)

                nii, mask = construct_nii(ct, outputs, color)
                nii = nib.Nifti1Image(nii, np.eye(4))
                nii.header['pixdim'][1:4] = [1, 1, test_ds.data_df['SliceThickness'].iloc[0]]
                if mask is not None:
                    mask = nib.Nifti1Image(mask, np.eye(4)) 
                    mask.header['pixdim'][1:4] = [1, 1, test_ds.data_df['SliceThickness'].iloc[0]]
                    mask.header['cal_min'] = 0
                    mask.header['cal_max'] = 4
                else:
                    mask = None

                filepath = os.path.join(args.output_dir, test_ds.data_df['SeriesInstanceUID'].iloc[0])
                os.makedirs(filepath, exist_ok=True)
                nib.save(nii, os.path.join(filepath,"{}.nii.gz".format(os.path.splitext(model_phase3[-1])[0])))
                nib.save(mask, os.path.join(filepath,"mask-{}.nii.gz".format(os.path.splitext(model_phase3[-1])[0])))
                 
            outs_phasex = []
            for model_other in tqdm(models_other,leave=False,desc='Old Models', total=len(models_other)):
                model_path = os.path.join(*model_other)

                with open(config_path, 'r') as f:
                    config = json.load(f)

                # Get model-specific dataset
                test_ds.thresh_hu_normal_lung = -750
                test_ds.thresh_hu_consolidation = -350
                test_ds.thresh_vesselness = 0.25        
                net = SegmentorUNet2D(num_channels=1, num_classes=5, reset_fc=False, 
                                      checkpoint=model_path, multi_gpu=args.multi_gpu, model_type='phasex', verbose=False) 
                
                test_dl = torch.utils.data.DataLoader(test_ds, batch_size=args.batch_size if args.batch_size > 0 else len(test_ds),
                                                      num_workers=args.num_workers, shuffle=False, drop_last=False)

                for batch_idx, batch in enumerate(tqdm(test_dl, leave=False)):
                    if batch_idx == 0:
                        outputs = net.predict_on_batch(batch['image'])
                        ct = batch['image']
                    else:
                        outputs = torch.concat((outputs, net.predict_on_batch(batch['image'])), dim=0)
                        ct = torch.concat((ct, batch['image']), dim=0)
                
                outs_phasex.append(outputs)

                nii, mask = construct_nii(ct, outputs, color)
                nii = nib.Nifti1Image(nii, np.eye(4))
                nii.header['pixdim'][1:4] = [1, 1, test_ds.data_df['SliceThickness'].iloc[0]]
                if mask is not None:
                    mask = nib.Nifti1Image(mask, np.eye(4)) 
                    mask.header['pixdim'][1:4] = [1, 1, test_ds.data_df['SliceThickness'].iloc[0]]
                    mask.header['cal_min'] = 0
                    mask.header['cal_max'] = 4
                else:
                    mask = None

                filepath = os.path.join(args.output_dir, test_ds.data_df['SeriesInstanceUID'].iloc[0])
                os.makedirs(filepath, exist_ok=True)
                nib.save(nii, os.path.join(filepath,"{}.nii.gz".format(os.path.splitext(model_other[-1])[0])))
                nib.save(mask, os.path.join(filepath,"mask-{}.nii.gz".format(os.path.splitext(model_other[-1])[0])))

            # Save plain CT
            nii, _ = construct_nii(ct)
            nii = nib.Nifti1Image(nii, np.eye(4))
            nii.header['pixdim'][1:4] = [1, 1, test_ds.data_df['SliceThickness'].iloc[0]]
            filepath = os.path.join(args.output_dir, test_ds.data_df['SeriesInstanceUID'].iloc[0])
            os.makedirs(filepath, exist_ok=True)
            nib.save(nii, os.path.join(filepath,"CT.nii.gz"))

            # Multi-volume Nifti
            vols = []
            vols.append(np.expand_dims(construct_nii(ct, outs_phase3[0], color=None, text='CT')[0], -1))
            for phasex, model_other in zip(outs_phasex, models_other):
                vols.append(np.expand_dims(construct_nii(ct, phasex, color=color, text=os.path.splitext(model_other[-1])[0])[0], -1))
            for phase3, model_phase3 in zip(outs_phase3, models_phase3):
                vols.append(np.expand_dims(construct_nii(ct, phase3, color=color, text=os.path.splitext(model_phase3[-1])[0])[0], -1))
            vols = np.concatenate(vols, -1)
            vols = nib.Nifti1Image(vols, np.eye(4))
            vols.header['pixdim'][1:4] = [1, 1, test_ds.data_df['SliceThickness'].iloc[0]]
            filepath = os.path.join(args.output_dir, test_ds.data_df['SeriesInstanceUID'].iloc[0])
            os.makedirs(filepath, exist_ok=True)
            nib.save(vols, os.path.join(filepath,"multi_volume.nii.gz"))



def main(args):
    # Show dataset statistics before evaluating all models
    datasets = construct_datasets(args, 0,0,0, slice_thickness=args.slice_thickness, verbose=True)
    print("\nDATASET STATISTICS")
    for ds_type, dataset in zip(['Train','Validate','Test'], datasets):
        if dataset is not None:
            print("{}: {} Studies --> {} Series --> {} Images".format(ds_type,
                                                                    len(dataset.data_df['StudyInstanceUID'].unique()),
                                                                    len(dataset.data_df['SeriesInstanceUID'].unique()),
                                                                    len(dataset.data_df['SOPInstanceUID'].unique())))
    print("(Test Set = Validation Set)\n") if args.val_as_test else None
    
    if len(args.phase3_models_search_path) > 0:
        models_phase3, configs = get_model_paths(args.phase3_models_search_path, args.model_filter)
    if len(args.other_models_search_path) > 0:
        models_other, _ = get_model_paths(args.other_models_search_path, '')

    if args.performance:
        performance(args, models_phase3, phase3_configs=configs, models_other=models_other)

    if args.visualize == 'png':
        visualize_png(args, models_phase3, phase3_configs=configs, models_other=models_other)
     
    if args.visualize == 'nii':
        visualize_nii(args, models_phase3, phase3_configs=configs, models_other=models_other)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # Run-time arguments
    parser.add_argument('--lungmask_model_path', type=str)
    parser.add_argument('--phase3_models_search_path', type=str)
    parser.add_argument('--other_models_search_path', type=str)
    parser.add_argument('--slice_thickness', type=str, default='all')
    parser.add_argument('--model_filter', type=str)
    parser.add_argument('--data_search_path', type=str)
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--val_as_test', type=str2bool, default=True)
    parser.add_argument('--desc',type=str, default='')
    parser.add_argument('--scheduled_start', type=str, default="")

    # Run modes
    parser.add_argument('--performance', type=str2bool, default=False)
    parser.add_argument('--visualize', type=str, default=None)  # png, nii
    parser.add_argument('--separate_figs', type=str2bool, default=True)
    
    # Background settings
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--multi_gpu', type=str2bool, default=True)

    args = parser.parse_args()

    if len(args.scheduled_start) > 0:
        scheduler = run_tools.RunScheduler(args.scheduled_start)

    timer = run_tools.RunTimer(log_path=None, name='evaluate.py')
    timer.add_marker('Start')
    atexit.register(timer.exit_handler)

    run_code = timer.time_points[0].strftime("%Y-%m-%d-%H-%M-%S-%f")
    desc = '_' + args.desc if len(args.desc) > 0 else ''
    args.output_dir = os.path.join(args.output_dir, '{}{}'.format(run_code, desc))
    os.makedirs(args.output_dir, exist_ok=True)
    timer.log_path = args.output_dir

    sys.stdout = run_tools.Printer(log_path=args.output_dir, filename='terminal_run.log', mode='a')

    main(args)