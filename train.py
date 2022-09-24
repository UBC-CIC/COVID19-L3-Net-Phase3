import json
import os
from collections import OrderedDict
import warnings

import torch
import torch.nn as nn
import torchnet.meter as meter
from torch.utils.tensorboard import SummaryWriter

import argparse
import sys
from datetime import datetime

import src.utilities.general_utils as gutils
from src.architecture.segmentor import SegmentorUNet2D
from src.datasets.phase3_dataset import Phase3HDF5, build_data_df
from src.architecture.metrics import jaccard_score

from src.utilities import run_tools
from src.lungmask import mask as mask_lungs
import atexit
import re
from tqdm import tqdm


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


def main(args):
    train_df, val_df, test_df = [None] * 3

    # Force validation if debugging using validation set as a test set for performance demo
    if args.val_as_test:
        args.val = True
        print("Validation forced to TRUE for debugging due to VAL_AS_TEST=TRUE flag. A validation hold-out will be used as the TEST set.")

    # Search data path for data summaries and build dataframes
    if args.train or args.val_as_test: 
        data_df = build_data_df(search_path=args.data_search_path, data_use_filter='train', holdout_size=0.2 if args.val else None, slice_thickness=args.slice_thickness, verbose=True)
        train_df = data_df.loc[data_df['DataUseAlloc'] == 'Train']
        if args.train_val or args.val_as_test:
            val_df = data_df.loc[data_df['DataUseAlloc'] == 'Holdout']

    if args.train_test or args.test_phase1 or args.test_phase2 or args.test_johof_lung:
        test_df = val_df if args.val_as_test else build_data_df(search_path=args.data_search_path, data_use_filter='test', slice_thickness=args.slice_thickness, verbose=True) 

    # Construct Pytorch Custom Datasets
    datasets = [Phase3HDF5(data_df=train_df, thresh_hu_normal_lung=args.thresh_hu_normal_lung, 
                           thresh_hu_consolidation=args.thresh_hu_consolidation, thresh_vesselness=args.thresh_vesselness,
                           unlabeled_data_mode=args.unlabeled_data_mode) if train_df is not None else None,
                Phase3HDF5(data_df=val_df, thresh_hu_normal_lung=args.thresh_hu_normal_lung, 
                           thresh_hu_consolidation=args.thresh_hu_consolidation, thresh_vesselness=args.thresh_vesselness,
                           unlabeled_data_mode=args.unlabeled_data_mode) if val_df is not None else None,
                Phase3HDF5(data_df=test_df, thresh_hu_normal_lung=args.thresh_hu_normal_lung, 
                           thresh_hu_consolidation=args.thresh_hu_consolidation, thresh_vesselness=args.thresh_vesselness,
                           unlabeled_data_mode=args.unlabeled_data_mode) if test_df is not None else None]
    train_ds, val_ds, test_ds = datasets

    # Verify dataset
    if args.verify_dataset_integrity:
        ds_error_counter = [0] * len(datasets)
        ds_error_idx = [[]] * len(datasets)
        for dsidx in tqdm(len(datasets), leave=False, desc='Verifying Datasets Integrity'):
            for idx in tqdm(range(len(datasets[dsidx])), leave=False, desc='\tSamples'):
                try: 
                    dataset[dsidx][idx]
                except Exception as e:
                    ds_error_counter[dsidx] += 1
                    ds_error_idx[dsidx].append(idx)
        # to add in the future: output the problem hdf5 files

    # Print dataset sizes
    print("\nDATASET STATISTICS")
    for ds_type, dataset in zip(['Train','Validate','Test'], datasets):
        if dataset is not None:
            print("{}: {} Studies --> {} Series --> {} Images".format(ds_type,
                                                                      len(dataset.data_df['StudyInstanceUID'].unique()),
                                                                      len(dataset.data_df['SeriesInstanceUID'].unique()),
                                                                      len(dataset.data_df['SOPInstanceUID'].unique())))
    print("(Test Set = Validation Set)\n") if args.val_as_test else None

    # pos_weights = train_ds.calculate_class_pixel_stats(return_pos_weights=True, exclude_unlabelled=True)

    # Construct Pytorch Dataloaders
    train_dl, val_dl, test_dl = [torch.utils.data.DataLoader(train_ds, batch_size=args.train_batch_size, 
                                                             num_workers=args.num_workers, shuffle=True, drop_last=True) if train_ds is not None else None,
                                 torch.utils.data.DataLoader(val_ds, batch_size=args.val_batch_size, 
                                                             num_workers=args.num_workers, shuffle=True, drop_last=False) if val_ds is not None else None,
                                 torch.utils.data.DataLoader(test_ds, batch_size=args.test_batch_size, 
                                                             num_workers=args.num_workers, shuffle=False, drop_last=False) if test_ds is not None else None]                                                         

    # Construct Pytorch Network and load latest available checkpoint for setting up phase 3 optimizer and scheduler
    reset_fc = args.reset_fc
    if args.checkpoint_phase3 != "":
        if os.path.exists(args.checkpoint_phase3):
            load_ckpt = args.checkpoint_phase3 
            if reset_fc:
                warnings.warn("Overriding reset_fc=True to reset_fc=False because checkpoint_phase3 exists and will be loaded. To avoid this warning, manually set reset_fc=False when providing a checkpoint_phase3")
                reset_fc = False
        else:
            load_ckpt = None
            raise ValueError("Unable to find {}".format(args.checkpoint_phase3))
    elif args.checkpoint_phase2 != "":
        if os.path.exists(args.checkpoint_phase2):
            load_ckpt = args.checkpoint_phase2
        else:
            load_ckpt = None
            raise ValueError("Unable to find {}".format(args.checkpoint_phase2))
    else:
        load_ckpt = None
        
    net = SegmentorUNet2D(num_channels=1, num_classes=5, reset_fc=reset_fc, 
                          checkpoint=load_ckpt, multi_gpu=args.multi_gpu, model_type='phase3') 
    net.to(device)

    # Construct Pytorch Optimizer and LR Scheduler
    optimizer = torch.optim.Adam([{'params': net.parameters(), 'lr': args.lr}])    
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.scheduler_milestones, gamma=args.scheduler_gamma)

    # Set-up Logging
    print('TensorBoard runID: ', log_dir)
    writer = SummaryWriter(log_dir)

    # Print Model Information
    print('\n### Model Information')
    print('Model Size: {:8.1f} mb'.format(gutils.model_size(net)))
    print('Number of Parameters: {:9d}\n'.format(gutils.num_params(net)))

    # Get Data Information
    for dl in [train_dl, val_dl, test_dl]:
        if dl is not None:  # Get the first available existing dataloader
            data_sample = dl.dataset[0]
            break

    assert data_sample is not None
    print('Input shape: ', data_sample['image'].shape)
    print('Label shape: ', data_sample['label'].shape)
    print('Input size %8.1f mb \n' % gutils.tensor_size(data_sample['image']))
    del data_sample

    # Begin Operations
    epochs = args.epochs if train_dl is not None else 0
    best_loss = float("inf")
    ckpt_epoch = None
    
    # Check if training run is NEW or CONTINUED from a previously stopped/crashed run
    if os.path.exists(args.checkpoint_phase3):
        filename = os.path.split(args.checkpoint_phase3)[-1]
        ckpt_epoch = re.search('epoch(.*){}'.format(os.path.splitext(filename)[-1]), filename)
        try:
            if ckpt_epoch is not None:
                ckpt_epoch = int(ckpt_epoch.group(1))  
                print ("Last saved checkpoint @ epoch {}. Continuing from epoch {}".format(ckpt_epoch, ckpt_epoch + 1))
                # Load last optimizer state if we are continuing a previous training cycle
                print("Loading Optimizer State from Model - Resuming Training from {}".format(ckpt_epoch))
                optimizer.load_state_dict(net.initial_optimizer_state) if net.initial_optimizer_state is not None else None
                if net.initial_loss is not None:
                    best_loss = net.initial_loss
        except ValueError:
            print ("Found checkpoint with EPOCH number, but could not convert [VALUE] in epoch[VALUE] to integer. Using EPOCH=0")
    
    best_epoch = ckpt_epoch if ckpt_epoch is not None else -1
    epoch_start = best_epoch + 1

    early_stop = 0

    if args.train:
        for epoch in range(epoch_start, epochs):
            # Train 
            train_on_loader(net, train_dl, optimizer, epoch, args.iter_per_update, writer, args.make_train_plots, figs_dir) if train_dl is not None and args.train else None

            # Validate
            if args.train_val:
                val_loss = val_on_loader(net, val_dl, epoch, writer, args.make_val_plots, figs_dir) if val_dl is not None else float("inf")
                if val_loss < best_loss:
                    early_stop = 0
                    best_loss = val_loss
                    print(run_tools.timestamp() + 'Saving best model - ', best_loss)
                    state = {
                        'epoch': epoch,
                        'model': net.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'loss': best_loss
                    }
                    os.makedirs(model_dir, exist_ok=True)
                    # Save current checkpoint
                    best_model_path = os.path.join(model_dir, 'best_model-epoch{:04d}.ckpt'.format(epoch))
                    torch.save(state, best_model_path)
                    # Remove old checkpoint
                    if best_epoch >= 0:
                        old_model_ckpt = 'best_model-epoch{:04d}.ckpt'.format(best_epoch)
                        print(run_tools.timestamp() + 'Removing outdated model - ', old_model_ckpt)
                        os.remove(os.path.join(model_dir, old_model_ckpt))
                    # Update best epoch tracker only after having removed the old best checkpoint
                    best_epoch = epoch
                else:
                    early_stop += 1
                    # temp_path = os.path.join(model_dir, 'temp-epoch{:04d}.ckpt'.format(epoch))
                    # torch.save(state, temp_path)
                    print("Early stopping @ {:d} of {:d}".format(early_stop, args.early_stop))
                    if early_stop >= args.early_stop and early_stop > 0:
                        break

            scheduler.step()
            sys.stdout.flush()

        # Final Performance on Test Set with BEST model, last stored in STATE dictionary after training
        test_on_loader(net, test_dl, best_epoch, writer, args.make_test_plots, figs_dir) if test_dl is not None and args.train_test else None

    # Other Model Tests
    if args.test_johof_lung:  # not yet tested - 2022-05-09
        net = mask_lungs.get_model('unet', 'R231CovidWeb', modelpath=args.lungmask_model_path)
        avg_iou = meter.AverageValueMeter()
        for idx in tqdm(range(len(test_df)), leave=False, dynamic_ncols=True, desc='LungMask_JoHof Evaluation'):
            data = test_ds[idx]
            ct = data['raw_ct'].detach().cpu().numpy()
            label = (data['parenchyma'].detach().cpu().numpy() != 0)
            mask = mask_lungs.apply(ct, net)[0] > 0
            avg_iou.add(jaccard_score(mask, label))

        print("LungMask_JoHof Lung Segmentation IoU: {:0.4f}\n".format(avg_iou.value()[0]))

    # Load and test Phase 1 Model Performance if specified
    if args.test_phase1: 
        if os.path.exists(args.checkpoint_phase1):  # not yet tested 2022-05-09
            net = SegmentorUNet2D(num_channels=1, num_classes=5, checkpoint=args.checkpoint_phase1, multi_gpu=args.multi_gpu, model_type='phase1') # build phase 1 net
            net.to(device)
            test_on_loader(net, test_dl, best_epoch, writer, args.make_test_plots, figs_dir) if test_dl is not None else None  # << Phase 1
            print("")
        else:
            print("Skipping testing for phase 1 since checkpoint_phase2 does not exist")

    # Load Test Phase 2 Net
    if args.test_phase2:
        if  os.path.exists(args.checkpoint_phase2) is not None and args.test_phase2:
            net = SegmentorUNet2D(num_channels=1, num_classes=5, checkpoint=args.checkpoint_phase2, multi_gpu=args.multi_gpu, model_type='phase2')  # reload phase 2 net
            net.to(device)
            test_on_loader(net, test_dl, best_epoch, writer, args.make_test_plots, figs_dir) if test_dl is not None else None  # << Phase 2
            print("")
        else:
            print("Skipping testing for phase 2 since checkpoint_phase2 does not exist")

    # Load Test Phase 3 Net
    if args.test_phase3:
        if  os.path.exists(args.checkpoint_phase3) is not None:
            net = SegmentorUNet2D(num_channels=1, num_classes=5, checkpoint=args.checkpoint_phase3, multi_gpu=args.multi_gpu, model_type='phase3')  # reload phase 3 net
            net.to(device)
            test_on_loader(net, test_dl, best_epoch, writer, args.make_test_plots, figs_dir) if test_dl is not None else None  # << Phase 3
            print("")
        else:
            print("Skipping testing for phase 3 since checkpoint_phase3 does not exist")


def train_on_loader(net, dataloader, optimizer, epoch, iter_per_update, writer, make_plots, figs_dir):
    meter_sen = meter.AverageValueMeter()
    meter_loss = meter.AverageValueMeter()
    meter_time = meter.TimeMeter(unit=False)
    net.train()

    for batch_idx, batch in enumerate(dataloader):

        optimizer.zero_grad()
        loss, outputs, metrics = net.process_batch(batch, device)
        optimizer.step()

        # Track stats
        meter_loss.add(loss.item())
        if metrics is not None:
            meter_sen.add(metrics['global_sen'].cpu())

        # Keep stats
        if batch_idx % iter_per_update == 0:
            t = meter_time.value()
            eps = iter_per_update * dataloader.batch_size / t

            avg_sen, _ = meter_sen.value()
            avg_loss, _ = meter_loss.value()

            print(run_tools.timestamp() + 'Epoch: {:4d} |  Batch: {:4d} | tLoss: {:6.4f} | tSen: {:6.4f} | tEPS: {:6.4f}'.format(epoch, batch_idx, avg_loss, avg_sen, eps), flush=True)

            tb_iter = epoch * len(dataloader.dataset) + batch_idx * dataloader.batch_size + 1
            writer.add_scalar('train_loss', avg_loss, tb_iter)
            writer.add_scalar('train_eps', eps, tb_iter)

            if make_plots:
                batch_figs_dir = os.path.join(figs_dir, 'train/epoch_{:06d}/batch_{:06d}'.format(epoch, batch_idx))
                os.makedirs(batch_figs_dir, exist_ok=True)
                net.viz_on_batch(batch, outputs, batch_figs_dir)

            meter_time.reset()
            meter_sen.reset()
            meter_loss.reset()


def val_on_loader(net, dataloader, epoch, writer, make_plots, figs_dir):
    meter_sen = meter.AverageValueMeter()
    meter_loss = meter.AverageValueMeter()
    meter_time = meter.TimeMeter(unit=False)
    net.eval()

    batch = None  # initialize
    for _, batch in enumerate(tqdm(dataloader, leave=False, desc='Validating (Epoch {})'.format(epoch))):
        with torch.no_grad():
            loss, outputs, metrics = net.process_batch(batch, device)
            meter_loss.add(loss.item())
            if metrics is not None:
                meter_sen.add(metrics['global_sen'].cpu())

    # Print stats from validation
    t = meter_time.value()
    num_examples = len(dataloader.dataset)
    eps = num_examples / t

    avg_loss, _ = meter_loss.value()
    avg_sen, _ = meter_sen.value()

    print(run_tools.timestamp() + 'Validation Epoch: {:4d} | vLoss: {:6.4f} | vSen: {:6.4f} | vEPS: {:6.4f}'.format(epoch, avg_loss, avg_sen, eps), flush=True)

    tb_iter = epoch + 1
    writer.add_scalar('val_loss', avg_loss, tb_iter)

    if make_plots:
        val_figs_dir = os.path.join(figs_dir, 'val/epoch_%06d' % (epoch))
        os.makedirs(val_figs_dir, exist_ok=True)
        net.viz_on_batch(batch, outputs, val_figs_dir)

    meter_time.reset()
    meter_sen.reset()
    meter_loss.reset()

    return avg_loss


def test_on_loader(net, dataloader, epoch, writer, make_plots, figs_dir):
    # n.b. if test_dl batch size > 1, the torchnet meters are average-over-batches, and not average of all samples.
    meters_metrics = {}
    meters_confusion = meter.ConfusionMeter(k=5)
    meters_loss = meter.AverageValueMeter()
    meters_time = meter.TimeMeter(unit=False)
    net.eval()

    for batch_idx, batch in enumerate(tqdm(dataloader, desc='Testing (Epoch {})'.format(epoch))):

        with torch.no_grad():
            loss, outputs, metrics = net.process_batch(batch, device)
            meters_loss.add(loss.item())

            if metrics is not None:
                if batch_idx == 0:
                    for key1 in metrics.keys():
                        if key1[0:5] == 'class':
                            for idx in range(len(metrics[key1])):
                                meters_metrics['c{}_{}'.format(key1[-3:], idx)] = meter.AverageValueMeter()    
                        else:
                            meters_metrics[key1] = meter.AverageValueMeter()

                for key1, values1 in metrics.items():
                    if key1[0:5] == 'class': # separate the class-wise accuracies
                        for idx in range(len(metrics[key1])): 
                            value = values1.detach().cpu()[idx]
                            if not torch.isnan(value):
                                meters_metrics['c{}_{}'.format(key1[-3:], idx)].add(value)
                                if key1[-3] == 'sen':
                                    print(meters_metrics['c{}_{}'.format(key1[-3:], idx)].value()[0], flush=True)
                    else:
                        meters_metrics[key1].add(values1.detach().cpu())

            if make_plots:
                test_figs_dir = os.path.join(figs_dir, 'test/epoch_%06d' % (epoch))
                net.viz_on_batch(batch, outputs, test_figs_dir, test_mode=True, plot_quantitative=args.plot_quantitative, plot_qualitative=args.plot_qualitative)

    # Print stats from validation
    t = meters_time.value()
    num_examples = len(dataloader.dataset)
    eps = num_examples / t

    
    metrics_msgs = []
    for key1, value1 in meters_metrics.items():
        metrics_msgs.append('{}: {:6.4f}'.format(key1, value1.value()[0])) if key1 != 'confusion' else None
    metrics_msgs = ' | '.join(metrics_msgs)
    print(run_tools.timestamp() + 'Test {}: {:4d} | {} | teEPS: {:6.4f}'.format(key.upper(), epoch, metrics_msgs, eps), flush=True)
    print('Confusion Matrix')
    print(meters_metrics[key]['confusion'].value()[0])
    
    # NOT IMPLEMENTED
    # print("Confusion Matrix:")
    # print(meters_confusion.value())

    tb_iter = epoch + 1
    writer.add_scalar('test_loss', meters_loss.value()[0], tb_iter)
    writer.add_scalar('test_eps', eps, tb_iter)


if __name__ == '__main__':
    str2bool = gutils.str2bool
    parser = argparse.ArgumentParser()

    # Hyperparameters
    parser.add_argument('--reset_fc', type=str2bool, default=True)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--iter_per_update', type=int, default=100)
    parser.add_argument('--train_batch_size', type=int, default=2)
    parser.add_argument('--val_batch_size', type=int, default=2)
    parser.add_argument('--test_batch_size', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--early_stop', type=int, default=0)
    parser.add_argument('--scheduler_milestones', '-ms', nargs='+', type=int, default=[10, 20])
    parser.add_argument('--scheduler_gamma', type=float, default=1e-1)

    # Runtime Arguments
    parser.add_argument('--verbose', '-v', type=str2bool, default=False)
    parser.add_argument('--multi_gpu', type=str2bool, default=True)
    parser.add_argument('--train', type=str2bool, default=True)
    parser.add_argument('--train_val', type=str2bool, default=True)
    parser.add_argument('--train_test', type=str2bool, default=True)
    parser.add_argument('--scheduled_start', type=str, default="")
    parser.add_argument('--lungmask_model_path', type=str, default=r'model_in\unet_r231covid-0de78a7e.pth')
    parser.add_argument('--checkpoint_phase1', type=str, default="")
    parser.add_argument('--checkpoint_phase2', type=str, default="", required=True)
    parser.add_argument('--checkpoint_phase3', type=str, default="")
    parser.add_argument('--desc',type=str, default='')

    # Overrides
    parser.add_argument('--override_run_code', type=str, default='')

    # Plot Settings
    parser.add_argument('--make_train_plots', type=str2bool, default=True)
    parser.add_argument('--make_val_plots', type=str2bool, default=True)
    parser.add_argument('--make_test_plots', type=str2bool, default=True)
    parser.add_argument('--plot_quantitative', type=str2bool, default=True)
    parser.add_argument('--plot_qualitative', type=str2bool, default=True)

    # Pre-train Test Switches
    parser.add_argument('--test_johof_lung', type=str2bool, default=False)
    parser.add_argument('--test_phase1', type=str2bool, default=False)
    parser.add_argument('--test_phase2', type=str2bool, default=False)
    parser.add_argument('--test_phase3', type=str2bool, default=False)

    # Data Switches
    parser.add_argument('--unlabeled_data_mode', type=str, default="a") # options: 'a', 'b', 'c' for leave as unlabeled, replace all as normal lung, replace unlabeled within GGO range as normal lung
    parser.add_argument('--verify_dataset_integrity', type=str2bool, default=False)
    parser.add_argument('--thresh_hu_normal_lung', type=int, default=None)
    parser.add_argument('--thresh_hu_consolidation', type=int, default=None)
    parser.add_argument('--thresh_vesselness', type=float, default=None) # if -1, then calculate based on thresh_hu_consolidation and targets
    parser.add_argument('--slice_thickness', type=str, default='all')

    # Debug Switches
    parser.add_argument('--val_as_test', type=str2bool, default=False)

    # Container environment
    parser.add_argument('--data_search_path', '-d', type=str,
                        default=os.environ.get('SM_CHANNEL_TRAINING'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--figs_dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR'))
    parser.add_argument('--log_dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR'))

    args = parser.parse_args()
    
    if len(args.scheduled_start) > 0:
        scheduler = run_tools.RunScheduler(args.scheduled_start)
    os.makedirs(args.log_dir, exist_ok=True)

    timer = run_tools.RunTimer(log_path=args.log_dir, name='train.py')
    timer.add_marker('Start')
    atexit.register(timer.exit_handler)

    run_code = timer.time_points[0].strftime("%Y-%m-%d-%H-%M-%S-%f")

    # Handle run_code overriding if continuing from another training job
    if len(args.override_run_code) > 0:
        # Directory handling
        temp_run_code = args.override_run_code
        if not os.path.exists(os.path.join(args.log_dir, 'run_{}_{}'.format(temp_run_code, args.desc))):
            warnings.warn("Could not locate output directories for provided override_run_code. Ensure matching desc is provided. Using override as new run_code")
        run_code = temp_run_code

    desc = '_' + args.desc if len(args.desc) > 0 else ''

    # Create directories
    figs_dir = os.path.join(args.figs_dir, '{}{}'.format(run_code, desc))
    log_dir = os.path.join(args.log_dir, '{}{}'.format(run_code, desc))
    model_dir = os.path.join(args.model_dir, '{}{}'.format(run_code, desc))
    for dir in [figs_dir, log_dir, model_dir]:
        os.makedirs(dir, exist_ok=True)

    sys.stdout = run_tools.Printer(log_path=log_dir, filename='terminal_run.log', mode='a')

    # Configuration file handling
    config_fn = os.path.join(model_dir,'config.json')
    if os.path.exists(config_fn):
        with open(config_fn, 'r') as f:
            config = json.load(f)
            for key, value in config.items():
                # only load threshold config items so that data is processed appropriately for the model being used
                if key[0:6] == 'thresh':
                    print("Loading {} = {}".format(key, value))
                    setattr(args, key, value)
    else:
        # Dump training configuration
        with open(config_fn, 'w') as f:
            config = vars(args)
            json.dump(config, f)

    main(args)