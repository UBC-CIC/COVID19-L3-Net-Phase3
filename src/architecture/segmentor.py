from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
from src.architecture.unet import UNet
import matplotlib.pyplot as plt
from matplotlib import cm
import torchmetrics


class SegmentorUNet2D(nn.Module):

    def __init__(self, num_channels, num_classes, checkpoint=None, reset_fc=False, multi_gpu=False, model_type='phase3', verbose=True):
        super().__init__()

        self.model_type = model_type
        self.num_classes = num_classes

        net = UNet(
            n_channels=num_channels,
            n_classes=num_classes,
        )

        # Load checkpoint
        state = None
        if checkpoint is not None:
            print('Loading Checkpoint: ', checkpoint) if verbose else None
            state = torch.load(checkpoint, map_location=torch.device('cpu'))
            state_num_classes = state['model']['outc.conv.weight'].shape[0]
            net_num_classes = net.outc.conv.weight.shape[0]
            if not (net_num_classes == state_num_classes):
                print('Classifier mismatch. Loaded checkpoint with {} output classes for num_channels={}...'.format(state_num_classes, net_num_classes)) if verbose else None
                reset_fc = True
            if reset_fc:
                print("\tResetting fully-connected layer with {} output classes and randomized weights".format(net_num_classes)) if verbose else None
                state['model']['outc.conv.weight'] = net.outc.conv.weight
                state['model']['outc.conv.bias'] = net.outc.conv.bias
            net.load_state_dict(state['model'])
            self.initial_optimizer_state = state['optimizer']
            if 'loss' in state.keys():
                self.initial_loss = state['loss']
            else:
                self.initial_loss = None
        else:
            self.initial_optimizer_state = None

        if multi_gpu and (torch.cuda.device_count() > 1):
            print('Using multi-gpu') if verbose else None
            net = nn.DataParallel(net)

        self.net = net

    def state_dict(self):
        if isinstance(self.net, nn.DataParallel):
            state_dict = self.net.module.state_dict()
        else:
            state_dict = self.net.state_dict()
        return state_dict

    def loss_fn(self, logits, labels):
        # Phase 2 Loss Function (Modified for Phase 3)
        outputs = torch.softmax(logits, dim=1) if self.model_type == 'phase3' else self.transform_outputs(torch.softmax(logits,dim=1))
        loss = F.kl_div(outputs.log(), labels, log_target=False, reduction='batchmean')

        if torch.isnan(loss):
            raise SystemExit("Loss = NaN. Stopping due to possible divergence; consider lowering learning rate.")

        return loss

    def transform_outputs(self, outputs):
        # used to transform the outputs from phase 1 and phase 2 models so they can be used to compare against phase 3 labels
        outputs = torch.cat([outputs[:,:1,...], torch.zeros(outputs[:,1:2,...].shape, device=outputs.device), outputs[:,1:2, ...], outputs[:,2:4,...].sum(dim=1, keepdim=True),outputs[:,4:,...]], dim=1)
            # merge the GGO and crazy-paving into one class (idx-3)
        return outputs

    def process_batch(self, batch, device):

        images = batch['image']
        labels = batch['label']

        images = images.to(device)
        labels = labels.to(device)

        if self.training:  
            logits = self.net(images)
            loss = self.loss_fn(logits, labels)
            loss.backward()

        else:
            with torch.no_grad():
                logits = self.net(images)
                loss = self.loss_fn(logits, labels)
        
        outputs = torch.softmax(logits,dim=1) if self.model_type == 'phase3' else self.transform_outputs(torch.softmax(logits,dim=1))
    
        metrics = self.get_metrics_dict(outputs, labels)

        return loss, outputs, metrics

    @torch.no_grad()
    def get_metrics_dict(self, outputs, labels): 
        outputs = outputs.detach()
        labels = labels.detach()
        metrics = OrderedDict()

        classes = self.num_classes

        targets_ = torch.argmax(labels.detach(), dim=1)
        pred_cls = torch.argmax(outputs.detach(), dim=1)

        # calculate metrics on flattened, non-negative targets/labels pixels
        labeled_idx = torch.where(torch.logical_and(targets_ >= 0, targets_ != 1)) # ignore pixels that are unlabelled, or vessels, in calculation of metrics
        labeled_tgt = targets_[labeled_idx[0], labeled_idx[1], labeled_idx[2]]
        labeled_prd = pred_cls[labeled_idx[0], labeled_idx[1], labeled_idx[2]]
        
        # calculate global accuracies, either all together and averaged across class
        metrics['global_acc'] = torchmetrics.functional.accuracy(labeled_prd, labeled_tgt, num_classes=classes, average='micro')
        metrics['avg_class_acc'] = torchmetrics.functional.accuracy(labeled_prd, labeled_tgt, num_classes=classes, average='macro')

        stat_scores = torchmetrics.functional.stat_scores(labeled_prd, labeled_tgt, reduce='macro', num_classes=5)[:,:-1] # [N-class, 4 (tp, fp, tn, fn)]
        # global sensitivity, specificity, ppv, npv
        metrics['global_sen'] = torch.div(stat_scores[:,0].sum(dim=0), stat_scores[:,0].sum(dim=0) + stat_scores[:,3].sum(dim=0))
        metrics['global_spc'] = torch.div(stat_scores[:,2].sum(dim=0), stat_scores[:,2].sum(dim=0) + stat_scores[:,1].sum(dim=0))
        metrics['global_ppv'] = torch.div(stat_scores[:,0].sum(dim=0), stat_scores[:,0].sum(dim=0) + stat_scores[:,1].sum(dim=0))
        metrics['global_npv'] = torch.div(stat_scores[:,2].sum(dim=0), stat_scores[:,2].sum(dim=0) + stat_scores[:,3].sum(dim=0))
        metrics['global_f1s'] = torch.div(2 * metrics['global_ppv'] * metrics['global_sen'], metrics['global_ppv'] + metrics['global_sen'])

        # calculate KLD
        metrics['class_kld'] = torch.ones(classes)
        for class_idx in range(0, classes):
            if class_idx != 1:
                flat_labels = torch.flatten(labels[:, class_idx])
                keep_pixels = torch.where(flat_labels >= 0)[0]
                flat_labels = torch.index_select(flat_labels, 0, keep_pixels)
                labels_ = torch.zeros((flat_labels.shape[0], 2))
                labels_[:, 1] = torch.round(flat_labels, decimals=5)
                labels_[:, 0] = 1 - labels_[:, 1]

                flat_output = torch.index_select(torch.flatten(outputs[:, class_idx]),0,keep_pixels)
                outputs_ = torch.zeros((flat_output.shape[0], 2))
                outputs_[:, 1] = torch.round(flat_output, decimals=5)  
                outputs_[:, 0] = 1 - outputs_[:, 1]

                # overcome numerical limits throwing error in kld
                outputs_ = torch.where(outputs_ == 1, 0.999999, outputs_)
                outputs_ = torch.where(outputs_ == 0, 0.000001, outputs_)
                
                torchmetrics.functional.kl_divergence(outputs_, labels_)
        
        # calculate stat_scores
        metrics['confusion'] = torchmetrics.functional.confusion_matrix(labeled_prd, labeled_tgt, num_classes=classes)
        metrics['class_acc'] = torchmetrics.functional.accuracy(labeled_prd, labeled_tgt, num_classes=classes, average='none')
        metrics['class_sen'] = torch.div(stat_scores[:,0], stat_scores[:,0] + stat_scores[:,3])
        metrics['class_spc'] = torch.div(stat_scores[:,2], stat_scores[:,2] + stat_scores[:,1])
        metrics['class_ppv'] = torch.div(stat_scores[:,0], stat_scores[:,0] + stat_scores[:,1])
        metrics['class_npv'] = torch.div(stat_scores[:,2], stat_scores[:,2] + stat_scores[:,3])
        metrics['class_f1s'] = torch.div(2 * metrics['class_ppv'] * metrics['class_sen'], metrics['class_ppv'] + metrics['class_sen'])

        # calculate class IoU
        class_jcc = torchmetrics.functional.jaccard_index(labeled_prd, labeled_tgt, num_classes=classes, reduction='none')
        class_jcc[torch.isnan(metrics['class_acc'])] = torch.nan # smush IoU calculations if accuracy cannot be calculated on the batch
        metrics['class_iou'] = class_jcc

        return metrics

    @torch.no_grad()
    def predict_on_batch(self, images):
        logits = self.net(images)
        outputs = torch.softmax(logits, dim=1) if self.model_type == 'phase3' else self.transform_outputs(torch.softmax(logits,dim=1))
        return outputs

    @torch.no_grad() 
    def plot_preds_qualitative(self, image, output, filepath):
        npy_output = np.squeeze(output.detach().cpu().numpy())
        npy_image = np.squeeze(image.detach().cpu().numpy())
        color = np.array([[0, 150, 255, 0],
                          [255, 189, 51, 0],
                          [255, 51, 51, 0]])/255

        fig = plt.figure(constrained_layout=True, dpi=150, figsize=(8, 4))
        gs = fig.add_gridspec(1, 2)

        # Plot original CT
        ax = fig.add_subplot(gs[0, 0])
        ax.imshow(npy_image, cmap='gray', vmin = -0.6, vmax = 1.5) 
        ax.set_ylim([0, 512])
        ax.set_xlim([0, 512])
        ax.invert_yaxis()
        ax.axis('off')

        # Image w/ Model Summed Outputs
        output = np.argmax(npy_output,axis=0)
        ax = fig.add_subplot(gs[0, 1])
        ax.imshow(npy_image, cmap='gray', vmin = -0.6, vmax = 1.5)   
        ax.axis('off')   
        alpha_scale_factor = 0.75
        for idx in range(2, npy_output.shape[0]):
            overlay = np.zeros([*npy_image.shape, 4])
            overlay[:,:,:] = color[idx-2] # get color
            idx_alpha = npy_output[idx] # get probabilities
            idx_alpha[output != idx] = 0 # set 
            idx_alpha = idx_alpha * alpha_scale_factor
            overlay[..., -1] = idx_alpha 
            ax.imshow(overlay)

        plt.savefig(filepath)
        plt.close()


    @torch.no_grad() 
    def plot_preds_quantitative(self, image, output, label, filepath):
        fig = plt.figure(constrained_layout=True, dpi=100, figsize=(20, 8))
        gs = fig.add_gridspec(2, 8)
        color = cm.tab10(np.linspace(0,1,10))

        metrics = self.get_metrics_dict(output.detach().cpu(), label.detach().cpu())
        npy_output = np.squeeze(output.detach().cpu().numpy())
        npy_label = np.squeeze(label.detach().cpu().numpy())
        npy_image = np.squeeze(image.detach().cpu().numpy())

        # Plot original CT
        ax = fig.add_subplot(gs[:2, :2])
        ax.set_title('CT Slice')
        ax.imshow(npy_image, cmap='gray', vmin = -0.6, vmax = 1.5) 
        ax.set_ylim([0, 512])
        ax.set_xlim([0, 512])
        ax.invert_yaxis()
        ax.axis('off')

        # Image Only w/ True Summed Labels
        ax = fig.add_subplot(gs[0, 2])
        ax.set_title('Labeled CT')
        ax.imshow(npy_image, cmap='gray', vmin = -0.6, vmax = 1.5) 
        ax.set_ylim([0, 512])
        ax.set_xlim([0, 512])
        ax.invert_yaxis()
        ax.axis('off')
        for idx in range(npy_label.shape[0]):
            rgb_label = np.zeros([*npy_image.shape, 4])  
            rgb_label[np.where(npy_label[idx] == 1)] = color[idx]
            rgb_label[rgb_label[:,:,-1] != 0,-1] = 0.5
            ax.imshow(rgb_label)

        # Image w/ True Individual Class Labels (One for each class)
        titles = ['True Not Lung', 'True Vessels' ,'True Healthy Lung', 'True Ground-glass', 'True Consolidation']
        for idx in range(npy_label.shape[0]):
            ax = fig.add_subplot(gs[0, 3+idx])
            ax.set_title("{}\n({:d} Labelled Pixels)".format(titles[idx], (npy_label[idx] == 1).sum()))
            ax.imshow(npy_image, cmap='gray', vmin = -0.6, vmax = 1.5)
            ax.set_ylim([0, 512])
            ax.set_xlim([0, 512])
            ax.invert_yaxis()
            ax.axis('off')
            rgb_label = np.zeros([*npy_image.shape, 4])
            rgb_label[np.where(npy_label[idx] == 1)] = color[idx]
            rgb_label[rgb_label[:,:,-1] != 0,-1] = 0.5
            ax.imshow(rgb_label)

        # Image w/ Model Summed Outputs
        ax = fig.add_subplot(gs[1, 2])
        ax.set_title('Model\n({:2.2f}% Accurate)'.format(metrics['pos']['gacc'] * 100))
        ax.imshow(npy_image, cmap='gray', vmin = -0.6, vmax = 1.5)   
        ax.axis('off')   
        for idx in range(npy_output.shape[0]):
            overlay = np.zeros([*npy_image.shape, 4])
            overlay[:,:,:] = color[idx]
            overlay[:,:,-1] = npy_output[idx] * 0.5
            ax.imshow(overlay)

        # Image w/ Model Individual Class Outputs (One for each class)
        titles = ['P(Not Lung)', 'P(Vessels)' ,'P(Healthy Lung)', 'P(Ground-glass)', 'P(Consolidation)']
        for idx in range(npy_output.shape[0]):
            ax = fig.add_subplot(gs[1, 3+idx])
            if idx == 0:
                ax.set_title(titles[idx] + "\n({:0.4f} IoU)".format(metrics['all']['liou']))
            else:
                ax.set_title(titles[idx] + "\n({:2.2f}% Accurate)".format(metrics['pos']['class_acc'][idx] * 100))
            ax.imshow(npy_image, cmap='gray', vmin = -0.6, vmax = 1.5)
            ax.set_ylim([0, 512])
            ax.set_xlim([0, 512])
            ax.invert_yaxis()
            ax.axis('off')
            overlay = np.zeros([*npy_image.shape, 4])
            overlay[:,:,:] = color[idx]
            overlay[:,:,-1] = npy_output[idx] * 0.5
            ax.imshow(overlay)
        ax.axis('off')

        plt.savefig(filepath)
        plt.close()

    @torch.no_grad()
    def viz_on_batch(self, batch, outputs, save_dir, test_mode=False, plot_quantitative=True, plot_qualitative=True):

        if plot_quantitative:
            quantitative = os.path.join(save_dir,'quantitative')
            os.makedirs(quantitative, exist_ok=True)

        if plot_qualitative:
            qualitative = os.path.join(save_dir,'qualitative')
            os.makedirs(qualitative, exist_ok=True)

        dataset = batch['dataset']
        images = batch['image']
        labels = batch['label']
        study_uids = batch['StudyInstanceUID']
        series_uids = batch['SeriesInstanceUID']
        sop_uids = batch['SOPInstanceUID']
        dim0idxs = batch['Dim0Index']

        for i in range(images.shape[0]):
            image = images[i:i+1,...]
            output = outputs[i:i+1,...].detach()
            label = labels[i:i+1,...]

            # save quantitative
            if plot_quantitative:
                save_dir = quantitative
                save_path = os.path.join(save_dir,'{}_{}_{:03d}.png'.format(dataset[i],study_uids[i],dim0idxs[i]))
                self.plot_preds_quantitative(image, output, label, save_path)

            if plot_qualitative:
                save_dir = qualitative
                save_path = os.path.join(save_dir,'{}_{}_{:03d}.png'.format(dataset[i],study_uids[i],dim0idxs[i]))
                self.plot_preds_qualitative(image, output, save_path)