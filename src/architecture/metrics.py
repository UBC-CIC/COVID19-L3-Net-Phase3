from collections import defaultdict

from scipy import spatial
import numpy as np
import torch
import torch.nn.functional as F


class SegMonitor:
    def __init__(self, other_metrics=False):
        self.cf = None
        self.n_samples = 0
        self.other_metrics = other_metrics
        if other_metrics:
            self.dice_dict = {}
            self.jaccard_dict = {}
            self.hausdorff_score_dict = {}
            self.precision_score_dict = {}
            self.recall_score_dict = {}
            self.specificity_score = {}
            self.accuracy_score = {}

    def val_on_batch(self, pred_mask, masks, num_classes):

        self.n_samples += masks.shape[0]
        ind = masks != -1
        masks = masks[ind]
        pred_mask = pred_mask[ind]

        labels = np.arange(num_classes)
        cf = confusion_multi_class(pred_mask.float(), masks.cuda().float(),
                                   labels=labels)

        if self.cf is None:
            self.cf = cf
        else:
            self.cf += cf

        if self.other_metrics:

            nclasses = labels.max() + 1
            for c in range(nclasses):
                true_mask = (masks == c).cpu().numpy().astype(int)
                pred_mask_c = (pred_mask == c).cpu().numpy().astype(int)
                if c not in self.dice_dict:
                    self.dice_dict[c] = 0
                    self.jaccard_dict[c] = 0
                    self.precision_score_dict[c] = 0
                    self.recall_score_dict[c] = 0
                    self.specificity_score[c] = 0
                    self.accuracy_score[c] = 0

                self.dice_dict[c] += dice_score(pred_mask_c, true_mask)
                self.jaccard_dict[c] += jaccard_score(pred_mask_c, true_mask)
                # self.hausdorff_score_dict[c] += hausdorff_score(pred_mask, true_mask)
                self.precision_score_dict[c] += precision_score(pred_mask_c, true_mask)
                self.recall_score_dict[c] += recall_score(pred_mask_c, true_mask)
                self.specificity_score[c] += specificity_score(pred_mask_c, true_mask)
                self.accuracy_score[c] += accuracy_score(pred_mask_c, true_mask)

    def get_avg_score(self):
        # return -1 
        Inter = np.diag(self.cf)
        G = self.cf.sum(axis=1)
        P = self.cf.sum(axis=0)
        union = G + P - Inter

        nz = union != 0
        iou = Inter / np.maximum(union, 1)
        mIoU = np.mean(iou[nz])
        iou[~nz] = np.nan
        val_dict = {'val_score': mIoU}
        if self.other_metrics:
            val_dict['iou'] = iou
            val_dict['dice'] = np.mean(list(self.dice_dict.values())) / self.n_samples
            val_dict['jaccard'] = np.mean(list(self.jaccard_dict.values())) / self.n_samples
            # val_dict['hausdorff_score_dict'] = np.mean(self.hausdorff_score_dict.values()) / self.n_samples
            val_dict['precision'] = np.mean(list(self.precision_score_dict.values())) / self.n_samples
            val_dict['recall'] = np.mean(list(self.recall_score_dict.values())) / self.n_samples
            val_dict['specificity'] = np.mean(list(self.specificity_score.values())) / self.n_samples
            val_dict['accuracy'] = np.mean(list(self.accuracy_score.values())) / self.n_samples

        return iou, mIoU, val_dict


def confusion_multi_class(prediction, truth, labels):
    """
    cf = confusion_matrix(y_true=prediction.cpu().numpy().ravel(),
            y_pred=truth.cpu().numpy().ravel(),
                    labels=labels)
    """
    nclasses = labels.max() + 1
    cf2 = torch.zeros(nclasses, nclasses, dtype=torch.float, device=prediction.device)
    prediction = prediction.view(-1).long()
    truth = truth.view(-1)
    to_one_hot = torch.eye(int(nclasses), dtype=cf2.dtype, device=prediction.device)
    for c in range(nclasses):
        true_mask = (truth == c)
        pred_one_hot = to_one_hot[prediction[true_mask]].sum(0)
        cf2[:, c] = pred_one_hot

    return cf2.cpu().numpy()


def confusion_binary_class(prediction, truth):
    confusion_vector = prediction / truth

    tp = torch.sum(confusion_vector == 1).item()
    fp = torch.sum(confusion_vector == float('inf')).item()
    tn = torch.sum(torch.isnan(confusion_vector)).item()
    fn = torch.sum(confusion_vector == 0).item()
    cm = np.array([[tn, fp],
                   [fn, tp]])
    return cm


class MetricManager(object):
    def __init__(self, metric_fns):
        self.metric_fns = metric_fns
        self.result_dict = defaultdict(float)
        self.num_samples = 0

    def __call__(self, prediction, ground_truth):
        self.num_samples += len(prediction)
        for metric_fn in self.metric_fns:
            for p, gt in zip(prediction, ground_truth):
                res = metric_fn(p, gt)
                dict_key = metric_fn.__name__
                self.result_dict[dict_key] += res

    def get_results(self):
        res_dict = {}
        for key, val in self.result_dict.items():
            res_dict[key] = val / self.num_samples
        return res_dict

    def reset(self):
        self.num_samples = 0
        self.result_dict = defaultdict(float)


def numeric_score(prediction, groundtruth):
    """Computation of statistical numerical scores:
    * FP = False Positives
    * FN = False Negatives
    * TP = True Positives
    * TN = True Negatives
    return: tuple (FP, FN, TP, TN)
    """
    FP = np.float(np.sum((prediction == 1) & (groundtruth == 0)))
    FN = np.float(np.sum((prediction == 0) & (groundtruth == 1)))
    TP = np.float(np.sum((prediction == 1) & (groundtruth == 1)))
    TN = np.float(np.sum((prediction == 0) & (groundtruth == 0)))
    return FP, FN, TP, TN


def dice_score(prediction, groundtruth):
    pflat = prediction.flatten()
    gflat = groundtruth.flatten()
    d = (1 - spatial.distance.dice(pflat, gflat))
    if np.isnan(d):
        return 0.0
    return d


def jaccard_score(prediction, groundtruth):
    pflat = prediction.flatten()
    gflat = groundtruth.flatten()
    return (1 - spatial.distance.jaccard(pflat, gflat))


def hausdorff_score(prediction, groundtruth):
    return spatial.distance.directed_hausdorff(prediction, groundtruth)


def precision_score(prediction, groundtruth):
    # PPV
    FP, FN, TP, TN = numeric_score(prediction, groundtruth)
    if (TP + FP) <= 0.0:
        return 0.0

    precision = np.divide(TP, TP + FP)
    return precision


def recall_score(prediction, groundtruth):
    # TPR, sensitivity
    FP, FN, TP, TN = numeric_score(prediction, groundtruth)
    if (TP + FN) <= 0.0:
        return 0.0
    TPR = np.divide(TP, TP + FN)
    return TPR


def specificity_score(prediction, groundtruth):
    FP, FN, TP, TN = numeric_score(prediction, groundtruth)
    if (TN + FP) <= 0.0:
        return 0.0
    TNR = np.divide(TN, TN + FP)
    return TNR


def intersection_over_union(prediction, groundtruth):
    FP, FN, TP, TN = numeric_score(prediction, groundtruth)
    if (TP + FP + FN) <= 0.0:
        return 0.0
    return TP / (TP + FP + FN)


def accuracy_score(prediction, groundtruth):
    FP, FN, TP, TN = numeric_score(prediction, groundtruth)
    N = FP + FN + TP + TN
    accuracy = np.divide(TP + TN, N)
    return accuracy
