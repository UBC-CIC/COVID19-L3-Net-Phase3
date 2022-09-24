import cv2 as cv
import numpy as np
from skimage import measure as skmeasure
from scipy.ndimage.morphology import binary_fill_holes
from scipy.ndimage.measurements import center_of_mass
from skimage.feature import peak_local_max
import pandas as pd
import mdai
from src.mdai_tools.mdai_lookups import *



def extract_max_prob_points(input_inference, min_size=10, mode='local'):
    inference = np.expand_dims(input_inference, 0) if len(input_inference.shape) == 3 else input_inference
    output_points = []

    for slice_idx in range(0, inference.shape[0]):
        for class_idx in range(0, inference.shape[1]):
            inferred = inference[slice_idx, class_idx, :, :]
            labeled, n_objects = skmeasure.label(inferred > 0, return_num=True)
            for obj_idx in range(1, n_objects + 1):
                selected_obj = inferred * np.isin(labeled, obj_idx)

                # skip objects with 0 probability, or if object size is too small
                if selected_obj.max() == 0 or np.count_nonzero(selected_obj) < min_size:
                    continue

                run_max = False
                if mode == 'local':
                    local_points = peak_local_max(selected_obj, min_distance=20)
                    for pidx in range(0, local_points.shape[0]):
                        local_point = np.concatenate((np.array([class_idx]), local_points[pidx]))
                        if len(input_inference.shape) > 3:
                            output_points.append(np.concatenate((np.array([slice_idx]), local_point)))
                        else:
                            output_points.append(local_point)
                    # failsafe to get global max point of object if none detected using local max
                    run_max = True if local_points.shape[0] == 0 else False

                if mode == 'abs' or run_max:
                    selected_obj[selected_obj != selected_obj.max()] = 0
                    max_point = np.concatenate((np.array([class_idx]), np.argwhere(selected_obj > 0)[0]))

                    if len(input_inference.shape) > 3:
                        output_points.append(np.concatenate((np.array([slice_idx]), max_point)))
                    else:
                        output_points.append(max_point)

    output_points = np.vstack(output_points) if len(output_points) >= 1 else output_points

    return output_points


def object_to_vertices(obj):
    image = np.uint8(np.repeat(obj[..., None], repeats=3, axis=-1)) * 255  # --> (X, Y , 3)
    imgray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    ret, thresh = cv.threshold(imgray, 127, 255, 0)
    output = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)[0]
    return output


def build_annotation_dict(sopinstanceuid, label_id, annotationNumber, data):
    annotation_dict = {'SOPInstanceUID': sopinstanceuid,
                       'labelId': label_id,
                       'annotationNumber': annotationNumber,
                       'data': data}

    return annotation_dict


def split_donut(selected_object):
    # assume object may have donut hole. fill the hole and compare areas
    object_area = np.count_nonzero(selected_object)
    filled_object = binary_fill_holes(selected_object)
    filled_area = np.count_nonzero(filled_object)
    donut = filled_area > object_area  # if filled area is larger than the original object area, there is donut or holes

    if donut:
        background = np.bitwise_not(filled_object)
        donut_hole = np.bitwise_not(background + selected_object)
        # count the number of holes
        labeled, n_holes = skmeasure.label(donut_hole, return_num=True)
        num_objects = n_holes + 1
        centres = pd.DataFrame(np.ceil(center_of_mass(donut_hole, labeled, range(1, num_objects))).astype(int))
        if n_holes > 1:
            centres = centres.sort_values(0)
            centres = centres.values.squeeze()
        else:
            centres = [centres.values.squeeze()]
        objects = []

        for obj_idx in range(0,num_objects):
            temp_object = np.copy(selected_object)
            if obj_idx == 0:
                temp_object[centres[obj_idx][0]:, :] = 0
            elif obj_idx == num_objects - 1:
                temp_object[:centres[-1][0], :] = 0
            else:
                temp_object[0:centres[obj_idx - 1][0], :] = 0
                temp_object[centres[obj_idx][0]:, :] = 0

            # after splitting an object, did it inadvertently result in more than one object?
            temp_labeled, n_objects = skmeasure.label(temp_object, return_num=True)
            for temp_lbl in range(1, n_objects+1):
                objects.append(np.isin(temp_labeled, temp_lbl))

    else:
        objects = [selected_object]

    return objects


def process_object(sopinstanceuid, mask, label_id, start_num=1, mode='mask'):
    annotations = []
    obj_num = start_num
    annotations.append(build_annotation_dict(sopinstanceuid, label_id, obj_num, mdai.common_utils.convert_mask_data(mask)))  
    return annotations, obj_num


def process_points(sopinstanceuid, points, start_num=1):
    annotations = []
    for obj_idx, point in enumerate(points):
        label_id = ailabel_ids[labels[point[0]]]
        data = {"x": point[2].item(),
                "y": point[1].item()}
        annotations.append(build_annotation_dict(sopinstanceuid, label_id, start_num + obj_idx, data))
    return annotations, start_num + len(points) - 1


def process_sopinstance(sopinstanceuid, inference, lung_mask, normal_lung_by_hu, process_lung_mask, process_opacity, mode='threshold'):
    master_annotations = []
    inference_labels = np.argmax(inference, axis=0)

    # process johof lung mask << upload validated functioning @ 2022-01-04 20:15
    lung_johof = lung_mask
    end_num = 0

    if process_lung_mask:
        mask_annotations, end_num = process_object(sopinstanceuid, lung_johof, ailabel_ids['lung_johof'], start_num=1, mode='mask')
        master_annotations += mask_annotations  # attach lung mask annotation dicts to master list

    if normal_lung_by_hu is not None:
        mask_annotations, end_num = process_object(sopinstanceuid, normal_lung_by_hu, ailabel_ids['normal'], start_num=end_num+1, mode='mask')
        master_annotations += mask_annotations  # attach lung mask annotation dicts to master list

    # process opacities --> max points
    if process_opacity:
        if len(np.unique(inference_labels)) > 1:
            if mode == 'pixel-max':  # generates more points (more granularity with argmax)
                inference_masked = np.zeros(inference.shape)
                for label in np.unique(inference_labels):
                    if label > 0:  # don't waste time processing background for points
                        inference_masked[label] = np.isin(inference_labels, label) * inference[label] * lung_johof
                points = extract_max_prob_points(inference_masked, min_size=10, mode='local')
            elif mode == 'threshold':  # << preferred
                for cidx in range(0, inference.shape[0]):
                    # threshold the inferences
                    threshold = 0.8  # if cidx == 0 else 0.8
                    inference[cidx, inference[cidx] < threshold] = 0
                    inference[cidx, :, :] = inference[cidx, :, :] * lung_johof
                points = extract_max_prob_points(inference, min_size=10, mode='local')
                # filter points to remove any from background class
                points = [point for point in points if point[0] > 0]
            else:
                points = None
            if points is not None:
                point_annotations, end_num = process_points(sopinstanceuid, points, start_num=end_num + 1)
                master_annotations += point_annotations
            else:
                raise ValueError("Mode for processing SOP Instances DOES NOT EXIST!")

    return master_annotations
