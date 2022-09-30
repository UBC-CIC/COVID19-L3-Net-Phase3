# COVID-19 L3-Net: Phase 3
## Introduction
This repository contains the development code for phase 3 of COVID-19 L3-Net, a machine learning model that detects areas of lung infection from axial slices of chest computed tomography (CT) images. This project is a collaboration between the University of British Columbia Cloud Innovation Centre (UBC-CIC), Amazon Web Services (AWS), and SapienML. Code and documentation for previous phases can be found in their own respective repositories. Data used to train is multinational, multi-site, and includes local data from Vancouver General Hospital and St. Paul's Hospital located in Vancouver, BC. This document will not discuss data specifics. 

<p align="middle">
  <img src="/documentation/sample_slice_ct.png" width="200" />
  <img src="/documentation/sample_slice_partial.png" width="200" /> 
  <img src="/documentation/sample_slice_solid.png" width="200" />
</p>

This project was developed using Python 3.9.7 and PyTorch 1.12.1. Additional details in ./requirements.txt

Link to the UBC-CIC website for Phase 2: [UBC-CIC L3-Net Version 2.0](https://cic.ubc.ca/project/open-source-ai-model-for-covid-19-ct-diagnostics-and-prognosis/version-2-0-current-open-source-ai-model-for-covid-19-ct-diagnostics-and-prognosis/)

The phase 3 model, like the phase 2 model, is based on the 2D UNet. Phase 2 model weights were used to initialize phase 3 model training:
- 0: Background
- 1: Vessels
- 2: Normal/Healthy Lung
- 3: GGO/Crazy Paving
- 4: Consolidation

## Quick Start
The trained model can be deployed from the command line or within a python script. In both cases, three paths are required:
1. DICOM path
    > The DICOM path should point to a folder of datasets, studies, or series. Both methods of deployment will search the provided path and infer on all identified CT series.
    > Sample DICOM paths:
    > `datasets/dataset_#/study_#/series_#/*.dcm` or `dataset_#/study_#/series_#/*.dcm` or `study_#/series_#/*.dcm` or `series_#/*.dcm`
2. Save path
    > Folder to save [Inference NPZ (Numpy Zipped Archive)](#inference-npz-numpy-zipped-archive) 
3. Model path
    > Path to model 3 checkpoint file (.ckpt)

### Method 1 - Command-line
    python infer.py --data_path path_to_dicoms  --output_dir output_path --checkpoint path_to_phase3_weights.ckpt

### Method 2 - In a python script
    from infer import infer_dicom
    infer_dicom(model_weights=path_to_weights_ckpt, dicom_folder=dicom_directory, output_directory=output_directory)

## Overall Data Flow
1. _Data Manager_: DICOMs uploaded to MD.ai
2. **pre_label.py**: Preliminary lung masks generated using automated lung segmentation code https://doi.org/10.1186/s41747-020-00173-2 at every 10 mm depth
3. **upload.py**: Upload lung masks to MD.ai
4. _Radiologists_: Apply parenchymal (GGO, consolidation) labels to high specificity areas on axial CT slices that contain a lung mask
5. _Radiologists_: Corrects lung masks by adding Normal Lung label (to add to lung mask) or Background label (to remove parts of the lung mask)
6. _Lead Radiologist_: Verifies labels on a dataset-by-dataset basis
7. **download.py**: Labels from completed dataset are downloaded from MD.ai and packaged with corresponding CT into HDF5 data packages, one for each Study_Instance_UID, and each assigned to one of train or test set.
8. **train.py**: A phase 3 model is trained, initialized using phase 2 model weights, and the HDF5 data packages.
9. **evaluate.py**: Evaluates phase 1, 2 and 3 models using the phase 3 HDF5 data packages, respecting 'train' and 'test' assignments.
10. **infer.py**: For deployment of phase 3 model and generate inference files with model predictions

## Repository Structure
### Core Directories
The core directories of the repository are as follows:
- **src**: contains all of the code for downloading labels, training models, and evaluating models
    - architecture: contains files for building the model
    - datasets: contains files for building dataset object for training/evaluation
    - lungmask: contains code from https://github.com/JoHof/lungmask used to initialize lung masks for labelers
    - mdai_tools: contains code that interface with the MD.ai platform (download or upload images, masks, labels, etc.)
    - resources: miscellaneous resource files for visualization, etc.
    - scripts: contains convenience scripts
    - utilities: contains other dependencies
- **model_in**: contains previous trained models or newly trained models to be used for initialization or evaluation
- **data_in**: contains the original DICOM images, arranged as ./data_in/dicom/"dataset"/"study_uid"/"series_uid"/"instance_uids.dcm"
- **.vscode**: contains the launch.json used with Visual Code for running scripts (parameters within)

The following directories will be generated if it does not already exist:
- **data_out**: contains sub-folders which hold the outputs of the various scripts
- **model_out**: contains models saved during the training process

### Core Scripts
The core scripts are:
- **pre_label.py**: infers labels using phase 2 model and automated lung mask **_(not yet updated for phase 3 use!)_**
- **download.py**: downloads labels/radiologist-generated data from MD.ai, packages with CT, and outputs HDF5 data packages
- **upload.py**: uploads labels from HDF5 packages back to MD.ai for visual inspection
- **train.py**: trains phase 3 models
- **evaluate.py**: evaluates phase 1, phase 2, and one or more phase 3 models at the same time against the train, validation, or test sets
- **infer.py**: given a folder containing multiple DICOM studies, will apply phase 3 model for inference, and save into .npz file
    - A convenience script **inference_to_nii.py** can be used to convert the .npz inference file into NIFTI CT.nii.gz and Mask.nii.gz

## Execution
The simplest way to deploy this project/repository is using Visual Code debugging, and running the launch configurations defined in ./.vscode/launch.json. Alternatively, scripts can be executed from command-line using argument-values specified in the launch.json file. The launch configurations can be separated into **Utilities**, **Data** and **Model** debugging/run configurations. 

### Utilities
The following launch.json configurations are utilities and may not be needed to be run:
- Refactor DICOM Directory
    > Used to restructure a folder containing .dcm files into the appropriate directory structure for **data_in** folder (see above)
- Remap UID Directories
    > Used to rename and shorten DICOM directory in instances where filepaths are too long
- Utility: Scrape DICOM series thicknesses
    > Used to scrape the slice thicknesses from DICOM series and append to data HDF5

### Data
The following launch.json configurations are used for data processing in preparation for training
- Phase 3 Data --Phase 2 Model-> Phase 3 Data AI Inference HDF5
    > Uses Phase 2 Model to pre-label the data _(not used in final implementation)_, as well as use automated lung segmentation to generate preliminary lung masks
    > - Input: Phase 3 DICOMs
    > - Output: Pre-label HDF5s, to be used with the **upload.py** script to push onto MD.ai
- Phase 3 AI Inference --Upload-> MD.ai
    > Uploads above inference HDF5s to MD.ai
    > - Input: Pre-label HDF5s
- MD.ai Labels --Download-> Phase 3 All-in-One HDF5
    > Downloads MD.ai labels, combines with **data_in** DICOM data, and applies Hessian Vesselness
    > Output: [Train/Test Data HDF5s](#traintest-data-hdf5)
- Phase 3 Ground Truth --Upload-> MD.ai
    > Uploads the final Train/Test data HDF5 to MD.ai for visual inspection, if desired
    > Input: [Train/Test Data HDF5s](#traintest-data-hdf5)

### Model
The following launch.json configurations are used for training a phase 3 model, infering, and evaluating models. Configuration names are self explanatory.
- Train Phase 3
- Evaluate on Phase 3 Data
- Infer with Phase 3 Model
    > - Outputs compressed [Inference NPZ (Numpy Zipped Archive)](#inference-npz-numpy-zipped-archive)
    > - Optionally, save NIFTI (.nii) files that can be visualized using a NIFTI viewer such as MRIcroGL in the same output directory

## Data Structures
### Train/Test Data HDF5
Training data are stored in HDF5 files. Each individual HDF5 contains the volumetric data for the CT image and labels. For a study with N axial slices:
- HDF5_file.hdf5
    - `<KeysViewHDF5 ['HDF5_file']>`
        - `<HDF5 dataset "CT": shape (N, 512, 512), type "<i4">`
            > Contains the CT data
        - `<HDF5 dataset "DataUse": shape (1,), type "|S75">`
            > Byte string specifying intended data use (train vs. test)
        - `<HDF5 dataset "Parenchyma": shape (N, 3, 512, 512), type "<f8">`
            > Contains parenchymal masks for each label group
            > - Dim-0: Slice Number (n)
            > - Dim-1: Label Masks (l)
            >    - [n, 0, ...] = Normal Lung
            >    - [n, 1, ...] = GGO
            >    - [n, 2, ...] = Consolidation
            > - Values: 
            >    - [n, l, ...] == -1 are Unlabeled
            >    - [n, l, ...] == 0 are Negative
            >    - [n, l, ...] == 1 are Positive
        - `<HDF5 dataset "Vesselness": shape (N, 512, 512), type "<f4">`
            > Contains the hessian vesselness measure at each voxel
        - `<HDF5 dataset "SOPInstanceUIDs": shape (N,), type "|S75">`
            > A list of the SOPInstanceUIDs for each axial slice
        - `<HDF5 dataset "Bone": shape (N, 512, 512), type "<f8">` _(not used. included due to its existence in MD.ai)_
        - `<HDF5 dataset "Effusion": shape (N, 512, 512), type "<f8">` _(not used. included due to its existence in MD.ai)_
        - `<HDF5 dataset "SOPLabels": shape (N, 1), type "|S75">` _(not used. included due to its existence in MD.ai)_

For example, to access the GGO labels for slice 15:
    
    import h5py

    file = h5py.File('HDF5_file.hdf5','r')
    ggo = file[HDF5_file]['Parenchyma'][15,1,...]

### Inference NPZ (Numpy Zipped Archive)
Inferences output by **infer.py** are compressed numpy files containing the following keys:
- in_: The processed CT volume (N, 512, 512)
- out_: The inferred labels (N, 512, 512)
- sops: The SOPInstanceUIDs for each slice n of N (N,)
- labels: Label names for values in _out_

To load an inference (ex. inference.npz):

    import numpy as np

    inferred = np.load('inference.npz')
    CT = inferred[in_]
    probs = inferred[probs]
    outputs = inferred[out_]
    SOPInstanceUIDs = inferred[sops]
    class_labels = dict(enumerate(inferred['labels']))

To convert a .npz inference file into separate NIFTI files (CT, Mask), the **inference_to_nii.py** script can be used.
> Accepts file path to inference.npz
> Outputs CT.nii.gz and Mask.nii.gz

    python3 inference_to_nii.py --inference_fp inference.npz
    python3 inference_to_nii.py --inference_fp inference.npz --output_dir output_folder

## Contributors
The key contributor to this repository is Marco Law. The COVID-19 L3-Net Phase 3 is based upon work from Phase 1 and 2, contributors being Keegan Lensink, Issam Laradji, and Marco Law.

## License
This project is distributed under the [Apache License 2.0](LICENSE)

