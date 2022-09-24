from ast import arg
import pydicom as dcm
import os
import shutil
from tqdm import tqdm
import argparse

study_folder = 'C:\\Users\\Marco\\PycharmProjects\\cic_covid19_phase3\\data_in\\SPH\\'
output_path = 'C:\\Users\\Marco\\PycharmProjects\\cic_covid19_phase3\\data_in\\refactoredSPH'


def refactor_dcm_data_directory(study_folder, output_path=None, silent=False):
    dcm_files = []
    for root, subdirs, files in os.walk(study_folder):
        if len(files) > 0:
            dcm_files = dcm_files + [os.path.join(root,file) for file in files]

    for source in tqdm(dcm_files):
        dicom = dcm.read_file(source)

        study_instance_uid = dicom.StudyInstanceUID
        series_instance_uid = dicom.SeriesInstanceUID
        sop_instance_uid = dicom.SOPInstanceUID
        refactor_path = os.path.join(output_path, study_instance_uid, series_instance_uid)
        os.makedirs(refactor_path, exist_ok=True)

        assert os.path.isdir(refactor_path)
        try:
            destination = os.path.join(refactor_path, sop_instance_uid + '.dcm')
            print("{} > {}".format(source, destination)) if not silent else None
            shutil.copy(source, destination)
        except:
            destination = os.path.join("\\\\?\\" + refactor_path, sop_instance_uid + '.dcm')
            print("{} > {}".format(source, destination)) if not silent else None
            shutil.copy(source, destination)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dicom_root', type=str)
    parser.add_argument('--output_path', type=str)
    args = parser.parse_args()
    refactor_dcm_data_directory(args.dicom_root, output_path=args.output_path, silent=True)

