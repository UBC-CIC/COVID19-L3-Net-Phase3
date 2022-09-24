import mdai
import glob
import os
import pickle
import numpy as np

class MDAIAnnotations:

    def __init__(self, download_folder, project_id, dataset_id, access_token, force_download):
        self.project_id = project_id
        self.dataset_id = dataset_id
        self.access_token = access_token
        self.force_download = force_download

        pkl_file = os.path.join(download_folder, 'MDAIAnnotations_{}_{}.pkl'.format(project_id, dataset_id))
        
        if os.path.isfile(pkl_file) and not self.force_download:
            print("Existing PICKLE File Found: {}".format(pkl_file))
            pkl_exists = True
            with open(pkl_file,'rb') as file:
                data = pickle.load(file)
        else:
            pkl_exists = False
            # download json if required
            self.json_file_path = self._fetch_json(download_folder)
            # convert json to dataframe and sort data if saved dataframe not provided
            data = mdai.common_utils.json_to_dataframe(self.json_file_path)

        self.annotations = data['annotations']
        self.studies = data['studies']
        self.test_studies = data['test_studies'] if 'test_studies' in data.keys() else list(self.annotations[self.annotations['labelName'] == 'Test Set']['StudyInstanceUID'].unique())
        self.labels = data['labels']

        # clean annotations dataframe
        self._clean_annotations()
        self.train_studies = data['train_studies'] if 'train_studies' in data.keys() else [study for study in self.studies['StudyInstanceUID'].unique() if study not in self.test_studies]
        
        if not pkl_exists:
            self.dump_pkl(pkl_file)

        self.pkl_file = pkl_file

    def dump_pkl(self, pkl_file):
        data = {'annotations': self.annotations, 'studies': self.studies, 'labels': self.labels, 'test_studies': self.test_studies, 'train_studies': self.train_studies}
        with open(pkl_file,'wb') as file:
            pickle.dump(data, file)

    def _clean_annotations(self):
        # pull list of exams/studies that have COMPLETE label to make sure that the 'claimed series' is also complete if detected
        completed_studiesuid = self.annotations[self.annotations['labelName'] == 'COMPLETE']['StudyInstanceUID']

        isolate_by = 'claimed'  # rad or claimed
        if isolate_by == 'rad':
            # isolate annotated series << FLAWED DUE TO NO FAIL-SAFE LOGIC
            labelers = list(self.annotations.createdById.unique())                                                                                   # build list of all users 
            labelers.remove('U_yokr9M') if 'U_yokr9M' in labelers else None                                                                          # exclude non-radiologist (Marco - U_yokr9M)
            labelers.remove('U_qgvnge') if 'U_qgvnge' in labelers else None                                                                          # exclude non-radiologist (Brian - U_qgvnge)
            rad_labeled = self.annotations[np.bitwise_and(self.annotations['createdById'].isin(labelers), self.annotations['scope'] == 'INSTANCE')]  # keep INSTANCE (image) -level annotations
            claimed_seriesuid = rad_labeled['SeriesInstanceUID'].unique()                                                                            # get unique SeriesInstanceUIDs that have been labeled by a rad
       
        if isolate_by == 'claimed':
            # PREFERRED: ideal isolation method, but requires review of all datasets and confirmed appropriate 'Claimed Series' tag
            claimed_seriesuid = self.annotations[self.annotations['labelName'] == 'Claimed Series']['SeriesInstanceUID']
        
        self.annotations = self.annotations[self.annotations['SeriesInstanceUID'].isin(claimed_seriesuid)]                                          # filter to keep only Series that have been claimed
        print("{:d} Exams w/ Claimed Series --> ".format(len(self.annotations['StudyInstanceUID'].unique())), end='')
        self.annotations = self.annotations[self.annotations['StudyInstanceUID'].isin(completed_studiesuid)]                                        # filter to keep only Series whose Study is COMPLETE
        print("{:d} COMPLETED Exams w/ Claimed Series".format(len(self.annotations['StudyInstanceUID'].unique())))
        self.studies = self.studies[self.studies['StudyInstanceUID'].isin(self.annotations['StudyInstanceUID'])]                                    # update study list based on kept annotations

    def _fetch_json(self, json_folder):
        json_file_path = glob.glob(os.path.join(json_folder, '*_{}_*_{}_*'.format(self.project_id, self.dataset_id)))
        json_file_path.sort()
        if len(json_file_path) > 0 and not self.force_download:
            print("Existing JSON File Found: {}".format(json_file_path[-1]))
            return json_file_path[-1]
        else:
            print("No existing JSON File for ProjectID & DatasetID. Will download.")
            mdai_client = mdai.Client(domain='vgh.md.ai', access_token=self.access_token)
            mdai_client.project(self.project_id, dataset_id=self.dataset_id, path=json_folder, annotations_only=True)
            json_file_path = glob.glob(os.path.join(json_folder, '*_{}_*_{}_*'.format(self.project_id, self.dataset_id)))
            json_file_path.sort()
            print("Using Downloaded JSON File: {}".format(json_file_path[-1]))
            return json_file_path[-1]