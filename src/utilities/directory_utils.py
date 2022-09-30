import os
import pandas as pd
import argparse


class UIDMapper:

    def __init__(self, folder_path) -> None:
        self.folder_path = folder_path
        if os.path.exists(os.path.join(folder_path,'lookup.csv')):
            self.lookup_data = pd.read_csv(os.path.join(folder_path,'lookup.csv'),index_col=[0],header=0)  
        else:
            self.lookup_data = None

    def lookup(self, uid):
        return self.lookup_data.loc[uid].values[0] if self.lookup_data is not None else uid

    def uid_to_generic(self):
        folder_path = self.folder_path
        if not os.path.exists(os.path.join(folder_path,'lookup.csv')):
            uids = [item for item in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, item))]
            lut_studies = {}

            # create an index
            for idx, uid in enumerate(uids):
                lut_studies[uid] = ['study{:04d}'.format(idx)]

            lut_uids = pd.DataFrame.from_dict(lut_studies,orient='index',columns=['folder_name'])
            lut_uids.index.rename('UID', inplace=True)
            lut_uids.to_csv(os.path.join(folder_path,'lookup.csv'))

            # rename based on new index
            for uid, folder_name in lut_uids.iterrows():
                original_path = os.path.join(folder_path,uid)
                new_path = os.path.join(folder_path,folder_name.values[0])
                os.rename(original_path, new_path)

            print("Study folders truncated! See lookup.csv for new mappings")
        else:
            print("Study folder has already been truncated. No changes have been made.")

    def generic_to_uid(self):
        folder_path = self.folder_path
        if self.lookup_data is not None:
            # rename based on new index
            for uid, folder_name in self.lookup_data.iterrows():
                original_path = os.path.join(folder_path,folder_name.values[0])
                new_path = os.path.join(folder_path,uid)
                if os.path.isdir(original_path):
                    os.rename(original_path, new_path)
                else:
                    print("{} not found. Skipping.".format(original_path))

            os.remove(os.path.join(folder_path,'lookup.csv'))
            print("Folders returned to original UID names. lookup.csv removed.")
        else:
            print("No lookup.csv exists. No changes will be made")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder_path", type=str, help="")
    parser.add_argument("--mode", type=str, help="encode: uid_to_generic. decode: generic_to_uid", default="encode")
    args = parser.parse_args()
    mapper = UIDMapper(args.folder_path)
    if args.mode == 'encode':
        mapper.uid_to_generic()
    if args.mode == 'decode':
        mapper.generic_to_uid()