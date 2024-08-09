import os

def delete_mfcc_files_in_noise_folders(path):
    for root, dirs, files in os.walk(path, topdown=False):
        for name in dirs:
            if name.endswith('_noise'):
                folder_path = os.path.join(root, name)
                mfcc_file_path = os.path.join(folder_path, 'mfcc.csv')
                if os.path.isfile(mfcc_file_path):
                    os.remove(mfcc_file_path)
                    print(f"Deleted: {mfcc_file_path}")

# Replace 'your_path_here' with the path you want to scan
delete_mfcc_files_in_noise_folders('../FINISHED_V6')
