import os
import shutil

def create_noise_folders_and_copy_files(root_dir):
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for dirname in dirnames:
            new_dir = os.path.join(dirpath, dirname + "_noise")
            os.makedirs(new_dir, exist_ok=True)
            print(f"Created: {new_dir}")

        for filename in filenames:
            source_file = os.path.join(dirpath, filename)
            # Ensure the destination directory exists
            destination_dir = dirpath + "_noise"
            os.makedirs(destination_dir, exist_ok=True)
            destination_file = os.path.join(destination_dir, filename)
            shutil.copy2(source_file, destination_file)
            print(f"Copied: {source_file} to {destination_file}")

# Set your root directory here
root_directory = '../FINISHED_V6'

create_noise_folders_and_copy_files(root_directory)
