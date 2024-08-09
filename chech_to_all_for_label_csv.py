import os

def check_label_csv(root_dir):
    for subdir, dirs, files in os.walk(root_dir):
        if not dirs:  # Check if there are no subdirectories
            if 'label.csv' in files:
                print(f"'label.csv' found in: {subdir}")
            else:
                print(f"No 'label.csv' in: {subdir}")



# Replace 'your_directory_path' with the path to your directory
check_label_csv('../../FINISHED_V6/ours_3')
