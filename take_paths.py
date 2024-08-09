import os
import csv


def find_mfcc_csv(root_dir):
    paths = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if 'mfcc.csv' in filenames:
            paths.append(dirpath)

    with open('ESC_not_drones.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Directory Path'])
        for path in paths:
            writer.writerow([path])


# Replace 'your_root_directory' with the path to the root directory you want to search
find_mfcc_csv('../../FINISHED_V7/esc_data')
