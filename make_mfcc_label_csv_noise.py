import os
import shutil
import pandas as pd


def process_directories(base_path, output_file):
    with open(output_file, 'w') as out_file:
        for root, dirs, files in os.walk(base_path):
            if root.endswith('_noise'):  # Process only directories ending with '_noise'
                label_path = os.path.join(root, 'label.csv')

                if os.path.isfile(label_path):
                    label_df = pd.read_csv(label_path, header=None)

                    if not label_df.empty and len(label_df.columns) > 0:
                        label = label_df.iloc[0, 0]
                        if label == 'drone':
                            label_value = 1
                        elif label == 'not_drone':
                            label_value = 0
                        else:
                            print(f"Warning: Not eligible label found in {label_path}")
                            label_value = 'not eligible'

                        for file in files:
                            if file.startswith('mfcc_'):
                                mfcc_path = os.path.join(root, file)
                                # Check if MFCC file exists
                                if os.path.isfile(mfcc_path):
                                    # Load the MFCC file
                                    mfcc_df = pd.read_csv(mfcc_path, header=None)

                                    # Check if the shape of the rows is 41
                                    if mfcc_df.shape[0] == 41:
                                        # Delete the first row
                                        mfcc_df = mfcc_df.drop(0)

                                        # Save the modified MFCC file back to the same path
                                        mfcc_df.to_csv(mfcc_path, index=False, header=False)

                                    out_file.write(f"{mfcc_path},{label_value}\n")
                    else:
                        print(f"Warning: Label column not found or empty in {label_path}")
                else:
                    print(f"Warning: Missing label.csv in {root}")
                    # Delete the folder if label.csv is missing
                    shutil.rmtree(root)
                    print(f"Folder deleted: {root}")


base_path = '../FINISHED_V6'
output_file = 'mfccs_labels.csv'
print("start")
process_directories(base_path, output_file)
