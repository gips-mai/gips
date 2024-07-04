import os
import pandas as pd
from tqdm import tqdm

def organize_raw_descr(path_dir='data/raw_descr', output_dir='data/comb_descr'):

    # Sort the files in the directory after split, e.g. '00', '01', '02', ...
    sorted_file_names = {}
    for file_name in tqdm(os.listdir(path_dir), desc='Sorting file names'):
        if file_name.endswith(".csv"):
            splitted_name = file_name.split('_')
            data_split = splitted_name[0]
            if data_split not in sorted_file_names:
                sorted_file_names[data_split] = [file_name]
            else:
                sorted_file_names[data_split].append(file_name)

    # Build combined dataframes for each split
    comb_splits = {}
    for split, file_names in tqdm(sorted_file_names.items(), desc='Building combined dataframes'):
        combined_df = pd.concat([pd.read_csv(os.path.join(path_dir, file)) for file in file_names],
                                ignore_index=True)
        comb_splits[split] = combined_df

    # For each split, reorganize the data into a dataframe with a single row per image
    for split in tqdm(sorted_file_names.keys(), desc='Reorganizing data'):
        extracted_data = []
        for col in comb_splits[split].columns:
            for item in comb_splits[split][col].dropna():
                nested_dict = eval(item)  # Convert the string into a dictionary
                extracted_data.append((nested_dict['image_file'], nested_dict['response']))
        extracted_df = pd.DataFrame(extracted_data, columns=['image_file', 'response'])
        extracted_df.to_csv(os.path.join(output_dir, f'comb_descr_{split}.csv'), index=False)

organize_raw_descr()
