import json
import csv
import os.path
from definitions import label_to_keypoint, connections
import pandas as pd

'''
Parse data from json to csv for DeepLanCut
goal format: https://github.com/DeepLabCut/DeepLabCut/blob/master/examples/Reaching-Mackenzie-2018-08-30/labeled-data/reachingvideo1/CollectedData_Mackenzie.csv
'''

with open(os.path.dirname(__file__) + '/../data/pose_dataset.json') as json_file:
    data = json.load(json_file)

df = pd.json_normalize(data, max_level=2)

# Remove rows of test dataset
remove_test_data = True
if remove_test_data:
    df = df[df.set != 'test']

# Remove rows with empty fields
df.dropna(axis=0,inplace=True)

# Remove the set and dataset columns
df = df.drop(columns=['set', 'dataset'])

# Split keypoint columns into x,y
prev_keypoint_columns = []
for column in df:
    if column == 'path':
        continue
    prev_keypoint_columns.append(column)
    new_column = column[10:]
    df[[column + '.x', column + '.y']] = pd.DataFrame(df[column].tolist(), index = df.index)
# Remove previous keypoint columns
df = df.drop(columns=prev_keypoint_columns)

# Create rows under header
scorer = ['scorer']
scorer.extend(['apic-ai']*(len(df.columns)-1))
bodyparts = ['bodyparts']
coords = ['coords']

for column in df:
    if column == 'path':
        continue
    _, b,c = column.split('.')
    bodyparts.append(b)
    coords.append(c)
top_rows = pd.DataFrame([scorer, bodyparts, coords], index = [2,3,4], columns = df.columns)
df = pd.concat([df.iloc[:0], top_rows, df.iloc[0:]]).reset_index(drop=True)

df.to_csv(os.path.dirname(__file__) + '/../data/pose_dataset_train_cleaned.csv', index=False, encoding='utf-8', header=False)

# coords = ['coords']
# coords.extend(['x', 'y']*((len(df.columns)-1)//2))

# df.loc[-2] = scorer
# df.loc[-1] = coords
# df.index = df.index + 2
# df = df.sort_index()

# df = pd.read_json(json_file)
# df.to_csv()

# f = csv.writer(open(os.path.dirname(__file__) + '/../data/pose_dataset.csv', 'w+'))

# num_keypoints = len(data[0]['keypoints'])
# num_elements_in_row = 1 + 2 * num_keypoints

# # Scorer
# scorer_list = ['scorer']
# scorer_list.extend(['apic-ai']*64)
# f.writerow(scorer_list)

# # Id, full name and coords
# bodyparts_id = ['bodyparts_id']
# full_name = ['bodyparts']
# coords = ['coords']
# for id in range(num_keypoints):
#     print(label_to_keypoint['0'])


# for id in range(num_keypoints):
#     id = str(id)
#     bodyparts_id.extend([id, id])
#     full_name.extend([label_to_keypoint[id]['name'], label_to_keypoint[id]['name']])
#     coords.extend(['x','y'])

# f.writerow(bodyparts_id)
# f.writerow(full_name)
# f.writerow(coords)

# for item in data:
#     row = [item['path']]
#     row.extend([item['keypoints'][str(id)] for id in item['keypoints'].keys()])
#     f.writerow(row)
