import os
import shutil
import random
train_tag_json = '../../data_0/traffic_data2/tag/train'
val_tag_json = "../../data_0/traffic_data2/tag/val/"
json_file = os.listdir(train_tag_json)
for files in json_file:
    json_path = os.path.join(train_tag_json, files)
    n = random.randint(1,20)
    if n== 2:
        shutil.copy(json_path, val_tag_json + files)

test_input_all_json = '../../data_0/traffic_data2/train/input'
val_input_path = "../../data_0/traffic_data2/val/input/"
json_file = os.listdir(val_tag_json)
for files in json_file:
    json_path = os.path.join(test_input_all_json, files)

    shutil.copy(json_path, val_input_path + files)
