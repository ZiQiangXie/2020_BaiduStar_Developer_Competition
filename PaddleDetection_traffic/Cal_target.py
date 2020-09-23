import os
import json
import glob

data_path = '../../data_0/traffic_data/base_data/tag/train'

josn_list = os.listdir(data_path)
print('Toal josn files: %d' % len(josn_list))
class_num_dict = {}
# 0~10, 10~20, 20~30, 30~40, 40~50, 50~60, 60~70, 70～80, 80~90, 90~
size_num_list = [0] * 10

for i, file_name in enumerate(josn_list):

    file_path = os.path.join(data_path, file_name)
    anno = json.load(open(file_path))
    signs = anno['signs']

    for i, sign in enumerate(signs):
        obj_type = sign['type']
        if obj_type not in class_num_dict.keys():
            class_num_dict.update({obj_type: 0})
        class_num_dict[obj_type] += 1

        h = sign['h']
        w = sign['w']
        if h > 90 and w > 90:
            size_num_list[9] += 1
        elif h > 80 and w > 80:
            size_num_list[8] += 1
        elif h > 70 and w > 70:
            size_num_list[7] += 1
        elif h > 60 and w > 60:
            size_num_list[6] += 1
        elif h > 50 and w > 50:
            size_num_list[5] += 1
        elif h > 40 and w > 40:
            size_num_list[4] += 1
        elif h > 30 and w > 30:
            size_num_list[3] += 1
        elif h > 20 and w > 20:
            size_num_list[2] += 1
        elif h > 10 and w > 10:
            size_num_list[1] += 1
        else:
            size_num_list[0] += 1

print('class num:')
for key, value in class_num_dict.items():
    print("'%s': %d" % (key, value))
print('\n')
print('size num:')
for i, num in enumerate(size_num_list):
    print('%d~%d:%d' % (i*10, (i+1)*10, num))



