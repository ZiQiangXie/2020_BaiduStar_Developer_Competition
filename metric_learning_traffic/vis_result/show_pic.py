from PIL import Image, ImageFont, ImageDraw
import colorsys
import numpy as np
import os
import json


def show_label_dir(view_path = '65217-64937_top_view_result_folder/' , image_dir='/home/sd/Downloads/star_baidu/test/pic/' ):
    match_result_file = os.listdir(view_path)
    match_result_file.sort(key=lambda x: int(x[:-5]))
    # print(match_result_file)
    detect_class_list = ['limit speed 20','limit speed 30','limit speed 40','limit speed 50','limit speed 60',\
                         'limit speed 70','limit speed 80','limit speed 90','limit speed 100','limit speed 110',\
                         'limit speed 120','ban turn left','ban turn right','ban straight','ban leftAright',\
                         'ban leftAstraight','ban straightAright','ban turn back','electronic eye']
    class_names = ['102', '103', '104', '105', '106', '107', '108', '109', '110', '111', '112', '201', '202', '203', '204', '205',
     '206', '207', '301']
    hsv_tuples = [(x / len(class_names), 1., 1.)
                  for x in range(len(class_names))]
    colors_ = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    i = 0;
    for colar in colors_:
        color_1 = tuple(map(lambda x: int(x * 255), colar))
        colors_[i] = color_1
        i += 1

    for i_file, img_json in enumerate(match_result_file):
        match_result = json.load(open(os.path.join(view_path,img_json)))
        match_result_groups = match_result['group']
        match_result_signs = match_result['signs']
        keep_pic = False
        for sign_i, sign in enumerate(match_result_signs):
            if keep_pic == False:
                img_path = os.path.join(image_dir,str(sign['pic_id']) + '.jpg')
                image = Image.open(img_path)
                draw = ImageDraw.Draw(image)
                font = ImageFont.truetype(font='font/FiraMono-Medium.otf',
                                          size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
                thickness = (image.size[0] + image.size[1]) // 800  # 300

            left, top, right, bottom, class_name_id = (sign['x'],sign['y'],sign['x'] + sign['w'],sign['y'] + sign['h'],sign['type'])
            for i,cla_nam in enumerate(class_names):
                if class_names[i] == class_name_id:
                    class_name_id = i
                    break
            label = detect_class_list[class_name_id] + '---' + sign['sign_id']
            print(label, (left, top), (right, bottom))
            label_size = draw.textsize(label, font)
            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])
            for i in range(thickness):
                draw.rectangle(
                    [int(left + i), int(top + i), int(right - i), int(bottom - i)],
                    outline=colors_[class_name_id])
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=colors_[class_name_id])
            draw.text(text_origin, label, fill=(0, 0, 0), font=font)
            image.show()
            if sign_i + 1 >= len(match_result_signs):
                continue
            elif match_result_signs[sign_i + 1]['pic_id'] == sign['pic_id']:
                keep_pic = True
                continue
            else:
                keep_pic = False
                del draw
            print("\nshow label : success")

def show_label_sigle(view_json = 'result_52000_64/11108213068.json' , image_dir='/home/sd/Downloads/star_baidu/test/pic/' ):
    detect_class_list = ['limit speed 20','limit speed 30','limit speed 40','limit speed 50','limit speed 60',\
                         'limit speed 70','limit speed 80','limit speed 90','limit speed 100','limit speed 110',\
                         'limit speed 120','ban turn left','ban turn right','ban straight','ban leftAright',\
                         'ban leftAstraight','ban straightAright','ban turn back','electronic eye']
    class_names = ['102', '103', '104', '105', '106', '107', '108', '109', '110', '111', '112', '201', '202', '203', '204', '205',
     '206', '207', '301']
    hsv_tuples = [(x / len(class_names), 1., 1.)
                  for x in range(len(class_names))]
    colors_ = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    i = 0;
    for colar in colors_:
        color_1 = tuple(map(lambda x: int(x * 255), colar))
        colors_[i] = color_1
        i += 1

    match_result = json.load(open(view_json))
    match_result_signs = match_result['signs']
    keep_pic = False
    for sign_i, sign in enumerate(match_result_signs):
        if keep_pic == False:
            img_path = os.path.join(image_dir,str(sign['pic_id']) + '.jpg')
            image = Image.open(img_path)
            draw = ImageDraw.Draw(image)
            font = ImageFont.truetype(font='font/FiraMono-Medium.otf',
                                      size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
            thickness = (image.size[0] + image.size[1]) // 800  # 300

        left, top, right, bottom, class_name_id = (sign['x'],sign['y'],sign['x'] + sign['w'],sign['y'] + sign['h'],sign['type'])
        for i,cla_nam in enumerate(class_names):
            if class_names[i] == class_name_id:
                class_name_id = i
                break
        label = detect_class_list[class_name_id] + '---' + sign['sign_id']
        print(label, (left, top), (right, bottom))
        label_size = draw.textsize(label, font)
        if top - label_size[1] >= 0:
            text_origin = np.array([left, top - label_size[1]])
        else:
            text_origin = np.array([left, top + 1])
        for i in range(thickness):
            draw.rectangle(
                [int(left + i), int(top + i), int(right - i), int(bottom - i)],
                outline=colors_[class_name_id])
        draw.rectangle(
            [tuple(text_origin), tuple(text_origin + label_size)],
            fill=colors_[class_name_id])
        draw.text(text_origin, label, fill=(0, 0, 0), font=font)
        image.show()
        if sign_i + 1 >= len(match_result_signs):
            continue
        elif match_result_signs[sign_i + 1]['pic_id'] == sign['pic_id']:
            keep_pic = True
            continue
        else:
            keep_pic = False
            del draw
        print("\nshow label : success")


if __name__ == '__main__':
    image_dir = '/home/sd/Downloads/star_baidu/test/pic/'        #   测试集图片文件夹
    view_path = '0.65217/' # 65217-64937_top_view_result_folder  #   匹配结果文件夹
    view_json = '0.65217/14108000518.json'                       #   单个匹配结果文件

    show_label_sigle(view_json, image_dir)      # 可视化 单个匹配文件
    # show_label_dir(view_path,image_dir)       # 可视化 整个结果匹配文件夹 需结合断点使用

'''
我的文件组织：
---font
---show_pic.py
---0.65217
         ---1*.json
         ---2*.json
         ---3*.json
         ...

'''