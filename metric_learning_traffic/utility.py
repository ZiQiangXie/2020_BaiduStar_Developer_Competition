#  Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserve.
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import six
import time
import subprocess
import distutils.util
import numpy as np
import sys
import paddle.fluid as fluid
from paddle.fluid import core
import multiprocessing as mp
import glob
import json
import cv2


def print_arguments(args):
    """Print argparse's arguments.

    Usage:

    .. code-block:: python

        parser = argparse.ArgumentParser()
        parser.add_argument("name", default="Jonh", type=str, help="User name.")
        args = parser.parse_args()
        print_arguments(args)

    :param args: Input argparse.Namespace for printing.
    :type args: argparse.Namespace
    """
    print("-----------  Configuration Arguments -----------")
    for arg, value in sorted(six.iteritems(vars(args))):
        print("%s: %s" % (arg, value))
    print("------------------------------------------------")


def add_arguments(argname, type, default, help, argparser, **kwargs):
    """Add argparse's argument.

    Usage:

    .. code-block:: python

        parser = argparse.ArgumentParser()
        add_argument("name", str, "Jonh", "User name.", parser)
        args = parser.parse_args()
    """
    type = distutils.util.strtobool if type == bool else type
    argparser.add_argument(
        "--" + argname,
        default=default,
        type=type,
        help=help + ' Default: %(default)s.',
        **kwargs)


def fmt_time():
    """ get formatted time for now
    """
    now_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    return now_str


def recall_topk_ori(fea, lab, k):
    fea = np.array(fea)
    fea = fea.reshape(fea.shape[0], -1)
    n = np.sqrt(np.sum(fea**2, 1)).reshape(-1, 1)
    fea = fea / n
    a = np.sum(fea**2, 1).reshape(-1, 1)
    b = a.T
    ab = np.dot(fea, fea.T)
    d = a + b - 2 * ab
    d = d + np.eye(len(fea)) * 1e8
    sorted_index = np.argsort(d, 1)
    res = 0
    for i in range(len(fea)):
        for j in range(k):
            pred = lab[sorted_index[i][j]]
            if lab[i] == pred:
                res += 1.0
                break
    res = res / len(fea)
    return res


def func(param):
    sharedlist, s, e = param
    fea, a, b = sharedlist
    ab = np.dot(fea[s:e], fea.T)
    d = a[s:e] + b - 2 * ab
    for i in range(e - s):
        d[i][s + i] += 1e8
    sorted_index = np.argsort(d, 1)[:, :10]
    return sorted_index


def recall_topk_parallel(fea, lab, k):
    fea = np.array(fea)
    fea = fea.reshape(fea.shape[0], -1)
    n = np.sqrt(np.sum(fea**2, 1)).reshape(-1, 1)
    fea = fea / n
    a = np.sum(fea**2, 1).reshape(-1, 1)
    b = a.T
    sharedlist = mp.Manager().list()
    sharedlist.append(fea)
    sharedlist.append(a)
    sharedlist.append(b)

    N = 100
    L = fea.shape[0] / N
    params = []
    for i in range(N):
        if i == N - 1:
            s, e = int(i * L), int(fea.shape[0])
        else:
            s, e = int(i * L), int((i + 1) * L)
        params.append([sharedlist, s, e])

    pool = mp.Pool(processes=4)
    sorted_index_list = pool.map(func, params)
    pool.close()
    pool.join()
    sorted_index = np.vstack(sorted_index_list)

    res = 0
    for i in range(len(fea)):
        for j in range(k):
            pred = lab[sorted_index[i][j]]
            if lab[i] == pred:
                res += 1.0
                break
    res = res / len(fea)
    return res


def recall_topk(fea, lab, k=1):
    if fea.shape[0] < 20:
        return recall_topk_ori(fea, lab, k)
    else:
        return recall_topk_parallel(fea, lab, k)


def get_gpu_num():
    visibledevice = os.getenv('CUDA_VISIBLE_DEVICES')
    if visibledevice:
        devicenum = len(visibledevice.split(','))
    else:
        devicenum = subprocess.check_output(
            [str.encode('nvidia-smi'), str.encode('-L')]).decode('utf-8').count(
                '\n')
    return devicenum

def check_cuda(use_cuda, err = \
    "\nYou can not set use_cuda = True in the model because you are using paddlepaddle-cpu.\n \
    Please: 1. Install paddlepaddle-gpu to run your models on GPU or 2. Set use_cuda = False to run models on CPU.\n"
                                                                                                                     ):
    try:
        if use_cuda == True and fluid.is_compiled_with_cuda() == False:
            print(err)
            sys.exit(1)
    except Exception as e:
        pass


def cosine_distance(fea, seq_id, thresh, k):
    fea = fea.reshape(fea.shape[0], -1)
    a = np.linalg.norm(fea, axis=1).reshape(-1, 1)

    d = 1 - np.dot(fea, fea.T) / (a * a.T)
    d = d + np.eye(len(fea)) * 1e8
    sorted_index = np.argsort(d, 1)
    matched_index = []
    cos_index = []
    k = min(len(fea), k)
    for i in range(len(fea)):
        matched_index.append([])
        cos_index.append([])
        matched = 0
        for j in range(k):
            # avoid matched index with same seq_id
            if seq_id[i] == seq_id[sorted_index[i, j]]: continue
            if d[i, sorted_index[i, j]] < thresh:
                matched_index[i].append(sorted_index[i, j])
                cos_index[i].append(d[i, sorted_index[i, j]])
                matched = 1
        if not matched:
            matched_index[i].append(-1)
            cos_index[i].append(-1)
    return matched_index, cos_index


def post_process(results, groups, labels, seq_id, thresh, k):
    group_num = np.max(groups) + 1
    res_group = [[] for i in range(group_num)]
    res_final = [[] for i in range(group_num)]
    cos_final = [[] for i in range(group_num)]
    im_id = 0
    for res, l, s, g in zip(results, labels, seq_id, groups):
        res_group[g].append([res, l, s, im_id])
        im_id += 1

    for group_id, res_pg in enumerate(res_group):
        res_list = []
        seq_list = []
        if len(res_pg) == 0: continue
        for res in res_pg:
            res_list.append(res[0])
            seq_list.append(res[2])
        matched_index, cos_index = cosine_distance(np.array(res_list), seq_list, thresh, k)
        for i, m_idxs in enumerate(matched_index):
            for j, m_i in enumerate(m_idxs):
                if m_i == -1:
                    res_final[group_id].append({i})
                    cos_final[group_id].append(-1)
                    continue
                if {i, m_i} not in res_final[group_id]:
                    res_final[group_id].append({i, m_i})
                    cos_final[group_id].append(cos_index[i][j])

    return res_final, cos_final


def generate_sign_id(anno, filename):
    group = anno['group']
    group_a = group[0]['pic_list']
    group_b = group[1]['pic_list']
    signs = anno['signs']
    id_a, id_b = 1, 1
    for sign in signs:
        if sign['pic_id'] in group_a:
            sign['sign_id'] = 'sign_A_{}'.format(id_a)
            id_a += 1
        elif sign['pic_id'] in group_b:
            sign['sign_id'] = 'sign_B_{}'.format(id_b)
            id_b += 1
        else:
            print('illegal pic id: {} in json file: {}'.format(sign['pic_id'],
                                                               filename))
    return anno


def save_result(res_final, cos_final, output_path, detect_path, test_pic_path = ''):  
    # convert type to interger for evaluation                                       
    convert_list = ['x', 'y', 'w', 'h']
    anno_files = glob.glob(os.path.join(detect_path, '*.json'))
    for i, anno_file in enumerate(anno_files):
        anno = json.load(open(anno_file))
        anno = generate_sign_id(anno, anno_file)
        signs = anno['signs']
        res = res_final[i]
        match_list = []
        for sign in signs:
            sign.pop('score')
            for k in sign.keys():
                if k in convert_list:
                    sign[k] = int(sign[k])
        cos = cos_final[i]

        sign_list = []      
        pic_id = []         
        cos_score = []      
        match_index = []  

        sign_list2 = [] 
        pic_id2 = []
        cos_score2 = []
        match_index2 = []

        index = 0  
        for j, match_pair in enumerate(res):
            match_pair = list(match_pair)
            if 'sign_id' not in signs[match_pair[0]]:
                continue
            if len(match_pair) > 1 and 'sign_id' not in signs[match_pair[1]]:
                continue
            if len(match_pair) > 1 and signs[match_pair[0]]['type'] != signs[match_pair[1]]['type']:  
                continue

            sign_id = signs[match_pair[0]]['sign_id']
            if 'B' in sign_id and len(match_pair) == 1:  
                continue
            # if 'A' in sign_id and len(match_pair) == 1: 
            #    continue
            if 'B' in sign_id and len(match_pair) > 1:     
                match_pair[0], match_pair[1] = match_pair[1], match_pair[0]
                sign_id = signs[match_pair[0]]['sign_id']

            if sign_id not in sign_list:
                sign_list.append(sign_id)
                pic_id.append([])
                cos_score.append([])
                match_index.append([])

            if len(match_pair) > 1 and signs[match_pair[1]]['sign_id'] not in sign_list2:
                sign_list2.append(signs[match_pair[1]]['sign_id'])
                pic_id2.append([])
                cos_score2.append([])
                match_index2.append([])

            match_list.append({'sign_id': sign_id})
            if len(match_pair) > 1:
                match_sign_id = signs[match_pair[1]]['sign_id']
                
                if signs[match_pair[1]]['pic_id'] not in pic_id[sign_list.index(sign_id)]:
                    pic_id[sign_list.index(sign_id)].append(signs[match_pair[1]]['pic_id'])
                    cos_score[sign_list.index(sign_id)].append([])
                    match_index[sign_list.index(sign_id)].append([])

                # pic_id[sign_list.index(sign_id)].append(signs[match_pair[1]]['pic_id'])  #
                pic_index = pic_id[sign_list.index(sign_id)].index(signs[match_pair[1]]['pic_id'])
                cos_score[sign_list.index(sign_id)][pic_index].append(cos[j])
                match_index[sign_list.index(sign_id)][pic_index].append(index)

                if signs[match_pair[0]]['pic_id'] not in pic_id2[sign_list2.index(match_sign_id)]:
                    pic_id2[sign_list2.index(match_sign_id)].append(signs[match_pair[0]]['pic_id'])
                    cos_score2[sign_list2.index(match_sign_id)].append([])
                    match_index2[sign_list2.index(match_sign_id)].append([])

                pic_index2 = pic_id2[sign_list2.index(match_sign_id)].index(signs[match_pair[0]]['pic_id'])
                cos_score2[sign_list2.index(match_sign_id)][pic_index2].append(cos[j])
                match_index2[sign_list2.index(match_sign_id)][pic_index2].append(index)

            else:
                match_sign_id = " "
            index += 1
            match_list[-1].update({'match_sign_id': match_sign_id})
        del_index = []
        repeat_indexs = []
        for p, cos_score_pre in enumerate(cos_score):              
            for q, cos_score_pre2 in enumerate(cos_score_pre):
                if len(cos_score_pre2) <= 1:
                    continue
                if len(cos_score_pre2) == 2:
                    error = abs(cos_score_pre2[0] - cos_score_pre2[1])
                    if error < 0.1:
                        repeat_indexs.append([])
                        repeat_indexs[-1].append(match_index[p][q][0])
                        repeat_indexs[-1].append(match_index[p][q][1])
                        continue
                    cos_min_ind = cos_score_pre2.index(max(cos_score_pre2))
                    del_index.append(match_index[p][q][cos_min_ind])
                if len(cos_score_pre2) > 2:
                    cos_score_pre2_copy = cos_score_pre2[:]
                    cos_score_pre2_copy.sort()
                    for pre2_copy in cos_score_pre2_copy[2:]:
                        ind = cos_score_pre2.index(pre2_copy)
                        del_index.append(match_index[p][q][ind])
                    error = abs(cos_score_pre2_copy[0] - cos_score_pre2_copy[1])
                    if error < 0.1:
                        repeat_indexs.append([])
                        repeat_indexs[-1].append(match_index[p][q][0])
                        repeat_indexs[-1].append(match_index[p][q][1])
                        continue
                    cos_min_ind = cos_score_pre2.index(cos_score_pre2_copy[1])
                    del_index.append(match_index[p][q][cos_min_ind])

        for p, cos_score_pre in enumerate(cos_score2):          
            for q, cos_score_pre2 in enumerate(cos_score_pre):
                if len(cos_score_pre2) <= 1:
                    continue
                if len(cos_score_pre2) == 2:
                    error = abs(cos_score_pre2[0] - cos_score_pre2[1])
                    if error < 0.1:
                        continue
                    cos_min_ind = cos_score_pre2.index(max(cos_score_pre2))
                    if match_index2[p][q][cos_min_ind] not in del_index:
                        del_index.append(match_index2[p][q][cos_min_ind])
                if len(cos_score_pre2) > 2:
                    cos_score_pre2_copy = cos_score_pre2[:]
                    cos_score_pre2_copy.sort()
                    for pre2_copy in cos_score_pre2_copy[2:]:
                        ind = cos_score_pre2.index(pre2_copy)
                        if match_index2[p][q][ind] not in del_index:
                            del_index.append(match_index2[p][q][ind])
                    error = abs(cos_score_pre2_copy[0] - cos_score_pre2_copy[1])
                    if error < 0.1:
                        continue
                    cos_min_ind = cos_score_pre2.index(cos_score_pre2_copy[1])
                    if match_index2[p][q][cos_min_ind] not in del_index:
                        del_index.append(match_index2[p][q][cos_min_ind])
        matched_repeat = []
        for m in range(len(repeat_indexs) - 1):
            sign_id = match_list[repeat_indexs[m][0]]['sign_id']
            match_sign_id = match_list[repeat_indexs[m][0]]['match_sign_id']
            match_sign_id2 = match_list[repeat_indexs[m][1]]['match_sign_id']
            for sign in signs:
                if sign['sign_id'] == sign_id:
                    pic_id = sign['pic_id']
                    x, y = sign['x'], sign['y']
                    type_sign = sign['type']
                if sign['sign_id'] == match_sign_id:
                    match_pic_id = sign['pic_id']
                    match_x, match_y = sign['x'], sign['y']
                if sign['sign_id'] == match_sign_id2:
                    match_pic_id2 = sign['pic_id']
                    match2_x, match2_y = sign['x'], sign['y']

            for n in range(m+1, len(repeat_indexs)):
                sign_id_re = match_list[repeat_indexs[n][0]]['sign_id']
                match_sign_id_re = match_list[repeat_indexs[n][0]]['match_sign_id']
                match_sign_id2_re = match_list[repeat_indexs[n][1]]['match_sign_id']
                for sign in signs:
                    if sign['sign_id'] == sign_id_re:
                        pic_id_re = sign['pic_id']
                        x_re, y_re = sign['x'], sign['y']
                        type_sign_re = sign['type']
                    if sign['sign_id'] == match_sign_id_re:
                        match_pic_id_re = sign['pic_id']
                        match_x_re, match_y_re = sign['x'], sign['y']
                    if sign['sign_id'] == match_sign_id2_re:
                        match_pic_id2_re = sign['pic_id']
                        match2_x_re, match2_y_re = sign['x'], sign['y']
                if pic_id == pic_id_re and match_pic_id == match_pic_id_re and type_sign == type_sign_re:
                    if x > x_re:
                        if match_x > match2_x:
                            del_index.append(repeat_indexs[m][1])
                        else:
                            del_index.append(repeat_indexs[m][0])
                        if match_x_re > match2_x_re:
                            del_index.append(repeat_indexs[n][0])
                        else:
                            del_index.append(repeat_indexs[n][1])
                    else:
                        if match_x > match2_x:
                            del_index.append(repeat_indexs[m][0])
                        else:
                            del_index.append(repeat_indexs[m][1])
                        if match_x_re > match2_x_re:
                            del_index.append(repeat_indexs[n][1])
                        else:
                            del_index.append(repeat_indexs[n][0])
                    break

            pic = os.path.join(test_pic_path, pic_id + '.jpg')
            match_pic = os.path.join(test_pic_path, match_pic_id + '.jpg')
            try:
                image = cv2.imread(pic)
                match_image = cv2.imread(match_pic)
                w = image.shape[1]
                match_w = match_image.shape[1]
            except:
                print('warning: pic', pic, match_pic)
                continue
            if x < w * 0.45:
                if match_x > match_w * 0.6 and repeat_indexs[m][0] not in del_index:
                    del_index.append(repeat_indexs[m][0])
                if match2_x > match_w * 0.6 and repeat_indexs[m][1] not in del_index:
                    del_index.append(repeat_indexs[m][1])
            elif x > w * 0.55:
                if match_x < match_w * 0.4 and repeat_indexs[m][0] not in del_index:
                    del_index.append(repeat_indexs[m][0])
                if match2_x < match_w * 0.4 and repeat_indexs[m][1] not in del_index:
                    del_index.append(repeat_indexs[m][1])

        match_list_new = []
        for k, match_li in enumerate(match_list):
            if k in del_index:
                continue
            match_list_new.append(match_li)

        match_list = match_list_new

        anno.update({'match': match_list})
        file_name = os.path.split(anno_file)[1]
        result_file = os.path.join(output_path, file_name)
        with open(result_file, 'w') as fp:
            json.dump(anno, fp)
        fp.close()


def load_pretrain(prog, pretrained_model):
    state = fluid.io.load_program_state(pretrained_model)
    ignore_set = set()
    # ignore the parameter which mismatch the shape 
    # between the model and pretrain weight.
    all_var_shape = {}
    all_var_name = set()
    for block in prog.blocks:
        for param in block.all_parameters():
            all_var_shape[param.name] = param.shape
    ignore_set.update([
        name for name, shape in all_var_shape.items()
        if name in state and shape != state[name].shape
    ])
    if len(ignore_set) > 0:
        for k in ignore_set:
            if k in state:
                print('warning: variable {} not used'.format(k))
                del state[k]
    for k in state.keys():
        if k not in all_var_shape:
            del state[k]
    fluid.io.set_program_state(prog, state)
