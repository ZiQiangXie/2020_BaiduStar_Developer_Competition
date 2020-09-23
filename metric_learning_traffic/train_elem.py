#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import math
import time
import logging
import argparse
import functools
import threading
import subprocess
import numpy as np
import paddle
import paddle.fluid as fluid
import models
import reader
from losses import SoftmaxLoss
from losses import ArcMarginLoss
from utility import add_arguments, print_arguments, load_pretrain
from utility import fmt_time, recall_topk, get_gpu_num, check_cuda

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
# yapf: disable
add_arg('model', str, "ResNet50", "Set the network to use.")
add_arg('embedding_size', int, 0, "Embedding size.")
add_arg('train_batch_size', int, 256, "Minibatch size.")
add_arg('image_shape', str, "3,64,64", "input image size")
add_arg('class_dim', int, 9993, "Class number.")
add_arg('lr', float, 0.01, "set learning rate.")
add_arg('lr_strategy', str, "piecewise_decay", "Set the learning rate decay strategy.")
add_arg('lr_steps', str, "9000,15000", "step of lr")
add_arg('total_iter_num', int, 18000, "total_iter_num")
add_arg('display_iter_step', int, 20, "display_iter_step.")
add_arg('save_iter_step', int, 1000, "save_iter_step.")
add_arg('use_gpu', bool, True, "Whether to use GPU or not.")
add_arg('pretrained_model', str, None, "Whether to use pretrained model.")
add_arg('checkpoint', str, None, "Whether to resume checkpoint.")
add_arg('model_save_dir', str, "output_elem", "model save directory")
add_arg('loss_name', str, "softmax", "Set the loss type to use.")
add_arg('arc_scale', float, 80.0, "arc scale.")
add_arg('arc_margin', float, 0.15, "arc margin.")
add_arg('arc_easy_margin', bool, False, "arc easy margin.")
add_arg('data_path', str, "../../data/traffic_data/train", "path of training data ")
# yapf: enable

model_list = [m for m in dir(models) if "__" not in m]


def optimizer_setting(params):
    ls = params["learning_strategy"]
    assert ls["name"] == "piecewise_decay", \
           "learning rate strategy must be {}, but got {}".format("piecewise_decay", lr["name"])

    bd = [int(e) for e in ls["lr_steps"].split(',')]
    base_lr = params["lr"]
    lr = [base_lr * (0.1**i) for i in range(len(bd) + 1)]
    optimizer = fluid.optimizer.Momentum(
        learning_rate=fluid.layers.piecewise_decay(
            boundaries=bd, values=lr),
        momentum=0.9,
        regularization=fluid.regularizer.L2Decay(1e-4))
    return optimizer


def net_config(image, label, model, args):
    assert args.model in model_list, "{} is not in lists: {}".format(args.model,
                                                                     model_list)

    out = model.net(input=image, embedding_size=args.embedding_size)

    if args.loss_name == "softmax":
        metricloss = SoftmaxLoss(class_dim=args.class_dim, )
    elif args.loss_name == "arcmargin":
        metricloss = ArcMarginLoss(
            class_dim=args.class_dim,
            margin=args.arc_margin,
            scale=args.arc_scale,
            easy_margin=args.arc_easy_margin, )
    cost, logit = metricloss.loss(out, label)
    avg_cost = fluid.layers.mean(x=cost)
    acc_top1 = fluid.layers.accuracy(input=logit, label=label, k=1)
    acc_top5 = fluid.layers.accuracy(input=logit, label=label, k=5)
    return avg_cost, acc_top1, acc_top5, out


def build_program(main_prog, startup_prog, args):
    image_shape = [int(m) for m in args.image_shape.split(",")]
    model = models.__dict__[args.model]()
    with fluid.program_guard(main_prog, startup_prog):
        queue_capacity = 64
        image = fluid.data(
            name='image', shape=[None] + image_shape, dtype='float32')
        label = fluid.data(name='label', shape=[None, 1], dtype='int64')
        loader = fluid.io.DataLoader.from_generator(
            feed_list=[image, label],
            capacity=queue_capacity,
            use_double_buffer=True,
            iterable=True)

        with fluid.unique_name.guard():
            avg_cost, acc_top1, acc_top5, out = net_config(image, label, model,
                                                           args)
            params = model.params
            params["lr"] = args.lr
            params["learning_strategy"]["lr_steps"] = args.lr_steps
            params["learning_strategy"]["name"] = args.lr_strategy
            optimizer = optimizer_setting(params)
            optimizer.minimize(avg_cost)
            global_lr = optimizer._global_learning_rate()
    return loader, avg_cost, acc_top1, acc_top5, global_lr


def train_async(args):
    # parameters from arguments

    logging.debug('enter train')
    model_name = args.model
    checkpoint = args.checkpoint
    pretrained_model = args.pretrained_model
    model_save_dir = args.model_save_dir

    startup_prog = fluid.Program()
    train_prog = fluid.Program()
    tmp_prog = fluid.Program()

    train_loader, train_cost, train_acc1, train_acc5, global_lr = build_program(
        main_prog=train_prog, startup_prog=startup_prog, args=args)

    train_fetch_list = [
        global_lr.name, train_cost.name, train_acc1.name, train_acc5.name
    ]

    place = fluid.CUDAPlace(0) if args.use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)
    num_trainers = int(os.environ.get('PADDLE_TRAINERS_NUM', 1))
    if num_trainers <= 1 and args.use_gpu:
        places = fluid.framework.cuda_places()
    else:
        places = place
    exe.run(startup_prog)

    logging.debug('after run startup program')

    if checkpoint is not None:
        fluid.load(program=train_prog, model_path=checkpoint, executor=exe)

    if pretrained_model:
        load_pretrain(train_prog, pretrained_model)

    if args.use_gpu:
        devicenum = get_gpu_num()
    else:
        devicenum = 1 
    assert (args.train_batch_size % devicenum) == 0
    train_batch_size = args.train_batch_size // devicenum

    train_loader.set_sample_generator(
        reader.train(args),
        batch_size=train_batch_size,
        drop_last=True,
        places=places)

    train_exe = fluid.ParallelExecutor(
        main_program=train_prog,
        use_cuda=args.use_gpu,
        loss_name=train_cost.name)

    totalruntime = 0
    iter_no = 0
    train_info = [0, 0, 0, 0]
    while iter_no <= args.total_iter_num:
        for train_batch in train_loader():
            t1 = time.time()
            lr, loss, acc1, acc5 = train_exe.run(feed=train_batch,
                                                 fetch_list=train_fetch_list)
            t2 = time.time()
            period = t2 - t1
            lr = np.mean(np.array(lr))
            train_info[0] += np.mean(np.array(loss))
            train_info[1] += np.mean(np.array(acc1))
            train_info[2] += np.mean(np.array(acc5))
            train_info[3] += 1
            if iter_no % args.display_iter_step == 0:
                avgruntime = totalruntime / args.display_iter_step
                avg_loss = train_info[0] / train_info[3]
                avg_acc1 = train_info[1] / train_info[3]
                avg_acc5 = train_info[2] / train_info[3]
                print("[%s] trainbatch %d, lr %.6f, loss %.6f, "\
                    "acc1 %.4f, acc5 %.4f, time %2.2f sec" % \
                    (fmt_time(), iter_no, lr, avg_loss, avg_acc1, avg_acc5, avgruntime))
                sys.stdout.flush()
                totalruntime = 0
            if iter_no % args.display_iter_step == 0:
                train_info = [0, 0, 0, 0]

            totalruntime += period

            if iter_no % args.save_iter_step == 0 and iter_no != 0:
                model_path = os.path.join(model_save_dir + '/' + model_name,
                                          str(iter_no))
                if not os.path.isdir(model_path):
                    os.makedirs(model_path)
                fluid.save(program=train_prog, model_path=model_path)

            iter_no += 1


def initlogging():
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    loglevel = logging.DEBUG
    logging.basicConfig(
        level=loglevel,
        # logger.BASIC_FORMAT,
        format="%(levelname)s:%(filename)s[%(lineno)s] %(name)s:%(funcName)s->%(message)s",
        datefmt='%a, %d %b %Y %H:%M:%S')


def main():
    args = parser.parse_args()
    print_arguments(args)
    check_cuda(args.use_gpu)
    train_async(args)


if __name__ == '__main__':
    main()
