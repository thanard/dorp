from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pdb
import argparse
import errno
import itertools
import json
import math
import os
import random
import time
from collections import OrderedDict

import numpy as np
import tensorflow as tf
from tensorflow.python.util import nest

from video_prediction import datasets, models
from video_prediction.utils import ffmpeg_gif, tf_utils


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dirs", "--input_dir", type=str, required=True,
                        help="either a directory containing subdirectories train, val, test, "
                             "etc, or a directory containing all datums")
    parser.add_argument("--dataset", type=str, help="dataset class name", default='NPYDataset')
    parser.add_argument("--dataset_hparams", type=str, help="a string of comma separated list of dataset hyperparameters")
    
    parser.add_argument("--logs_dir", default='', help="ignored if output_dir is specified")
    parser.add_argument("--output_dir", help="output directory where json files, summary, model, gifs, etc are saved. "
                                             "default is logs_dir/model_fname, where model_fname consists of "
                                             "information from model and model_hparams")
    parser.add_argument("--checkpoint", help="directory with checkpoint or checkpoint name (e.g. checkpoint_dir/model-200000)")
    parser.add_argument("--resume", action='store_true', help='resume from lastest checkpoint in output_dir.')

    parser.add_argument("--experiment_dir", type=str, default='', help="folder with all config files")
    parser.add_argument("--model", type=str, help="model class name", default='savp')
    parser.add_argument("--model_hparams", type=str, help="a string of comma separated list of model hyperparameters")
    parser.add_argument("--model_hparams_dict", type=str, help="a json file of model hyperparameters")

    parser.add_argument("--summary_freq", type=int, default=1000, help="save summaries (except for image and eval summaries) every summary_freq steps")
    parser.add_argument("--image_summary_freq", type=int, default=5000, help="save image summaries every image_summary_freq steps")
    parser.add_argument("--eval_summary_freq", type=int, default=0, help="save eval summaries every eval_summary_freq steps")
    parser.add_argument("--progress_freq", type=int, default=100, help="display progress every progress_freq steps")
    parser.add_argument("--metrics_freq", type=int, default=0, help="run and display metrics every metrics_freq step")
    parser.add_argument("--gif_freq", type=int, default=0, help="save gifs of predicted frames every gif_freq steps")
    parser.add_argument("--save_freq", type=int, default=5000, help="save model every save_freq steps, 0 to disable")

    parser.add_argument("--gpu_mem_frac", type=float, default=0, help="fraction of gpu memory to use")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--timing_file", type=str, help="")
    args = parser.parse_args()

    logsdir = args.logs_dir

    if args.seed is not None:
        tf.set_random_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)

    if args.experiment_dir != '':
        dataset_hparams_file = args.experiment_dir + '/dataset_hparams.json'
        model_hparams_file = args.experiment_dir + '/model_hparams.json'
    else:
        dataset_hparams_file = args.dataset_hparams
        model_hparams_file = args.model_hparams_dict

    if args.output_dir is None:
        list_depth = 0
        model_fname = ''
        for t in ('model=%s,%s' % (args.model, args.model_hparams)):
            if t == '[':
                list_depth += 1
            if t == ']':
                list_depth -= 1
            if list_depth and t == ',':
                t = '..'
            if t in '=,':
                t = '.'
            if t in '[]':
                t = ''
            model_fname += t
        args.output_dir = os.path.join(logsdir, model_fname)

    if args.resume:
        if args.checkpoint:
            raise ValueError('resume and checkpoint cannot both be specified')
        args.checkpoint = args.output_dir

    model_hparams_dict = {}
    if model_hparams_file:
        with open(model_hparams_file) as f:
            model_hparams_dict.update(json.loads(f.read()))
    if args.checkpoint:
        checkpoint_dir = os.path.normpath(args.checkpoint)
        if not os.path.exists(checkpoint_dir):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), checkpoint_dir)
        if not os.path.isdir(args.checkpoint):
            checkpoint_dir, _ = os.path.split(checkpoint_dir)
        with open(os.path.join(checkpoint_dir, "options.json")) as f:
            print("loading options from checkpoint %s" % args.checkpoint)
            options = json.loads(f.read())
            args.dataset = args.dataset or options['dataset']
            args.model = args.model or options['model']
        try:
            with open(os.path.join(checkpoint_dir, "model_hparams.json")) as f:
                model_hparams_dict.update(json.loads(f.read()))
                model_hparams_dict.pop('num_gpus', None)  # backwards-compatibility
        except FileNotFoundError:
            print("model_hparams.json was not loaded because it does not exist")
    print('----------------------------------- Options ------------------------------------')
    for k, v in args._get_kwargs():
        print(k, "=", v)
    print('------------------------------------- End --------------------------------------')

    #initialize train models
    VideoPredictionModel = models.get_model_class(args.model)
    train_model = VideoPredictionModel(mode='train', hparams_dict=model_hparams_dict, hparams=args.model_hparams)
    val_model = VideoPredictionModel(mode='val', hparams_dict=model_hparams_dict, hparams=args.model_hparams)
    batch_size = train_model.hparams.batch_size

    # load datasets
    dataset_class = datasets.get_dataset_class(args.dataset)
    dataset = dataset_class(args.input_dirs, dataset_hparams_file, batch_size)

    with tf.variable_scope('') as training_scope:
        inputs, targets = dataset.make_input_targets(model_hparams_dict['sequence_length'], model_hparams_dict['context_frames'], 'train')
        train_model.build_graph(inputs, targets)
    
    with tf.variable_scope(training_scope, reuse=True):
        inputs, targets = dataset.make_input_targets(model_hparams_dict['sequence_length'], model_hparams_dict['context_frames'], 'val')
        val_model.build_graph(inputs, targets)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    with open(os.path.join(args.output_dir, "options.json"), "w") as f:
        f.write(json.dumps(vars(args), sort_keys=True, indent=4))
    with open(os.path.join(args.output_dir, "dataset_hparams.json"), "w") as f:
        f.write(json.dumps(dataset.hparams.values(), sort_keys=True, indent=4))  # save hparams from first dataset
    with open(os.path.join(args.output_dir, "model_hparams.json"), "w") as f:
        f.write(json.dumps(train_model.hparams.values(), sort_keys=True, indent=4))

    if args.gif_freq:
        val_tensors = OrderedDict()
        context_images = val_model.inputs['images'][:, :val_model.hparams.context_frames]
        val_tensors['gen_images_vis'] = tf.concat([context_images, val_model.gen_images], axis=1)
        if val_model.gen_images_enc is not None:
            val_tensors['gen_images_enc_vis'] = tf.concat([context_images, val_model.gen_images_enc], axis=1)
        val_tensors.update({name: tensor for name, tensor in val_model.inputs.items() if tensor.shape.ndims >= 4})
        val_tensors['targets'] = val_model.targets
        val_tensors.update({name: tensor for name, tensor in val_model.outputs.items() if tensor.shape.ndims >= 4})
        val_tensor_clips = OrderedDict([(name, tf_utils.tensor_to_clip(output)) for name, output in val_tensors.items()])

    with tf.name_scope("parameter_count"):
        parameter_count = tf.reduce_sum([tf.reduce_prod(tf.shape(v)) for v in tf.trainable_variables()])

    saver = tf.train.Saver(max_to_keep=3)
    summaries = tf.get_collection(tf.GraphKeys.SUMMARIES)
    image_summaries = set(tf.get_collection(tf_utils.IMAGE_SUMMARIES))
    eval_summaries = set(tf.get_collection(tf_utils.EVAL_SUMMARIES))
    eval_image_summaries = image_summaries & eval_summaries
    image_summaries -= eval_image_summaries
    eval_summaries -= eval_image_summaries
    if args.summary_freq:
        summary_op = tf.summary.merge(summaries)
    if args.image_summary_freq:
        image_summary_op = tf.summary.merge(list(image_summaries))
    if args.eval_summary_freq:
        eval_summary_op = tf.summary.merge(list(eval_summaries))
        eval_image_summary_op = tf.summary.merge(list(eval_image_summaries))

    if args.summary_freq or args.image_summary_freq or args.eval_summary_freq:
        summary_writer = tf.summary.FileWriter(args.output_dir)

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_mem_frac)
    config = tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True)
    global_step = tf.train.get_or_create_global_step()
    max_steps = train_model.hparams.max_steps
    elapsed_times = []
    with tf.Session(config=config) as sess:
        print("parameter_count =", sess.run(parameter_count))

        sess.run(tf.global_variables_initializer())
        train_model.restore(sess, args.checkpoint)

        start_step = sess.run(global_step)
        # start at one step earlier to log everything without doing any training
        # step is relative to the start_step
        for step in range(-1, max_steps - start_step):
            if step == 0:
                start = time.time()

            def should(freq):
                return freq and ((step + 1) % freq == 0 or (step + 1) in (0, max_steps - start_step))

            fetches = {}
            if step >= 0:
                fetches["train_op"] = train_model.train_op

            if should(args.progress_freq):
                fetches['d_losses'] = train_model.d_losses
                fetches['g_losses'] = train_model.g_losses
                if isinstance(train_model.learning_rate, tf.Tensor):
                    fetches["learning_rate"] = train_model.learning_rate
            if should(args.metrics_freq):
                fetches['metrics'] = train_model.metrics
            if should(args.summary_freq):
                fetches["summary"] = summary_op
            if should(args.image_summary_freq):
                fetches["image_summary"] = image_summary_op
            if should(args.eval_summary_freq):
                fetches["eval_summary"] = eval_summary_op
                fetches["eval_image_summary"] = eval_image_summary_op

            run_start_time = time.time()
            results = sess.run(fetches)
            run_elapsed_time = time.time() - run_start_time
            elapsed_times.append(run_elapsed_time)
            if run_elapsed_time > 1.5:
                print('session.run took %0.1fs' % run_elapsed_time)

            # print("average t_iter {} \n".format(np.mean(elapsed_times[-20:])))
            # if step == 88:
            #     with open(args.timing_file, 'w') as f:
            #         f.write("{}\n".format(np.mean(elapsed_times)))
            #     import sys; sys.exit("finished")

            if should(args.progress_freq) or should(args.summary_freq):
                if step >= 0:
                    elapsed_time = time.time() - start
                    average_time = elapsed_time / (step + 1)
                    images_per_sec = batch_size / average_time
                    remaining_time = (max_steps - (start_step + step)) * average_time

            if should(args.progress_freq):
                # global_step will have the correct step count if we resume from a checkpoint
                steps_per_epoch = math.ceil(dataset.num_examples_per_epoch() / batch_size)
                train_epoch = math.ceil(global_step.eval() / steps_per_epoch)
                train_step = (global_step.eval() - 1) % steps_per_epoch + 1
                print("progress  global step %d  epoch %d  step %d" % (global_step.eval(), train_epoch, train_step))
                if step >= 0:
                    print("          image/sec %0.1f  remaining %dm (%0.1fh) (%0.1fd)" %
                          (images_per_sec, remaining_time / 60, remaining_time / 60 / 60, remaining_time / 60 / 60 / 24))

                for name, loss in itertools.chain(results['d_losses'].items(), results['g_losses'].items()):
                    print(name, loss)
                if isinstance(train_model.learning_rate, tf.Tensor):
                    print("learning_rate", results["learning_rate"])
            if should(args.metrics_freq):
                for name, metric in results['metrics']:
                    print(name, metric)

            if should(args.summary_freq):
                print("recording summary")
                summary_writer.add_summary(results["summary"], global_step.eval())
                if step >= 0:
                    try:
                        from tensorboard.summary import scalar_pb
                        for name, scalar in zip(['images_per_sec', 'remaining_hours'],
                                                [images_per_sec, remaining_time / 60 / 60]):
                            summary_writer.add_summary(scalar_pb(name, scalar), global_step.eval())
                    except ImportError:
                        pass

                print("done")
            if should(args.image_summary_freq):
                print("recording image summary")
                summary_writer.add_summary(
                    tf_utils.convert_tensor_to_gif_summary(results["image_summary"]), global_step.eval())
                print("done")
            if should(args.eval_summary_freq):
                print("recording eval summary")
                summary_writer.add_summary(results["eval_summary"], global_step.eval())
                summary_writer.add_summary(
                    tf_utils.convert_tensor_to_gif_summary(results["eval_image_summary"]), global_step.eval())
                print("done")



            if should(args.summary_freq) or should(args.image_summary_freq) or should(args.eval_summary_freq):
                summary_writer.flush()

            if should(args.save_freq):
                print("saving model to", args.output_dir)
                saver.save(sess, os.path.join(args.output_dir, "model"), global_step=global_step)
                print("done")


            if should(args.gif_freq):
                image_dir = os.path.join(args.output_dir, 'images')
                if not os.path.exists(image_dir):
                    os.makedirs(image_dir)

                gif_clips = sess.run(val_tensor_clips)
                gif_step = global_step.eval()
                for name, clip in gif_clips.items():
                    filename = "%08d-%s.gif" % (gif_step, name)
                    print("saving gif to", os.path.join(image_dir, filename))
                    ffmpeg_gif.save_gif(os.path.join(image_dir, filename), clip, fps=4)
                    print("done")


if __name__ == '__main__':
    main()
