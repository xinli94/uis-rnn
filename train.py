# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import numpy as np
import os
import pandas as pd
import random

from model import arguments
from model import evals
from model import uisrnn
from model import utils

random.seed(12345)

# avoid OOM
MAX_SIZE = 20000 * 256
# test chunk size
CHUNK_SIZE = 100

# SAVED_MODEL_NAME = 'saved_model.uisrnn'
TRAIN_SEQUENCE = 'train_sequence'
TRAIN_CLUSTER = 'train_cluster_id'
TEST_SEQUENCE = 'test_sequences'
TEST_CLUSTER = 'test_cluster_ids'

def data_path_helper(data_path):
  data_ext = os.path.splitext(data_path)[-1]
  if data_ext == '.npz':
    data_list = [data_path]
  elif data_ext == '.csv':
    data_list = pd.read_csv(data_path, names=['npz_path'])['npz_path'].values.tolist()
  else:
    raise ValueError('==> Input data type unsupported')
  return data_list

def train_uis_rnn(model_args, training_args, inference_args, data_args):
  """Experiment pipeline.

  Load data --> train model --> test model --> output result

  Args:
    model_args: model configurations
    training_args: training configurations
    inference_args: inference configurations
  """
  model = uisrnn.UISRNN(model_args)

  ###############################################################################################
  #                                       training                                              #
  ###############################################################################################
  if not data_args.test_only:
    train_data_list = data_path_helper(data_args.train_data_path)

    train_idx = 0
    while train_idx < len(train_data_list):
      new_cluster_flag = True
      train_data, train_sequence, train_cluster_id = None, None, None

      while train_idx < len(train_data_list) and (new_cluster_flag or train_sequence.size < MAX_SIZE):
        try:
          train_data = np.load(train_data_list[train_idx])
        except:
          print('==> Skip npz file: {}'.format(train_data_list[train_idx]))
          continue

        train_idx += 1
        # if train_sequence == None:
        if new_cluster_flag == True:
          train_sequence = train_data[TRAIN_SEQUENCE]
          train_cluster_id = train_data[TRAIN_CLUSTER]
          new_cluster_flag = False
        else:
          train_sequence = np.append(train_sequence, train_data[TRAIN_SEQUENCE], axis=0)
          train_cluster_id = np.append(train_cluster_id, train_data[TRAIN_CLUSTER], axis=0)

      if train_data != None:
        print('==> Train_sequence idx {}, shape {}'.format(train_idx, train_sequence.shape))
        print('==> Train_cluster_id idx {}, shape {}'.format(train_idx, train_cluster_id.shape))
        model.fit(train_sequence, train_cluster_id, training_args)
        model.save(data_args.checkpoint_path)

  else:
    # we can also skip training by calling：
    model.load(data_args.checkpoint_path)
    print('==> Loaded checkpoint from {}'.format(data_args.checkpoint_path))

  ###############################################################################################
  #                                       testing                                               #
  ###############################################################################################
  predicted_labels = []
  test_record = []
  records = []

  test_sequences, test_cluster_ids = None, None
  test_data_list = data_path_helper(data_args.test_data_path)
  random.shuffle(test_data_list)

  if not data_args.reformat:
    for test_data_path in test_data_list:
      test_data = np.load(test_data_path)
      test_sequences = test_data[TEST_SEQUENCE]
      test_cluster_ids = test_data[TEST_CLUSTER]

      for (test_sequence, test_cluster_id) in zip(test_sequences, test_cluster_ids):
        predicted_label = model.predict(test_sequence, inference_args)
        predicted_labels.append(predicted_label)

        accuracy = evals.compute_sequence_match_accuracy(
            test_cluster_id, predicted_label)
        test_record.append((accuracy, len(test_cluster_id)))
        print('Ground truth labels:')
        print(test_cluster_id)
        print('Predicted labels:')
        print(predicted_label)
        print('Accuracy: {}'.format(accuracy))
        print('-' * 80)
        records += zip(test_cluster_id, predicted_label)

  else:
    test_idx = 0
    while test_idx < len(test_data_list):
      new_cluster_flag = True
      test_data, test_sequence, test_cluster_id = None, None, None
      while test_idx < len(test_data_list) and (new_cluster_flag or test_sequence.shape[0] < CHUNK_SIZE):
        try:
          test_data = np.load(test_data_list[test_idx])
        except:
          print('==> Skip npz file: {}'.format(test_data_list[test_idx]))
          continue

        test_idx += 1
        # if test_sequence == None:
        if new_cluster_flag == True:
          test_sequence = test_data[TRAIN_SEQUENCE]
          test_cluster_id = test_data[TRAIN_CLUSTER]
          new_cluster_flag = False
        else:
          test_sequence = np.append(test_sequence, test_data[TRAIN_SEQUENCE], axis=0)
          test_cluster_id = np.append(test_cluster_id, test_data[TRAIN_CLUSTER], axis=0)

      if test_data != None:
        print('==> test_sequence idx {}, shape {}'.format(test_idx, test_sequence.shape))
        print('==> test_cluster_id idx {}, shape {}'.format(test_idx, test_cluster_id.shape))
        predicted_label = model.predict(test_sequence, inference_args)

        accuracy = evals.compute_sequence_match_accuracy(
            list(test_cluster_id), list(predicted_label))
        test_record.append((accuracy, len(test_cluster_id)))
        print('Ground truth labels:')
        print(test_cluster_id)
        print('Predicted labels:')
        print(predicted_label)
        print('Accuracy: {}'.format(accuracy))
        print('-' * 80)
        records += zip(test_cluster_id, predicted_label)
  
  df = pd.DataFrame.from_records(records, columns=['ground_truth', 'predicted_label'])
  df.to_csv(data_args.output_path, index=False)

  output_string = utils.output_result(model_args, training_args, test_record)

  print('Finished diarization experiment')
  print(output_string)


def main():
  model_args, training_args, inference_args = arguments.parse_arguments()

  data_parser = argparse.ArgumentParser(
    description='Training data.', add_help=False)

  data_parser.add_argument(
    '--train_data_path',
    default='./data/training_data.npz',
    help='The path for training data.')

  data_parser.add_argument(
    '--test_data_path',
    default='./data/testing_data.npz',
    help='The path for testing data.')

  data_parser.add_argument(
    '--test_only',
    action='store_true',
    help='If set, skip training and load a ckpt directly.')

  data_parser.add_argument(
    '--reformat',
    action='store_true',
    help='If set, reformat test data from 2d to 3d.')

  data_parser.add_argument(
    '--checkpoint_path',
    default='tmp.uisrnn',
    help='checkpoint path.')

  data_parser.add_argument(
    '--output_path',
    default='tmp.csv',
    help='Output path.')

  data_args, _ = data_parser.parse_known_args()
  train_uis_rnn(model_args, training_args, inference_args, data_args)


if __name__ == '__main__':
  main()
