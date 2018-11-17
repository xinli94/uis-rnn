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

from model import arguments
from model import evals
from model import uisrnn
from model import utils

SAVED_MODEL_NAME = 'saved_model.uisrnn'
TRAIN_SEQUENCE = 'train_sequence'
TRAIN_CLUSTER = 'train_cluster_id'
TEST_SEQUENCE = 'test_sequences'
TEST_CLUSTER = 'test_cluster_ids'

def train_uis_rnn(model_args, training_args, inference_args, data_args):
  """Experiment pipeline.

  Load data --> train model --> test model --> output result

  Args:
    model_args: model configurations
    training_args: training configurations
    inference_args: inference configurations
  """

  predicted_labels = []
  test_record = []

  train_data = np.load(data_args.train_data_path)
  test_data = np.load(data_args.test_data_path)
  train_sequence = train_data[TRAIN_SEQUENCE]
  train_cluster_id = train_data[TRAIN_CLUSTER]
  test_sequences = test_data[TEST_SEQUENCE]
  test_cluster_ids = test_data[TEST_CLUSTER]

  model = uisrnn.UISRNN(model_args)

  # training
  model.fit(train_sequence, train_cluster_id, training_args)
  model.save(SAVED_MODEL_NAME)
  # we can also skip training by calling：
  # model.load(SAVED_MODEL_NAME)

  # testing
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
    print('----------------------')

  utils.output_result(model_args, training_args, test_record)

  print('Finished diarization experiment with --sigma_alpha {} --sigma_beta {} '
        '--crp_alpha {} -l {} -r {}'
        .format(training_args.sigma_alpha, training_args.sigma_beta,
                model_args.crp_alpha, training_args.learning_rate,
                training_args.regularization_weight))


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

  data_args, _ = data_parser.parse_known_args()
  train_uis_rnn(model_args, training_args, inference_args, data_args)


if __name__ == '__main__':
  main()
