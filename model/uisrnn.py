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
"""The UISRNN model."""
from model import utils
import numpy as np
import os
import tempfile
import torch
from torch import autograd
from torch import nn
from torch import optim
import torch.nn.functional as F
import zipfile

_INITIAL_SIGMA2_VALUE = 0.1
_SAVED_STATES_FILE = 'saved_model.states'
_SAVED_NPZ_FILE = 'saved_model.npz'


class NormalRNN(nn.Module):
  """Normal Recurent Neural Networks."""

  def __init__(self, input_dim, hidden_size, depth, dropout, observation_dim):
    super(NormalRNN, self).__init__()
    self.hidden_size = hidden_size
    if depth >= 2:
      self.gru = nn.GRU(input_dim, hidden_size, depth, dropout=dropout)
    else:
      self.gru = nn.GRU(input_dim, hidden_size, depth)
    self.linear_mean1 = nn.Linear(hidden_size, hidden_size)
    self.linear_mean2 = nn.Linear(hidden_size, observation_dim)

  def forward(self, input_seq, hidden=None):
    output_seq, hidden = self.gru(input_seq, hidden)
    if isinstance(output_seq, torch.nn.utils.rnn.PackedSequence):
      output_seq, _ = torch.nn.utils.rnn.pad_packed_sequence(
          output_seq, batch_first=False)
    mean = self.linear_mean2(F.relu(self.linear_mean1(output_seq)))
    return mean, hidden


class BeamState(object):
  """Structure that contains necessary states for beam search."""

  def __init__(self):
    self.mean_set = []
    self.hidden_set = []
    self.neg_lhood = 0
    self.trace = []
    self.block_counts = []


class UISRNN(object):
  """Unbounded Interleaved-State Recurrent Neural Networks."""

  def __init__(self, args):
    """Construct the UISRNN object.

    Args:
      args: Model configurations. See arguments.py for details.
    """
    self.observation_dim = args.observation_dim
    self.device = torch.device(
        'cuda:0' if torch.cuda.is_available() else 'cpu')
    self.rnn_model = NormalRNN(self.observation_dim, args.rnn_hidden_size,
                               args.rnn_depth, args.rnn_dropout,
                               self.observation_dim).to(self.device)
    self.rnn_init_hidden = nn.Parameter(
        torch.zeros(args.rnn_depth, 1, args.rnn_hidden_size).to(self.device))
    self.estimate_sigma2 = (args.sigma2 is None)
    sigma2 = _INITIAL_SIGMA2_VALUE if self.estimate_sigma2 else args.sigma2
    self.sigma2 = nn.Parameter(
        sigma2 * torch.ones(self.observation_dim).to(self.device))
    self.transition_bias = args.transition_bias
    self.crp_alpha = args.crp_alpha

  def _get_optimizer(self, optimizer, learning_rate):
    """Get optimizer for UISRNN.

    Args:
      optimizer: string - name of the optimizer.
      learning_rate: - learning rate for the entire model.
        We do not customize learning rate for separate parts.

    Returns:
      a pytorch "optim" object
    """
    params = [
        {
            'params': self.rnn_model.parameters()
        },  # rnn parameters
        {
            'params': self.rnn_init_hidden
        }  # rnn initial hidden state
    ]
    if self.estimate_sigma2:  # train sigma2
      params.append({
          'params': self.sigma2
      }  # variance parameters
                   )
    assert optimizer == 'adam', 'Only adam optimizer is supported.'
    return optim.Adam(params, lr=learning_rate)

  def save(self, filepath):
    """Save the model to a file.

    Args:
      filepath: the path of the file.
    """
    tempdir = tempfile.mkdtemp()
    # save states
    states_file = os.path.join(tempdir, _SAVED_STATES_FILE)
    torch.save(self.rnn_model.state_dict(), states_file)

    # save other parameters
    npz_file = os.path.join(tempdir, _SAVED_NPZ_FILE)
    np.savez(npz_file,
             transition_bias=self.transition_bias,
             crp_alpha=self.crp_alpha,
             sigma2=self.sigma2.detach().cpu().numpy())

    # create combined model file
    with zipfile.ZipFile(filepath, 'w') as myzip:
      myzip.write(states_file, _SAVED_STATES_FILE)
      myzip.write(npz_file, _SAVED_NPZ_FILE)

  def load(self, filepath):
    """Load the model from a file.

    Args:
      filepath: the path of the file.
    """
    tempdir = tempfile.mkdtemp()
    # extract zip file
    with zipfile.ZipFile(filepath) as myzip:
      myzip.extract(_SAVED_STATES_FILE, path=tempdir)
      myzip.extract(_SAVED_NPZ_FILE, path=tempdir)

    # load states
    states_file = os.path.join(tempdir, _SAVED_STATES_FILE)
    self.rnn_model.load_state_dict(torch.load(states_file))

    # load other parameters
    npz_file = os.path.join(tempdir, _SAVED_NPZ_FILE)
    data = np.load(npz_file)
    self.transition_bias = float(data['transition_bias'])
    self.crp_alpha = float(data['crp_alpha'])
    self.sigma2 = nn.Parameter(
        torch.from_numpy(data['sigma2']).to(self.device))

  def fit(self, train_sequence, train_cluster_id, args):
    """Fit UISRNN model.

    Args:
      train_sequence: 2-dim numpy array of real numbers, size: N * D
        - the training observation sequence.
        N - summation of lengths of all utterances
        D - observation dimension
        For example, train_sequence =
        [[1.2 3.0 -4.1 6.0]    --> an entry of speaker #0 from utterance 'iaaa'
         [0.8 -1.1 0.4 0.5]    --> an entry of speaker #1 from utterance 'iaaa'
         [-0.2 1.0 3.8 5.7]    --> an entry of speaker #0 from utterance 'iaaa'
         [3.8 -0.1 1.5 2.3]    --> an entry of speaker #0 from utterance 'ibbb'
         [1.2 1.4 3.6 -2.7]]   --> an entry of speaker #0 from utterance 'ibbb'
        Here N=5, D=4.
        We concatenate all training utterances into a single sequence.
      train_cluster_id: 1-dim list or numpy array of strings, size: N
        - the speaker id sequence.
        For example, train_cluster_id =
        ['iaaa_0', 'iaaa_1', 'iaaa_0', 'ibbb_0', 'ibbb_0']
        'iaaa_0' means the entry belongs to speaker #0 in utterance 'iaaa'.
        Note that the order of entries within an utterance are preserved,
        and all utterances are simply concatenated together.
      args: Training configurations. See arguments.py for details.

    Raises:
      TypeError: If train_sequence or train_cluster_id is of wrong type.
      ValueError: If train_sequence or train_cluster_id has wrong dimension.
    """
    # check type
    if (not isinstance(train_sequence, np.ndarray) or
        train_sequence.dtype != float):
      raise TypeError('train_sequence should be a numpy array of float type.')
    if isinstance(train_cluster_id, list):
      train_cluster_id = np.array(train_cluster_id)
    if (not isinstance(train_cluster_id, np.ndarray) or
        not train_cluster_id.dtype.name.startswith('str')):
      raise TypeError('train_cluster_id type be a numpy array of strings.')
    # check dimension
    if train_sequence.ndim != 2:
      raise ValueError('train_sequence must be 2-dim array.')
    if train_cluster_id.ndim != 1:
      raise ValueError('train_cluster_id must be 1-dim array.')
    # check length and size
    train_total_length, observation_dim = train_sequence.shape
    if observation_dim != self.observation_dim:
      raise ValueError('train_sequence does not match the dimension specified '
                       'by args.observation_dim.')
    if train_total_length != len(train_cluster_id):
      raise ValueError('train_sequence length is not equal to '
                       'train_cluster_id length.')

    self.rnn_model.train()
    optimizer = self._get_optimizer(optimizer=args.optimizer,
                                    learning_rate=args.learning_rate)

    sub_sequences, seq_lengths, transition_bias = utils.resize_sequence(
        sequence=train_sequence,
        cluster_id=train_cluster_id,
        num_permutations=args.num_permutations)
    if self.transition_bias is None:
      self.transition_bias = transition_bias
    # For batch learning, pack the entire dataset.
    if args.batch_size is None:
      packed_train_sequence, rnn_truth = utils.pack_sequence(
          sub_sequences,
          seq_lengths,
          args.batch_size,
          self.observation_dim,
          self.device)
    train_loss = []
    for t in range(args.train_iteration):
      # Update learning rate if half life is specified.
      if args.learning_rate_half_life > 0:
        if t > 0 and t % args.learning_rate_half_life == 0:
          optimizer.param_groups[0]['lr'] /= 2.0
          print('Changing learning rate to: {}'.format(
              optimizer.param_groups[0]['lr']))
      optimizer.zero_grad()
      # For online learning, pack a subset in each iteration.
      if args.batch_size is not None:
        packed_train_sequence, rnn_truth = utils.pack_sequence(
            sub_sequences,
            seq_lengths,
            args.batch_size,
            self.observation_dim,
            self.device)
      hidden = self.rnn_init_hidden.repeat(1, args.batch_size, 1)
      mean, _ = self.rnn_model(packed_train_sequence, hidden)
      # use mean to predict
      mean = torch.cumsum(mean, dim=0)
      mean_size = mean.size()
      mean = torch.mm(
          torch.diag(
            1.0 / torch.arange(1, mean_size[0] + 1).float().to(self.device)),
          mean.view(mean_size[0], -1))
      mean = mean.view(mean_size)

      # Likelihood part.
      loss1 = utils.weighted_mse_loss(
          input_tensor=(rnn_truth != 0).float() * mean[:-1, :, :],
          target_tensor=rnn_truth,
          weight=1 / (2 * self.sigma2))

      weight = (((rnn_truth != 0).float() * mean[:-1, :, :] - rnn_truth)
                **2).view(-1, observation_dim)
      num_non_zero = torch.sum((weight != 0).float(), dim=0).squeeze()
      loss2 = ((2 * args.sigma_alpha + num_non_zero + 2) /
               (2 * num_non_zero) * torch.log(self.sigma2)).sum() + (
                   args.sigma_beta / (self.sigma2 * num_non_zero)).sum()
      # regularization
      l2_reg = 0
      for param in self.rnn_model.parameters():
        l2_reg += torch.norm(param)
      loss3 = args.regularization_weight * l2_reg

      loss = loss1 + loss2 + loss3
      loss.backward()
      nn.utils.clip_grad_norm_(self.rnn_model.parameters(), 5.0)
      # nn.utils.clip_grad_norm_(self.sigma2, 1.0)
      optimizer.step()
      # avoid numerical issues
      self.sigma2.data.clamp_(min=1e-6)

      if np.remainder(t, 10) == 0:
        print('Iter: {:d}  \t'
              'Training Loss: {:.4f}    \n'
              '    Negative Log Likelihood: {:.4f}\t'
              'Sigma2 Prior: {:.4f}\t'
              'Regularization: {:.4f}'.format(t, float(loss.data),
                float(loss1.data), float(loss2.data), float(loss3.data)))
      train_loss.append(float(loss1.data))  # only save the likelihood part
    print('Done training with {} iterations'.format(args.train_iteration))

  def copy_beam_state(self, beam_state):
    """Copy a state."""

    new_beam_state = BeamState()
    new_beam_state.mean_set = beam_state.mean_set.copy()
    new_beam_state.hidden_set = beam_state.hidden_set.copy()
    new_beam_state.trace = beam_state.trace.copy()
    new_beam_state.block_counts = beam_state.block_counts.copy()
    new_beam_state.neg_lhood = beam_state.neg_lhood
    return new_beam_state

  def update_beam_state(self, beam_state, look_ahead_seq, cluster_seq):
    """Update a beam state given a look ahead sequence and known cluster
    assignments.

    Args:
      beam_state: A BeamState object.
      look_ahead_seq: Look ahead sequence, size: look_ahead*D.
        look_ahead: number of step to look ahead in the beam search.
        D: observation dimension
      cluster_seq: Cluster assignment sequence for look_ahead_seq.

    Returns:
      new_beam_state: An updated BeamState object.
    """

    loss = 0  
    new_beam_state = self.copy_beam_state(beam_state)
    for sub_idx, cluster in enumerate(cluster_seq):
      if cluster > len(new_beam_state.mean_set):  # invalid trace
        new_beam_state.neg_lhood = float('inf')
        break
      elif cluster < len(new_beam_state.mean_set):  # existing cluster
        last_cluster = new_beam_state.trace[-1]
        loss = utils.weighted_mse_loss(
            input_tensor=torch.squeeze(new_beam_state.mean_set[cluster]),
            target_tensor=look_ahead_seq[sub_idx, :],
            weight=1 / (2 * self.sigma2)).cpu().detach().numpy()
        if cluster == last_cluster:
          loss -= np.log(1 - self.transition_bias)
        else:
          loss -= np.log(self.transition_bias) + np.log(
              new_beam_state.block_counts[cluster]) - np.log(
                  sum(new_beam_state.block_counts) + self.crp_alpha)
        # update new mean and new hidden
        mean, hidden = self.rnn_model(
            look_ahead_seq[sub_idx, :].unsqueeze(0).unsqueeze(0),
            new_beam_state.hidden_set[cluster])
        new_beam_state.mean_set[cluster] = (new_beam_state.mean_set[cluster]*(
            (np.array(new_beam_state.trace) == cluster).sum() -
            1).astype(float) + mean.clone()) / (
                np.array(new_beam_state.trace) == cluster).sum().astype(
                    float)  # use mean to predict
        new_beam_state.hidden_set[cluster] = hidden.clone()
        if cluster != last_cluster:
          new_beam_state.block_counts[cluster] += 1
        new_beam_state.trace.append(cluster)
      else:  # new cluster
        init_input = autograd.Variable(
            torch.zeros(self.observation_dim)
            ).unsqueeze(0).unsqueeze(0).to(self.device)
        mean, hidden = self.rnn_model(init_input,
                                      self.rnn_init_hidden)
        loss = utils.weighted_mse_loss(
            input_tensor=torch.squeeze(mean),
            target_tensor=look_ahead_seq[sub_idx, :],
            weight=1 / (2 * self.sigma2)).cpu().detach().numpy()
        loss -= np.log(self.transition_bias) + np.log(
            self.crp_alpha) - np.log(
                sum(new_beam_state.block_counts) + self.crp_alpha)
        # update new min and new hidden
        mean, hidden = self.rnn_model(
            look_ahead_seq[sub_idx, :].unsqueeze(0).unsqueeze(0),
            hidden)
        new_beam_state.mean_set.append(mean.clone())
        new_beam_state.hidden_set.append(hidden.clone())
        new_beam_state.block_counts.append(1)
        new_beam_state.trace.append(cluster)
      new_beam_state.neg_lhood += loss
    return new_beam_state

  def calc_score(self, beam_state, look_ahead_seq):
    """Calculate negative log likelihoods for all possible state allocations
       of a look ahead sequence, according to the current beam state.

    Args:
      beam_state: A BeamState object.
      look_ahead_seq: Look ahead sequence, size: look_ahead*D.
        look_ahead: number of step to look ahead in the beam search.
        D: observation dimension

    Returns:
      beam_score_set: a set of scores for each possible state allocation.
    """

    look_ahead, _  = look_ahead_seq.shape
    beam_num_clusters = len(beam_state.mean_set)
    beam_score_set = float('inf') * np.ones(beam_num_clusters + 1 + np.arange(
                                            look_ahead))
    for cluster_seq, _ in np.ndenumerate(beam_score_set):
      updated_beam_state = self.update_beam_state(beam_state,
                                             look_ahead_seq, cluster_seq)
      beam_score_set[cluster_seq] = updated_beam_state.neg_lhood
    return beam_score_set

  def predict(self, test_sequence, args):
    """Predict test sequence labels using UISRNN model.

    Args:
      test_sequence: 2-dim numpy array of real numbers, size: N * D
        - the test observation sequence.
        N - length of one test utterance
        D - observation dimension
        For example, test_sequence =
        [[2.2 -1.0 3.0 5.6]    --> 1st entry of utterance 'iccc'
         [0.5 1.8 -3.2 0.4]    --> 2nd entry of utterance 'iccc'
         [-2.2 5.0 1.8 3.7]    --> 3rd entry of utterance 'iccc'
         [-3.8 0.1 1.4 3.3]    --> 4th entry of utterance 'iccc'
         [0.1 2.7 3.5 -1.7]]   --> 5th entry of utterance 'iccc'
        Here N=5, D=4.
      args: Inference configurations. See arguments.py for details.

    Returns:
      predicted_cluster_id: (integer array, size: N)
        - predicted speaker id sequence.
        For example, predicted_cluster_id = [0, 1, 0, 0, 1]

    Raises:
      TypeError: If test_sequence is of wrong type.
      ValueError: If test_sequence has wrong dimension.
    """
    # check type
    if (not isinstance(test_sequence, np.ndarray) or
        test_sequence.dtype != float):
      raise TypeError('test_sequence should be a numpy array of float type.')
    # check dimension
    if test_sequence.ndim != 2:
      raise ValueError('test_sequence must be 2-dim array.')
    # check size
    test_sequence_length, observation_dim = test_sequence.shape
    if observation_dim != self.observation_dim:
      raise ValueError('test_sequence does not match the dimension specified '
                       'by args.observation_dim.')

    self.rnn_model.eval()
    test_sequence = np.tile(test_sequence, (args.test_iteration, 1))
    test_sequence = autograd.Variable(
        torch.from_numpy(test_sequence).float()).to(self.device)
    # bookkeeping for beam search
    beam_set = [BeamState()]
    for t in np.arange(0, args.test_iteration * test_sequence_length,
                       args.look_ahead):
      max_clusters = max([len(beam_state.mean_set) for beam_state in beam_set])
      look_ahead_seq = test_sequence[t:t+args.look_ahead,:]
      look_ahead_seq_length = look_ahead_seq.shape[0]
      score_set = float('inf') * np.ones(
          np.append(
              args.beam_size, max_clusters + 1 + np.arange(
                  look_ahead_seq_length)))
      for beam_rank, beam_state in enumerate(beam_set):
        beam_score_set = self.calc_score(beam_state, look_ahead_seq)
        score_set[beam_rank,:] = np.pad(beam_score_set,
            np.tile([[0, max_clusters-len(beam_state.mean_set)]],
                (look_ahead_seq_length,1)), 'constant',
                    constant_values=float('inf'))
      # find top scores
      score_ranked = np.sort(score_set, axis=None)
      score_ranked[score_ranked == float('inf')] = 0
      score_ranked = np.trim_zeros(score_ranked)
      idx_ranked = np.argsort(score_set, axis=None)
      updated_beam_set = []
      for new_beam_rank in range(
          np.min((len(score_ranked), args.beam_size))):
        total_idx = np.unravel_index(idx_ranked[new_beam_rank],
                                     score_set.shape)
        prev_beam_rank = total_idx[0]
        cluster_seq = total_idx[1:]
        updated_beam_state = self.update_beam_state(
            beam_set[prev_beam_rank], look_ahead_seq, cluster_seq)
        updated_beam_set.append(updated_beam_state)
      beam_set = updated_beam_set
    predicted_cluster_id = beam_set[0].trace[-test_sequence_length:]
    return predicted_cluster_id
