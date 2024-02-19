import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import nn

from EEGminer import CorrEEGminer

# Model parameters
model_path = 'CorrEEGminer_female_male_f1.pt'
n_top_features = 5
sample_rate = 128.
ch_names = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']

# Load data
eeg = torch.tensor([])  # (trials, electrodes, time)
targets = torch.tensor([])  # (trials)

# Load model
model_dict = torch.load(model_path, map_location='cpu')
model = CorrEEGminer(in_shape=[eeg.shape[-2], eeg.shape[-1]], n_out=1)
model.load_state_dict(model_dict)
model.float()
model.eval()
# Order the features based on their classifier weight
weight_order = np.argsort(-np.abs(model.fc_out.weight.data.numpy().reshape(-1)))

# Define the parameter range to test over
center_freq_linspace = np.linspace(-12, 12, 11) / (sample_rate / 2)
bandwidth_linspace = np.linspace(-12, 12, 11) / (sample_rate / 2)

# Prepare tracking arrays
tr_center_freq = np.zeros([n_top_features, 2])
tr_bandwidth = np.zeros([n_top_features, 2])
tr_electrode_ix = np.zeros([n_top_features, 2], dtype=int)
tr_center_freq_loss = np.zeros([n_top_features, len(center_freq_linspace), 2])
tr_bandwidth_loss = np.zeros([n_top_features, len(bandwidth_linspace), 2])

# Specify loss
loss = nn.MSELoss()

# Loop over the most important features
for i_ft, ft_ix in enumerate(weight_order[:n_top_features]):
    channel_ix = np.mod(ft_ix, model.n_filters)  # filter channel
    feature_ix = np.floor(ft_ix / model.n_filters).astype(int)  # index within the filter channel
    triu_indices = np.triu_indices(eeg.shape[-2], k=1)
    electrode_1_ix = triu_indices[0][feature_ix]  # first electrode in the connection
    electrode_2_ix = triu_indices[1][feature_ix]  # second electrode in the connection
    filter_ind = [electrode_1_ix * model.n_filters + channel_ix,
                  electrode_2_ix * model.n_filters + channel_ix]  # corresponding indices in the filters
    tr_electrode_ix[i_ft, 0] = electrode_1_ix
    tr_electrode_ix[i_ft, 1] = electrode_2_ix

    # Loop over the filters in the feature
    for i_filter, f_ix in enumerate(filter_ind):
        tr_center_freq[i_ft, i_filter] = model.filter.f_mean.data[f_ix]  # track the center freq of the filter
        tr_bandwidth[i_ft, i_filter] = model.filter.bandwidth.data[f_ix]  # track the bandwidth of the filter

        # Loop over the center frequency range
        for i_freq, freq in enumerate(center_freq_linspace):
            # Calculate the loss for the altered center frequency
            model.filter.f_mean.data[f_ix] = model.filter.f_mean.data[f_ix] + freq
            x = model(eeg)
            x = loss(x.squeeze(-1), targets).detach().numpy()
            tr_center_freq_loss[i_ft, i_freq, i_filter] = x
            # Reset model to the original filter parameters
            model.load_state_dict(model_dict)

        # Loop over the bandwidth range
        for i_bandwidth, bw in enumerate(bandwidth_linspace):
            # Calculate the loss for the altered bandwidth
            model.filter.bandwidth.data[f_ix] = model.filter.bandwidth.data[f_ix] + bw
            x = model(eeg)
            x = loss(x.squeeze(-1), targets).detach().numpy()
            tr_bandwidth_loss[i_ft, i_bandwidth, i_filter] = x
            # Reset model to the original filter parameters
            model.load_state_dict(model_dict)

# Scale the center frequency and bandwidth from [0,1] to [0, fs/2]
tr_center_freq = tr_center_freq * sample_rate / 2
tr_bandwidth = tr_bandwidth * sample_rate / 2

plt.figure(figsize=(16, 3))
# Loop over top features to plot the loss sensitivity
for i_ft in range(n_top_features):
    # Loop over the two electrodes in each connection
    for i_elec in range(2):
        plt.subplot(1, 2 * n_top_features, i_ft * 2 + i_elec + 1)
        plt.plot(center_freq_linspace * (sample_rate / 2), tr_center_freq_loss[i_ft, :, i_elec])
        plt.plot(center_freq_linspace * (sample_rate / 2), tr_bandwidth_loss[i_ft, :, i_elec])
        ft_electrode = ch_names[tr_electrode_ix[i_ft, i_elec]]
        plt.title(
            f"{ft_electrode} {np.clip(tr_center_freq[i_ft, i_elec] - tr_bandwidth[i_ft, i_elec] / 2, 0, sample_rate / 2):.0f}"
            f"-{np.clip(tr_center_freq[i_ft, i_elec] + tr_bandwidth[i_ft, i_elec] / 2, 0, sample_rate / 2):.0f}Hz")
        plt.axvline(0, linestyle='--', alpha=0.5)
        plt.xticks([-10, 0, 10], ['-10Hz', '', '+10Hz'])
        losses = np.stack([tr_center_freq_loss, tr_bandwidth_loss])
        margin = (np.max(losses) - np.min(losses)) * 0.05
        plt.ylim(np.min(losses) - margin, 0.25)
        if (i_ft == 0) & (i_elec == 0):
            plt.ylabel('MSE loss')
            plt.legend(['center', 'width'], loc='upper left')
        else:
            plt.yticks([])
plt.tight_layout()
plt.subplots_adjust(wspace=0.1)
plt.show()
