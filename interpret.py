import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.patches import ConnectionPatch
import matplotlib.lines as lines

from EEGminer import GeneralizedGaussianFilter

# Model parameters
model_path = 'CorrEEGminer_female_male_f1.pt'
classes = ['Male', 'Female']
n_top_features = 10
sample_rate = 128.
n_channels = 2  # filters per electrode
ch_names = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']
n_electrodes = len(ch_names)
freq_limits = [4, 8, 13, 20, 30]  # for vertical lines
labelled_freq_limits = [0, 20, 45]  # for labelled x-ticks

# Load model
model_dict = torch.load(model_path, map_location='cpu')

# Initialize generalized gaussian filters
sequence_length = (len(model_dict['filter.n_range']) - 1) * 2  # length of EEG signals
filters = GeneralizedGaussianFilter(in_channels=n_electrodes, out_channels=n_electrodes * n_channels,
                                    sequence_length=sequence_length, sample_rate=sample_rate,
                                    f_mean=[23.] * n_channels,
                                    bandwidth=[44.] * n_channels,
                                    shape=[2.] * n_channels,
                                    group_delay=[20.] * n_channels)
filter_functions_init = filters.construct_filters()
mag_response_init = torch.sqrt(filter_functions_init[0, :, 0] ** 2 + filter_functions_init[0, :, 1] ** 2)
mag_response_init = mag_response_init.detach().numpy()  # (freqs,)

# Extract trained filter functions and compute their magnitude response
filters.f_mean.data = model_dict['filter.f_mean']
filters.bandwidth.data = model_dict['filter.bandwidth']
filters.shape.data = model_dict['filter.shape']
filters.group_delay.data = model_dict['filter.group_delay']
filter_functions = filters.construct_filters()  # (out_channels, freqs, 2), real and imag dimensions
mag_response = torch.sqrt(filter_functions[..., 0] ** 2 + filter_functions[..., 1] ** 2)
mag_response = mag_response.reshape(n_electrodes, n_channels, mag_response.shape[-1])
mag_response = mag_response.detach().numpy()  # (n_electrodes, channels, freqs)

# Extract logistic regression weights, average over n_out
# The order of weights stems from the order of model dimensions when flattening the feature vector
linreg_weights_flat = model_dict['fc_out.weight'].numpy()  # (n_out, feat_per_ch * channels)
linreg_weights_flat = linreg_weights_flat[0]  # assume single output, binary classification
features_per_channel = linreg_weights_flat.shape[0] // n_channels

# Find most important features by their absolute regression weight
top_fts_ix = np.argsort(-np.abs(linreg_weights_flat))  # descending order
top_fts_ix = top_fts_ix[:n_top_features]  # slice top n
top_feat_weights = linreg_weights_flat[top_fts_ix]
max_feat_weight = np.max(np.abs(top_feat_weights))

# Attribute top features to the correct filter channel and electrodes forming the connection.
# Filter channels and connections are interleaved, e.g. [a, b, a, b, a, b] for filter channels a and b.
channel_ix = np.mod(top_fts_ix, n_channels)  # filter channel
feature_ix = np.floor(top_fts_ix / n_channels).astype(int)  # index within the filter channel
triu_indices = np.triu_indices(n_electrodes, k=1)
electrode_1_ix = triu_indices[0][feature_ix]
electrode_2_ix = triu_indices[1][feature_ix]

# Plot
fig = plt.figure(figsize=(16, 2.5))
n_plot_cols = n_top_features + 1
freqs = np.linspace(0, sample_rate // 2, mag_response.shape[-1])
for i in range(n_top_features):
    # Create subplots for frequency filters in electrode indexing order

    # First electrode
    ax1 = plt.subplot(2, n_plot_cols, i + 1)
    for f in freq_limits:
        plt.axvline(f, linestyle='--', linewidth=0.5, alpha=0.5)
    mag = mag_response[electrode_1_ix[i], channel_ix[i], :]  # select correct filter function
    mag = mag * np.abs(top_feat_weights[i])  # scale by importance
    plt.plot(freqs, mag, color='dimgrey')
    plt.ylim(bottom=0, top=max_feat_weight)
    ax1.set_xticks(labelled_freq_limits)
    ax1.set_xticks(freq_limits, minor=True)
    plt.yticks([])
    ax1.set_title(ch_names[electrode_1_ix[i]], loc='left')

    # Second electrode
    ax2 = plt.subplot(2, n_plot_cols, n_plot_cols + i + 1)
    for f in freq_limits:
        plt.axvline(f, linestyle='--', linewidth=0.5, alpha=0.5)
    mag = mag_response[electrode_2_ix[i], channel_ix[i], :]  # select correct filter function
    mag = mag * np.abs(top_feat_weights[i])  # scale by importance
    plt.plot(freqs, mag, color='dimgrey')
    plt.ylim(bottom=0, top=max_feat_weight)
    ax2.set_xticks(labelled_freq_limits)
    ax2.set_xticks(freq_limits, minor=True)
    plt.yticks([])
    ax2.set_title(ch_names[electrode_2_ix[i]], loc='left')

    # Connection
    color = np.array(to_rgb('tab:red')) if top_feat_weights[i] >= 0 else np.array(to_rgb('tab:blue'))
    c_width = np.abs(top_feat_weights[i] / max_feat_weight)
    cp_weight = ConnectionPatch((.5, .5), (.5, .5), "axes fraction", "axes fraction",
                                axesA=ax1, axesB=ax2,
                                color=color, alpha=c_width,
                                linewidth=5 * c_width)
    cp_weight.set_zorder(-1)
    fig.add_artist(cp_weight)

# Legend
legend_h = 0.7
fig.add_artist(lines.Line2D([0.92, 0.945], [legend_h + 0.08, legend_h + 0.08],
                            color='tab:blue', alpha=1, linewidth=3 + 2))
fig.add_artist(lines.Line2D([0.92, 0.945], [legend_h, legend_h],
                            color='tab:red', alpha=1, linewidth=3 + 2))
fig.text(0.95, legend_h + 0.065, classes[0])
fig.text(0.95, legend_h - 0.02, classes[1])

# Filter initialization
ax = plt.subplot(2, n_plot_cols, n_plot_cols)
for f in freq_limits:
    plt.axvline(f, linestyle='--', linewidth=0.5, alpha=0.5)
ax.set_ylim(bottom=0, top=max_feat_weight)
ax.set_xticks(labelled_freq_limits)
ax.set_xticks(freq_limits, minor=True)
ax.set_yticks([])
ax.set_title('Init')
plt.plot(freqs, mag_response_init * 0.98 * max_feat_weight, color='tab:grey', linestyle='--')
plt.subplots_adjust(wspace=0.3, hspace=0.7)
box = ax.get_position()
box.x0 = box.x0 + 0.01
box.x1 = box.x1 + 0.01
ax.set_position(box)
