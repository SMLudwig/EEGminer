import torch.optim as optim
import geoopt

from EEGminer import GeneralizedGaussianFilter


# Define the model
model = None
n_epochs = None
lr = None  # learning rate
l1_penalty = None  # L1 penalty applied on the last linear layer (fc_out)
weight_decay = None  # L2 penalty applied by the optimizer

# Optimizer and lr_scheduler for EEGminer models
temporal_filter_params = []
model_params = []
for m in model.children():
    if isinstance(m, (GeneralizedGaussianFilter,)):
        temporal_filter_params += [m.f_mean, m.bandwidth, m.shape, m.group_delay]
    else:
        model_params += list(m.parameters())
optimizer = optim.SGD([{'params': temporal_filter_params, 'lr': lr, 'weight_decay': 0.,
                        'momentum': 0.99, 'nesterov': True},
                       {'params': model_params, 'lr': lr, 'weight_decay': 0.,
                        'momentum': 0.9, 'nesterov': True}])
lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs - 1, eta_min=0)

# Optimizer for deep learning baselines: ShallowNet, EEGNet, SyncNet
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

# Optimizer for deep learning baselines: SPDNet
optimizer = geoopt.optim.RiemannianAdam(model.parameters(), lr=lr, weight_decay=weight_decay)

# Loss
criterion = None  # MSELoss/BCELoss
x = None
target = None
pred = model(x)
loss = criterion(pred, target) + l1_penalty * model.fc_out.weight.abs().sum()
