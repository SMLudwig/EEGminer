import torch
from torch import nn
import torch.nn.functional as F
import geoopt

from utils import max_norm_


class EEGNet(nn.Module):
    """EEGNet implementation based on https://iopscience.iop.org/article/10.1088/1741-2552/aace8c/pdf.
    Assumes sampling frequency of 128 for kernel size choices. Uses odd kernel sizes to fit paddings."""

    def __init__(self, in_shape, n_out, dropout=0.25):
        super(EEGNet, self).__init__()
        self.in_shape = in_shape  # (electrodes, time)
        self.n_out = n_out
        self.dropout = dropout

        # Model size parameters
        self.n_filters = 4
        self.n_spatial = 2

        # Block 1: kernel 1/2 of sampling freq + 1
        # Temporal Filter
        self.conv1 = nn.Conv2d(1, self.n_filters,
                               kernel_size=(1, 65), padding=(0, 32),
                               bias=False, groups=1)
        self.bn1 = nn.BatchNorm2d(self.n_filters)
        # Spatial Filter
        self.conv2 = nn.Conv2d(self.n_filters, self.n_spatial * self.n_filters,
                               kernel_size=(in_shape[0], 1),
                               bias=False, groups=self.n_filters)
        self.bn2 = nn.BatchNorm2d(self.n_spatial * self.n_filters)
        self.pool1 = nn.AvgPool2d(kernel_size=(1, 4))
        self.drop1 = nn.Dropout(dropout)

        # Block 2: kernel 1/8 of sampling freq + 1
        # Temporal Filter
        self.conv3 = nn.Conv2d(self.n_spatial * self.n_filters, self.n_spatial * self.n_filters,
                               kernel_size=(1, 17), padding=(0, 8),
                               bias=False, groups=self.n_spatial * self.n_filters)
        # Pointwise Convolution
        self.conv4 = nn.Conv2d(self.n_filters * self.n_spatial, self.n_filters * self.n_spatial,
                               kernel_size=(1, 1),
                               bias=False, groups=1)
        self.bn3 = nn.BatchNorm2d(self.conv4.out_channels, affine=True)
        self.pool2 = nn.AvgPool2d(kernel_size=(1, 8))
        self.drop2 = nn.Dropout(dropout)

        # Linear Classifier
        self.n_features = int(self.n_spatial * self.n_filters * (in_shape[1] // 4 // 8))
        self.fc_out = nn.Linear(self.n_features, n_out)
        nn.init.zeros_(self.fc_out.bias)

    def forward(self, x):
        """x: (batch, electrodes, time)"""
        batch, _, _ = x.shape

        # Add artificial image channel dimension for Conv2d
        x = x.unsqueeze(1)

        # Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = max_norm_(self.conv2, c=1., return_module=True)(x)
        x = self.bn2(x)
        x = F.elu(x)
        x = self.pool1(x)
        x = self.drop1(x)

        # Block 2
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.bn3(x)
        x = F.elu(x)
        x = self.pool2(x)
        x = self.drop2(x)

        # Classifier
        x = x.reshape(batch, -1)
        x = max_norm_(self.fc_out, c=0.25, return_module=True)(x)
        x = torch.sigmoid(x)

        return x


class SyncNet(nn.Module):
    """SyncNet implementation based on https://github.com/yitong91/SyncNet/blob/master/SyncNetModel.py"""

    def __init__(self, in_shape, n_out, dropout=0.):
        super(SyncNet, self).__init__()
        self.in_shape = in_shape  # (electrodes, time)
        self.n_out = n_out
        self.dropout = dropout

        # Model size parameters
        self.n_filters = 10  # K in the original code
        self.kernel_size = 41  # Nt in the original code
        self.pool_size = 40

        # Filter parameters (out_channels, in_channels, kernel_size)
        self.b = nn.Parameter(torch.rand([self.n_filters, in_shape[-2], 1]) / 10 - 0.05)  # [-0.05, 0.05]
        self.omega = nn.Parameter(torch.rand([self.n_filters, 1, 1]))  # [0, 1]
        self.phi_init = nn.Parameter(torch.normal(size=[self.n_filters, in_shape[-2] - 1, 1], mean=0, std=0.05))
        self.beta = nn.Parameter(torch.rand([self.n_filters, 1, 1]) / 20)  # [0, 0.05]
        self.filter_bias = nn.Parameter(torch.zeros(self.n_filters))

        # Time range symmetric around 0
        self.t = torch.arange(-(self.kernel_size - 1) // 2, (self.kernel_size - 1) // 2 + 1)
        self.t = self.t.reshape([1, 1, self.kernel_size])
        self.t = nn.Parameter(self.t, requires_grad=False)

        # Linear Classifier
        self.n_features = self.n_filters * in_shape[-1] // self.pool_size
        self.fc_out = nn.Linear(self.n_features, n_out)
        nn.init.zeros_(self.fc_out.bias)

    def syncnet_filters(self):
        """Generates the filter function from trainable parameters."""
        # Clamp beta >= 0
        self.beta.data = torch.clamp(self.beta.data, min=0.)

        # Add channel with phase shift fixed to zero to the phi parameter
        phi = torch.cat([torch.zeros([self.n_filters, 1, 1],
                                     dtype=self.phi_init.dtype,
                                     device=self.phi_init.device),
                         self.phi_init], dim=1)

        # Construct filter weights
        W_osc = self.b * torch.cos(self.t * self.omega + phi)
        W_decay = torch.exp(- self.t ** 2 * self.beta)
        W = W_osc * W_decay

        return W

    def forward(self, x):
        """x: (batch, electrodes, time)"""
        batch, _, _ = x.shape

        # Input channel dropout
        x = F.dropout2d(x.unsqueeze(-1), p=self.dropout).squeeze(-1)

        # Apply filters
        # NOTE: pytorch does cross-correlation not convolution
        # x -> (batch, filters, time)
        x = F.conv1d(x, weight=self.syncnet_filters(), bias=self.filter_bias,
                     padding=(self.kernel_size - 1) // 2)
        x = torch.relu(x)
        # x -> (batch, filters, time / pool_size)
        x = F.max_pool1d(x, kernel_size=self.pool_size, stride=self.pool_size)

        # Classifier
        x = x.reshape(batch, self.n_features)
        x = self.fc_out(x)
        x = torch.sigmoid(x)

        return x


class ShallowConvNet(nn.Module):
    """ShallowConvNet implementation based on https://onlinelibrary.wiley.com/doi/full/10.1002/hbm.23730"""

    def __init__(self, in_shape, n_out, dropout=0.5):
        super(ShallowConvNet, self).__init__()
        self.in_shape = in_shape  # (electrodes, time)
        self.n_out = n_out
        self.dropout = dropout

        # Model size parameters
        self.n_filters = 40

        # Convolution block
        self.temporal = nn.Conv2d(1, self.n_filters, kernel_size=(1, 13), bias=False, groups=1)
        self.spatial = nn.Conv2d(self.n_filters, self.n_filters, kernel_size=(in_shape[0], 1),
                                 bias=False, groups=1)
        self.bn = nn.BatchNorm2d(self.spatial.out_channels)
        self.drop = nn.Dropout(dropout)
        self.pooling = nn.AvgPool2d((1, 36), stride=(1, 8))

        # Linear Classifier
        self.n_features = self.n_filters * in_shape[-1] // 8 - 200
        self.fc_out = nn.Linear(self.n_features, n_out)
        nn.init.zeros_(self.fc_out.bias)

    def forward(self, x):
        """x: (batch, electrodes, time)"""

        # Add artificial image channel dimension for Conv2d
        x = x.unsqueeze(1)

        # Convolution
        x = self.temporal(x)
        x = self.spatial(x)
        x = self.bn(x)
        x = torch.square(x)
        x = self.pooling(x)
        x = torch.clamp(x, min=1e-6)
        x = torch.log(x)
        x = self.drop(x)

        # Classifier
        x = x.reshape(x.shape[0], self.n_features)
        x = self.fc_out(x)
        x = torch.sigmoid(x)

        return x


class SPDNet(nn.Module):
    """SPDNet implementation based on https://ojs.aaai.org/index.php/AAAI/article/view/10866"""

    def __init__(self, in_shape, n_out, dropout=0.5):
        super(SPDNet, self).__init__()
        self.in_shape = in_shape  # (electrodes, time)
        self.n_out = n_out
        self.dropout = dropout

        # Model size parameters
        self.n_filters = 1  # only supports 1 filter with 1.8.1 implementation
        self.n_hidden = 14

        # BiMap layers (orthogonal weight init)
        self.stiefel = geoopt.Stiefel()
        # Layer 1
        self.w1 = geoopt.ManifoldParameter(torch.empty(self.in_shape[-2], self.n_hidden),
                                           manifold=self.stiefel)
        w1 = nn.init.orthogonal(torch.empty([self.in_shape[-2], self.n_hidden]))
        self.w1.data = w1.unsqueeze(0)
        # Layer 2
        self.w2 = geoopt.ManifoldParameter(torch.empty(self.w1.shape[-1], self.n_hidden),
                                           manifold=self.stiefel)
        w2 = nn.init.orthogonal(torch.empty([self.w1.shape[-1], self.n_hidden]))
        self.w2.data = w2.unsqueeze(0)
        # Layer 3
        self.w3 = geoopt.ManifoldParameter(torch.empty(self.w2.shape[-1], self.n_hidden),
                                           manifold=self.stiefel)
        w3 = nn.init.orthogonal(torch.empty([self.w2.shape[-1], self.n_hidden]))
        self.w3.data = w3.unsqueeze(0)

        # Linear Classifier
        self.n_features = self.n_filters * self.w3.shape[-1] ** 2
        self.fc_out = nn.Linear(self.n_features, n_out)
        nn.init.zeros_(self.fc_out.bias)

    def re_eig(self, x):
        """ReEig layer
        Args:
            x (batch, filters, channels, channels)
        """
        # eval (batch, filters, channels)
        # evec (batch, filters, channels, channels)
        eval, evec = torch.linalg.eigh(x)
        eval = torch.clamp(eval, min=1e-4)  # rectification threshold
        x = torch.matmul(evec, torch.matmul(torch.diag_embed(eval), evec.transpose(-2, -1)))
        return x

    def forward(self, x):
        """x: (batch, electrodes, time)"""
        batch, channels, time = x.shape

        # Make covariance matrix
        # x -> (batch, 1, electrodes, electrodes)
        x = x - x.mean(dim=(-1), keepdim=True)
        x = torch.matmul(x, x.transpose(-2, -1)) / x.shape[-1]
        x = x.unsqueeze(-3)

        # BiMap layers
        # x -> (batch, filters, channels, channels)
        x = torch.matmul(self.w1.transpose(-2, -1), torch.matmul(x, self.w1))
        x = self.re_eig(x)
        x = torch.matmul(self.w2.transpose(-2, -1), torch.matmul(x, self.w2))
        x = self.re_eig(x)
        x = torch.matmul(self.w3.transpose(-2, -1), torch.matmul(x, self.w3))

        # LogEig layer
        # eval (batch, filters, channels)
        # evec (batch, filters, channels, channels)
        eval, evec = torch.linalg.eigh(x)
        eval = torch.log(torch.clamp(eval, min=1e-6))
        x = torch.matmul(evec, torch.matmul(torch.diag_embed(eval), evec.transpose(-2, -1)))

        # Classifier
        x = x.reshape(batch, self.n_features)
        x = self.fc_out(x)
        x = torch.sigmoid(x)

        return x
