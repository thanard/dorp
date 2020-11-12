import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.gen_utils import *
from utils.dataset import *

class CPC(nn.Module):
    def __init__(self,
                 encoder,
                 z_dim,
                 batch_size,
                 in_channels=3,
                 temp=1,
                 mode='single_encoder',
                 input_dim=0,
                 W_form=0,
                 num_filters=32,
                 num_onehots=2,
                 separate_W=0,
                 num_layers=3,
                 normalization=None,
                 n_key_onehots=1, # only used for key encoders
                 n_agent_onehots=1, # only used for key encoders
                 alpha=1, # tried 10, 100
                 ):

        super(CPC, self).__init__()
        self.num_onehots = num_onehots
        if encoder == 'cnn':

            self.encoder = FactoredEncoder(input_dim,
                                           in_channels=in_channels,
                                           out_onehots=num_onehots,
                                           z_dim=z_dim,
                                           num_filters=num_filters,
                                           mode=mode,
                                           temp=temp)
        elif encoder == 'cswm':
            self.encoder = CSWM(input_dim,
                                in_channels=in_channels,
                                num_filters=num_filters,
                                z_dim=z_dim,
                                out_onehots=num_onehots,
                                mode=mode,
                                temp=temp,
                                num_layers=num_layers,
                                normalization=normalization)
                                # circular_padding=True,
                                # downsampling_by=4,
        elif encoder == 'cswm-scaled-down':
            conv_seq = ['same']*num_layers
            conv_seq.append('half')
            conv_seq.append('half')
            conv_seq = '-'.join(conv_seq)
            self.encoder = CSWMCircular(input_dim,
                                in_channels=in_channels,
                                num_filters=num_filters,
                                z_dim=z_dim,
                                out_onehots=num_onehots,
                                mode=mode,
                                temp=temp,
                                num_layers=num_layers,
                                normalization=normalization,
                                conv_seq=conv_seq)
        elif encoder == 'cswm-scaled-down-first':
            conv_seq = ['same']*num_layers
            conv_seq.insert(0, 'half')
            conv_seq.insert(0, 'half')
            conv_seq = '-'.join(conv_seq)
            self.encoder = CSWMCircular(input_dim,
                                in_channels=in_channels,
                                num_filters=num_filters,
                                z_dim=z_dim,
                                out_onehots=num_onehots,
                                mode=mode,
                                temp=temp,
                                num_layers=num_layers,
                                normalization=normalization,
                                conv_seq=conv_seq)
        elif encoder == 'cswm-gt':
            self.encoder = CSWM(input_dim,
                                in_channels=in_channels,
                                num_filters=num_filters,
                                z_dim=z_dim,
                                out_onehots=num_onehots,
                                mode=mode,
                                temp=temp,
                                num_layers=num_layers,
                                normalization=normalization,
                                gt_extractor=True)
        elif encoder == 'cswm-key':
            self.num_onehots = n_key_onehots + n_agent_onehots
            self.encoder = CSWMKey(input_dim,
                                in_channels=in_channels,
                                num_filters=num_filters,
                                z_dim=z_dim,
                                out_key_onehots=n_key_onehots,
                                out_agent_onehots=n_agent_onehots,
                                mode=mode,
                                temp=temp,
                                num_layers=num_layers,
                                normalization=normalization)
        else:
            raise NotImplementedError("Encoder not recognized: {}".format(encoder))

        self.encoder_type = encoder
        self.num_layers = num_layers
        self.alpha = alpha
        self.encoder_form = encoder
        self.batch_size = batch_size
        self.mode = mode
        self.z_dim = z_dim
        self.W_form = W_form
        self.separate_W = separate_W

        self.k = nn.Parameter(torch.tensor(1.))
        self.W = nn.Parameter(torch.rand(z_dim * self.num_onehots, z_dim * self.num_onehots))
        self.I = torch.eye(z_dim * self.num_onehots).cuda()

    def encode(self, x, vis=False, continuous=False):
        if vis:
            res = self.encoder.vis(x)
            return res
        else:
            x = self.encoder(x, continuous)
            return x

    def get_W(self):
        if self.W_form == 0: # random W matrix
            W = self.W
        elif self.W_form == 1: # identity matrix
            W = self.I
        elif self.W_form == 3:
            if self.separate_W:
                base_submatrix = 2 * torch.eye(self.z_dim) - torch.ones(self.z_dim, self.z_dim)
                base = torch.zeros(self.z_dim*self.num_onehots, self.z_dim*self.num_onehots)
                for i in range(self.num_onehots):
                    base[self.z_dim*i:self.z_dim*i+self.z_dim, self.z_dim*i:self.z_dim*i+self.z_dim] = base_submatrix
            else:
                base = 2 * torch.eye(self.z_dim * self.num_onehots) - torch.ones(self.z_dim * self.num_onehots,
                                                                                 self.z_dim * self.num_onehots)
            W = torch.exp(self.k) * ((torch.sigmoid(self.W) + self.alpha*self.I) * base.cuda())
        else:
            raise NotImplementedError("W form %d not used" % self.W_form)
        return W

    def log_density(self, x_next, z):
        assert x_next.size(0) == z.size(0)  # batch sizes must match
        if self.mode == 'double_encoder':
            z_next = self.encode(x_next, continuous=True)
        elif self.mode == 'single_encoder' or self.mode == 'continuous':
            z_next = self.encode(x_next)
        else:
            raise NotImplementedError("Mode not recognized: {}".format(self.mode))

        z_next = z_next.view(z_next.size(0), -1)
        z = z.view(z.size(0), -1)
        z = z.unsqueeze(2)  # bs x z_dim x 1
        z_next = z_next.unsqueeze(2)
        W = self.get_W()
        w = W.repeat(z.size(0), 1, 1)
        f_out = torch.bmm(torch.bmm(z_next.permute(0, 2, 1), w), z)
        f_out = f_out.squeeze()

        return f_out

    def compute_logits(self, z_a, z_pos, ce_temp=1.):
        """
        Uses logits trick from CURL:
        - compute (B,B) matrix z_a (W z_pos.T)
        - positives are all diagonal elements
        - negatives are all other elements
        - to compute loss use multiclass cross entropy with identity matrix for labels
        """
        assert z_a.size(0) == z_pos.size(0)
        z_pos = z_pos.reshape(z_pos.size(0), -1)
        z_a = z_a.reshape(z_a.size(0), -1)
        W = self.get_W()
        Wz = torch.matmul(W, z_pos.T)  # (z_dim,B)
        logits = torch.matmul(z_a, Wz)  # (B,B)
        logits = logits - torch.max(logits, 1)[0][:, None]
        return logits/ce_temp


    def energy(self, state, next_state, sigma=.5):
        """Energy function based on normalized squared L2 norm."""

        norm = 0.5 / (sigma ** 2)
        diff = state - next_state
        return norm * diff.pow(2).sum(2).mean(1)

    def forward(self, *input):
        return self.log_density(*input)

class FactoredEncoder(nn.Module):
    def __init__(self, input_dim, out_onehots=2, in_channels=2, z_dim=8, num_filters=32, mode='single_encoder', temp=1):
        super(FactoredEncoder, self).__init__()
        self.z_dim = z_dim
        self.num_filters = num_filters
        self.temp = temp
        self.mode = mode
        self.input_dim = input_dim
        self.out_onehots = out_onehots

        self._conv_1 = nn.Conv2d(in_channels=in_channels,
                                 out_channels=num_filters,
                                 kernel_size=3,
                                 stride=1)

        self._conv_2 = nn.Conv2d(in_channels=num_filters,
                                 out_channels=num_filters,
                                 kernel_size=1,
                                 stride=1)

        self._conv_3 = nn.Conv2d(in_channels=num_filters,
                                 out_channels=num_filters,
                                 kernel_size=1,
                                 stride=1)

        self.h_dim = input_dim - 2
        self.fc = nn.Linear(num_filters * self.h_dim * self.h_dim, z_dim * out_onehots)
        self.ln = nn.LayerNorm(z_dim)

    def vis(self, inputs):
        x = inputs
        x = F.relu(self._conv_1(x))
        x = F.relu(self._conv_2(x))
        x = F.relu(self._conv_3(x))
        x = x.reshape(-1, self.num_filters * self.h_dim * self.h_dim)
        x = self.fc(x)
        x = x.view(-1, self.out_onehots, self.z_dim)
        x = self.ln(x)
        x = torch.argmax(x, dim=2)
        return x

    def forward(self, inputs, continuous=False):
        x = inputs
        x = F.relu(self._conv_1(x))
        x = F.relu(self._conv_2(x))
        x = F.relu(self._conv_3(x))
        x = x.reshape(-1, self.num_filters * self.h_dim * self.h_dim)
        x = self.fc(x)
        x = x.view(-1, self.out_onehots, self.z_dim)

        if self.mode == 'continuous':
            return x

        elif self.mode == 'single_encoder':
            x = self.ln(x)
            x = F.gumbel_softmax(x, tau=self.temp, hard=True)
            return x
        elif self.mode == 'double_encoder':
            if continuous:
                x = F.softmax(x, dim=2)
                return x
            else:
                x = self.ln(x)
                x = F.gumbel_softmax(x, tau=self.temp, hard=True)
                return x

        return x

class CircularConv2d(nn.Module):
    def __init__(self, size, in_channels, out_channels, circular_padding=False):
        super(CircularConv2d, self).__init__()
        self.circular_padding = circular_padding
        if size == 'same':
            self.layer = nn.Conv2d(in_channels=in_channels,
                  out_channels=out_channels,
                  kernel_size=3,
                  padding=0) if circular_padding \
                else nn.Conv2d(in_channels=in_channels,
                      out_channels=out_channels,
                      kernel_size=3,
                      padding=1)

        elif size == 'half':
            self.layer =  nn.Conv2d(in_channels=in_channels,
                          out_channels=out_channels,
                          kernel_size=4,
                          stride=2,
                          padding=0) if circular_padding \
            else nn.Conv2d(in_channels=in_channels,
                      out_channels=out_channels,
                      kernel_size=4,
                      stride=2,
                      padding=1)

    def pad_circular(self, x, pad):
        """
        :param x: shape [H, W]
        :param pad: int >= 0
        :return:
        """
        x = torch.cat([x, x[:, :, 0:pad]], dim=2)
        x = torch.cat([x, x[:, :, :, 0:pad]], dim=3)
        x = torch.cat([x[:, :, -2 * pad:-pad], x], dim=2)
        x = torch.cat([x[:, :, :, -2 * pad:-pad], x], dim=3)

        return x

    def forward(self, x):
        if self.circular_padding:
            return self.layer(self.pad_circular(x, 1))
        return self.layer(x)

class CSWM(nn.Module):
    def __init__(self, input_dim, out_onehots, in_channels, z_dim, num_filters, mode='single_encoder', temp=1,
                 num_layers=3, normalization="none", gt_extractor=False):
        super(CSWM, self).__init__()
        self.z_dim = z_dim
        self.num_filters = num_filters
        self.temp = temp
        self.mode = mode
        self.input_dim = input_dim
        self.hdim = input_dim
        self.out_onehots = out_onehots
        self.in_channels = 3
        self.num_layers = num_layers
        self.normalization = normalization
        self.gt_extractor = gt_extractor

        self.ln = nn.LayerNorm(self.z_dim) # normalize before gumbel softmax

        # Object Extractor CNN
        self.object_extractor = nn.ModuleList()
        self.object_extractor.append(nn.Conv2d(in_channels=self.in_channels,
                                             out_channels=num_filters,
                                             kernel_size=3,
                                             padding=1))
        self.object_extractor.append(self.get_norm_layer(normalization))
        self.object_extractor.append(nn.ReLU())
        for i in range(self.num_layers):
            self.object_extractor.append(nn.Conv2d(in_channels=num_filters,
                                                 out_channels=num_filters,
                                                 kernel_size=3,
                                                 padding=1))
            self.object_extractor.append(self.get_norm_layer(normalization))
            self.object_extractor.append(nn.ReLU())
        self.object_extractor.append(nn.Conv2d(in_channels=num_filters,
                                             out_channels=self.out_onehots,
                                             kernel_size=3,
                                             padding=1))
        self.object_extractor.append(nn.Sigmoid())

        # Object Encoder MLP
        self.num_hiddens = num_filters * 16
        self.object_encoder = nn.ModuleList([
            nn.Linear(self.hdim * self.hdim, self.num_hiddens),
            nn.ReLU(),
            nn.Linear(self.num_hiddens, self.num_hiddens),
            nn.LayerNorm(self.num_hiddens),
            nn.ReLU(),
            nn.Linear(self.num_hiddens, self.z_dim),
        ])

    def get_norm_layer(self, normalization):
        if normalization == 'batchnorm':
            return nn.BatchNorm2d(self.num_filters)
        elif normalization == 'layernorm':
            return nn.LayerNorm([self.num_filters, self.hdim, self.hdim])
        elif normalization == "none":
            return nn.Identity()
        else:
            raise NotImplementedError("normalization type not recognized: %s" % normalization)
        return

    def expand_input(self, input):
        return input.repeat_interleave(4, dim=3).repeat_interleave(4, dim=2)

    def conv_forward(self, inputs):
        batch_size = inputs.size(0)
        x = inputs*10
        # if inputs.size(2) == 16:
        #     x = self.expand_input(x)
        if self.gt_extractor:
            attn_maps = x
        else:
            for layer in self.object_extractor:
                x = layer(x)
            attn_maps = x
        x = x.view(batch_size, self.out_onehots, -1)
        for layer in self.object_encoder:
            x = layer(x)
        return x, attn_maps

    def vis(self, inputs):
        x, attn_maps = self.conv_forward(inputs)
        x = torch.argmax(x, dim=2)
        return x

    def get_attn_map_reg(self, attn_maps):
        reg = 0
        for m_k in attn_maps:
            reg += torch.mean(torch.min(m_k**2, (1-m_k)**2))
        return -reg

    def forward(self, inputs, continuous=False):
        x, attn_maps = self.conv_forward(inputs)
        reg = self.get_attn_map_reg(attn_maps)
        if self.mode == 'continuous':
            return x, reg, attn_maps
        elif self.mode == 'single_encoder':
            x = self.ln(x)
            x = F.gumbel_softmax(x, dim=2, tau=self.temp, hard=True)
            return x, reg, attn_maps
        elif self.mode == 'double_encoder':
            if continuous:
                x = F.softmax(x, dim=2)
                return x, reg, attn_maps
            else:
                x = self.ln(x)
                x = F.gumbel_softmax(x, dim=2, tau=self.temp, hard=True)
                return x, reg, attn_maps
        else:
            raise NotImplementedError


class CSWMCircular(nn.Module): # more expressive attention module directly on input
    def __init__(self, input_dim, out_onehots, in_channels, z_dim, num_filters, mode='single_encoder', temp=1,
                 num_layers=2, normalization="none", gt_extractor=False, conv_seq='same', circular_padding=False, downsampling_by=1):
        super(CSWM, self).__init__()
        self.z_dim = z_dim
        self.num_filters = num_filters
        self.temp = temp
        self.mode = mode
        self.input_dim = input_dim
        self.out_onehots = out_onehots
        self.in_channels = 3
        self.num_layers = num_layers
        self.normalization = normalization
        self.gt_extractor = gt_extractor
        self.conv_seq = conv_seq.split('-')
        if len(self.conv_seq) == 1:
            self.conv_seq = self.conv_seq*(2+num_layers)
        else:
            assert len(self.conv_seq) == 2+num_layers

        self.ln = nn.LayerNorm(self.z_dim) # normalize before gumbel softmax

        self.downsampling_by = downsampling_by

        # Object Extractor CNN
        self.object_extractor = nn.ModuleList()
        if downsampling_by > 1:
            self.object_extractor.append(nn.AvgPool2d(
                downsampling_by, downsampling_by
            ))
        self.object_extractor.append(CircularConv2d(self.conv_seq[0], self.in_channels, num_filters, circular_padding))
        self.object_extractor.append(self.get_norm_layer(normalization))
        self.object_extractor.append(nn.ReLU())
        for i in range(self.num_layers):
            self.object_extractor.append(CircularConv2d(self.conv_seq[i+1], num_filters, num_filters, circular_padding))
            self.object_extractor.append(self.get_norm_layer(normalization))
            self.object_extractor.append(nn.ReLU())
        self.object_extractor.append(CircularConv2d(self.conv_seq[-1], num_filters, self.out_onehots, circular_padding))
        self.object_extractor.append(nn.Sigmoid())

        # Get h_dim
        self.hdim = self.get_h_dim()

        # Object Encoder MLP
        self.num_hiddens = num_filters * 16
        self.object_encoder = nn.ModuleList([
            nn.Linear(self.hdim * self.hdim, self.num_hiddens),
            nn.ReLU(),
            nn.Linear(self.num_hiddens, self.num_hiddens),
            nn.LayerNorm(self.num_hiddens),
            nn.ReLU(),
            nn.Linear(self.num_hiddens, self.z_dim),
        ])

    def get_h_dim(self):
        size = self.input_dim / self.downsampling_by
        for conv_size in self.conv_seq:
            if conv_size == 'half':
                size /= 2
        return int(size)
        # return self.object_extractor(torch.zeros((1, self.in_channels, self.input_dim, self.input_dim))).shape[-1]

    def get_norm_layer(self, normalization):
        if normalization == 'batchnorm':
            return nn.BatchNorm2d(self.num_filters)
        elif normalization == 'layernorm':
            return nn.LayerNorm([self.num_filters, self.hdim, self.hdim])
        elif normalization == "none":
            return nn.Identity()
        else:
            raise NotImplementedError("normalization type not recognized: %s" % normalization)
        return

    def expand_input(self, input):
        return input.repeat_interleave(4, dim=3).repeat_interleave(4, dim=2)

    def conv_forward(self, inputs):
        batch_size = inputs.size(0)
        x = inputs * 10
        # if inputs.size(2) == 16:
        #     x = self.expand_input(x)
        if self.gt_extractor:
            attn_maps = x
        else:

            for layer in self.object_extractor:
                x = layer(x)
            attn_maps = x
        x = x.view(batch_size, self.out_onehots, -1)
        for layer in self.object_encoder:
            x = layer(x)
        return x, attn_maps

    def vis(self, inputs):
        x, attn_maps = self.conv_forward(inputs)
        x = torch.argmax(x, dim=2)
        return x

    def get_attn_map_reg(self, attn_maps):
        reg = 0
        for m_k in attn_maps:
            reg += torch.mean(torch.min(m_k**2, (1-m_k)**2))
        return -reg

    def forward(self, inputs, continuous=False):
        x, attn_maps = self.conv_forward(inputs)
        reg = self.get_attn_map_reg(attn_maps)
        if self.mode == 'continuous':
            return x, reg, attn_maps
        elif self.mode == 'single_encoder':
            x = self.ln(x)
            x = F.gumbel_softmax(x, dim=2, tau=self.temp, hard=True)
            return x, reg, attn_maps
        elif self.mode == 'double_encoder':
            if continuous:
                x = F.softmax(x, dim=2)
                return x, reg, attn_maps
            else:
                x = self.ln(x)
                x = F.gumbel_softmax(x, dim=2, tau=self.temp, hard=True)
                return x, reg, attn_maps
        else:
            raise NotImplementedError

class CSWMKey(nn.Module):
    def __init__(self, input_dim, in_channels, z_dim, num_filters=32, out_agent_onehots=1, out_key_onehots=1,  mode='single_encoder', temp=1,
                 num_layers=3, normalization="none"):
        super(CSWMKey, self).__init__()
        self.z_dim = z_dim
        self.num_filters = num_filters
        self.temp = temp
        self.mode = mode
        self.input_dim = input_dim
        self.hdim = input_dim
        self.out_key_onehots = out_key_onehots
        self.out_agent_onehots = out_agent_onehots
        self.in_channels = in_channels
        self.num_layers = num_layers
        self.normalization = normalization
        self.out_onehots = self.out_agent_onehots + self.out_key_onehots

        self.ln = nn.LayerNorm(self.z_dim) # normalize before gumbel softmax
        self.ln_k = nn.LayerNorm(2) # for key output

        # Object Extractor CNN
        self.object_extractor = nn.ModuleList()
        self.object_extractor.append(nn.Conv2d(in_channels=self.in_channels,
                                             out_channels=num_filters,
                                             kernel_size=3,
                                             padding=1))
        self.object_extractor.append(self.get_norm_layer(normalization))
        self.object_extractor.append(nn.ReLU())
        for i in range(self.num_layers):
            self.object_extractor.append(nn.Conv2d(in_channels=num_filters,
                                                 out_channels=num_filters,
                                                 kernel_size=3,
                                                 padding=1))
            self.object_extractor.append(self.get_norm_layer(normalization))
            self.object_extractor.append(nn.ReLU())
        self.object_extractor.append(nn.Conv2d(in_channels=num_filters,
                                             out_channels=self.out_agent_onehots+self.out_key_onehots,
                                             kernel_size=3,
                                             padding=1))
        self.object_extractor.append(nn.Sigmoid())


        # Object Encoder MLP
        self.num_hiddens = num_filters * 16
        self.object_encoder_agent = nn.ModuleList([
            nn.Linear(self.hdim * self.hdim, self.num_hiddens),
            nn.ReLU(),
            nn.Linear(self.num_hiddens, self.num_hiddens),
            nn.LayerNorm(self.num_hiddens),
            nn.ReLU(),
            nn.Linear(self.num_hiddens, self.z_dim),
        ])

        self.object_encoder_key = nn.ModuleList([
            nn.Linear(self.hdim * self.hdim, self.num_hiddens),
            nn.ReLU(),
            nn.Linear(self.num_hiddens, self.num_hiddens),
            nn.LayerNorm(self.num_hiddens),
            nn.ReLU(),
            nn.Linear(self.num_hiddens, 2),]) # binary for key

    def get_norm_layer(self, normalization):
        if normalization == 'batchnorm':
            return nn.BatchNorm2d(self.num_filters)
        elif normalization == 'layernorm':
            return nn.LayerNorm([self.num_filters, self.hdim, self.hdim])
        elif normalization == "none":
            return nn.Identity()
        else:
            raise NotImplementedError("normalization type not recognized: %s" % normalization)

    def expand_input(self, input):
        return input.repeat_interleave(4, dim=3).repeat_interleave(4, dim=2)

    def conv_forward(self, inputs):
        batch_size = inputs.size(0)
        x = inputs*10
        for layer in self.object_extractor:
            x = layer(x)
        attn_maps = x
        x = x.reshape(batch_size, self.out_agent_onehots+self.out_key_onehots, -1).contiguous()
        x_a = x[:, :self.out_agent_onehots, :]
        x_k = x[:, self.out_agent_onehots:, :]
        for layer in self.object_encoder_agent:
            x_a = layer(x_a)

        for layer in self.object_encoder_key:
            x_k = layer(x_k)

        return x_a, x_k, attn_maps

    def vis(self, inputs):
        x_a, x_k, attn_maps = self.conv_forward(inputs)
        x_a = torch.argmax(x_a, dim=2)
        x_k = torch.argmax(x_k, dim=2)
        x = torch.cat((x_a, x_k), dim=1)  # [batch_size, out_onehots, z_dim]
        return x

    def forward(self, inputs, continuous=False):
        batch_size = inputs.size(0)
        x_a, x_k, attn_maps = self.conv_forward(inputs)
        _ = None
        if self.mode == 'continuous':
            key_padded = torch.zeros((batch_size, self.out_key_onehots, self.z_dim)).cuda()
            key_padded[:, :, :2] = x_k
            x = torch.cat((x_a, key_padded), dim=1)  # [batch_size, out_onehots, z_dim]
            return x, _, attn_maps
        elif self.mode == 'single_encoder':
            x_a = self.ln(x_a)
            x_a = F.gumbel_softmax(x_a, dim=2, tau=self.temp, hard=True)
            x_k = F.softmax(x_k, dim=2)
            key_padded = torch.zeros_like(x_a)
            key_padded[:, :, :2] = x_k
            x = torch.cat((x_a, key_padded), dim=1)  # [batch_size, out_onehots, z_dim]
            return x_a, x_k, attn_maps
        elif self.mode == 'double_encoder':
            if continuous:
                x_a = F.softmax(x_a, dim=2)
                x_k = F.softmax(x_k, dim=2)
                key_padded = torch.zeros((batch_size, self.out_key_onehots, self.z_dim)).cuda()
                key_padded[:, :, :2] = x_k
                x = torch.cat((x_a, key_padded), dim=1)  # [batch_size, out_onehots, z_dim]
                return x, _, attn_maps
            else:
                x_a = self.ln(x_a)
                x_a = F.gumbel_softmax(x_a, dim=2, tau=self.temp, hard=True)
                # x_k = self.ln_k(x_k)
                # x_k = F.gumbel_softmax(x_k, dim=2, tau=self.temp, hard=True)
                x_k = F.softmax(x_k, dim=2)
                key_padded = torch.zeros((batch_size, self.out_key_onehots, self.z_dim)).cuda()
                key_padded[:, :, :2] = x_k
                x = torch.cat((x_a, key_padded), dim=1)  # [batch_size, out_onehots, z_dim]
                return x, _, attn_maps
        else:
            raise NotImplementedError

def get_hinge_loss(model, state, next_state, hinge=1):
    batch_size = state.size(0)
    perm = np.random.permutation(batch_size)
    neg_state = state[perm]

    pos_loss = model.energy(state, next_state)
    zeros = torch.zeros_like(pos_loss)
    pos_loss = pos_loss.mean()

    neg_loss = torch.max(
        zeros, hinge - model.energy(
            state, neg_state)).mean()

    loss = pos_loss + neg_loss
    return loss

def get_loss(loss_form, model, z_a, z_pos, ce_temp=1.):
    if loss_form == 'ce':
        CE = nn.CrossEntropyLoss()
        logits = model.compute_logits(z_a, z_pos, ce_temp)
        labels = torch.arange(logits.shape[0]).long().cuda()
        return CE(logits, labels)
    elif loss_form == 'hinge':
        return get_hinge_loss(model, z_a, z_pos)
    else:
        raise NotImplementedError("Loss form not recognized: " + loss_form)

class init_weights_func(object):
    def __init__(self, scale_factor=1.):
        self.scale_factor=scale_factor

    def __call__(self, m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            torch.nn.init.xavier_uniform_(m.weight, self.scale_factor)

def process_batch(model, sample_ims, single=False):
    '''
    Computes and returns forward pass of CPC model for a batch of processed images
    :param model: CPC model
    :param sample_ims: batch of input images (any length)
    :return: np array of z outputs [sample_size, model.z_dim]
    '''
    if single:
        return model.encode(np_to_var(sample_ims).unsqueeze(0).permute(0, 3, 1, 2), vis=True).squeeze(
            0).cpu().numpy()
    max_batch_size = 64
    idx = 0
    z_labels = []
    while idx < len(sample_ims):
        zs = model.encode(np_to_var(sample_ims[idx:idx + max_batch_size]).permute(0, 3, 1, 2),
                          vis=True).cpu().numpy()
        z_labels.append(zs)
        idx += max_batch_size
    return np.concatenate(z_labels)

# def process_batch_from_vp_grid(model, sample_ims, single=False):
#     # images from vp model are tensors and rgb,
#     # so they do not need to be preprocessed as in process_batch
#     if single:
#         return model.encode(sample_ims.unsqueeze(0), vis=True).squeeze(0).cpu().numpy()
#     max_batch_size = 128
#     idx = 0
#     z_labels = []
#     while idx < len(sample_ims):
#         zs = model.encode(sample_ims[idx:idx + max_batch_size], vis=True).cpu().numpy()
#         z_labels.append(zs)
#         idx += max_batch_size
#     return np.concatenate(z_labels)

def get_labels(model, im):
    '''
    returns labels for model.encode(im), converted from onehot to int
    '''
    z_labels = process_batch(model, im)
    zs = tensor_to_label_array(z_labels, model.num_onehots, model.z_dim)
    return zs

def get_labels_uncompressed(model, im):
    '''
    returns onehot labels model.encode(im)
    '''
    z_labels = process_batch(model, im)
    return z_labels

def get_hamming_dists_samples(env, model, n_batches=64, batch_size=512):
    distances = np.array([])
    for batch in range(n_batches):
        data = get_sample_transitions(env, batch_size, 2)
        anchors, positives = data[:, 0], data[:, 1]
        anchors = np_to_var(np.transpose(anchors, (0, 3, 1, 2)))
        positives = np_to_var(np.transpose(positives, (0, 3, 1, 2)))
        z = model.encode(anchors, vis=True).cpu().numpy()
        z_next = model.encode(positives, vis=True).cpu().numpy()
        distances_batch = np.sum((z != z_next), axis=1)
        if not distances.any():
            distances = distances_batch
        else:
            distances = np.concatenate((distances, distances_batch))
    print("hamming dist shape", distances.shape)
    return distances