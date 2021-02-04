

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
        self.in_channels = in_channels
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

    def conv_forward(self, inputs):
        batch_size = inputs.size(0)
        x = inputs*10
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
        # reg = self.get_attn_map_reg(attn_maps)
        if self.mode == 'continuous':
            x_pred = torch.argmax(x, dim=2)
            return x, x_pred, attn_maps
        elif self.mode == 'single_encoder':
            x = self.ln(x)
            x_pred = torch.argmax(x, dim=2)
            x = F.gumbel_softmax(x, dim=2, tau=self.temp, hard=True)
            return x, x_pred, attn_maps
        elif self.mode == 'double_encoder':
            if continuous:
                x_pred = torch.argmax(x, dim=2)
                x = F.softmax(x, dim=2)
                return x, x_pred, attn_maps
            else:
                x = self.ln(x)
                x_pred = torch.argmax(x, dim=2)
                x = F.gumbel_softmax(x, dim=2, tau=self.temp, hard=True)
                return x, x_pred, attn_maps
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

class CSWMKeyV2(nn.Module):
    '''
    Same as CSWMKey but switches the order of channel input to the object encoder
    '''
    def __init__(self, input_dim, in_channels, z_dim, num_filters=32, out_agent_onehots=1, out_key_onehots=1,  mode='single_encoder', temp=1,
                 num_layers=3, normalization="none", scope=0):
        super(CSWMKeyV2, self).__init__()
        self.z_dim = z_dim
        self.num_filters = num_filters
        self.temp = temp
        self.mode = mode
        self.input_dim = input_dim
        self.hdim = input_dim
        self.out_key_onehots = out_key_onehots
        self.out_agent_onehots = out_agent_onehots
        self.scope = scope
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
        x_k = x[:, :self.out_key_onehots, :]
        x_a = x[:, self.out_key_onehots:, :]
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
            # x_k = self.ln_k(x_k)
            # x_k = F.gumbel_softmax(x_k, dim=2, tau=self.temp, hard=True)
            x_k = F.softmax(x_k, dim=2)
            # key_padded = torch.zeros_like(x_a)
            # key_padded[:, :, :2] = x_k
            # x = torch.cat((x_a, key_padded), dim=1)  # [batch_size, out_onehots, z_dim]
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
