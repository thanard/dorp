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