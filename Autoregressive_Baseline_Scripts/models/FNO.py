import torch
import torch.nn as nn
import torch.nn.functional as F
# from netCDF4 import Dataset
import yaml
with open("./config/config.yaml", "r") as f:
    config = yaml.safe_load(f)

class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-2), x.size(-1) // 2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        # Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x

class FNO(nn.Module):
    def __init__(self, input_dim, output_dim, modes1, modes2, width, n_layers, retrain_fno, device=None, padding_frac=1/4):
        super(FNO, self).__init__()
        """
        The model expects:
          x[..., 6] => mask with 1=valid, 0=hole
        """
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.n_layers = n_layers
        self.retrain_fno = retrain_fno
        self.output_channels = output_dim

        torch.manual_seed(self.retrain_fno)
        self.padding_frac = padding_frac

        # Infer input channel distribution
        self.in_channels = input_dim  # 7 for CE, 5 for NS
        if self.in_channels == 7:
            self.in_channels_physical = 4
            self.in_channels_coords = 2
        elif self.in_channels == 6:
            self.in_channels_physical = 5
            self.in_channels_coords = 2
        else:
            raise ValueError(f"Unsupported in_channels: {self.in_channels}. Expected 5 or 7.")

        # Compute total input channels for fc0
        total_input = self.in_channels_physical + self.in_channels_coords

        # We'll lift (rho, Ux, Uy, P, x, y) => 6 channels
        self.fc0 = nn.Linear(total_input, self.width)

        self.conv_list = nn.ModuleList(
            [nn.Conv2d(self.width, self.width, 1) for _ in range(self.n_layers)]
        )
        self.spectral_list = nn.ModuleList(
            [SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
             for _ in range(self.n_layers)]
        )
        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, self.output_channels)
        self.to(device)

    def forward(self, x):
        """
        x shape: (batch, H, W, 7)
          channels = [rho, Ux, Uy, P, xcoord, ycoord, mask]
        The mask is 1=valid, 0=hole (no flipping here).
        """
        mask = x[..., -1]  # shape (B,H,W)
        physical = x[..., :self.in_channels_physical]  # (B,H,W,4)
        coords   = x[..., self.in_channels_physical:self.in_channels_physical + self.in_channels_coords] # (B,H,W,2)

        # zero out holes
        physical = physical * mask.unsqueeze(-1)

        # combine physical + coords => 6 channels
        x_in = torch.cat([physical, coords], dim=-1)
        x_in = self.fc0(x_in)
        x_in = x_in.permute(0, 3, 1, 2)

        # padding
        pad_x = int(round(x_in.shape[-1] * self.padding_frac))
        pad_y = int(round(x_in.shape[-2] * self.padding_frac))
        x_in = F.pad(x_in, [0, pad_x, 0, pad_y])

        # Fourier layers
        for k, (s, c) in enumerate(zip(self.spectral_list, self.conv_list)):
            x1 = s(x_in)
            x2 = c(x_in)
            x_in = x1 + x2
            if k != self.n_layers - 1:
                x_in = F.gelu(x_in)
        x_in = x_in[..., :-pad_x, :-pad_y]

        x_in = x_in.permute(0, 2, 3, 1)
        x_in = self.fc1(x_in)
        x_in = F.gelu(x_in)
        out = self.fc2(x_in)  # shape (B,H,W,4)

        # re-apply mask
        out = out * mask.unsqueeze(-1)
        return out