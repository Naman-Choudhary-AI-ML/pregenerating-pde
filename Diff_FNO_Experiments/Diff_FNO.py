import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import wandb
from torch.optim.lr_scheduler import StepLR
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from netCDF4 import Dataset
from utils import read_mesh_coordinates, transform_coordinates, sigma_formation
from matplotlib.colors import Normalize

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")

PATH_Sigma = '/home/vhsingh/Geo-UPSplus/Diff_FNO_Experiments/CE-CRP_openfoam/CE-RP_Openfoam_Irregular.npy'
PATH_XY = '/home/vhsingh/Geo-UPSplus/Diff_FNO_Experiments/CE-CRP_openfoam/C'
dataset_name = os.path.basename(os.path.dirname(PATH_Sigma))
XY = read_mesh_coordinates(PATH_XY)
XY = XY[:,:2]

X, Y = sigma_formation(PATH_Sigma)
sigma = np.load(PATH_Sigma)
coords_expanded = XY[np.newaxis, np.newaxis, :, :]  # shape: (1, 1, 16320, 2)
# print(coords_expanded.shape)
xy_data = np.tile(coords_expanded, (sigma.shape[0], sigma.shape[1], 1, 1))
print(xy_data.shape)
print(sigma.shape)
xy_torch = torch.tensor(xy_data, dtype=torch.float)
X_expanded = np.concatenate([sigma, xy_data], axis=-1)

input = X_expanded[:, 0, :, :]
target = X_expanded[:, 1, :, :4]

X_torch = torch.tensor(input, dtype=torch.float)         # shape (20000, 16320, 6)
Y_torch = torch.tensor(target, dtype=torch.float)         # shape (20000, 16320, 4)

print(xy_torch.shape, X_torch.shape, Y_torch.shape)

n_samples = X_torch.shape[0]
n_train = int(0.8 * n_samples)
n_test = n_samples - n_train

# Option 1: a simple split
train_X = X_torch[:n_train]
train_xy = xy_torch[:n_train]
train_Y = Y_torch[:n_train]

test_X = X_torch[n_train:]
test_xy = xy_torch[n_train:]
test_Y = Y_torch[n_train:]

train_dataset = TensorDataset(train_X, train_Y)
test_dataset  = TensorDataset(test_X, test_Y)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader  = DataLoader(test_dataset,  batch_size=32, shuffle=False)

################################################################
#  2d fourier layer
################################################################
class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, s1=32, s2=32):
        super(SpectralConv2d, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2
        self.s1 = s1
        self.s2 = s2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, u, x_in=None, x_out=None, iphi=None, code=None):
        batchsize = u.shape[0]

        # Compute Fourier coeffcients up to factor of e^(- something constant)
        if x_in == None:
            u_ft = torch.fft.rfft2(u)
            s1 = u.size(-2)
            s2 = u.size(-1)
        else:
            u_ft = self.fft2d(u, x_in, iphi, code)
            s1 = self.s1
            s2 = self.s2

        # Multiply relevant Fourier modes
        # print(u.shape, u_ft.shape)
        factor1 = self.compl_mul2d(u_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        factor2 = self.compl_mul2d(u_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        # Return to physical space
        if x_out == None:
            out_ft = torch.zeros(batchsize, self.out_channels, s1, s2 // 2 + 1, dtype=torch.cfloat, device=u.device)
            out_ft[:, :, :self.modes1, :self.modes2] = factor1
            out_ft[:, :, -self.modes1:, :self.modes2] = factor2
            u = torch.fft.irfft2(out_ft, s=(s1, s2))
        else:
            out_ft = torch.cat([factor1, factor2], dim=-2)
            u = self.ifft2d(out_ft, x_out, iphi, code)

        return u

    def fft2d(self, u, x_in, iphi=None, code=None):
        # u (batch, channels, n)
        # x_in (batch, n, 2) locations in [0,1]*[0,1]
        # iphi: function: x_in -> x_c

        batchsize = x_in.shape[0]
        N = x_in.shape[1]
        device = x_in.device
        m1 = 2 * self.modes1
        m2 = 2 * self.modes2 - 1

        # wavenumber (m1, m2)
        k_x1 =  torch.cat((torch.arange(start=0, end=self.modes1, step=1), \
                            torch.arange(start=-(self.modes1), end=0, step=1)), 0).reshape(m1,1).repeat(1,m2).to(device)
        k_x2 =  torch.cat((torch.arange(start=0, end=self.modes2, step=1), \
                            torch.arange(start=-(self.modes2-1), end=0, step=1)), 0).reshape(1,m2).repeat(m1,1).to(device)

        # print(x_in.shape)
        if iphi == None:
            x = x_in
        else:
            x = iphi(x_in, code)

        # print(x.shape)
        # K = <y, k_x>,  (batch, N, m1, m2)
        K1 = torch.outer(x[...,0].view(-1), k_x1.view(-1)).reshape(batchsize, N, m1, m2)
        K2 = torch.outer(x[...,1].view(-1), k_x2.view(-1)).reshape(batchsize, N, m1, m2)
        K = K1 + K2

        # basis (batch, N, m1, m2)
        basis = torch.exp(-1j * 2 * np.pi * K).to(device)

        # Y (batch, channels, N)
        u = u + 0j
        Y = torch.einsum("bcn,bnxy->bcxy", u, basis)
        return Y

    def ifft2d(self, u_ft, x_out, iphi=None, code=None):
        # u_ft (batch, channels, kmax, kmax)
        # x_out (batch, N, 2) locations in [0,1]*[0,1]
        # iphi: function: x_out -> x_c

        batchsize = x_out.shape[0]
        N = x_out.shape[1]
        device = x_out.device
        m1 = 2 * self.modes1
        m2 = 2 * self.modes2 - 1

        # wavenumber (m1, m2)
        k_x1 =  torch.cat((torch.arange(start=0, end=self.modes1, step=1), \
                            torch.arange(start=-(self.modes1), end=0, step=1)), 0).reshape(m1,1).repeat(1,m2).to(device)
        k_x2 =  torch.cat((torch.arange(start=0, end=self.modes2, step=1), \
                            torch.arange(start=-(self.modes2-1), end=0, step=1)), 0).reshape(1,m2).repeat(m1,1).to(device)

        if iphi == None:
            x = x_out
        else:
            x = iphi(x_out, code)

        # K = <y, k_x>,  (batch, N, m1, m2)
        K1 = torch.outer(x[:,:,0].view(-1), k_x1.view(-1)).reshape(batchsize, N, m1, m2)
        K2 = torch.outer(x[:,:,1].view(-1), k_x2.view(-1)).reshape(batchsize, N, m1, m2)
        K = K1 + K2

        # basis (batch, N, m1, m2)
        basis = torch.exp(1j * 2 * np.pi * K).to(device)

        # coeff (batch, channels, m1, m2)
        u_ft2 = u_ft[..., 1:].flip(-1, -2).conj()
        u_ft = torch.cat([u_ft, u_ft2], dim=-1)

        # Y (batch, channels, N)
        Y = torch.einsum("bcxy,bnxy->bcn", u_ft, basis)
        Y = Y.real
        return Y

class FNO2d(nn.Module):
    def __init__(self, fno_architecture, device=None, padding_frac=1 / 4):
        super(FNO2d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .

        input: the solution of the coefficient function and locations (a(x, y), x, y)
        input shape: (batchsize, x=s, y=s, c=3)
        output: the solution
        output shape: (batchsize, x=s, y=s, c=1)
        """
        self.modes1 = fno_architecture["modes"]
        self.modes2 = fno_architecture["modes"]
        self.width = fno_architecture["width"]
        self.n_layers = fno_architecture["n_layers"]
        self.retrain_fno = fno_architecture["retrain_fno"]
        self.is_mesh = fno_architecture["is_mesh"]
        self.s1 = fno_architecture["s1"]
        self.s2 = fno_architecture["s2"]

        torch.manual_seed(self.retrain_fno)
        # self.padding = 9 # pad the domain if input is non-periodic
        self.padding_frac = padding_frac
        self.fc0 = nn.Linear(4, self.width)  # input channel is 3: (a(x, y), x, y)

        self.conv_list = nn.ModuleList(
            [nn.Conv2d(self.width, self.width, 1) for _ in range(self.n_layers)])
        self.spectral_list = nn.ModuleList(
            [SpectralConv2d(self.width, self.width, self.modes1, self.modes2, self.s1, self.s2) for _ in range(self.n_layers)])
        self.b_list = nn.ModuleList(
            [nn.Conv2d(2, self.width, 1) for _ in range(self.n_layers - 1)]
        )
        self.b4 = nn.Conv1d(2, self.width, 1)
        

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 4)

        self.to(device)

    def forward(self, u, code = None, x_in = None, x_out = None, iphi=None):
        if self.is_mesh and x_in == None:
            x_in = u
        if self.is_mesh and x_out == None:
            x_out = u
        grid = self.get_grid([u.shape[0], self.s1, self.s2], u.device).permute(0, 3, 1, 2)
        # print("The shape of x before fc0", u.shape)
        u = self.fc0(u)
        # print("The shape of x after fc0 is:", u.shape)
        # x = x.permute(0, 3, 1, 2)
        u = u.permute(0, 2, 1)

        # x1_padding = int(round(x.shape[-1] * self.padding_frac))
        # x2_padding = int(round(x.shape[-2] * self.padding_frac))
        # x = F.pad(x, [0, x1_padding, 0, x2_padding])

        for k, (s, c, b) in enumerate(zip(self.spectral_list, self.conv_list, self.b_list)):
            if k == 0:
                # print(f"Before layer {k}, shape of u = {u.shape}")
                x1 = s(u, x_in=x_in,iphi=iphi,code=code)
                x3 = b(grid)
                uc = x1 + x3
                # print(f"After layer {k}, shape of u = {uc.shape}")
            elif k == 3:
                # print(f"Before layer {k}, shape of u = {u.shape}")
                u = s(uc, x_out=x_out, iphi=iphi, code=code)
                x3 = self.b4(x_out.permute(0, 2, 1))
                u = u + x3
                # print(f"After layer {k}, shape of u = {u.shape}")
            else:
                # print(f"Before layer {k}, shape of u = {u.shape}")
                x1 = s(uc)
                x2 = c(uc)
                x3 = b(grid)
                uc = x1 + x2 + x3
                # print(f"After layer {k}, shape of u = {uc.shape}")
            if k != self.n_layers - 1:
                uc = F.gelu(uc)
        # x = x[..., :-x1_padding, :-x2_padding]
        # print(u.shape)
        u = u.permute(0, 2, 1)
        u = self.fc1(u)
        u = F.gelu(u)
        u = self.fc2(u)
        return u
    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)

class IPHI(nn.Module):
    def __init__(self, width=32):
        super(IPHI, self).__init__()
        """
        inverse phi: x -> xi
        """
        self.width = width
        self.fc0 = nn.Linear(4, self.width)
        self.fc_code = nn.Linear(42, self.width)
        self.fc_no_code = nn.Linear(3*self.width, 4*self.width)
        self.fc1 = nn.Linear(4*self.width, 4*self.width)
        self.fc2 = nn.Linear(4*self.width, 4*self.width)
        self.fc3 = nn.Linear(4*self.width, 2)
        self.center = torch.tensor([0.5,0.5], device="cuda").reshape(1,1,2)
        # self.center = torch.tensor([0.5,0.5]).reshape(1,1,2)

        self.B = np.pi*torch.pow(2, torch.arange(0, self.width//4, dtype=torch.float, device="cuda")).reshape(1,1,1,self.width//4)
        # self.B = np.pi*torch.pow(2, torch.arange(0, self.width//4, dtype=torch.float)).reshape(1,1,1,self.width//4)

    def forward(self, x, code=None):
        # x (batch, N_grid, 2)
        # code (batch, N_features)
        # some feature engineering
        angle = torch.atan2(x[:,:,1] - self.center[:,:, 1], x[:,:,0] - self.center[:,:, 0])
        radius = torch.norm(x - self.center, dim=-1, p=2)
        xd = torch.stack([x[:,:,0], x[:,:,1], angle, radius], dim=-1)

        # sin features from NeRF
        b, n, d = xd.shape[0], xd.shape[1], xd.shape[2]
        x_sin = torch.sin(self.B * xd.view(b,n,d,1)).view(b,n,d*self.width//4)
        x_cos = torch.cos(self.B * xd.view(b,n,d,1)).view(b,n,d*self.width//4)
        xd = self.fc0(xd)
        xd = torch.cat([xd, x_sin, x_cos], dim=-1).reshape(b,n,3*self.width)

        if code!= None:
            cd = self.fc_code(code)
            cd = cd.unsqueeze(1).repeat(1,xd.shape[1],1)
            xd = torch.cat([cd,xd],dim=-1)
        else:
            xd = self.fc_no_code(xd)

        xd = self.fc1(xd)
        xd = F.gelu(xd)
        xd = self.fc2(xd)
        xd = F.gelu(xd)
        xd = self.fc3(xd)
        return x + x * xd


#Wandb Logging
#######################################################
# Initialize wandb
wandb.init(
    project="Research",  # Your personal project name
    entity="ved100-carnegie-mellon-university",  # Replace with your WandB username
    name=f"Diffeomorphic-FNO_{dataset_name}",  # Optional, gives each run a unique name
    config={  # Optional configuration logging
        "learning_rate": 0.001,
        "epochs": 401,
        "batch_size": 32,
        "modes": 12,
        "width": 32,
        "output_dim": 2,
        "n_layers": 4,
        "scheduler_step_size": 100,
        "scheduler_gamma": 0.5
    }
)

# Model architecture and optimizer setup

fno_architecture = {
    "modes": wandb.config.modes,
    "width": wandb.config.width,
    "n_layers": wandb.config.n_layers,
    "retrain_fno": 42,
    "is_mesh": True,
    "s1": 40,
    "s2": 40,
}
model = FNO2d(fno_architecture, device=device).to(device)
model_iphi = IPHI().cuda()
# model_iphi = IPHI()
optimizer_fno = optim.Adam(model.parameters(), lr=wandb.config.learning_rate, weight_decay=1e-4)
scheduler_fno = StepLR(optimizer_fno, step_size=wandb.config.scheduler_step_size, gamma=wandb.config.scheduler_gamma)
optimizer_iphi = optim.Adam(model.parameters(), lr=wandb.config.learning_rate, weight_decay=1e-4)
scheduler_iphi = StepLR(optimizer_fno, step_size=wandb.config.scheduler_step_size, gamma=wandb.config.scheduler_gamma)
criterion = torch.nn.MSELoss()
output_folder = './output_folder/'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Training loop
epochs = wandb.config.epochs
# output_folder = "./output_images"
# os.makedirs(output_folder, exist_ok=True)

for epoch in range(1, epochs + 1):
    model.train()
    train_loss = 0.0
    for input_data, target in train_loader:
        input_data, target = input_data.cuda(), target.cuda()
        pde_b = input_data[:,:,:4]
        coords_b = input_data[:,:,4:]
        optimizer_fno.zero_grad()
        optimizer_iphi.zero_grad()
        outputs = model(u=pde_b,x_in=coords_b, x_out=coords_b, iphi=model_iphi)

        loss = criterion(outputs, target)  # L2 loss
        loss.backward()
        optimizer_fno.step()
        optimizer_iphi.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)

    # Evaluate
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for input_data, target in test_loader:
            input_data, target = input_data.cuda(), target.cuda()
            pde_b = input_data[:,:,:4]
            coords_b = input_data[:,:,4:]
            outputs = model(u=pde_b, x_in=coords_b, x_out=coords_b, iphi=model_iphi)

            loss = criterion(outputs, target)  # L2 loss
            test_loss += loss.item()

        test_loss /= len(test_loader)

    # Scheduler step
    scheduler_fno.step()
    scheduler_iphi.step()

    # Log metrics to wandb
    wandb.log({
        "epoch": epoch-1,
        "train_loss": train_loss,
        "test_loss": test_loss,
        "learning_rate_fno": scheduler_fno.get_last_lr()[0],  # Log current learning rate
        "learning_rate_iphi": scheduler_iphi.get_last_lr()[0]
    })
    print(f"Output:", epoch, train_loss, test_loss)

    # Save prediction plots every 50 epochs
    if (epoch-1) % 50 == 0:
        model.eval()

        checkpoint_path = os.path.join(output_folder, f"models_epoch_og_{epoch}.pth")
        torch.save({
            'model_fno_state_dict': model.state_dict(),
            'model_iphi_state_dict': model_iphi.state_dict(),
            'optimizer_fno_state_dict': optimizer_fno.state_dict(),
            'optimizer_iphi_state_dict': optimizer_iphi.state_dict(),
            'scheduler_fno_state_dict': scheduler_fno.state_dict(),
            'scheduler_iphi_state_dict': scheduler_iphi.state_dict(),
            'epoch': epoch
        }, checkpoint_path)
        wandb.save(checkpoint_path)
        # Flatten spatial data for scatter plot
        channel_names = ["Density", "Horizontal Velocity", "Vertical Velocity", "Pressure"]

        # Flatten spatial data for scatter plot
        XY_flat = coords_b.reshape(-1, 2).detach().cpu().numpy()  # Flatten XY to [num_points, 2]
        truth_flat = target.reshape(-1, 4).detach().cpu().numpy()   # Flatten truth to [num_points, 4]
        pred_flat = outputs.reshape(-1, 4).detach().cpu().numpy()   # Flatten pred to [num_points, 4]

        # Number of channels
        num_channels = truth_flat.shape[1]

        # Global normalization using min/max across all channels and both truth/prediction
        # global_min = min(truth_flat.min(), pred_flat.min())
        # global_max = max(truth_flat.max(), pred_flat.max())
        global_min = truth_flat.min()
        global_max = truth_flat.max()
        norm = Normalize(vmin=global_min, vmax=global_max)

        fig, axes = plt.subplots(nrows=num_channels, ncols=3, figsize=(15, 5 * num_channels))

        for i in range(num_channels):
            # Extract channel data
            truth_channel = truth_flat[:, i]
            pred_channel = pred_flat[:, i]
            diff_channel = truth_channel - pred_channel

            # Normalize using the global min/max for consistent coloring
            truth_normalized = norm(truth_channel)
            pred_normalized = norm(pred_channel)
            diff_normalized = norm(diff_channel)

            # Truth scatter plot
            scatter = axes[i, 0].scatter(XY_flat[:, 0], XY_flat[:, 1], c=truth_normalized, s=10, cmap='RdBu_r', edgecolor='w', lw=0.1)
            axes[i, 0].set_title(f"Truth ({channel_names[i]})")
            axes[i, 0].set_xlabel('X')
            axes[i, 0].set_ylabel('Y')
            fig.colorbar(scatter, ax=axes[i, 0])

            # Prediction scatter plot
            scatter = axes[i, 1].scatter(XY_flat[:, 0], XY_flat[:, 1], c=pred_normalized, s=10, cmap='RdBu_r', edgecolor='w', lw=0.1)
            axes[i, 1].set_title(f"Prediction ({channel_names[i]})")
            axes[i, 1].set_xlabel('X')
            fig.colorbar(scatter, ax=axes[i, 1])

            # Difference scatter plot
            scatter = axes[i, 2].scatter(XY_flat[:, 0], XY_flat[:, 1], c=diff_normalized, s=10, cmap='RdBu_r', edgecolor='w', lw=0.1)
            axes[i, 2].set_title(f"Difference ({channel_names[i]})")
            axes[i, 2].set_xlabel('X')
            fig.colorbar(scatter, ax=axes[i, 2])

        # Adjust layout
        plt.tight_layout()
        wandb.log({"Epoch Images": wandb.Image(fig, caption=f"Epoch {epoch}")})
        plt.close(fig)

wandb.finish()