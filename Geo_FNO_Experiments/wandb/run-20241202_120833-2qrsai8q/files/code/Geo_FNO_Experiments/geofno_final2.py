"""
script adopted from Zongyi Li's work on Geo-FNO
"""
import torch
from torch.utils.data import DataLoader, TensorDataset
from netCDF4 import Dataset
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from timeit import default_timer
from utilities3 import *
from Adam import Adam
import wandb
import os
# from torch.optim import Adam


torch.manual_seed(0)
np.random.seed(0)
torch.cuda.manual_seed(0)
torch.backends.cudnn.deterministic = True

################################################################
# 3d fourier layers
################################################################

class SpectralConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, modes3):
        super(SpectralConv3d, self).__init__()

        """
        3D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2
        self.modes3 = modes3

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3,
                                    dtype=torch.cfloat))
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3,
                                    dtype=torch.cfloat))
        self.weights3 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3,
                                    dtype=torch.cfloat))
        self.weights4 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3,
                                    dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul3d(self, input, weights):
        # (batch, in_channel, x,y,t ), (in_channel, out_channel, x,y,t) -> (batch, out_channel, x,y,t)
        return torch.einsum("bixyz,ioxyz->boxyz", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        # print("input shape to spectral forward:", x.shape)
        x_ft = torch.fft.rfftn(x, dim=[-3, -2, -1])
        # print("x_ft shape (after FFT):", x_ft.shape)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-3), x.size(-2), x.size(-1) // 2 + 1,
                             dtype=torch.cfloat, device=x.device)
        # print("out_ft shape (before mode selection):", out_ft.shape)
        out_ft[:, :, :self.modes1, :self.modes2, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, :self.modes1, :self.modes2, :self.modes3], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, -self.modes1:, :self.modes2, :self.modes3], self.weights2)
        out_ft[:, :, :self.modes1, -self.modes2:, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, :self.modes1, -self.modes2:, :self.modes3], self.weights3)
        out_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3], self.weights4)

        # Return to physical space
        x = torch.fft.irfftn(out_ft, s=(x.size(-3), x.size(-2), x.size(-1)))
        # print("Output shape (after inverse FFT)", x.shape)
        return x


class FNO3d(nn.Module):
    def __init__(self, modes1, modes2, modes3, width):
        super(FNO3d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .

        input: the solution of the first 10 timesteps + 3 locations (u(1, x, y), ..., u(10, x, y),  x, y, t). It's a constant function in time, except for the last index.
        input shape: (batchsize, x=64, y=64, t=40, c=13)
        output: the solution of the next 40 timesteps
        output shape: (batchsize, x=64, y=64, t=40, c=1)
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        self.width = width
        self.padding = 5  # pad the domain if input is non-periodic
        self.fc0 = nn.Linear(5, self.width) # 3 velocity channels + 2 positional encodings
        # input channel is 12: the solution of the first 10 timesteps + 3 locations (u(1, x, y), ..., u(10, x, y),  x, y, t)

        self.conv0 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv1 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv2 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv3 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.w0 = nn.Conv3d(self.width, self.width, 1)
        self.w1 = nn.Conv3d(self.width, self.width, 1)
        self.w2 = nn.Conv3d(self.width, self.width, 1)
        self.w3 = nn.Conv3d(self.width, self.width, 1)
        self.bn0 = torch.nn.BatchNorm3d(self.width)
        self.bn1 = torch.nn.BatchNorm3d(self.width)
        self.bn2 = torch.nn.BatchNorm3d(self.width)
        self.bn3 = torch.nn.BatchNorm3d(self.width)

        self.fc1 = nn.Linear(self.width, 128)
        # self.fc2 = nn.Linear(128, 3)
        self.fc2 = nn.Linear(128, 2)  # Output 33 features (3 channels * 11 timesteps)

    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        # grid = grid.unsqueeze(3).repeat(1, 1, 1, x.size(3), 1) #To add timesteps to grid
        # print("Shape of x before concatenation:", x.shape)
        # print("Shape of grid:", grid.shape)
        x = torch.cat((x, grid), dim=-1)
        # print("Shape of x after concatenation", x.shape)
        x = self.fc0(x)
        # print("Shape of x after fc0:", x.shape)
        x = x.permute(0, 4, 1, 2, 3)
        x = F.pad(x, [0, self.padding, 0, self.padding, 0, self.padding])  # pad the domain if input is non-periodic

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2
        # print("Shape after final convolution:", x.shape)

        x = x[..., :-self.padding, :-self.padding, :-self.padding]
        x = x.permute(0, 2, 3, 4, 1)  # pad the domain if input is non-periodic
        x = self.fc1(x)
        # print("Shaper after fc1:",x.shape)
        x = F.gelu(x)
        # print("Shape after gelu activateion on fc1:",x.shape)
        x = self.fc2(x)
        print("Shape after fc2 (model output):", x.shape)
        x = x.view(x.size(0), x.size(1), x.size(2), -1, 2)
        print("Shape after reshaping after the final fc2 layer", x.shape)
        return x

    def get_grid(self, shape, device):
        batchsize, size_x, size_y, size_z = shape[0], shape[1], shape[2], shape[3]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1, 1).repeat([batchsize, 1, size_y, size_z, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1, 1).repeat([batchsize, size_x, 1, size_z, 1])
        gridz = torch.tensor(np.linspace(0, 1, size_z), dtype=torch.float)
        gridz = gridz.reshape(1, 1, 1, size_z, 1).repeat([batchsize, size_x, size_y, 1, 1])
        return torch.cat((gridx, gridy, gridz), dim=-1).to(device)

################################################################
# configs
################################################################
DATA_PATH = '/home/namancho/datasets/FNS-KF-Poseidon/solution_0.nc'
dataset_name = os.path.basename(os.path.dirname(DATA_PATH))  # This gives the folder name like 'NS-PwC'
output_folder = dataset_name
# output_folder = "NS-Sines-Poseidon-2-trial"

# Create the directory if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

N = 1177 #987
ntrain = 900
ntest = 277 #80

batch_size = 20
learning_rate = 0.001

epochs = 401
step_size = 100
gamma = 0.5

modes = 20
width = 32
out_dim = 2 #4, changed to 3 as we have 3 channels.

s1 = 128 #101, x dimension match
s2 = 128 #31, y dimension match
t = 21

r1 = 1
r2 = 1
# s1 = int(((s1 - 1) / r1) + 1)
# s2 = int(((s2 - 1) / r2) + 1)

################################################################
# load data and data normalization
################################################################

#open the .nc file nad read the data
with Dataset(DATA_PATH, mode='r') as nc:
    velocity = nc.variables['solution'][:] #Shape: (samples, time, channels, x, y) velocity for NS, solution for FNS-KF

x_train = velocity[:ntrain, :10, :2, :, :] #Input first 10 time steps for training
print("Shape of xtrain:", x_train.shape)
y_train = velocity[:ntrain, 10:20, :2, :, :] #Output all remaining time steps for training
print("Shape of ytrain:", y_train.shape)

x_test = velocity[-ntest:, :10, :2, :, :] #input first 10 time steps for testing
y_test = velocity[-ntest:, 10:20, :2, :, :] #output remanining time steps for testing

#Permuting axes to match (batch, x, y, time, channels)
print("x_train shape before transpose:", x_train.shape)
x_train = np.transpose(x_train, (0, 3, 4, 1, 2))
print("xtrain shape after transpose", x_train.shape)
y_train = np.transpose(y_train, (0, 3, 4, 1, 2))
print("x_test shape before transpose:", x_test.shape)
x_test = np.transpose(x_test, (0, 3, 4, 1, 2))
y_test = np.transpose(y_test, (0, 3, 4, 1, 2))

#converting to Pytorch tensors
x_train = torch.tensor(x_train, dtype = torch.float32)
y_train = torch.tensor(y_train, dtype = torch.float32)
x_test = torch.tensor(x_test, dtype = torch.float32)
y_test = torch.tensor(y_test, dtype = torch.float32)

print(x_train.shape, y_train.shape)

#Create dataloader objects
train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=False)

#Previous implementation
# reader = MatReader(DATA_PATH)
# x_train = reader.read_field('input')[:ntrain, ::r1][:, :s1].reshape(ntrain,s1,1,1,1).repeat(1,1,s2,t,1)
# y_train = reader.read_field('output')[:ntrain, ::r1, ::r2][:, :s1, :s2]
# reader.load_file(DATA_PATH)
# x_test = reader.read_field('input')[-ntest:, ::r1][:, :s1].reshape(ntest,s1,1,1,1).repeat(1,1,s2,t,1)
# y_test = reader.read_field('output')[-ntest:, ::r1, ::r2][:, :s1, :s2]
# print(x_train.shape, y_train.shape)

# train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), batch_size=batch_size,
#                                            shuffle=True)
# test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=batch_size,
#                                           shuffle=False)

#Wandb Logging
#######################################################
# Initialize wandb
wandb.init(
    project="GeoFNO1",  # Your personal project name
    entity="namancho",  # Replace with your WandB username
    name=f"{dataset_name}_{1}_geofno",  # Optional, gives each run a unique name
    config={  # Optional configuration logging
        "learning_rate": learning_rate,
        "epochs": epochs,
        "batch_size": batch_size,
        "modes": modes,
        "width": width,
        "output_dim": out_dim
    }
)

################################################################
# training and evaluation
################################################################
model = FNO3d(modes, modes, 8, width).cuda()
# model = torch.load('../model/plas_101'+str(500))
print(count_params(model))
optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
myloss = LpLoss(size_average=False, p=2)


for ep in range(epochs):
    model.train()
    t1 = default_timer()
    train_l2 = 0
    train_reg = 0
    for x, y in train_loader:
        x, y = x.cuda(), y.cuda()

        optimizer.zero_grad()
        out = model(x)
        # print("Model output shape before reshape:", out.shape)
        # print("Output numel:", out.numel())
        # print("Expected numel:", batch_size * s1 * s2 * 10 * out_dim)

        # out = out.reshape(batch_size, s1, s2, t-10, out_dim)

        # Debug shapes and values
        # print(f"Epoch {ep}, Batch:")
        # print(f"Prediction shape: {out.shape}, Ground truth shape: {y.shape}")
        # print(f"Prediction sample: {out[0, :5, :5, 0, :]}")
        # print(f"Ground truth sample: {y[0, :5, :5, 0, :]}")

        loss = myloss(out.view(batch_size, -1), y.view(batch_size, -1))
        # print(f"Loss: {loss.item()}")
        loss.backward()

        train_l2 += loss.item()

    
    # Debug weight norms
    # for name, param in model.named_parameters():
    #     print(f"Weight norm {name}: {param.data.norm()}")
    optimizer.step()
    scheduler.step()
    model.eval()
    test_l2 = 0.0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.cuda(), y.cuda()

            # out = model(x).reshape(batch_size, s1, s2, t, out_dim)
            out = model(x)
            test_l2 += myloss(out.view(batch_size, -1), y.view(batch_size, -1)).item()

    train_l2 /= ntrain
    train_reg /= ntrain
    test_l2 /= ntest

    t2 = default_timer()
    print("OUTPUT", ep, t2 - t1, train_l2, train_reg, test_l2)

    # Log metrics to wandb
    wandb.log({
        "epoch": ep,
        "train_loss": train_l2,
        "test_loss": test_l2,
        "learning_rate": scheduler.get_last_lr()[0]  # Log current learning rate
    })
    
    if ep % 50 == 0:
        truth = y[0].squeeze().detach().cpu().numpy()
        pred = out[0].squeeze().detach().cpu().numpy()

        fig, ax = plt.subplots(4, 5, figsize=(20, 18))  # 4 rows: 2 for each channel (truth + prediction)
        time_indices = [0, 2, 4, 6, 8]  # Example timesteps to visualize
        channels = ['Horizontal Velocity (u)', 'Vertical Velocity (v)']

        # Compute global vmin and vmax for each variable
        vmin_u = truth[:, :, :, 0].min()
        vmax_u = truth[:, :, :, 0].max()
        pred_u = pred[:, :, :, 0].min()
        predx_u = pred[:, :, :, 0].max()
        vmin_v = truth[:, :, :, 1].min()
        vmax_v = truth[:, :, :, 1].max()
        pred_v = pred[:, :, :, 1].min()
        predx_v = pred[:, :, :, 1].max()

        # Loop over channels and plot with consistent scales
        for ch in range(2):  # Loop over channels (0 for horizontal, 1 for vertical)
            vmin = vmin_u if ch == 0 else vmin_v
            vmax = vmax_u if ch == 0 else vmax_v
            pred_min = pred_u if ch == 0 else pred_v
            pred_max = predx_u if ch ==0 else predx_v

            for i, t in enumerate(time_indices):
                # Plot truth for current channel
                im1 = ax[2 * ch, i].imshow(truth[:, :, t, ch], cmap='gist_ncar', vmin=vmin, vmax=vmax)
                ax[2 * ch, i].set_title(f"Truth - {channels[ch]} - Time {t+10}")
                ax[2 * ch, i].axis('off')
                fig.colorbar(im1, ax=ax[2 * ch, i], orientation='vertical', fraction=0.046, pad=0.04)

                # Plot prediction for current channel
                im2 = ax[2 * ch + 1, i].imshow(pred[:, :, t, ch], cmap='gist_ncar', vmin=pred_min, vmax=pred_max)
                ax[2 * ch + 1, i].set_title(f"Prediction - {channels[ch]} - Time {t+10}")
                ax[2 * ch + 1, i].axis('off')
                fig.colorbar(im2, ax=ax[2 * ch + 1, i], orientation='vertical', fraction=0.046, pad=0.04)

        # Add loss as text to the figure
        train_loss_text = f"Train Loss: {train_l2:.4f}"
        test_loss_text = f"Test Loss: {test_l2:.4f}"
        fig.text(0.5, 0.01, train_loss_text + " | " + test_loss_text, ha='center', fontsize=12)

        plt.tight_layout()
        fig.savefig(os.path.join(output_folder, f"output_epoch_{ep}.png"))
        plt.close(fig)  # Close the figure to avoid memory issues

wandb.finish()