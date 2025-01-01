import sys
sys.path.append("/home/vhsingh/Geo-UPSplus/Diff_FNO_Experiments")
import torch.nn.functional as F
import matplotlib.pyplot as plt
from timeit import default_timer
from utilities3 import *
from Adam import Adam
import os
from torch.utils.data import DataLoader, TensorDataset
import utils
from utils import read_mesh_coordinates, transform_coordinates, sigma_formation
from matplotlib.colors import Normalize
import wandb

def set_seed(seed):    
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)

torch.backends.cudnn.deterministic = True
set_seed(0)

Ntotal = 775
ntrain = 500
ntest = 225

batch_size = 20
learning_rate_fno = 0.001
learning_rate_iphi = 0.0001

epochs = 401

modes = 12
width = 32

################################################################
# load data and data normalization
################################################################
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

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)

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
    def __init__(self, modes1, modes2, width, in_channels, out_channels, is_mesh=True, s1=40, s2=40):
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

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.is_mesh = is_mesh
        self.s1 = s1
        self.s2 = s2

        self.fc0 = nn.Linear(in_channels, self.width)  # input channel is 3: (a(x, y), x, y)

        self.conv0 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2, s1, s2)
        self.conv1 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv4 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2, s1, s2)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        self.w3 = nn.Conv2d(self.width, self.width, 1)
        self.b0 = nn.Conv2d(2, self.width, 1)
        self.b1 = nn.Conv2d(2, self.width, 1)
        self.b2 = nn.Conv2d(2, self.width, 1)
        self.b3 = nn.Conv2d(2, self.width, 1)
        self.b4 = nn.Conv1d(2, self.width, 1)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, out_channels)

    def forward(self, u, code=None, x_in=None, x_out=None, iphi=None):
        # u (batch, Nx, d) the input value
        # code (batch, Nx, d) the input features
        # x_in (batch, Nx, 2) the input mesh (sampling mesh)
        # xi (batch, xi1, xi2, 2) the computational mesh (uniform)
        # x_in (batch, Nx, 2) the input mesh (query mesh)

        if self.is_mesh and x_in == None:
            x_in = u
        if self.is_mesh and x_out == None:
            x_out = u
        # print(u.shape)
        grid = self.get_grid([u.shape[0], self.s1, self.s2], u.device).permute(0, 3, 1, 2)
        # print(u.shape)
        u = self.fc0(u)
        u = u.permute(0, 2, 1)

        uc1 = self.conv0(u, x_in=x_in, iphi=iphi, code=code)
        uc3 = self.b0(grid)
        uc = uc1 + uc3
        uc = F.gelu(uc)

        uc1 = self.conv1(uc)
        uc2 = self.w1(uc)
        uc3 = self.b1(grid)
        uc = uc1 + uc2 + uc3
        uc = F.gelu(uc)

        uc1 = self.conv2(uc)
        uc2 = self.w2(uc)
        uc3 = self.b2(grid)
        uc = uc1 + uc2 + uc3
        uc = F.gelu(uc)

        uc1 = self.conv3(uc)
        uc2 = self.w3(uc)
        uc3 = self.b3(grid)
        uc = uc1 + uc2 + uc3
        uc = F.gelu(uc)

        u = self.conv4(uc, x_out=x_out, iphi=iphi, code=code)
        u3 = self.b4(x_out.permute(0, 2, 1))
        u = u + u3
        # print("The shape of u before the last permute",u.shape)
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

# Training and Evaluation

model = FNO2d(modes, modes, width, in_channels=4, out_channels=4).cuda()
# model = FNO2d(modes, modes, width, in_channels=4, out_channels=4)
model_iphi = IPHI().cuda()
# model_iphi = IPHI()
print(count_params(model), count_params(model_iphi))

optimizer_fno = Adam(model.parameters(), lr=learning_rate_fno, weight_decay=1e-4)
scheduler_fno = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_fno, T_max = 200)
optimizer_iphi = Adam(model_iphi.parameters(), lr=learning_rate_iphi, weight_decay=1e-4)
scheduler_iphi = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_iphi, T_max = 200)

wandb.init(
    project="Research",  # Replace with your project name
    entity="ved100-carnegie-mellon-university",
    name=f"Diffeomorphic_GeoFNO_{dataset_name}",  # Optional: A specific run name
    config={  # Add any configuration parameters you want to track
        "epochs": epochs,
        "learning_rate_fno": learning_rate_fno,
        "learning_rate_iphi": learning_rate_iphi,
        "optimizer_fno": Adam(model.parameters(), lr=learning_rate_fno, weight_decay=1e-4),
        "scheduler_fno": torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_fno, T_max = 200),
        "optimizer_iphi" : Adam(model_iphi.parameters(), lr=learning_rate_iphi, weight_decay=1e-4),
        "scheduler_iphi" : torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_iphi, T_max = 200)
    }
)

myloss = LpLoss(size_average=False)
N_sample = 1000
save_dir = './saved_models/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Model file paths
model_fno_path = os.path.join(save_dir, 'elas_og_fno.pth')
model_iphi_path = os.path.join(save_dir, 'elas_og_iphi.pth')
criterion = torch.nn.MSELoss()
# Lists to store losses for plotting later
train_losses = []
test_losses = []
for ep in range(epochs):
    model.train()
    t1 = default_timer()
    train_l2 = 0
    train_reg = 0
    for input_data, target in train_loader:
        input_data, target = input_data.cuda(), target.cuda()
        pde_b= input_data[:,:,:4]
        coords_b = input_data[:,:,4:]

        optimizer_fno.zero_grad()
        optimizer_iphi.zero_grad()
        out = model(u=pde_b, x_in=coords_b, x_out=coords_b, iphi=model_iphi)

        # loss = myloss(out.view(batch_size, -1), sigma.view(batch_size, -1))
        loss = criterion(out, target)
        loss.backward()

        optimizer_fno.step()
        optimizer_iphi.step()
        train_l2 += loss.item()

    scheduler_fno.step()
    scheduler_iphi.step()

    model.eval()
    test_l2 = 0.0
    with torch.no_grad():
        for input_data, target in test_loader:
            input_data, target = input_data.cuda(), target.cuda()
            pde_b= input_data[:,:,:4]
            coords_b = input_data[:,:,4:]
            out = model(u=pde_b, x_in=coords_b, x_out=coords_b, iphi=model_iphi)
            # test_l2 += myloss(out.view(batch_size, -1), sigma.view(batch_size, -1)).item()
            test_l2 += criterion(out, target).item()

    train_l2 /= ntrain
    test_l2 /= ntest
    train_losses.append(train_l2)
    test_losses.append(test_l2)

    # Log metrics to WandB
    wandb.log({
        "epoch": ep,
        "train_loss": train_l2,
        "test_loss": test_l2,
        "learning_rate_fno": scheduler_fno.get_last_lr()[0],
        "learning_rate_iphi": scheduler_iphi.get_last_lr()[0]
    })

    t2 = default_timer()
    print(ep, t2 - t1, train_l2, test_l2)

    if ep%50==0:
        # torch.save(model, '../model/elas_v2_'+str(ep))
        # torch.save(model_iphi, '../model/elas_v2_iphi_'+str(ep))
        checkpoint_path = os.path.join(save_dir, f"models_epoch_og_{ep}.pth")
        torch.save({
            'model_fno_state_dict': model.state_dict(),
            'model_iphi_state_dict': model_iphi.state_dict(),
            'optimizer_fno_state_dict': optimizer_fno.state_dict(),
            'optimizer_iphi_state_dict': optimizer_iphi.state_dict(),
            'scheduler_fno_state_dict': scheduler_fno.state_dict(),
            'scheduler_iphi_state_dict': scheduler_iphi.state_dict(),
            'epoch': ep
        }, checkpoint_path)
        wandb.save(checkpoint_path)
        # Flatten spatial data for scatter plot
        # Channel names for better readability
        channel_names = ["Density", "Horizontal Velocity", "Vertical Velocity", "Pressure"]

        # Flatten spatial data for scatter plot
        XY_flat = coords_b.reshape(-1, 2).detach().cpu().numpy()  # Flatten XY to [num_points, 2]
        truth_flat = target.reshape(-1, 4).detach().cpu().numpy()   # Flatten truth to [num_points, 4]
        pred_flat = out.reshape(-1, 4).detach().cpu().numpy()   # Flatten pred to [num_points, 4]

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
        wandb.log({"Epoch Images": wandb.Image(fig, caption=f"Epoch {ep}")})
        plt.close(fig)


# Saving final models
torch.save(model.state_dict(), model_fno_path)
torch.save(model_iphi.state_dict(), model_iphi_path)

# Plotting the training and test losses
plt.figure(figsize=(10, 6))
plt.plot(range(epochs), train_losses, label='Training Loss')
plt.plot(range(epochs), test_losses, label='Test Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Test Loss Over Epochs')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(save_dir, 'loss_plot.png'))
plt.show()