import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from DataLoaders.load_utils import _load_dataset
from DataLoaders.CNO_TimeLoaders import NSFlowTimeDataset
from torch_utils.debug_tools import format_tensor_size
from torch.utils.data import DataLoader
# from netCDF4 import Dataset
import yaml
# with open("/home/vhsingh/Geo-UPSplus/Autoregressive_Baseline_Scripts/config/config.yaml", "r") as f:
#     config = yaml.safe_load(f)

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

class FNO_time(pl.LightningModule):
    def __init__(self, in_dim, out_dim, modes1, modes2, width, n_layers, retrain_fno, padding_frac=1/4, lr=0.001, weight_decay=0.0, loader_dictionary = dict()):
        super(FNO_time, self).__init__()
        self.save_hyperparameters()  # Save hyperparameters for logging and checkpointing
        
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.n_layers = n_layers
        self.retrain_fno = retrain_fno
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.padding_frac = padding_frac
        self.lr = lr
        self.weight_decay = weight_decay
        self.loader_dictionary = loader_dictionary
        self.validation_errs  = dict()
        self.validation_times = dict()
        self.batch_size = 8
        if ("separate" in self.loader_dictionary) and self.loader_dictionary["separate"]:
            assert "separate_dim" in self.loader_dictionary
            print(self.loader_dictionary["separate_dim"], "separate_dim")
            self.validation_errs_sep = dict()
            self.test_errs_sep = dict()
        
        self.best_val_loss_mean = 1000
        self.best_val_loss_median = 1000
        self.best_val_loss_mean_last = 1000
        self.best_val_loss_median_last = 1000

        # Set manual seed for reproducibility
        torch.manual_seed(self.retrain_fno)
        
        # Infer channel distribution based on input_dim
        # self.in_dim = in_dim  # e.g., 7 for CE, 6 for NS
        # if self.in_dim == 6:
        #     self.in_channels_physical = 5
        #     self.in_channels_coords = 2
        # elif self.in_dim == 7:
        #     self.in_channels_physical = 6
        #     self.in_channels_coords = 2
        # elif self.in_dim == 4:
        #     self.in_channels_physical = 3
        #     self.in_channels_coords = 2
        # else:
        #     raise ValueError(f"Unsupported in_channels: {self.in_dim}.")

        # total_input = self.in_channels_physical + self.in_channels_coords
        # Lift input into the desired channel space
        self.fc0 = nn.Linear(8, self.width)
        
        # Define lists of Fourier and standard convolution layers
        self.conv_list = nn.ModuleList([nn.Conv2d(self.width, self.width, 1) for _ in range(self.n_layers)])
        self.spectral_list = nn.ModuleList(
            [SpectralConv2d(self.width, self.width, self.modes1, self.modes2) for _ in range(self.n_layers)]
        )
        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, self.out_dim)

    def forward(self, x):
        """
        x: (batch, H, W, channels)
        Channels = [physical variables, x_coord, y_coord, mask]
        """
        if x.dim() == 4:
            x = x.permute(0, 2, 3, 1)
        mask = x[..., -1]  # (B, H, W)
        physical = x[..., :6]
        coords = x[..., 6:8]
        
        # Zero out holes using the mask
        physical = physical * mask.unsqueeze(-1)
        # Concatenate physical variables with coordinates
        x_in = torch.cat([physical, coords], dim=-1)
        x_in = self.fc0(x_in)
        x_in = x_in.permute(0, 3, 1, 2)  # (B, C, H, W)
        
        # Apply padding
        pad_x = int(round(x_in.shape[-1] * self.padding_frac))
        pad_y = int(round(x_in.shape[-2] * self.padding_frac))
        x_in = F.pad(x_in, [0, pad_x, 0, pad_y])
        
        # Apply the Fourier and standard convolution layers
        for k, (spec_conv, conv) in enumerate(zip(self.spectral_list, self.conv_list)):
            x1 = spec_conv(x_in)
            x2 = conv(x_in)
            x_in = x1 + x2
            if k != self.n_layers - 1:
                x_in = F.gelu(x_in)
        
        # Remove padding
        x_in = x_in[..., :-pad_x, :-pad_y]
        x_in = x_in.permute(0, 2, 3, 1)  # Back to (B, H, W, C)
        x_in = self.fc1(x_in)
        x_in = F.gelu(x_in)
        out = self.fc2(x_in)
        
        # Re-apply mask
        out = out * mask.unsqueeze(-1)
        return out

    def training_step(self, batch):
        
        # Are the physical quantities separated in the loss function?
        is_separate = ("separate" in self.loader_dictionary) and self.loader_dictionary["separate"] and "separate_dim" in self.loader_dictionary
                
        # What kind of separation do we use?
        if is_separate:
            separate_dim = self.loader_dictionary["separate_dim"]
            assert type(separate_dim) is list             
        
        #---------
        # Are we interested in all the channels or we want to predict just a few of them and ignore others?
        #---------
        if "is_masked" in self.loader_dictionary:
            is_masked = self.loader_dictionary["is_masked"] is not None
        else:
            is_masked = False
        
        if not is_masked:
            t_batch, input_batch, output_batch = batch
        else:
            # Relevant dim tells us what channels we need to care about (it's a mask)
            t_batch, input_batch, output_batch, masked_dim = batch
        
        # Predict:
        output_pred_batch = self(input_batch)
        #---------
        
        # If airfoil, mask it
        which = self.loader_dictionary["which"]
        if "airfoil" in which:
            output_pred_batch[input_batch==1] = 1.0
            output_batch[input_batch==1] = 1.0
            
        #---------------
        # Compute the loss - Relative L1
        #---------------
        
        if not is_separate:
            if output_batch.dim() == 4 and output_batch.shape[1] in [3, 9]:
                output_batch = output_batch.permute(0, 2, 3, 1)
            loss = nn.L1Loss()(output_batch, output_pred_batch) / nn.L1Loss()(torch.zeros_like(output_batch), output_batch)
 
        else:
            # How are the variables separated?
            diff = [0, separate_dim[0]]
            for i in range(1,len(separate_dim)):
                diff.append(diff[-1]+separate_dim[i])
            self.num_separate = len(diff)-1 

            loss = 0.0
            if not is_masked:
                # Compute the loss over each block in 'separated' output
                weight = 1.0/self.num_separate
                for i in range(self.num_separate):
                    loss = loss + weight*nn.L1Loss()(output_pred_batch[:,diff[i]:diff[i+1]], output_batch[:,diff[i]:diff[i+1]])/ (nn.L1Loss()(output_batch[:,diff[i]:diff[i+1]],torch.zeros_like(output_batch[:,diff[i]:diff[i+1]])) + 1e-10)

            else:
                
                # Mask and compute the loss
                for i in range(self.num_separate):
                    mask = masked_dim[:,diff[i]:diff[i+1]]
                    mask = mask.unsqueeze(-1).unsqueeze(-1).expand(mask.shape[0], mask.shape[1], self.encoder_sizes[0], self.encoder_sizes[0])
                    output_pred_batch[:,diff[i]:diff[i+1]][mask==0.0] = 1.0
                    output_batch[:,diff[i]:diff[i+1]][mask==0.0] = 1.0

                    loss = loss + nn.L1Loss()(output_pred_batch[:,diff[i]:diff[i+1]], output_batch[:,diff[i]:diff[i+1]])/ nn.L1Loss()(output_batch[:,diff[i]:diff[i+1]],torch.zeros_like(output_batch[:,diff[i]:diff[i+1]]) + 1e-10) 
           
        
        # self.log("loss", loss, prog_bar=True, on_step=True, sync_dist=True)
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)

        return loss

    def configure_optimizers(self):
        
        # Are we in the FT regime?
        if "fine_tuned" in self.loader_dictionary and self.loader_dictionary["fine_tuned"]:
            
            print("=========================")
            print("Configure Optimizers - FT")
            print("=========================")
            assert hasattr(self, 'lr_emb')
            assert hasattr(self, 'lr_norm')

            params_1 = [param for name, param in self.named_parameters() if ("project" not in name) and ("lift" not in name) and ("in_norm_conditiner" not in name)]

            params_2 = [param for name, param in self.named_parameters() if (("project" in name) or ("lift" in name)) and ("in_norm_conditiner" not in name)]
            
            params_3 = [param for name, param in self.named_parameters() if ("in_norm_conditiner" in name) and ("project" not in name) and ("lift" not in name)]

            optimizer = torch.optim.AdamW([{'params': params_1},
                                           {'params': params_2,
                                            'lr': self.lr_emb},
                                           {'params': params_3,
                                            'lr': self.lr_norm}],
                                           lr=self.lr, weight_decay = self.loader_dictionary["weight_decay"])     
        else:
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        # Choose scheduler type
        scheduler_type = self.loader_dictionary.get("lr_scheduler", "step").lower()

        if scheduler_type == "cosine":
            T_max = self.loader_dictionary["max_epochs"]
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=T_max, eta_min=0
            )
        elif scheduler_type == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=self.scheduler_step, gamma=self.scheduler_gamma
            )
        else:
            raise ValueError(f"Unsupported scheduler type: {scheduler_type}")
         #Scheduler does not depend on the regime
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.scheduler_step, gamma=self.scheduler_gamma)
        
        return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}] 
    
    def train_dataloader(self):
        
        which = self.loader_dictionary["which"]
        mixing = self.loader_dictionary.get("mixing", False)

        if mixing:
            alpha = self.loader_dictionary.get("alpha", 0.0)
            total_samples = self.loader_dictionary.get("num_samples", 400)
            hole_path = self.loader_dictionary["hole_path"]
            nohole_path = self.loader_dictionary["nohole_path"]

            num_hole = int(alpha * total_samples)
            num_nohole = total_samples - num_hole

            datasets = []

            if num_hole > 0:
                hole_dataset = NSFlowTimeDataset(
                    max_num_time_steps = self.loader_dictionary["time_steps"],
                    time_step_size     = self.loader_dictionary["dt"],
                    fix_input_to_time_step = None,
                    which = "train",
                    resolution = 128,
                    in_dist = True,
                    num_trajectories = num_hole,
                    data_path = hole_path,
                    time_input = self.loader_dictionary["time_input"],
                    masked_input = None,
                    allowed_transitions = self.loader_dictionary["allowed_tran"]
                )
                datasets.append(hole_dataset)

            if num_nohole > 0:
                nohole_dataset = NSFlowTimeDataset(
                    max_num_time_steps = self.loader_dictionary["time_steps"],
                    time_step_size     = self.loader_dictionary["dt"],
                    fix_input_to_time_step = None,
                    which = "train",
                    resolution = 128,
                    in_dist = True,
                    num_trajectories = num_nohole,
                    data_path = nohole_path,
                    time_input = self.loader_dictionary["time_input"],
                    masked_input = None,
                    allowed_transitions = self.loader_dictionary["allowed_tran"]
                )
                datasets.append(nohole_dataset)

            assert len(datasets) > 0, "Both hole and no-hole sample counts are zero!"

            # Combine and return
            train_dataset = torch.utils.data.ConcatDataset(datasets)
            train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=6)
            return train_loader
    
        else:
            if which == "eul_ns_mix1":
                train_dataset1 = _load_dataset(dic = self.loader_dictionary, 
                                            which = "eul_riemann", 
                                            which_loader = "train",
                                            in_dim = self.in_dim,
                                            out_dim = self.out_dim,
                                            masked_input = [1.0, 1.0, 1.0, 1.0])
                
                train_dataset2 = _load_dataset(dic = self.loader_dictionary, 
                                            which = "eul_riemann_cur", 
                                            which_loader = "train",
                                            in_dim = self.in_dim,
                                            out_dim = self.out_dim,
                                            masked_input = [1.0, 1.0, 1.0, 1.0])
                
                train_dataset3 = _load_dataset(dic = self.loader_dictionary, 
                                            which = "eul_gauss", 
                                            which_loader = "train",
                                            in_dim = self.in_dim,
                                            out_dim = self.out_dim,
                                            masked_input = [1.0, 1.0, 1.0, 1.0])
                
                
                train_dataset4 = _load_dataset(dic = self.loader_dictionary, 
                                            which = "eul_kh", 
                                            which_loader = "train",
                                            in_dim = self.in_dim,
                                            out_dim = self.out_dim,
                                            masked_input = [1.0, 1.0, 1.0, 1.0])
                
                
                train_dataset5 = _load_dataset(dic = self.loader_dictionary, 
                                            which = "ns_gauss", 
                                            which_loader = "train",
                                            in_dim = self.in_dim,
                                            out_dim = self.out_dim,
                                            masked_input = [1.0, 1.0, 1.0, 0.0])
                
                train_dataset6 = _load_dataset(dic = self.loader_dictionary, 
                                            which = "ns_sin", 
                                            which_loader = "train",
                                            in_dim = self.in_dim,
                                            out_dim = self.out_dim,
                                            masked_input = [1.0, 1.0, 1.0, 0.0])
                
                train_datasets = [train_dataset1, train_dataset2, train_dataset3,
                                train_dataset4, train_dataset5, train_dataset6]
                train_dataset = torch.utils.data.ConcatDataset(train_datasets)
            
            else:
                is_masked =  "is_masked" in self.loader_dictionary and self.loader_dictionary["is_masked"] is not None
                if is_masked:  
                    if which[:2] == "ns":
                        mask = [1.0, 1.0, 1.0, 0.0]
                    elif which[:3] == "eul":
                        mask = [1.0, 1.0, 1.0, 1.0]
                    else:
                        mask = [1.0, 1.0, 1.0, 1.0]
                else:
                    mask = None
                    
                
                train_dataset = _load_dataset(dic = self.loader_dictionary, 
                                            which = which, 
                                            which_loader = "train",
                                            in_dim = self.in_dim,
                                            out_dim = self.out_dim,
                                            masked_input = mask)
            
            train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=6)
            return train_loader
        
    def validation_step(self, batch, batch_idx, dataloader_idx = 0):
        
        # Are the physical quantities separated in the loss function?
        is_separate = ("separate" in self.loader_dictionary) and self.loader_dictionary["separate"] and "separate_dim" in self.loader_dictionary
                
        # What kind of separation do we use?
        if is_separate:
            separate_dim = self.loader_dictionary["separate_dim"]
            assert type(separate_dim) is list  
        
        #---------
        # Are we interested in all the channels or we want to predict just a few of them and ignore others?
        #---------
        if "is_masked" in self.loader_dictionary:
            is_masked = self.loader_dictionary["is_masked"] is not None
        else:
            is_masked = False
        
        if not is_masked:
            t_batch, input_batch, output_batch = batch
        else:
            # Relevant dim tells us what channels we need to care about (it's a mask)
            t_batch, input_batch, output_batch, masked_dim = batch
        
        print(f"[VAL STEP] batch_idx: {batch_idx}")
        print(f"[VAL STEP] time shape: {t_batch.shape}")  # might be just scalar or (B,) if batched
        print(f"[VAL STEP] x_in shape: {input_batch.shape}")
        print(f"[VAL STEP] y_in shape: {output_batch.shape}")

        # Predict:
        output_pred_batch = self(input_batch)
        #---------
        
        # If airfoil, mask it
        which = self.loader_dictionary["which"]
        if "airfoil" in which:
            output_pred_batch[input_batch==1] = 1.0
            output_batch[input_batch==1] = 1.0
            
        #---------------
        # Compute the loss
        #---------------
        if not is_masked:
            # loss = (torch.mean(abs(output_pred_batch - output_batch), (-3, -2, -1)) / (torch.mean(abs(output_batch), (-3, -2, -1))+ 1e-10))* 100
            # Ensure both tensors are channel last.
            if output_pred_batch.dim() == 4 and output_pred_batch.shape[1] in [3, 9]:
                output_pred_batch = output_pred_batch.permute(0, 2, 3, 1)
            if output_batch.dim() == 4 and output_batch.shape[1] in [3, 9]:
                output_batch = output_batch.permute(0, 2, 3, 1)

            loss = (torch.mean(torch.abs(output_pred_batch - output_batch), dim=(-3, -2, -1)) /
                    (torch.mean(torch.abs(output_batch), dim=(-3, -2, -1)) + 1e-10)) * 100
        else:
            mask = masked_dim.unsqueeze(-1).unsqueeze(-1).expand(masked_dim.shape[0], masked_dim.shape[1], self.encoder_sizes[0], self.encoder_sizes[0])
            output_pred_batch[mask==0.0] = 1.0
            output_batch[mask==0.0] = 1.0

            
            loss = (torch.mean(abs(output_pred_batch - output_batch), (-3, -2, -1)) / (torch.mean(abs(output_batch), (-3, -2, -1)) + 1e-10))* 100
            
        #---------------
        # If it is separate - compute loss over the dimension
        #---------------
        if is_separate:

            diff = [0, self.loader_dictionary["separate_dim"][0]]
            for i in range(1,len(self.loader_dictionary["separate_dim"])):
                diff.append(diff[-1]+self.loader_dictionary["separate_dim"][i])
            self.num_separate = len(diff)-1 
            
            # Masked?
            if not is_masked:

                loss_sep = []
                for i in range(self.num_separate):
                    _loss = (torch.mean(abs(output_pred_batch[:,diff[i]:diff[i+1]] - output_batch[:,diff[i]:diff[i+1]]), (-3, -2, -1)) / (torch.mean(abs(output_batch[:,diff[i]:diff[i+1]]), (-3, -2, -1))+ 1e-10))* 100
                    loss_sep.append(_loss)

            else:
                loss_sep = []
                for i in range(self.num_separate):
                    mask = masked_dim[:,diff[i]:diff[i+1]]
                    mask = mask.unsqueeze(-1).unsqueeze(-1).expand(mask.shape[0], mask.shape[1], self.encoder_sizes[0], self.encoder_sizes[0])
                    output_pred_batch[:,diff[i]:diff[i+1]][mask==0.0] = 1.0
                    output_batch[:,diff[i]:diff[i+1]][mask==0.0] = 1.0

                    loss_sep.append((torch.mean(abs(output_pred_batch[:,diff[i]:diff[i+1]] - output_batch[:,diff[i]:diff[i+1]]), (-3, -2, -1)) / (torch.mean(abs(output_batch[:,diff[i]:diff[i+1]]), (-3, -2, -1))+1e-10))* 100)   
        
        #---------------
        # Save validation errs:
        #---------------
        if batch_idx==0:
            self.validation_times[str(dataloader_idx)] = t_batch
            self.validation_errs[str(dataloader_idx)] = loss
            
            if is_separate:
                self.validation_errs_sep[str(dataloader_idx)] = []
                for i in range(self.num_separate):
                    self.validation_errs_sep[str(dataloader_idx)].append(loss_sep[i])
                        
        else:
            
            self.validation_times[str(dataloader_idx)] = torch.cat((self.validation_times[str(dataloader_idx)], t_batch))
            self.validation_errs[str(dataloader_idx)] = torch.cat((self.validation_errs[str(dataloader_idx)], loss))
                   
            if is_separate:
                for i in range(self.num_separate):
                    self.validation_errs_sep[str(dataloader_idx)][i] = torch.cat((self.validation_errs_sep[str(dataloader_idx)][i], loss_sep[i]))
                
        return loss
        
        
    def val_dataloader(self):
        which = self.loader_dictionary["which"]      # which benchmark
        mixing = self.loader_dictionary.get("mixing", False)

        if mixing:
            hole_path = self.loader_dictionary["hole_path"]
            nohole_path = self.loader_dictionary["nohole_path"]

            datasets = []

            hole_dataset = NSFlowTimeDataset(
                max_num_time_steps = self.loader_dictionary["time_steps"],
                time_step_size     = self.loader_dictionary["dt"],
                fix_input_to_time_step = None,
                which = "val",
                resolution = 128,
                in_dist = True,
                num_trajectories = 50,
                N_val = 50,
                data_path = hole_path,
                time_input = self.loader_dictionary["time_input"],
                masked_input = None,
                allowed_transitions = self.loader_dictionary["allowed_tran"]
            )
            datasets.append(hole_dataset)

            nohole_dataset = NSFlowTimeDataset(
                max_num_time_steps = self.loader_dictionary["time_steps"],
                time_step_size     = self.loader_dictionary["dt"],
                fix_input_to_time_step = None,
                which = "val",
                resolution = 128,
                in_dist = True,
                num_trajectories = 50,
                N_val = 50,
                data_path = nohole_path,
                time_input = self.loader_dictionary["time_input"],
                masked_input = None,
                allowed_transitions = self.loader_dictionary["allowed_tran"]
            )
            datasets.append(nohole_dataset)

            val_dataset = torch.utils.data.ConcatDataset(datasets)
            val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=6)

            self.num_validation_loaders = 1
            self.num_out_loaders = 0
            self.val_labels = ["val_mixed_50_50"]

            return [val_loader]
        else :
            val_datasets = []
            num_datasets = 1
            num_out      = 0

            if which == "eul_ns_mix1":
                val_dataset1  =  _load_dataset(dic = self.loader_dictionary, 
                                            which = "eul_riemann", 
                                            which_loader = "val",
                                            in_dim = self.in_dim,
                                            out_dim = self.out_dim,
                                            masked_input = [1.0, 1.0, 1.0, 1.0])
                val_dataset2  =  _load_dataset(dic = self.loader_dictionary, 
                                            which = "eul_riemann_cur", 
                                            which_loader = "val",
                                            in_dim = self.in_dim,
                                            out_dim = self.out_dim,
                                            masked_input = [1.0, 1.0, 1.0, 1.0])
                val_dataset3  =  _load_dataset(dic = self.loader_dictionary, 
                                            which = "eul_gauss", 
                                            which_loader = "val",
                                            in_dim = self.in_dim,
                                            out_dim = self.out_dim,
                                            masked_input = [1.0, 1.0, 1.0, 1.0])
                val_dataset4  =  _load_dataset(dic = self.loader_dictionary, 
                                            which = "eul_kh", 
                                            which_loader = "val",
                                            in_dim = self.in_dim,
                                            out_dim = self.out_dim,
                                            masked_input = [1.0, 1.0, 1.0, 1.0])
                
                val_dataset5  =  _load_dataset(dic = self.loader_dictionary, 
                                            which = "ns_gauss", 
                                            which_loader = "val",
                                            in_dim = self.in_dim,
                                            out_dim = self.out_dim,
                                            masked_input = [1.0, 1.0, 1.0, 0.0])
                val_dataset6  =  _load_dataset(dic = self.loader_dictionary, 
                                            which = "ns_sin", 
                                            which_loader = "val",
                                            in_dim = self.in_dim,
                                            out_dim = self.out_dim,
                                            masked_input = [1.0, 1.0, 1.0, 0.0])
    
                val_datasets = [val_dataset1, val_dataset2, val_dataset3,
                                val_dataset4, val_dataset5, val_dataset6]
                num_datasets = 6
                num_out = 0
                
                self.val_labels = ["CE_Ri_", "CE_RiCu_", "CE_Gau_", "CE_KH_", "IC_Gau_", "IC_Sin_"]
                
                val_loaders = []
                for dataset in val_datasets:
                    val_loaders.append(DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=6))
                    
            else:
                is_masked = "is_masked" in self.loader_dictionary and self.loader_dictionary["is_masked"] is not None
                if is_masked:
                    if which[:2] == "ns":
                        mask = [1.0, 1.0, 1.0, 0.0]
                    elif which[:3] == "eul":
                        mask = [1.0, 1.0, 1.0, 1.0]
                    else:
                        mask = [1.0, 1.0, 1.0, 1.0]
                else:
                    mask = None
                
                val_dataset  =  _load_dataset(dic = self.loader_dictionary, 
                                            which = which, 
                                            which_loader = "val",
                                            in_dim = self.in_dim,
                                            out_dim = self.out_dim,
                                            masked_input = mask)
                self.val_labels =[which+"_"]
                val_loaders = [DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=6)]
            
            self.num_validation_loaders = num_datasets
            self.num_out_loaders        = num_out
            
            return val_loaders
        
        
    def on_validation_epoch_end(self):

        # Are the physical quantities separated in the loss function?
        is_separate = ("separate" in self.loader_dictionary) and self.loader_dictionary["separate"] and "separate_dim" in self.loader_dictionary
                
        # What kind of separation do we use?
        if is_separate:
            separate_dim = self.loader_dictionary["separate_dim"]
            assert type(separate_dim) is list
        
        # What to do with all the loaders?
        for dataloader_idx in range(self.num_validation_loaders + self.num_out_loaders):
            
            _stack = self.validation_errs[str(dataloader_idx)]
            
            if is_separate:
                _stack_sep = self.validation_errs_sep[str(dataloader_idx)] 
                
            if dataloader_idx == 0:
                _stack_all = _stack
                
            elif dataloader_idx < self.num_validation_loaders:
                _stack_all = torch.cat((_stack_all, _stack))
            
            idx_label = self.val_labels[dataloader_idx]
            median_loss = torch.median(_stack).item()
            mean_loss = torch.mean(_stack).item()
                        
            prog_bar = True
            
            if self.num_validation_loaders + self.num_out_loaders > 1:
                self.log(idx_label + "med_val_l", median_loss, prog_bar=False, on_step=False, on_epoch=True,sync_dist=True)
                self.log(idx_label + "mean_val_l",  mean_loss, prog_bar=False, on_step=False, on_epoch=True,sync_dist=True)

            if is_separate:
                
                for i in range(self.num_separate):
                    median_loss_s = torch.median(_stack_sep[i]).item()
                    mean_loss_s = torch.mean(_stack_sep[i]).item()

                    self.log(idx_label+"mean_val_" + str(i),  mean_loss_s, on_step=False, on_epoch=True,sync_dist=True)
                    self.log(idx_label+"med_val_"  + str(i),  median_loss_s, on_step=False, on_epoch=True,sync_dist=True)
             
        
        
            
        median_loss = torch.median(_stack_all).item()
        mean_loss = torch.mean(_stack_all).item() 
        
        self.log("med_val_l", median_loss, prog_bar=True, on_step=False, on_epoch=True,sync_dist=True)
        self.log("mean_val_l",  mean_loss, prog_bar=True, on_step=False, on_epoch=True,sync_dist=True)
        
        # Save the best loss
        if mean_loss<self.best_val_loss_mean:
            self.best_val_loss_mean = mean_loss
            self.best_val_loss_median = median_loss
        
        self.log("best_mean_val_loss",self.best_val_loss_mean,on_step=False, on_epoch=True,sync_dist=True)
        self.log("best_med_val_loss",self.best_val_loss_median,on_step=False, on_epoch=True,sync_dist=True)
                
        return {"med_val_l": median_loss, "mean_val_l": mean_loss,} 
    
    def get_n_params(self):
        pp = 0
        
        for p in list(self.parameters()):
            nn = 1
            for s in list(p.size()):
                nn = nn * s
            pp += nn
        return pp
    
    def print_size(self):
        nparams = 0
        nbytes = 0

        for param in self.parameters():
            nparams += param.numel()
            nbytes += param.data.element_size() * param.numel()

        print(f'Total number of model parameters: {nparams} (~{format_tensor_size(nbytes)})')

        return nparams
    
    def test_dataloader(self):
        if self.loader_dictionary.get("mixing", False):
            test_hole = NSFlowTimeDataset(
                max_num_time_steps=self.loader_dictionary["time_steps"],
                time_step_size=self.loader_dictionary["dt"],
                which="test",
                resolution=128,
                in_dist=True,
                num_trajectories=80,
                data_path=self.loader_dictionary["hole_path"],
                time_input=self.loader_dictionary["time_input"],
                allowed_transitions=self.loader_dictionary["allowed_tran"]
            )
            test_nohole = NSFlowTimeDataset(
                max_num_time_steps=self.loader_dictionary["time_steps"],
                time_step_size=self.loader_dictionary["dt"],
                which="test",
                resolution=128,
                in_dist=True,
                num_trajectories=80,
                data_path=self.loader_dictionary["nohole_path"],
                time_input=self.loader_dictionary["time_input"],
                allowed_transitions=self.loader_dictionary["allowed_tran"]
            )
            loaders = [
                DataLoader(test_hole, batch_size=8, shuffle=False, num_workers=6),
                DataLoader(test_nohole, batch_size=8, shuffle=False, num_workers=6)
            ]
            self.num_test_loaders = len(loaders)  # Set the number of test loaders
            self.test_labels = ["test_hole_", "test_nohole_"]
            return loaders

        else:
            test_dataset = _load_dataset(
                dic=self.loader_dictionary,
                which=self.loader_dictionary["which"],
                which_loader="test",
                in_dim=self.in_dim,
                out_dim=self.out_dim,
            )
            self.num_test_loaders = 1
            self.test_labels = ["test_"]
            return [DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=6)]
        
    def test_step(self, batch, batch_idx, dataloader_idx=0):
        t_batch, input_batch, output_batch = batch
        output_pred_batch = self(input_batch)
        if output_pred_batch.dim() == 4 and output_pred_batch.shape[1] in [3, 9]:
                output_pred_batch = output_pred_batch.permute(0, 2, 3, 1)
        if output_batch.dim() == 4 and output_batch.shape[1] in [3, 9]:
            output_batch = output_batch.permute(0, 2, 3, 1)
        loss = (torch.mean(abs(output_pred_batch - output_batch), (-3, -2, -1)) /
                (torch.mean(abs(output_batch), (-3, -2, -1)) + 1e-10)) * 100

        key = str(dataloader_idx)
        # Initialize the dictionary if it doesn't exist
        if not hasattr(self, "test_errs"):
            self.test_errs = {}
        # If key doesn't exist, create it
        if key not in self.test_errs:
            self.test_errs[key] = loss.unsqueeze(0)
        else:
            self.test_errs[key] = torch.cat((self.test_errs[key], loss.unsqueeze(0)))
        
        return loss


    def on_test_epoch_end(self):
        _stack_all = None
        for dataloader_idx in range(self.num_test_loaders):
            key = str(dataloader_idx)
            # Skip if no batches were processed for this loader
            if key not in self.test_errs:
                continue
            _stack = self.test_errs[key]
            idx_label = self.test_labels[dataloader_idx]
            median_loss = torch.median(_stack).item()
            mean_loss = torch.mean(_stack).item()
            
            self.log(idx_label + "med_test_l", median_loss, on_step=False, on_epoch=True, sync_dist=True)
            self.log(idx_label + "mean_test_l", mean_loss, on_step=False, on_epoch=True, sync_dist=True)
            
            if _stack_all is None:
                _stack_all = _stack
            else:
                _stack_all = torch.cat((_stack_all, _stack))
        
        if _stack_all is not None:
            global_median = torch.median(_stack_all).item()
            global_mean = torch.mean(_stack_all).item()
            self.log("med_test_l", global_median, on_step=False, on_epoch=True, sync_dist=True)
            self.log("mean_test_l", global_mean, on_step=False, on_epoch=True, sync_dist=True)
            return {"med_test_l": global_median, "mean_test_l": global_mean}
        else:
            return {}
    