import torch
import numpy as np
from utils.base import BaseTimeDataset
import yaml

with open("config/config.yaml", "r") as f:
    config = yaml.safe_load(f)

class CEDataset(BaseTimeDataset):
    def __init__(
        self,
        *args,
        file_path,
        autoregressive=True,  # NEW: autoregressive mode flag
        max_num_time_steps=20,
        time_step_size=1,
        fix_input_to_time_step=None,
        allowed_time_transitions=None,
        resolution=128,
        num_trajectories=100,
        N_val=100,
        N_test=80,
        which=None,
        use_all_channels=False,
        **kwargs,
    ):
        self.autoregressive = autoregressive  # store the mode flag
        super().__init__(
            *args,
            max_num_time_steps=max_num_time_steps,
            time_step_size=time_step_size,
            fix_input_to_time_step=fix_input_to_time_step,
            allowed_time_transitions=allowed_time_transitions,
            which=which,
            num_trajectories=num_trajectories,
            **kwargs,
        )
        self.data = np.load(file_path)  # expected shape: (num_trajectories, timesteps, H, W, 7)
        self.N_max = self.data.shape[0]
        self.resolution = resolution
        self.N_val = N_val
        self.N_test = N_test
        self.num_trajectories = (
            self.N_max - self.N_val - self.N_test if num_trajectories is None else num_trajectories
        )
        self.use_all_channels = use_all_channels
        input_dim = config["model"]["input_dim"]
        if input_dim == 6:
            self.data = self.data[..., [0, 1, 2, 4, 5, 3]]
        elif input_dim == 7:
            self.data = self.data[..., [0,1,2,3,6,4,5]]
        # When autoregressive, we want one sample per trajectory rather than multiple time pairs.
        if autoregressive:
            self.length = self.num_trajectories
            self.start = 0
        else:
            # existing splitting logic based on 'which'
            if which == "train":
                self.start = 0
                self.length = self.num_trajectories
            elif which == "val":
                self.start = self.num_trajectories
                self.length = self.N_val
            elif which == "test":
                self.start = self.num_trajectories + self.N_val
                self.length = self.N_test
            else:
                raise ValueError(f"Unknown dataset split: {which}")
        
        self.output_dim = config["model"]["output_dim"]
        if self.output_dim == 4:
            self.label_description = "[u, v, pressure, density]"
            self.printable_channel_description = ["u", "v", "p", "d"]
        elif self.output_dim == 3:
            self.label_description = "[u, v, pressure]"
            self.printable_channel_description = ["u", "v", "p"]

        self.constants = {
            "mean": self.data[..., :self.output_dim].mean(axis=(0, 1, 2, 3)),
            "std": self.data[..., :self.output_dim].std(axis=(0, 1, 2, 3)),
            "time": float(max_num_time_steps)
        }
        self.pixel_mask = None

        self.post_init()
    def __getitem__(self, idx):
        if self.autoregressive:
            # Here, we treat idx as a trajectory index.
            i = idx  # Each trajectory gives one sample.
            # Choose the input as the first timestep (or a fixed timestep, if you prefer)
            input_frame = torch.tensor(
                self.data[i, 0, :, :, :],
                dtype=torch.float32,
            )
            # Define the target sequence as all subsequent timesteps up to max_num_time_steps.
            # You might also choose a subset here.
            target_seq = torch.tensor(
                self.data[i, 1:self.max_num_time_steps, :, :, :self.output_dim],
                dtype=torch.float32,
            )

            input_frame[..., :self.output_dim] = (input_frame[..., :self.output_dim] - self.constants["mean"]) / self.constants["std"]
            target_seq = (target_seq - self.constants["mean"]) / self.constants["std"]

            # Permute dimensions.
            # Input: (H, W, channels) -> (channels, H, W)
            input_frame = input_frame.permute(2, 0, 1)
            # Target sequence: (T, H, W, channels) -> (T, channels, H, W)
            target_seq = target_seq.permute(0, 3, 1, 2)

            # Optionally, you can also return a time vector for each target frame if needed.
            # times = torch.linspace(0, 1, steps=target_seq.shape[0])

            return {
                "pixel_values": input_frame,
                "labels": target_seq,
            }
        else:
            # Original behavior for next-timestep prediction using _idx_map.
            i, t, t1, t2 = self._idx_map(idx)
            time = t / self.constants["time"]
            if self.use_all_channels:
                inputs = torch.tensor(self.data[i, t1, :, :, :], dtype=torch.float32)
            else:
                inputs = torch.tensor(self.data[i, t1, :, :, :self.output_dim], dtype=torch.float32)
            labels = torch.tensor(self.data[i, t2, :, :, :self.output_dim], dtype=torch.float32)
            inputs[..., :self.output_dim] = (inputs[..., :self.output_dim] - self.constants["mean"]) / self.constants["std"]
            labels = (labels - self.constants["mean"]) / self.constants["std"]
            inputs = inputs.permute(2, 0, 1)
            labels = labels.permute(2, 0, 1)
            return {
                "pixel_values": inputs,
                "labels": labels,
                "time": time,
                "pixel_mask": self.pixel_mask,
            }