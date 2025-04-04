import torch
import h5py
import numpy as np
import copy
from scOT.problems.base import BaseTimeDataset
from scOT.problems.fluids.normalization_constants import CONSTANTS
import torch
import numpy as np

class GaussianNumpyDataset(BaseTimeDataset):
    def __init__(
        self,
        *args,
        file_path,
        max_num_time_steps=19,
        time_step_size=1,
        fix_input_to_time_step=None,
        allowed_time_transitions=None,
        just_velocities=False,
        resolution=128,
        num_trajectories = 100,
        N_val=100,
        N_test=80,
        which=None,
        **kwargs,
    ):
        """
        Custom dataset for `.npy` data adapted from `BaseTimeDataset`.

        Args:
            file_path (str): Path to the `.npy` dataset.
            max_num_time_steps (int): Maximum number of timesteps (default: 20).
            time_step_size (int): Size of timesteps (default: 1).
            fix_input_to_time_step (int): Fix input to a specific time step (optional).
            allowed_time_transitions (list): Allowed transitions between timesteps (optional).
            just_velocities (bool): Whether to use only velocity channels.
            resolution (int): Resolution of the grid (default: 128).
            N_val (int): Number of validation samples.
            N_test (int): Number of test samples.
            which (str): Specifies the subset of the dataset ('train', 'val', 'test').
        """
        # Initialize BaseTimeDataset
        super().__init__(
            *args,
            max_num_time_steps=max_num_time_steps,
            time_step_size=time_step_size,
            fix_input_to_time_step=fix_input_to_time_step,
            allowed_time_transitions=allowed_time_transitions,
            which=which,
            num_trajectories = num_trajectories,
            **kwargs,
        )

        # Load `.npy` data
        self.data = np.load(file_path)  # Shape: (num_trajectories, timesteps, height, width, channels)
        self.N_max = self.data.shape[0]  # Total number of trajectories
        self.resolution = resolution
        self.N_val = N_val  # Validation set size
        self.N_test = N_test  # Test set size
        self.num_trajectories = self.N_max - self.N_val - self.N_test  if num_trajectories is None else num_trajectories
        self.just_velocities = just_velocities

        # Split the dataset based on `which`
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

        # Set normalization constants (for velocity channels only)
        self.constants = {
            "mean": self.data[..., :3].mean(axis=(0, 1, 2, 3)),  # Mean of u, v
            "std": self.data[..., :3].std(axis=(0, 1, 2, 3)),    # Std of u, v
            "time": 19.0
        }

        # Define input and output dimensions
        self.input_dim = self.data.shape[-1]   # Includes mask channel if `just_velocities` is False, 3 with mask, 2 without mask
        self.output_dim = 3
        # self.label_description = "[u,v],[mask]" if not just_velocities else "[u,v]"
        self.label_description = "[u,v,p]"
        self.channel_slice_list = [0, 1, 2]  # For two channels, adjust as needed [0, 1, 2] for with mask
        self.printable_channel_description = ["u", "v", "p"]  # Describe the channels

        # Channel-wise mask (e.g., velocities are valid, mask is optional)
        self.pixel_mask = torch.tensor([False, False])

        # Finalize initialization
        self.post_init()

    def __getitem__(self, idx):
        """
        Get an item from the dataset.
        """
        i, t, t1, t2 = self._idx_map(idx)
        time = t / self.constants["time"]
        # Extract inputs (at t1) and labels (at t2)
        inputs = torch.tensor(self.data[i, t1, ..., :3], dtype=torch.float32)  # u, v, p
        labels = torch.tensor(self.data[i, t2, ..., :3], dtype=torch.float32)  # u, v, p (next timestep)
        if self.input_dim > 2:
            hole_info = torch.tensor(self.data[i, t1, ..., 3:], dtype=torch.float32)  # Hole info
            # Add hole info channel to inputs
            inputs = torch.cat([inputs, hole_info], dim=-1)
            # inputs = torch.cat([inputs, hole_info.unsqueeze(-1)], dim=-1)
        # # Add mask channel to inputs if `just_velocities` is False
        # if not self.just_velocities:
        #     mask = torch.tensor(self.data[i, t1, ..., 2], dtype=torch.float32)  # Mask
        #     # print(f"Shape of inputs: {inputs.shape}")
        #     # print(f"Shape of mask before reshaping: {mask.shape}")
        #     # print(f"Shape of mask after reshaping: {mask.shape}")
        #     inputs = torch.cat([inputs, mask.unsqueeze(-1)], dim=-1)
        # Normalize velocity channels
        inputs[..., :3] = (inputs[..., :3] - self.constants["mean"]) / self.constants["std"]
        labels = (labels - self.constants["mean"]) / self.constants["std"]
        # print(f"Shape of inputs in dataset finally: {inputs.shape}")
        # Permute to match (channels, height, width)
        inputs = inputs.permute(2, 0, 1)  # From (height, width, channels) to (channels, height, width)
        labels = labels.permute(2, 0, 1)

        return {
            "pixel_values": inputs,
            "labels": labels,
            "time": time,
            "pixel_mask": None,
        }




class IncompressibleBase(BaseTimeDataset):
    def __init__(
        self,
        N_max,
        file_path,
        *args,
        tracer=False,
        just_velocities=False,
        transpose=False,
        resolution=None,
        **kwargs
    ):
        """
        just_velocities: If True, only the velocities are used as input and output.
        transpose: If True, the input and output are transposed.
        """
        super().__init__(*args, **kwargs)
        assert self.max_num_time_steps * self.time_step_size <= 20

        self.N_max = N_max
        self.N_val = 120
        self.N_test = 240
        self.resolution = 128
        self.tracer = tracer
        self.just_velocities = just_velocities
        self.transpose = transpose

        data_path = self.data_path + file_path
        data_path = self._move_to_local_scratch(data_path)
        self.reader = h5py.File(data_path, "r")

        self.constants = copy.deepcopy(CONSTANTS)
        if just_velocities:
            self.constants["mean"] = self.constants["mean"][1:3]
            self.constants["std"] = self.constants["std"][1:3]

        self.density = torch.ones(1, self.resolution, self.resolution)
        self.pressure = torch.zeros(1, self.resolution, self.resolution)

        self.input_dim = 4 if not tracer else 5
        if just_velocities:
            self.input_dim -= 2
        self.label_description = "[u,v]"
        if not self.just_velocities:
            self.label_description = "[rho],[u,v],[p]"
        if tracer:
            self.label_description += ",[tracer]"

        self.pixel_mask = torch.tensor([False, False])
        if not self.just_velocities:
            self.pixel_mask = torch.tensor([False, False, False, True])
        if tracer:
            self.pixel_mask = torch.cat(
                [self.pixel_mask, torch.tensor([False])],
                dim=0,
            )

        if resolution is None:
            self.res = None
        else:
            if resolution > 128:
                raise ValueError("Resolution must be <= 128")
            self.res = resolution

        self.post_init()

    def _downsample(self, image, target_size):
        image = image.unsqueeze(0)
        image_size = image.shape[-2]
        freqs = torch.fft.fftfreq(image_size, d=1 / image_size)
        sel = torch.logical_and(freqs >= -target_size / 2, freqs <= target_size / 2 - 1)
        image_hat = torch.fft.fft2(image, norm="forward")
        image_hat = image_hat[:, :, sel, :][:, :, :, sel]
        image = torch.fft.ifft2(image_hat, norm="forward").real
        return image.squeeze(0)

    def __getitem__(self, idx):
        i, t, t1, t2 = self._idx_map(idx)
        time = t / self.constants["time"]

        inputs_v = (
            torch.from_numpy(self.reader["velocity"][i + self.start, t1, 0:2])
            .type(torch.float32)
            .reshape(2, self.resolution, self.resolution)
        )
        label_v = (
            torch.from_numpy(self.reader["velocity"][i + self.start, t2, 0:2])
            .type(torch.float32)
            .reshape(2, self.resolution, self.resolution)
        )
        if self.transpose:
            inputs_v = inputs_v.transpose(-2, -1)
            label_v = label_v.transpose(-2, -1)

        if not self.just_velocities:
            inputs = torch.cat([self.density, inputs_v, self.pressure], dim=0)
            label = torch.cat([self.density, label_v, self.pressure], dim=0)
        else:
            inputs = inputs_v
            label = label_v

        inputs = (inputs - self.constants["mean"]) / self.constants["std"]
        label = (label - self.constants["mean"]) / self.constants["std"]

        if self.tracer:
            input_tracer = (
                torch.from_numpy(self.reader["velocity"][i + self.start, t1, 2:3])
                .type(torch.float32)
                .reshape(1, self.resolution, self.resolution)
            )
            output_tracer = (
                torch.from_numpy(self.reader["velocity"][i + self.start, t2, 2:3])
                .type(torch.float32)
                .reshape(1, self.resolution, self.resolution)
            )
            if self.transpose:
                input_tracer = input_tracer.transpose(-2, -1)
                output_tracer = output_tracer.transpose(-2, -1)
            input_tracer = (
                input_tracer - self.constants["tracer_mean"]
            ) / self.constants["tracer_std"]
            output_tracer = (
                output_tracer - self.constants["tracer_mean"]
            ) / self.constants["tracer_std"]

            inputs = torch.cat([inputs, input_tracer], dim=0)
            label = torch.cat([label, output_tracer], dim=0)

        if self.res is not None:
            inputs = self._downsample(inputs, self.res)
            label = self._downsample(label, self.res)

        return {
            "pixel_values": inputs,
            "labels": label,
            "time": time,
            "pixel_mask": self.pixel_mask,
        }


class KolmogorovFlow(BaseTimeDataset):
    def __init__(self, *args, tracer=False, just_velocities=False, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.max_num_time_steps * self.time_step_size <= 20

        assert tracer == False

        self.N_max = 20000
        self.N_val = 120
        self.N_test = 240
        self.resolution = 128
        self.just_velocities = just_velocities

        data_path = self.data_path + "/FNS-KF.nc"
        data_path = self._move_to_local_scratch(data_path)
        self.reader = h5py.File(data_path, "r")

        self.constants = copy.deepcopy(CONSTANTS)
        self.constants["mean"][1] = -2.2424793e-13
        self.constants["mean"][2] = 4.1510376e-12
        self.constants["std"][1] = 0.22017328
        self.constants["std"][2] = 0.22078253
        if just_velocities:
            self.constants["mean"] = self.constants["mean"][1:3]
            self.constants["std"] = self.constants["std"][1:3]

        self.density = torch.ones(1, self.resolution, self.resolution)
        self.pressure = torch.zeros(1, self.resolution, self.resolution)
        X, Y = torch.meshgrid(
            torch.linspace(0, 1, self.resolution),
            torch.linspace(0, 1, self.resolution),
            indexing="ij",
        )
        f = lambda x, y: 0.1 * torch.sin(2.0 * np.pi * (x + y))
        self.forcing = f(X, Y).unsqueeze(0)
        self.constants["mean_forcing"] = -1.2996679288335145e-09
        self.constants["std_forcing"] = 0.0707106739282608
        self.forcing = (self.forcing - self.constants["mean_forcing"]) / self.constants[
            "std_forcing"
        ]

        self.input_dim = 5 if not tracer else 6
        if just_velocities:
            self.input_dim -= 2
        self.label_description = "[u,v],[g]"
        if not self.just_velocities:
            self.label_description = "[rho],[u,v],[p],[g]"
        if tracer:
            self.label_description += ",[tracer]"

        self.pixel_mask = torch.tensor([False, False, False])
        if not self.just_velocities:
            self.pixel_mask = torch.tensor([False, False, False, True, False])
        if tracer:
            self.pixel_mask = torch.cat(
                [self.pixel_mask, torch.tensor([False])],
                dim=0,
            )

        self.post_init()

    def __getitem__(self, idx):
        i, t, t1, t2 = self._idx_map(idx)
        time = t / self.constants["time"]

        inputs_v = (
            torch.from_numpy(self.reader["solution"][i + self.start, t1, 0:2])
            .type(torch.float32)
            .reshape(2, self.resolution, self.resolution)
        )
        label_v = (
            torch.from_numpy(self.reader["solution"][i + self.start, t2, 0:2])
            .type(torch.float32)
            .reshape(2, self.resolution, self.resolution)
        )

        if not self.just_velocities:
            inputs = torch.cat([self.density, inputs_v, self.pressure], dim=0)
            label = torch.cat([self.density, label_v, self.pressure], dim=0)
        else:
            inputs = inputs_v
            label = label_v

        inputs = (inputs - self.constants["mean"]) / self.constants["std"]
        label = (label - self.constants["mean"]) / self.constants["std"]

        inputs = torch.cat([inputs, self.forcing], dim=0)
        label = torch.cat([label, self.forcing], dim=0)

        return {
            "pixel_values": inputs,
            "labels": label,
            "time": time,
            "pixel_mask": self.pixel_mask,
        }


class BrownianBridge(IncompressibleBase):
    def __init__(self, *args, tracer=False, just_velocities=False, **kwargs):
        if tracer:
            raise ValueError("BrownianBridge does not have a tracer")
        file_path = "/NS-BB.nc"
        super().__init__(
            20000,
            file_path,
            *args,
            tracer=False,
            just_velocities=just_velocities,
            **kwargs
        )


class PiecewiseConstants(IncompressibleBase):
    def __init__(self, *args, tracer=False, just_velocities=False, **kwargs):
        file_path = "/NS-PwC.nc"
        super().__init__(
            20000,
            file_path,
            *args,
            tracer=tracer,
            just_velocities=just_velocities,
            **kwargs
        )


class Gaussians(IncompressibleBase):
    def __init__(self, *args, tracer=False, just_velocities=False, **kwargs):
        if tracer:
            raise ValueError("Gaussians does not have a tracer")
        file_path = "/NS-Gauss.nc"
        super().__init__(
            20000,
            file_path,
            *args,
            tracer=False,
            just_velocities=just_velocities,
            **kwargs
        )


class ShearLayer(IncompressibleBase):
    def __init__(self, *args, tracer=False, just_velocities=False, **kwargs):
        if tracer:
            raise ValueError("Shear layer does not have a tracer")
        super().__init__(
            40000,
            "/NS-SL.nc",
            *args,
            transpose=True,
            tracer=False,
            just_velocities=just_velocities,
            **kwargs
        )


class VortexSheet(IncompressibleBase):
    def __init__(self, *args, tracer=False, just_velocities=False, **kwargs):
        if tracer:
            raise ValueError("VortexSheet does not have a tracer")
        file_path = "/NS-SVS.nc"
        super().__init__(
            20000,
            file_path,
            *args,
            tracer=False,
            just_velocities=just_velocities,
            **kwargs
        )


class Sines(IncompressibleBase):
    def __init__(self, *args, tracer=False, just_velocities=False, **kwargs):
        if tracer:
            raise ValueError("Sines does not have a tracer")
        file_path = "/NS-Sines.nc"
        super().__init__(
            20000,
            file_path,
            *args,
            tracer=False,
            just_velocities=just_velocities,
            **kwargs
        )
