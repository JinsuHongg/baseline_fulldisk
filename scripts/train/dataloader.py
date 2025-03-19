import os
import random
import numpy as np
import xarray as xr
import pandas as pd

import torch
from torch.utils.data import Dataset
from torchvision.io import read_image

# import torchvision.transforms as transforms


class SolarFlSets(Dataset):
    def __init__(
        self,
        annotations_df: pd.DataFrame,
        img_dir: str,
        num_sample=False,
        random_state: int = 1004,
        transform=None,
        target_transform=None,
        normalization=False,
    ):

        if num_sample:
            self.df = annotations_df.sample(n=num_sample, random_state=random_state)
        else:
            self.df = annotations_df

        self.img_dir = img_dir
        self.transform = transform

        self.target_transform = target_transform
        self.norm = normalization

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        # deploy channel if necessary
        timestamp = self.df.iloc[idx, 0]
        img_path = os.path.join(
            self.img_dir,
            f"{timestamp.year}/{timestamp.month:02d}/{timestamp.day:02d}/"
            + f"HMI.m{timestamp.year}.{timestamp.month:02d}.{timestamp.day:02d}_"
            + f"{timestamp.hour:02d}.{timestamp.minute:02d}.{timestamp.second:02d}.jpg",
        )
        image = read_image(img_path).repeat(3, 1, 1).float()
        label = self.df.iloc[idx, 2]  # 0: timestamp 1: GOES class 2: target label

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        if self.norm:
<<<<<<< HEAD
            image = image / 255 # zero to one normalization
        return image, label

class heliofm_FLDataset:
    """
    Usage example
    ```
    SolarFlareDataset(
        index_path=sdo_index,
        fl_path=fl_index,
        time_delta_input_minutes=[0],
        time_delta_target_minutes=60,
        n_input_timestamps=1,
        scalers=scalers,
        channels=['0094', '0131', '0171', '0193', '0211', '0304', '0335', 'hmi']
    )
    ```
    Here, `sdo_index` is the path to one of the index files used for pretraining.
    `fl_path` points to an index file containing solar flare information.
    """

    def __init__(
        self,
        index_path: str,
        fl_path: str,
        time_delta_input_minutes: list[int],
        time_delta_target_minutes: int,
        n_input_timestamps: int,
        scalers=None,
        channels: list[str] | None = None,
    ):
        self.scalers = scalers
        self.channels = channels
        self.n_input_timestamps = n_input_timestamps

        if self.channels is None:
            # AIA + HMI channels
            self.channels = [
                "0094",
                "0131",
                "0171",
                "0193",
                "0211",
                "0304",
                "0335",
                "hmi",
            ]
        self.in_channels = len(self.channels)

        # Convert time delta to numpy timedelta64
        self.time_delta_input_minutes = sorted(
            np.timedelta64(t, "m") for t in time_delta_input_minutes
        )
        self.time_delta_target_minutes = [
            np.timedelta64(time_delta_target_minutes, "m")
        ]

        # Create the index
        self.sdo_index = self._get_index(index_path)
        self.fl_index = self._get_index(fl_path)

        # Filter out rows where the sequence is not fully present
        self.valid_indices = self.filter_valid_indices()
        self.adjusted_length = len(self.valid_indices)

    def filter_valid_indices(self):
        """
        Extracts timestamps from the combination of self.sdo_index
        and self.fl_index that define valid samples.

        Args:
        Returns:
            List of timestamps.
        """

        valid_indices = []
        time_deltas = np.unique(self.time_delta_input_minutes)

        for reference_timestep in self.sdo_index.index.intersection(
            self.fl_index.index
        ):
            required_timesteps = reference_timestep + time_deltas

            if all(t in self.sdo_index.index for t in required_timesteps):
                valid_indices.append(reference_timestep)

        return valid_indices

    def _get_index(self, file_name: str) -> pd.DataFrame:
        """
        Reads index files into data frames.
        """
        index = pd.read_csv(file_name)
        # index = index[index["present"] == 1]
        index["timestep"] = pd.to_datetime(index["timestep"]).values.astype(
            "datetime64[ns]"
        )
        index.set_index("timestep", inplace=True)
        index.sort_index(inplace=True)

        return index

    def __len__(self):
        return self.adjusted_length

    def __getitem__(self, idx: int) -> dict:
        """
        Args:
            idx: Index of sample to load. (Pytorch standard.)
        Returns:
            Dictionary with following keys. The values are tensors with shape as follows:
                ts (torch.Tensor):                C, T, H, W
                time_delta_input (torch.Tensor):  T
                lead_time_delta (torch.Tensor):   L
                label (torch.Tensor):             L==1
            C - Channels, T - Input times, H - Image height, W - Image width, L - Lead time.
        """

        time_deltas = np.array(
            sorted(
                random.sample(
                    self.time_delta_input_minutes[:-1], self.n_input_timestamps - 1
                )
            )
            + [self.time_delta_input_minutes[-1]]
        )
        reference_timestep = self.valid_indices[idx]
        required_timesteps = reference_timestep + time_deltas

        sequence_data = [
            # self.transform_data(
            #     self.load_nc_data(
            #         self.sdo_index.loc[timestep, "path"], timestep, self.channels
            #     )
            # )
            # for timestep in required_timesteps

            self.load_nc_data(
                    self.sdo_index.loc[timestep, "path"], timestep, self.channels
            )
            for timestep in required_timesteps
        ]

        stacked_inputs = torch.stack(sequence_data, dim=1)

        time_delta_input_float = time_deltas / np.timedelta64(1, "h")
        time_delta_input_float = torch.from_numpy(time_delta_input_float).to(
            dtype=torch.float32
        )

        lead_time_delta_float = self.time_delta_target_minutes / np.timedelta64(1, "h")
        lead_time_delta_float = -torch.tensor(lead_time_delta_float).to(
            dtype=torch.float32
        )

        label = self.fl_index.loc[reference_timestep, "label"]
        label = torch.Tensor([label])

        return {
            "ts": stacked_inputs,
            "time_delta_input": time_delta_input_float,
            "label": label,
            "lead_time_delta": lead_time_delta_float,
        }

    def load_nc_data(
        self, filepath: str, timestep: pd.Timestamp, channels: list[str]
    ) -> np.ndarray:
        """
        Args:
            filepath: String or Pathlike. Points to NetCDF file to open.
            timestep: Identifies timestamp to retrieve.
        Returns:
            Numpy array of shape (C, H, W).
        """
        # with xr.open_dataset(filepath, engine="h5netcdf") as ds:
        #     # Use .sel() with the actual timestep
        #     data = ds.sel(time=timestep, channel=channels)["value"].values
        data = read_image(filepath).float()
        return data

    def transform_data(self, data: np.ndarray) -> torch.Tensor:
        """
        Applies scalers.

        Args:
            data: Numpy array of shape (C, H, W)
        Returns:
            Tensor of shape (C, H, W). Data type float32.
        """
        for idx, channel in enumerate(self.channels):
            data[idx] = self.scalers[channel].signum_log_transform(data[idx])

        return torch.tensor(data, dtype=torch.float32)
=======
            image = image / 255  # zero to one normalization
        return image, label
>>>>>>> main
