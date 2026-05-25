import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

class VicRoadsExcelParser:
    """
    Parses and transforms the VicRoads Excel sheet.
    Converts horizontal 15-minute intervals (V00-V95) into a vertical hourly timeline.
    """
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.sequential_df = None

    def parse_to_hourly_sequence(self) -> pd.DataFrame:
        # Load Excel sheet
        df = pd.read_excel(self.file_path)
        
        # Identify the 96 interval columns
        volume_cols = [col for col in df.columns if col.startswith('V')]
        id_vars = ['SCATS Number', 'Date']
        
        # Unpivot from wide horizontal matrix to a long vertical timeline
        melted_df = pd.melt(
            df, id_vars=id_vars, value_vars=volume_cols, 
            var_name='Interval', value_name='Count'
        )
        
        # Sort chronologically by site, date, and interval index
        melted_df['Interval_Idx'] = melted_df['Interval'].str.extract(r'(\d+)').astype(int)
        melted_df = melted_df.sort_values(by=['SCATS Number', 'Date', 'Interval_Idx']).reset_index(drop=True)
        melted_df['Count'] = melted_df['Count'].fillna(0)
        
        # Aggregate four 15-minute intervals into 1 Hour Blocks (Vehicles/Hour)
        melted_df['Hour_Group'] = melted_df.index // 4
        hourly_df = melted_df.groupby(['SCATS Number', 'Hour_Group']).agg({
            'Count': 'sum',
            'Date': 'first'
        }).reset_index()
        
        self.sequential_df = hourly_df.rename(columns={'Count': 'Vehicles_Per_Hour'})
        return self.sequential_df

    def get_dataset_loaders(self, scats_number: int, seq_len: int = 4, batch_size: int = 16):
        """Filters by SCATS ID and returns standard PyTorch DataLoaders."""
        site_data = self.sequential_df[self.sequential_df['SCATS Number'] == scats_number]
        volumes = site_data['Vehicles_Per_Hour'].values.astype('float32')
        
        X, y = [], []
        for i in range(len(volumes) - seq_len):
            X.append(volumes[i : i + seq_len])
            y.append(volumes[i + seq_len])
            
        dataset = PyTorchTrafficDataset(X, y)
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)


class PyTorchTrafficDataset(Dataset):
    """Feeds aggregated sequence tensors cleanly into LSTM and GRU networks."""
    def __init__(self, X, y):
        self.X = torch.tensor(X).unsqueeze(-1)  # Shape: (Samples, Seq_Len, 1)
        self.y = torch.tensor(y).unsqueeze(-1)  # Shape: (Samples, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
