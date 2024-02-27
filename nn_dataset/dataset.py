sys.path.append('../toolbox')
from toolbox.plotstock import plotstock
import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import sys
from PIL import Image 

train_str_data_dir = '../train_data/mock.csv'
stock_data_dir = '../stock_data/'
WEEKS_IN_YEAR = 52

class StockDataset(Dataset):
    def __init__(self, train_str_data_dir, transform=None):
        self.train_str_data = pd.read_csv(train_str_data_dir)
        self.transform = transform

    def __len__(self):
        return len(self.train_str_data)

    def __getitem__(self, index):
        train_data_dir = pd.read_csv(train_str_data_dir)

        stock = train_data_dir.iloc[index, 0]
        start_date_str = train_data_dir.iloc[index, 1]
        start_date = pd.Timestamp(start_date_str)
        stage = train_data_dir.iloc[index, 2]

        pickle_dir = os.path.join(stock_data_dir, f"{stock}.pkl")
        stock_data = pd.read_pickle(pickle_dir)

        start_date_index = stock_data.index.get_loc(start_date)
        end_date_index = start_date_index + WEEKS_IN_YEAR * 2

        desired_data = stock_data.iloc[start_date_index:end_date_index]
        plotstock(desired_data, "cache")

        stock_img = Image.open(f"{desired_data}.png")  
        stock_volume_img = Image.open(f"{desired_data}_volume.png")



        # if self.transform is not None:
        #     augmentations = self.transform(image=image, mask=mask)
        #     image = augmentations["image"]
        #     mask = augmentations["mask"]

        return stock_img, stock_volume_img, stage
    

if __name__ == "__main__":
