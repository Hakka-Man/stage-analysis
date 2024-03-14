from plotstock import plotstock
import pandas as pd
import os
from PIL import Image
import random
import torchvision.transforms as transforms
# from torchvision.transforms.functional import pil_to_tensor
from torchvision.transforms.functional import to_pil_image


train_str_data_dir = '../train_data/mock.csv'
stock_data_dir = '../stock_data/'
images_folder = "../stock_image"

WEEKS_IN_YEAR = 52

train_data_dir = pd.read_csv(train_str_data_dir)

def get_images(ind):
    stock = train_data_dir.iloc[ind, 0]
    start_date_str = train_data_dir.iloc[ind, 1]
    start_date = pd.Timestamp(start_date_str)
    stage = str(int(float(train_data_dir.iloc[ind, 2])))
    pickle_dir = os.path.join(stock_data_dir, f"{stock}.pkl")
    stock_data = pd.read_pickle(pickle_dir)
    start_date_index = stock_data.index.get_loc(start_date)
    end_date_index = start_date_index + WEEKS_IN_YEAR * 2
    desired_data = stock_data.iloc[start_date_index:end_date_index]
    plotstock(desired_data, stage + "/" + stock + "_" + start_date_str)
    return stock

len(train_data_dir)
prev_stock = ""
for i in range(len(train_data_dir)):
    ticker = get_images(i)
    if ticker != prev_stock:
        print(ticker)
        prev_stock = ticker