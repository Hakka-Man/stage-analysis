import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from utils import create_dir_if_not_exist
from tqdm import tqdm

train_str_data_dir = '../train_data/mock.csv'
stock_data_dir = '../stock_data/'
images_folder = "../stock_image"
WEEKS_IN_YEAR = 52

bull_color = "red"
bear_color = "green"
wma30_color = "blue"
fyh_color = "orange"

bar_width = 0.8
stick_width = 0.1
fig_size = (10, 1)
line_width = 2

width = [0.8] * 104

images_folder = "../stock_image"
create_dir_if_not_exist(images_folder)


def plotstock(df: pd.DataFrame, output_plot_dir: str):
    df = df.reset_index()
    up = df[df.close >= df.open]
    down = df[df.close < df.open]

    plt.style.use('dark_background')
    plt.figure(figsize=fig_size, dpi=80)
    figure, axis = plt.subplots(2, height_ratios=[3,1])

    axis[0].margins(x=0.0, y=0.1)
    axis[1].margins(x=0.0, y=0.1)

    # Draw the price history
    axis[0].bar(up.index, up.close-up.open, bar_width, bottom=up.open, color=bull_color) 
    axis[0].bar(up.index, up.high-up.close, stick_width, bottom=up.close, color=bull_color) 
    axis[0].bar(up.index, up.low-up.open, stick_width, bottom=up.open, color=bull_color)

    axis[0].bar(down.index, down.close-down.open, bar_width, bottom=down.open, color=bear_color) 
    axis[0].bar(down.index, down.high-down.open, stick_width, bottom=down.open, color=bear_color) 
    axis[0].bar(down.index, down.low-down.close, stick_width, bottom=down.close, color=bear_color)

    axis[0].plot(df.wma30, color=wma30_color, linewidth=line_width)
    axis[0].axhline(df.iloc[-1].fyh, color=fyh_color, linewidth=line_width)

    axis[0].axes.get_xaxis().set_visible(False)
    axis[0].axes.get_yaxis().set_visible(False)

    axis[1].bar(df.index, df["volume"].values, width=width)
    axis[1].axes.get_xaxis().set_visible(False)
    axis[1].axes.get_yaxis().set_visible(False)

    plt.gcf().set_facecolor('black')
    plt.savefig(os.path.join(images_folder, f"{output_plot_dir}.png"), bbox_inches='tight', pad_inches=0, dpi=100)
    plt.close('all')

def get_images():
    train_data = pd.read_csv(train_str_data_dir)
    for ind in tqdm(range(len(train_data))):
        stock = train_data.iloc[ind, 0]
        start_date_str = train_data.iloc[ind, 1]
        start_date = pd.Timestamp(start_date_str)
        stage = str(int(float(train_data.iloc[ind, 2])))
        pickle_dir = os.path.join(stock_data_dir, f"{stock}.pkl")
        stock_data = pd.read_pickle(pickle_dir)
        start_date_index = stock_data.index.get_loc(start_date)
        end_date_index = start_date_index + WEEKS_IN_YEAR * 2
        desired_data = stock_data.iloc[start_date_index:end_date_index]
        plotstock(desired_data, stage + "/" + stock + "_" + start_date_str)


if __name__ == "__main__":
    get_images()