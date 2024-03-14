import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from utils import create_dir_if_not_exist

bull_color = "red"
bear_color = "green"
wma30_color = "blue"
fyh_color = "orange"

bar_width = 0.8
stick_width = 0.1
fig_size = (10, 1)
line_width = 2

width = [0.8]*104


images_folder = "../stock_image"
create_dir_if_not_exist(images_folder)


def plotstock(df: pd.DataFrame, output_plot_dir: str):
    df = df.reset_index()
    up = df[df.close >= df.open]
    down = df[df.close < df.open]

    plt.figure(figsize=fig_size, dpi=80)
    plt.style.use('default')
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

    # Draw the volume
    # print(df["volume"].values)
    axis[1].bar(df.index, df["volume"].values, width=width)
    
    axis[1].axes.get_xaxis().set_visible(False)
    axis[1].axes.get_yaxis().set_visible(False)

    # fig, ax1 = plt.subplots(figsize=fig_size)
    

    # ax1.bar(df.index, df.volume*scale_factor, width=5)

    # ax2 = ax1.twinx()

    # ax2.bar(up.index, up.close-up.open, box_width, bottom=up.open, color=bull_color) 
    # ax2.bar(up.index, up.high-up.close, stick_width, bottom=up.close, color=bull_color) 
    # ax2.bar(up.index, up.low-up.open, stick_width, bottom=up.open, color=bull_color)

    # ax2.bar(down.index, down.close-down.open, box_width, bottom=down.open, color=bear_color) 
    # ax2.bar(down.index, down.high-down.open, stick_width, bottom=down.open, color=bear_color) 
    # ax2.bar(down.index, down.low-down.close, stick_width, bottom=down.close, color=bear_color) 

    # ax2.plot(df.wma30, color=wma30_color, linewidth=line_width)
    # ax2.axhline(df.fyh[-1], color=fyh_color, linewidth=line_width)

    # # ax1.plot(x_data1, y_data1, color='blue', label='Graph 1')

    # # plt.figure(figsize=fig_size)
    # # plt.xticks(rotation=30, ha='right') 
    # ax1.set_xticks([])
    # ax1.set_yticks([])
    # ax2.set_xticks([])
    # ax2.set_yticks([])

    plt.gcf().set_facecolor('white')
    plt.savefig(os.path.join(images_folder, f"{output_plot_dir}.png"), bbox_inches='tight', pad_inches=0, dpi=100)
    plt.close('all')

    # plt.figure(figsize=fig_size)
    # # plt.xticks(rotation=30, ha='right')
    # plt.xticks([]) 
    # plt.yticks([]) 

    # plt.savefig(os.path.join(images_folder, f"{output_plot_dir}_volume.png"), bbox_inches='tight', pad_inches=0, dpi=100)