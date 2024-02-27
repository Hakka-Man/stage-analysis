import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from utils import create_dir_if_not_exist

bull_color = "red"
bear_color = "green"
wma30_color = "blue"
fyh_color = "orange"

box_width = 5
stick_width = 1
fig_size = (10, 5)
line_width = 2
scale_factor = 0.003

images_folder = "../stock_image"
create_dir_if_not_exist(images_folder)


def plotstock(df: pd.DataFrame, output_plot_dir: str):
    up = df[df.close >= df.open]
    down = df[df.close < df.open]


    fig, ax1 = plt.subplots(figsize=fig_size)
    # fig = plt.figure()
    # ax1.set_ylim(0, 2000)
    
    ax1.set_ylim(bottom=0, top=10000000)

    # ax1 = fig.add_axes([0.1, 0.1, 0.8, 0.4])

    ax1.bar(df.index, df.volume*scale_factor, width=5)

    ax2 = ax1.twinx()
    # ax2 = fig.add_axes([0.1, 0.1, 0.8, 1])


    ax2.bar(up.index, up.close-up.open, box_width, bottom=up.open, color=bull_color) 
    ax2.bar(up.index, up.high-up.close, stick_width, bottom=up.close, color=bull_color) 
    ax2.bar(up.index, up.low-up.open, stick_width, bottom=up.open, color=bull_color)

    ax2.bar(down.index, down.close-down.open, box_width, bottom=down.open, color=bear_color) 
    ax2.bar(down.index, down.high-down.open, stick_width, bottom=down.open, color=bear_color) 
    ax2.bar(down.index, down.low-down.close, stick_width, bottom=down.close, color=bear_color) 

    ax2.plot(df.wma30, color=wma30_color, linewidth=line_width)
    ax2.axhline(df.fyh[-1], color=fyh_color, linewidth=line_width)

    # ax1.plot(x_data1, y_data1, color='blue', label='Graph 1')

    # plt.figure(figsize=fig_size)
    # plt.xticks(rotation=30, ha='right') 
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax2.set_xticks([])
    ax2.set_yticks([])

    plt.gcf().set_facecolor('white')
    plt.savefig(os.path.join(images_folder, f"{output_plot_dir}.png"), bbox_inches='tight', pad_inches=0, dpi=100)

    # plt.figure(figsize=fig_size)
    # # plt.xticks(rotation=30, ha='right')
    # plt.xticks([]) 
    # plt.yticks([]) 

    # plt.savefig(os.path.join(images_folder, f"{output_plot_dir}_volume.png"), bbox_inches='tight', pad_inches=0, dpi=100)