#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
November 2024
@author: Thomas Bonald <bonald@enst.fr>
"""
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation as anim
plt.rcParams["animation.html"] = "jshtml"


def display_position(image, position=None, positions=None, marker='o', marker_size=200, marker_color='b', interval=200):
    fig, ax = plt.subplots()
    ax.axis('off')
    ax.imshow(image)
    if positions is not None:
        if type(marker_color) == str:
            marker_color = len(positions) * [marker_color]
        
        y, x = positions[0]
        image = ax.scatter(x, y, marker=marker, s=marker_size, c=marker_color[0])

        def update(i):
            y_, x_ = positions[i]
            image.set_offsets(np.vstack((x_, y_)).T)
            image.set_color(marker_color[i])

        animation = anim.FuncAnimation(fig, update, frames=len(positions), interval=interval, repeat=False)
        return animation
    elif position is not None:
        y, x = position
        ax.scatter(x, y, marker=marker, s=marker_size, c=marker_color)


def display_board(image, board=None, boards=None, marker1='x', marker2='o', marker_size=200,
                  color1='b', color2='r', interval=200):
    fig, ax = plt.subplots()
    ax.axis('off')
    ax.imshow(image)
    if boards is not None:
        board = boards[0]
        y, x = np.where(board > 0)
        player1 = ax.scatter(x, y, marker=marker1, s=marker_size, c=color1)
        y, x = np.where(board < 0)
        player2 = ax.scatter(x, y, marker=marker2, s=marker_size, c=color2)

        def update(i):
            board_ = boards[i]
            y_, x_ = np.where(board_ > 0)
            player1.set_offsets(np.vstack((x_, y_)).T)
            y_, x_ = np.where(board_ < 0)
            player2.set_offsets(np.vstack((x_, y_)).T)

        animation = anim.FuncAnimation(fig, update, frames=len(boards), interval=interval, repeat=False)
        return animation
    elif board is not None:
        y, x = np.where(board > 0)
        ax.scatter(x, y, marker=marker1, s=marker_size, c=color1)
        y, x = np.where(board < 0)
        ax.scatter(x, y, marker=marker2, s=marker_size, c=color2)
