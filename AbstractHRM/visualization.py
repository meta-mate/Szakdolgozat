import cv2
import numpy as np

class Visualization():

    def draw_grid(grid, square_size=15):

        color_list = [
            [0, 0, 0],
            [0, 0, 0],
            [30, 147, 255],
            [249, 60, 49],
            [79, 204, 48],
            [255, 220, 0],
            [153, 153, 153],
            [229, 58, 163],
            [255, 133, 27],
            [135, 216, 241],
            [146, 18, 49]
        ]

        for color in color_list:
            color.reverse()

        y = len(grid)
        x = len(grid[0])

        img = np.zeros((y * square_size, x * square_size, 3), dtype=np.uint8)

        for i in range(y):
            for j in range(x):
                start = (j * square_size, i * square_size)
                end = tuple(s + square_size for s in start)
                index = grid[i][j]
                color = color_list[index]
                cv2.rectangle(img, start, end, color, -1)
                if index != 0:
                    cv2.rectangle(img, start, end, (127,127,127), 2)

        cv2.rectangle(img, (0, 0), (y * square_size, x * square_size), (255, 255, 255), 2)

        return img