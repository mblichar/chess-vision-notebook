import math
from typing import List

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt


class Line:
    def __init__(self, rho, theta):
        self.rho = rho
        self.theta = theta

    def __str__(self):
        return f'rho={self.rho} theta={self.theta}'


def process_hough_lines(lines):
    return list(map(lambda l: Line(l[0][0], l[0][1]), lines))


def group_lines(lines: List[Line], rho_threshold, theta_threshold):
    grouped_lines = []
    for line in lines:
        found = False
        rho = line.rho
        theta = line.theta

        for idx, grouped_line in enumerate(grouped_lines):
            if abs(grouped_line.rho - rho) < rho_threshold and abs(grouped_line.theta - theta) < theta_threshold:
                grouped_lines[idx] = Line((grouped_line.rho + rho) / 2, (grouped_line.theta + theta) / 2)
                found = True
                break
        if not found:
            grouped_lines.append(Line(rho, theta))

    return grouped_lines


def invert(point):
    return point[1], point[0]


def show_image(image, convert=cv.COLOR_BGR2RGB):
    fig, ax = plt.subplots(figsize=(20, 20))
    plt.axis('off')
    ax.imshow(cv.cvtColor(image, convert))


def draw_lines(lines: List[Line], image, color):
    for line in lines:
        rho = line.rho
        theta = line.theta
        a = math.cos(theta)
        b = math.sin(theta)
        x0 = a * rho
        y0 = b * rho
        pt1 = (int(x0 + 1000 * -b), int(y0 + 1000 * a))
        pt2 = (int(x0 - 1000 * -b), int(y0 - 1000 * a))
        cv.line(image, pt1, pt2, color, 3, cv.LINE_AA)


def distinct_colors(n):
    step = 360 / n
    hue = 0
    for i in range(0, n):
        hue += step
        hsl = [hue / 2, 255, 255]
        dst = cv.cvtColor(np.uint8([[hsl]]), cv.COLOR_HSV2BGR)
        yield dst[0][0].tolist()


def show_lines(lines: List[Line], image, color):
    image_copy = image.copy()
    draw_lines(lines, image_copy, color)
    show_image(image_copy)
