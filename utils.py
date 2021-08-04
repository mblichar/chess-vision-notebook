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


def merge_lines(lines: List[Line], rho_threshold, theta_threshold):
    merged_lines = []
    for line in lines:
        found = False
        rho = line.rho
        theta = line.theta

        for idx, grouped_line in enumerate(merged_lines):
            if abs(grouped_line.rho - rho) < rho_threshold and abs(grouped_line.theta - theta) < theta_threshold:
                merged_lines[idx] = Line((grouped_line.rho + rho) / 2, (grouped_line.theta + theta) / 2)
                found = True
                break
        if not found:
            merged_lines.append(Line(rho, theta))

    return merged_lines


def lines_intersection_point(a: Line, b: Line):
    sin_theta_a = math.sin(a.theta)
    sin_theta_b = math.sin(b.theta)
    cos_theta_a = math.cos(a.theta)
    cos_theta_b = math.cos(b.theta)

    x = (b.rho * sin_theta_a - a.rho * sin_theta_b) / (cos_theta_b * sin_theta_a - cos_theta_a * sin_theta_b)
    y = (a.rho - x * cos_theta_a) / sin_theta_a

    return x, y


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
