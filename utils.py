import sys

import numpy as np


def compute_rotation_matrices(angle_x, angle_y, angle_z):
    cos_x, sin_x = np.cos(angle_x), np.sin(angle_x)
    cos_y, sin_y = np.cos(angle_y), np.sin(angle_y)
    cos_z, sin_z = np.cos(angle_z), np.sin(angle_z)

    rotation_x = np.array([[1, 0, 0], [0, cos_x, -sin_x], [0, sin_x, cos_x]])
    rotation_y = np.array([[cos_y, 0, sin_y], [0, 1, 0], [-sin_y, 0, cos_y]])
    rotation_z = np.array([[cos_z, -sin_z, 0], [sin_z, cos_z, 0], [0, 0, 1]])

    return rotation_x, rotation_y, rotation_z


def progress_bar(current, total, bar_length=50):
    percent = float(current) / total
    arrow = "█" * int(round(percent * bar_length))
    spaces = " " * (bar_length - len(arrow))

    sys.stdout.write(f"\rProgress: [{arrow}{spaces}] {int(percent * 100)}%")
    sys.stdout.flush()

    if current == total:
        sys.stdout.write("\n")
