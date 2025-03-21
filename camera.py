import numpy as np


class Camera:
    def __init__(self, position=np.array([0, 0, -5]), fov=60):
        self.position = position
        self.fov = fov
        self.fov_rad = np.radians(fov)
        self.focal_length = 1.0 / np.tan(self.fov_rad / 2)

    def get_view_vector(self, point):
        return point - self.position
