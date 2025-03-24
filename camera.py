import numpy as np


class Camera:
    def __init__(self, position=np.array([0, 0, 0]), fov=60):
        self.position = position
        self.focal_length = 1.0 / np.tan(np.radians(fov) / 2)

    def get_view_vector(self, point):
        return point - self.position
