from time import time

import cv2
import numpy as np
from PIL import Image

from camera import Camera
from cube import Cube
from renderer import Renderer


class CubeAnimation:
    def __init__(self, texture_path: str, screen_width=600, screen_height=600):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.texture = np.array(Image.open(texture_path).convert("RGB"))

        self.camera = Camera(position=np.array([0, 0, -5]), fov=60)
        self.cube = Cube(size=2, texture=self.texture)
        self.renderer = Renderer(screen_width, screen_height, self.camera)

        self.start_time = time()
        self.last_update = 0
        self.frame_count = 0

    def update(self):
        current_time = time() - self.start_time

        angle_x = current_time * 0.3
        angle_y = current_time * 0.5
        angle_z = current_time * 0.1

        self.cube.rotate(angle_x, angle_y, angle_z)

    def render(self):
        return self.renderer.render(self.cube)

    def run(self):
        while True:
            self.update()
            frame = self.render()
            cv2.imshow("Rotating Cube", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cv2.destroyAllWindows()


if __name__ == "__main__":
    animation = CubeAnimation("image.png")
    animation.run()
