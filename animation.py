import argparse
from time import time

import cv2
import numpy as np
from PIL import Image

from camera import Camera
from cube import Cube
from renderer import Renderer


class CubeAnimation:
    def __init__(
        self, texture_path: str, screen_width=600, screen_height=600, speed=1.0
    ):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.speed = speed
        self.texture = np.array(Image.open(texture_path))

        self.camera = Camera(position=np.array([0, 0, -5]), fov=60)
        self.cube = Cube(size=2, texture=self.texture)
        self.renderer = Renderer(screen_width, screen_height, self.camera)

        self.start_time = time()

    def update(self):
        dt = time() - self.start_time

        angle_x = dt * 0.3 * self.speed
        angle_y = dt * 0.5 * self.speed
        angle_z = dt * 0.1 * self.speed

        self.cube.rotate(angle_x, angle_y, angle_z)

    def render(self):
        return self.renderer.render(self.cube)

    def run(self):
        while True:
            self.update()
            frame = self.render()
            cv2.imshow("Cube", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cube Animation")
    parser.add_argument("--speed", type=float, default=1.5)
    args = parser.parse_args()

    animation = CubeAnimation("image.png", speed=args.speed)
    animation.run()
