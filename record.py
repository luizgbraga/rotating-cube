import argparse

import cv2

from animation import CubeAnimation
from utils import progress_bar


class CubeVideoRecorder:
    def __init__(self, texture_path: str, duration=10.0, speed=1.0):
        self.output_path = "cube_animation.mp4"
        self.duration = duration
        self.fps = 30
        self.screen_width = 600
        self.screen_height = 600

        self.animation = CubeAnimation(
            texture_path=texture_path,
            screen_width=self.screen_width,
            screen_height=self.screen_height,
            speed=speed,
        )

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.video_writer = cv2.VideoWriter(
            self.output_path, fourcc, self.fps, (self.screen_width, self.screen_height)
        )

    def record(self):
        total_frames = int(self.duration * self.fps)

        for i in range(total_frames):
            self.animation.update()
            frame = self.animation.render()

            self.video_writer.write(frame)

            progress_bar(i + 1, total_frames)

        self.video_writer.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Record Cube Animation")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--duration", type=float, default=10.0)
    args = parser.parse_args()

    recorder = CubeVideoRecorder(
        texture_path="image.png", duration=args.duration, speed=args.speed
    )

    recorder.record()
