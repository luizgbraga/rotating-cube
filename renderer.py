import cv2
import numpy as np


class Renderer:
    def __init__(self, width, height, camera):
        self.width = width
        self.height = height
        self.camera = camera
        self.aspect_ratio = width / height

    def project_vertices(self, vertices):
        vertices_pos_cam = vertices - self.camera.position
        z = vertices_pos_cam[:, 2]

        z_safe = np.where(np.abs(z) < 1e-10, 1e-10, z)  # avoid division by zero

        focal_z_ratio = self.camera.focal_length / z_safe
        x_projected = vertices_pos_cam[:, 0] * focal_z_ratio
        y_projected = vertices_pos_cam[:, 1] * focal_z_ratio

        projected = np.zeros((len(vertices), 2))
        projected[:, 0] = (
            x_projected * self.aspect_ratio * self.height / 2 + self.width / 2
        )
        projected[:, 1] = y_projected * self.height / 2 + self.height / 2

        return projected

    def map_texture_to_face(self, texture, face_vertices_2d):
        face_vertices_2d = np.array(face_vertices_2d, dtype=np.int32)

        mask = np.zeros((self.height, self.width), dtype=np.uint8)
        cv2.fillConvexPoly(mask, face_vertices_2d, 255)

        h, w = texture.shape[:2]
        src_points = np.array(
            [[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32
        )

        dst_points = np.array(face_vertices_2d, dtype=np.float32)

        H, _ = cv2.findHomography(src_points, dst_points)
        H_inv = np.linalg.inv(H)

        face_image = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        y_indices, x_indices = np.where(mask > 0)

        pixels = np.column_stack((x_indices, y_indices, np.ones_like(x_indices)))
        src_pixels = np.dot(pixels, H_inv.T)

        non_zero_mask = np.abs(src_pixels[:, 2]) > 1e-10  # avoid division by zero

        src_pixels_safe = src_pixels[non_zero_mask]
        x_indices_safe = x_indices[non_zero_mask]
        y_indices_safe = y_indices[non_zero_mask]

        src_pixels_safe[:, 0] /= src_pixels_safe[:, 2]
        src_pixels_safe[:, 1] /= src_pixels_safe[:, 2]

        src_x = src_pixels_safe[:, 0].astype(int)
        src_y = src_pixels_safe[:, 1].astype(int)

        valid_mask = (src_x >= 0) & (src_x < w) & (src_y >= 0) & (src_y < h)

        valid_dest_x = x_indices_safe[valid_mask]
        valid_dest_y = y_indices_safe[valid_mask]
        valid_src_x = src_x[valid_mask]
        valid_src_y = src_y[valid_mask]

        if len(valid_dest_x) > 0 and len(valid_src_x) > 0:
            channels = min(texture.shape[2], 3)
            for i in range(len(valid_dest_x)):
                if (
                    0 <= valid_src_y[i] < texture.shape[0]
                    and 0 <= valid_src_x[i] < texture.shape[1]
                ):
                    face_image[valid_dest_y[i], valid_dest_x[i]] = texture[
                        valid_src_y[i], valid_src_x[i]
                    ][:channels]

        return face_image, mask

    def render(self, obj):
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        projected_vertices = self.project_vertices(obj.transformed_vertices)

        visible_faces = [face for face in obj.faces]
        visible_faces.sort(key=lambda face: face.depth, reverse=True)

        for face in visible_faces:
            face_vertices_2d = projected_vertices[face.vertices_indices]
            face_image, mask = self.map_texture_to_face(obj.texture, face_vertices_2d)
            mask_3d = np.stack([mask, mask, mask], axis=2)
            frame = np.where(mask_3d > 0, face_image, frame)

        return frame
