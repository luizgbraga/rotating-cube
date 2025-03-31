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

        x_projected = (vertices_pos_cam[:, 0] * self.camera.focal_length) / z
        y_projected = (vertices_pos_cam[:, 1] * self.camera.focal_length) / z

        projected = np.zeros((len(vertices), 2))
        projected[:, 0] = (
            x_projected * self.aspect_ratio * self.height / 2 + self.width / 2
        )
        projected[:, 1] = y_projected * self.height / 2 + self.height / 2

        return projected

    def get_face_homography(self, texture, face_vertices_2d):
        h, w = texture.shape[:2]
        src_points = np.array(face_vertices_2d, dtype=np.float32)
        dst_points = np.array(
            [[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32
        )

        H, _ = cv2.findHomography(src_points, dst_points)

        return H

    def map_texture_to_face(self, texture, face_vertices_2d):
        face_vertices_2d = np.array(face_vertices_2d, dtype=np.int32)

        H = self.get_face_homography(texture, face_vertices_2d)

        mask = np.zeros((self.height, self.width), dtype=np.uint8)
        cv2.fillConvexPoly(
            mask, face_vertices_2d, 255
        )  # Coordinates inside the face are set to 255
        y_face_indices, x_face_indices = np.where(mask == 255)
        face_pixels = np.column_stack(
            (x_face_indices, y_face_indices, np.ones_like(x_face_indices))
        )

        face_image = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        src_pixels = np.dot(face_pixels, H.T)
        src_pixels[:, 0] /= src_pixels[:, 2]
        src_pixels[:, 1] /= src_pixels[:, 2]

        src_x = src_pixels[:, 0].astype(int)
        src_y = src_pixels[:, 1].astype(int)

        for i in range(len(x_face_indices)):
            if 0 <= src_y[i] < texture.shape[0] and 0 <= src_x[i] < texture.shape[1]:
                texture_color = texture[src_y[i], src_x[i]][:3]
                bgr_color = texture_color[::-1]  # OpenCV uses BGR
                face_image[y_face_indices[i], x_face_indices[i]] = bgr_color

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
