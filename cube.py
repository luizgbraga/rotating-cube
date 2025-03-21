import numpy as np

from utils import compute_rotation_matrices, normalize


class TexturedFace:
    def __init__(self, vertices_indices):
        self.vertices_indices = vertices_indices
        self.normal = None
        self.center = None
        self.depth = None

    def update(self, transformed_vertices):
        face_vertices = transformed_vertices[self.vertices_indices]
        self.center = np.mean(face_vertices, axis=0)
        self.depth = self.center[2]

        v0, v1, v2 = face_vertices[0], face_vertices[1], face_vertices[2]
        edge1 = v1 - v0
        edge2 = v2 - v0
        self.normal = normalize(np.cross(edge1, edge2))


class Cube:
    def __init__(self, size=1.0, texture=None):
        self.size = size
        self.texture = texture
        self.vertices = self._define_vertices()
        self.faces = self._define_faces()
        self.transformed_vertices = self.vertices.copy()

    def _define_vertices(self):
        return (self.size / 2) * np.array(
            [
                [-1, -1, -1],  # 0: front bottom left
                [1, -1, -1],  # 1: front bottom right
                [1, 1, -1],  # 2: front top right
                [-1, 1, -1],  # 3: front top left
                [-1, -1, 1],  # 4: back bottom left
                [1, -1, 1],  # 5: back bottom right
                [1, 1, 1],  # 6: back top right
                [-1, 1, 1],  # 7: back top left
            ],
            dtype=float,
        )

    def _define_faces(self):
        face_indices = [
            [0, 1, 2, 3],  # front face (-z)
            [4, 5, 6, 7],  # back face (+z)
            [0, 1, 5, 4],  # bottom face (-y)
            [2, 3, 7, 6],  # top face (+y)
            [0, 3, 7, 4],  # left face (-x)
            [1, 2, 6, 5],  # right face (+x)
        ]

        return [TexturedFace(indices) for indices in face_indices]

    def rotate(self, angle_x, angle_y, angle_z):
        rotation_x, rotation_y, rotation_z = compute_rotation_matrices(
            angle_x, angle_y, angle_z
        )

        composed_rotations = np.dot(np.dot(rotation_x, rotation_y), rotation_z)
        self.transformed_vertices = np.dot(self.vertices, composed_rotations.T)

        for face in self.faces:
            face.update(self.transformed_vertices)
