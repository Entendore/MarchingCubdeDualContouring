import sys
import time
import json
import os
import struct
import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.Qt3DCore import Qt3DCore
from PySide6.Qt3DExtras import Qt3DExtras
from PySide6.Qt3DRender import Qt3DRender

try:
    from skimage.measure import marching_cubes
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False

try:
    from numba import njit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

# ==========================================
# ALGORITHMS (NUMBA OPTIMIZED)
# ==========================================

if HAS_NUMBA:
    @njit(cache=True)
    def compute_vertex_normals_numba(vertices, faces):
        normals = np.zeros_like(vertices)
        for i in range(faces.shape[0]):
            i0, i1, i2 = faces[i, 0], faces[i, 1], faces[i, 2]
            p0, p1, p2 = vertices[i0], vertices[i1], vertices[i2]
            ux, uy, uz = p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]
            vx, vy, vz = p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]
            nx = uy * vz - uz * vy
            ny = uz * vx - ux * vz
            nz = ux * vy - uy * vx
            length = np.sqrt(nx*nx + ny*ny + nz*nz)
            if length > 1e-8:
                inv_len = 1.0 / length
                nx *= inv_len; ny *= inv_len; nz *= inv_len
            normals[i0, 0] += nx; normals[i0, 1] += ny; normals[i0, 2] += nz
            normals[i1, 0] += nx; normals[i1, 1] += ny; normals[i1, 2] += nz
            normals[i2, 0] += nx; normals[i2, 1] += ny; normals[i2, 2] += nz
        for i in range(normals.shape[0]):
            length = np.sqrt(normals[i,0]**2 + normals[i,1]**2 + normals[i,2]**2)
            if length > 1e-8:
                inv_len = 1.0 / length
                normals[i,0] *= inv_len; normals[i,1] *= inv_len; normals[i,2] *= inv_len
        return normals

    @njit(cache=True)
    def faces_to_lines_numba(faces):
        if len(faces) == 0: return np.zeros(0, dtype=np.uint32)
        line_indices = np.empty(len(faces) * 6, dtype=np.uint32)
        for i in range(len(faces)):
            idx = i * 6
            line_indices[idx]     = faces[i, 0]
            line_indices[idx + 1] = faces[i, 1]
            line_indices[idx + 2] = faces[i, 1]
            line_indices[idx + 3] = faces[i, 2]
            line_indices[idx + 4] = faces[i, 2]
            line_indices[idx + 5] = faces[i, 0]
        return line_indices

    @njit(cache=True)
    def calculate_surface_area_numba(verts, faces):
        if len(faces) == 0: return 0.0
        area = 0.0
        for i in range(len(faces)):
            p0 = verts[faces[i, 0]]
            p1 = verts[faces[i, 1]]
            p2 = verts[faces[i, 2]]
            ux, uy, uz = p1[0]-p0[0], p1[1]-p0[1], p1[2]-p0[2]
            vx, vy, vz = p2[0]-p0[0], p2[1]-p0[1], p2[2]-p0[2]
            cx = uy*vz - uz*vy
            cy = uz*vx - ux*vz
            cz = ux*vy - uy*vx
            area += np.sqrt(cx*cx + cy*cy + cz*cz) * 0.5
        return area

    @njit(cache=True)
    def calculate_volume_numba(verts, faces):
        if len(faces) == 0: return 0.0
        vol = 0.0
        for i in range(len(faces)):
            p0 = verts[faces[i, 0]]
            p1 = verts[faces[i, 1]]
            p2 = verts[faces[i, 2]]
            vol += (p0[0]*(p1[1]*p2[2] - p1[2]*p2[1]) +
                    p0[1]*(p1[2]*p2[0] - p1[0]*p2[2]) +
                    p0[2]*(p1[0]*p2[1] - p1[1]*p2[0])) / 6.0
        return abs(vol)

    @njit(cache=True)
    def get_heatmap_colors_numba(verts, errors, max_error):
        if len(verts) == 0: return np.zeros((0, 3), dtype=np.float32)
        colors = np.zeros((len(verts), 3), dtype=np.float32)
        for i in range(len(verts)):
            t = errors[i] / max_error if max_error > 0 else 0.0
            if t > 1.0: t = 1.0
            colors[i, 0] = t
            colors[i, 1] = 0.2
            colors[i, 2] = 1.0 - t
        return colors

    @njit(cache=True)
    def find_active_voxels_numba(field, isovalue):
        N = field.shape[0]
        max_voxels = (N - 1) * (N - 1) * (N - 1)
        active_voxels = np.zeros((max_voxels, 3), dtype=np.int32)
        count = 0
        for x in range(N - 1):
            for y in range(N - 1):
                for z in range(N - 1):
                    f_min = field[x, y, z]
                    f_max = field[x, y, z]
                    v = field[x+1, y, z]; f_min = min(f_min, v); f_max = max(f_max, v)
                    v = field[x, y+1, z]; f_min = min(f_min, v); f_max = max(f_max, v)
                    v = field[x+1, y+1, z]; f_min = min(f_min, v); f_max = max(f_max, v)
                    v = field[x, y, z+1]; f_min = min(f_min, v); f_max = max(f_max, v)
                    v = field[x+1, y, z+1]; f_min = min(f_min, v); f_max = max(f_max, v)
                    v = field[x, y+1, z+1]; f_min = min(f_min, v); f_max = max(f_max, v)
                    v = field[x+1, y+1, z+1]; f_min = min(f_min, v); f_max = max(f_max, v)
                    if f_min < isovalue and f_max > isovalue:
                        active_voxels[count, 0] = x
                        active_voxels[count, 1] = y
                        active_voxels[count, 2] = z
                        count += 1
        return active_voxels[:count]

    @njit(cache=True)
    def _solve_qef_3x3(A, b):
        det = A[0,0]*(A[1,1]*A[2,2] - A[1,2]*A[2,1]) - \
              A[0,1]*(A[1,0]*A[2,2] - A[1,2]*A[2,0]) + \
              A[0,2]*(A[1,0]*A[2,1] - A[1,1]*A[2,0])
        if abs(det) < 1e-7: return False, 0.0, 0.0, 0.0
        inv_det = 1.0 / det
        x0 = ((A[1,1]*A[2,2] - A[1,2]*A[2,1])*b[0] - (A[0,1]*A[2,2] - A[0,2]*A[2,1])*b[1] + (A[0,1]*A[1,2] - A[0,2]*A[1,1])*b[2]) * inv_det
        x1 = (-(A[1,0]*A[2,2] - A[1,2]*A[2,0])*b[0] + (A[0,0]*A[2,2] - A[0,2]*A[2,0])*b[1] - (A[0,0]*A[1,2] - A[0,2]*A[1,0])*b[2]) * inv_det
        x2 = ((A[1,0]*A[2,1] - A[1,1]*A[2,0])*b[0] - (A[0,0]*A[2,1] - A[0,1]*A[2,0])*b[1] + (A[0,0]*A[1,1] - A[0,1]*A[1,0])*b[2]) * inv_det
        return True, x0, x1, x2

    @njit(cache=True)
    def run_dual_algorithms(field, grad, active_voxels, N, isovalue, max_error):
        num_active = len(active_voxels)
        sn_verts = np.zeros((num_active, 3), dtype=np.float32)
        dc_verts = np.zeros((num_active, 3), dtype=np.float32)
        dc_errors = np.zeros(num_active, dtype=np.float32)
        vert_idx_map = np.full((N+2, N+2, N+2), -1, dtype=np.int32)
        
        int_points = np.zeros((num_active * 12, 3), dtype=np.float32)
        int_normals = np.zeros((num_active * 12, 3), dtype=np.float32)
        int_count = 0
        
        corners = np.array([[0,0,0], [1,0,0], [0,1,0], [1,1,0], [0,0,1], [1,0,1], [0,1,1], [1,1,1]], dtype=np.int32)
        edges = np.array([[0,1], [2,3], [4,5], [6,7], [0,2], [1,3], [4,6], [5,7], [0,4], [1,5], [2,6], [3,7]], dtype=np.int32)
        
        for i in range(num_active):
            x, y, z = active_voxels[i]
            c_vals = np.zeros(8, dtype=np.float32)
            for c in range(8):
                c_vals[c] = field[x + corners[c,0], y + corners[c,1], z + corners[c,2]]
                
            A = np.zeros((3, 3), dtype=np.float32)
            b = np.zeros(3, dtype=np.float32)
            cx_sum, cy_sum, cz_sum = 0.0, 0.0, 0.0
            count = 0
            
            for e in range(12):
                i1, i2 = edges[e, 0], edges[e, 1]
                v1, v2 = c_vals[i1], c_vals[i2]
                if (v1 - isovalue) * (v2 - isovalue) < 0:
                    t = (v1 - isovalue) / (v1 - v2)
                    p1x = x + corners[i1,0]; p1y = y + corners[i1,1]; p1z = z + corners[i1,2]
                    p2x = x + corners[i2,0]; p2y = y + corners[i2,1]; p2z = z + corners[i2,2]
                    px = p1x + t * (p2x - p1x); py = p1y + t * (p2y - p1y); pz = p1z + t * (p2z - p1z)
                    
                    n1x, n1y, n1z = grad[x + corners[i1,0], y + corners[i1,1], z + corners[i1,2]]
                    n2x, n2y, n2z = grad[x + corners[i2,0], y + corners[i2,1], z + corners[i2,2]]
                    nx = n1x * (1 - t) + n2x * t; ny = n1y * (1 - t) + n2y * t; nz = n1z * (1 - t) + n2z * t
                    nn = np.sqrt(nx*nx + ny*ny + nz*nz)
                    if nn > 0:
                        inv_nn = 1.0 / nn; nx *= inv_nn; ny *= inv_nn; nz *= inv_nn
                        
                    cx_sum += px; cy_sum += py; cz_sum += pz
                    A[0,0] += nx * nx; A[0,1] += nx * ny; A[0,2] += nx * nz
                    A[1,0] += ny * nx; A[1,1] += ny * ny; A[1,2] += ny * nz
                    A[2,0] += nz * nx; A[2,1] += nz * ny; A[2,2] += nz * nz
                    dot_pn = px*nx + py*ny + pz*nz
                    b[0] += dot_pn * nx; b[1] += dot_pn * ny; b[2] += dot_pn * nz
                    
                    int_points[int_count, 0] = px; int_points[int_count, 1] = py; int_points[int_count, 2] = pz
                    int_normals[int_count, 0] = nx; int_normals[int_count, 1] = ny; int_normals[int_count, 2] = nz
                    int_count += 1
                    count += 1
                    
            if count > 0:
                inv_count = 1.0 / count
                sn_verts[i, 0] = cx_sum * inv_count; sn_verts[i, 1] = cy_sum * inv_count; sn_verts[i, 2] = cz_sum * inv_count
                
                success, x0, x1, x2 = _solve_qef_3x3(A, b)
                vx, vy, vz = 0.0, 0.0, 0.0
                err = 0.0
                if success:
                    vx, vy, vz = x0, x1, x2
                    for j in range(count):
                        idx = int_count - count + j
                        diff0 = vx - int_points[idx, 0]; diff1 = vy - int_points[idx, 1]; diff2 = vz - int_points[idx, 2]
                        dot = diff0*int_normals[idx, 0] + diff1*int_normals[idx, 1] + diff2*int_normals[idx, 2]
                        err += dot * dot
                else:
                    vx, vy, vz = sn_verts[i, 0], sn_verts[i, 1], sn_verts[i, 2]
                    err = 1e10
                    
                if err > max_error:
                    vx, vy, vz = sn_verts[i, 0], sn_verts[i, 1], sn_verts[i, 2]
                    
                if vx < x: vx = x
                elif vx > x + 1: vx = x + 1
                if vy < y: vy = y
                elif vy > y + 1: vy = y + 1
                if vz < z: vz = z
                elif vz > z + 1: vz = z + 1
                
                dc_verts[i, 0] = vx; dc_verts[i, 1] = vy; dc_verts[i, 2] = vz
                dc_errors[i] = err
                vert_idx_map[x+1, y+1, z+1] = i
                
        max_faces = num_active * 12
        faces_list = np.zeros((max_faces, 3), dtype=np.uint32)
        face_count = 0
        
        for i in range(num_active):
            x, y, z = active_voxels[i]
            if (field[x,y,z] - isovalue) * (field[x+1,y,z] - isovalue) < 0:
                v0 = vert_idx_map[x, y, z]; v1 = vert_idx_map[x, y+1, z]
                v2 = vert_idx_map[x, y+1, z+1]; v3 = vert_idx_map[x, y, z+1]
                if v0 != -1 and v1 != -1 and v2 != -1 and v3 != -1:
                    if field[x,y,z] > isovalue:
                        faces_list[face_count, 0] = v0; faces_list[face_count, 1] = v1; faces_list[face_count, 2] = v2; face_count += 1
                        faces_list[face_count, 0] = v0; faces_list[face_count, 1] = v2; faces_list[face_count, 2] = v3; face_count += 1
                    else:
                        faces_list[face_count, 0] = v0; faces_list[face_count, 1] = v2; faces_list[face_count, 2] = v1; face_count += 1
                        faces_list[face_count, 0] = v0; faces_list[face_count, 1] = v3; faces_list[face_count, 2] = v2; face_count += 1
            if (field[x,y,z] - isovalue) * (field[x,y+1,z] - isovalue) < 0:
                v0 = vert_idx_map[x, y, z]; v1 = vert_idx_map[x, y, z+1]
                v2 = vert_idx_map[x+1, y, z+1]; v3 = vert_idx_map[x+1, y, z]
                if v0 != -1 and v1 != -1 and v2 != -1 and v3 != -1:
                    if field[x,y,z] > isovalue:
                        faces_list[face_count, 0] = v0; faces_list[face_count, 1] = v1; faces_list[face_count, 2] = v2; face_count += 1
                        faces_list[face_count, 0] = v0; faces_list[face_count, 1] = v2; faces_list[face_count, 2] = v3; face_count += 1
                    else:
                        faces_list[face_count, 0] = v0; faces_list[face_count, 1] = v2; faces_list[face_count, 2] = v1; face_count += 1
                        faces_list[face_count, 0] = v0; faces_list[face_count, 1] = v3; faces_list[face_count, 2] = v2; face_count += 1
            if (field[x,y,z] - isovalue) * (field[x,y,z+1] - isovalue) < 0:
                v0 = vert_idx_map[x, y, z]; v1 = vert_idx_map[x+1, y, z]
                v2 = vert_idx_map[x+1, y+1, z]; v3 = vert_idx_map[x, y+1, z]
                if v0 != -1 and v1 != -1 and v2 != -1 and v3 != -1:
                    if field[x,y,z] > isovalue:
                        faces_list[face_count, 0] = v0; faces_list[face_count, 1] = v1; faces_list[face_count, 2] = v2; face_count += 1
                        faces_list[face_count, 0] = v0; faces_list[face_count, 1] = v2; faces_list[face_count, 2] = v3; face_count += 1
                    else:
                        faces_list[face_count, 0] = v0; faces_list[face_count, 1] = v2; faces_list[face_count, 2] = v1; face_count += 1
                        faces_list[face_count, 0] = v0; faces_list[face_count, 1] = v3; faces_list[face_count, 2] = v2; face_count += 1
                        
        return sn_verts, dc_verts, dc_errors, faces_list[:face_count], int_points[:int_count], int_normals[:int_count], active_voxels

def compute_vertex_normals(vertices, faces):
    if len(faces) == 0: return np.zeros_like(vertices)
    if HAS_NUMBA: return compute_vertex_normals_numba(vertices, faces)
    v0, v1, v2 = vertices[faces[:, 0]], vertices[faces[:, 1]], vertices[faces[:, 2]]
    face_normals = np.cross(v1 - v0, v2 - v0)
    norm = np.linalg.norm(face_normals, axis=1, keepdims=True)
    norm[norm == 0] = 1
    face_normals = face_normals / norm
    vertex_normals = np.zeros_like(vertices)
    np.add.at(vertex_normals, faces[:, 0], face_normals)
    np.add.at(vertex_normals, faces[:, 1], face_normals)
    np.add.at(vertex_normals, faces[:, 2], face_normals)
    norm = np.linalg.norm(vertex_normals, axis=1, keepdims=True)
    norm[norm == 0] = 1
    return (vertex_normals / norm).astype(np.float32)

def calculate_metrics(verts, faces):
    if len(verts) == 0 or len(faces) == 0: return 0.0, 0.0
    if HAS_NUMBA:
        return calculate_surface_area_numba(verts, faces), calculate_volume_numba(verts, faces)
    v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    area = np.sum(np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1) / 2.0)
    vol = np.abs(np.sum(np.einsum('ij,ij->i', v0, np.cross(v1, v2)) / 6.0))
    return area, vol

def marching_cubes_algorithm(f, isovalue):
    if not HAS_SKIMAGE: return np.array([]), np.array([])
    verts, faces, _, _ = marching_cubes(f, level=isovalue, spacing=(1.0, 1.0, 1.0))
    return verts.astype(np.float32), faces.astype(np.uint32)

# ==========================================
# DYNAMIC QT3D ENTITY (ZERO ALLOCATION)
# ==========================================

class DynamicEntity:
    def __init__(self, root, primitive_type, color, offset=None):
        self.entity = Qt3DCore.QEntity(root)
        self.color = color
        
        self.transform = Qt3DCore.QTransform()
        if offset: self.transform.setTranslation(QtGui.QVector3D(*offset))
        self.entity.addComponent(self.transform)

        self.geometry = Qt3DCore.QGeometry(self.entity)
        
        self.vbuf = Qt3DCore.QBuffer(self.geometry)
        self.pos_attr = Qt3DCore.QAttribute(self.geometry)
        self.pos_attr.setName(Qt3DCore.QAttribute.defaultPositionAttributeName())
        self.pos_attr.setVertexBaseType(Qt3DCore.QAttribute.Float)
        self.pos_attr.setVertexSize(3)
        self.pos_attr.setAttributeType(Qt3DCore.QAttribute.VertexAttribute)
        self.pos_attr.setBuffer(self.vbuf)
        self.geometry.addAttribute(self.pos_attr)
        
        self.nbuf = Qt3DCore.QBuffer(self.geometry)
        self.norm_attr = Qt3DCore.QAttribute(self.geometry)
        self.norm_attr.setName(Qt3DCore.QAttribute.defaultNormalAttributeName())
        self.norm_attr.setVertexBaseType(Qt3DCore.QAttribute.Float)
        self.norm_attr.setVertexSize(3)
        self.norm_attr.setAttributeType(Qt3DCore.QAttribute.VertexAttribute)
        self.norm_attr.setBuffer(self.nbuf)
        self.geometry.addAttribute(self.norm_attr)
        
        self.cbuf = Qt3DCore.QBuffer(self.geometry)
        self.col_attr = Qt3DCore.QAttribute(self.geometry)
        self.col_attr.setName(Qt3DCore.QAttribute.defaultColorAttributeName())
        self.col_attr.setVertexBaseType(Qt3DCore.QAttribute.Float)
        self.col_attr.setVertexSize(3)
        self.col_attr.setAttributeType(Qt3DCore.QAttribute.VertexAttribute)
        self.col_attr.setBuffer(self.cbuf)
        
        self.ibuf = Qt3DCore.QBuffer(self.geometry)
        self.idx_attr = Qt3DCore.QAttribute(self.geometry)
        self.idx_attr.setVertexBaseType(Qt3DCore.QAttribute.UnsignedInt)
        self.idx_attr.setAttributeType(Qt3DCore.QAttribute.IndexAttribute)
        self.idx_attr.setBuffer(self.ibuf)
        self.geometry.addAttribute(self.idx_attr)
        
        self.renderer = Qt3DRender.QGeometryRenderer(self.entity)
        self.renderer.setPrimitiveType(primitive_type)
        self.renderer.setGeometry(self.geometry)
        self.entity.addComponent(self.renderer)
        
        self.material = None
        self.vertex_count = 0
        self.index_count = 0
        self.entity.setEnabled(False)

    def update_vertex_data(self, vertices, normals, colors=None):
        if len(vertices) == 0:
            self.entity.setEnabled(False)
            return
        self.vbuf.setData(vertices.tobytes())
        self.nbuf.setData(normals.tobytes())
        self.pos_attr.setCount(len(vertices))
        self.norm_attr.setCount(len(vertices))
        self.vertex_count = len(vertices)
        
        if colors is not None:
            self.cbuf.setData(colors.tobytes())
            self.col_attr.setCount(len(colors))
            if not isinstance(self.material, Qt3DExtras.QPerVertexColorMaterial):
                if self.material: self.entity.removeComponent(self.material)
                self.material = Qt3DExtras.QPerVertexColorMaterial(self.entity)
                self.entity.addComponent(self.material)
        else:
            if not isinstance(self.material, Qt3DExtras.QPhongAlphaMaterial):
                if self.material: self.entity.removeComponent(self.material)
                self.material = Qt3DExtras.QPhongAlphaMaterial(self.entity)
                self.material.setDiffuse(QtGui.QColor(self.color.red(), self.color.green(), self.color.blue(), 255))
                self.material.setAmbient(QtGui.QColor(30, 30, 30))
                self.material.setSpecular(QtGui.QColor(200, 200, 200))
                self.entity.addComponent(self.material)

    def update_index_data(self, indices):
        if len(indices) == 0:
            self.entity.setEnabled(False)
            return
        self.ibuf.setData(indices.tobytes())
        self.idx_attr.setCount(len(indices))
        self.index_count = len(indices)
        self.entity.setEnabled(True)

    def set_visible(self, visible):
        self.entity.setEnabled(visible and self.index_count > 0)

class PointEntity:
    def __init__(self, root, color, size=4.0, offset=None):
        self.entity = Qt3DCore.QEntity(root)
        self.transform = Qt3DCore.QTransform()
        if offset: self.transform.setTranslation(QtGui.QVector3D(*offset))
        self.entity.addComponent(self.transform)
        self.geometry = Qt3DCore.QGeometry(self.entity)
        
        self.vbuf = Qt3DCore.QBuffer(self.geometry)
        self.pos_attr = Qt3DCore.QAttribute(self.geometry)
        self.pos_attr.setName(Qt3DCore.QAttribute.defaultPositionAttributeName())
        self.pos_attr.setVertexBaseType(Qt3DCore.QAttribute.Float)
        self.pos_attr.setVertexSize(3)
        self.pos_attr.setAttributeType(Qt3DCore.QAttribute.VertexAttribute)
        self.pos_attr.setBuffer(self.vbuf)
        self.geometry.addAttribute(self.pos_attr)
        
        self.cbuf = Qt3DCore.QBuffer(self.geometry)
        self.col_attr = Qt3DCore.QAttribute(self.geometry)
        self.col_attr.setName(Qt3DCore.QAttribute.defaultColorAttributeName())
        self.col_attr.setVertexBaseType(Qt3DCore.QAttribute.Float)
        self.col_attr.setVertexSize(3)
        self.col_attr.setAttributeType(Qt3DCore.QAttribute.VertexAttribute)
        self.col_attr.setBuffer(self.cbuf)
        self.geometry.addAttribute(self.col_attr)
        
        self.renderer = Qt3DRender.QGeometryRenderer(self.entity)
        self.renderer.setPrimitiveType(Qt3DRender.QGeometryRenderer.Points)
        self.renderer.setGeometry(self.geometry)
        self.entity.addComponent(self.renderer)
        
        self.material = Qt3DExtras.QPerVertexColorMaterial(self.entity)
        point_size = Qt3DRender.QPointSize(self.material)
        point_size.setSizeMode(Qt3DRender.QPointSize.Fixed)
        point_size.setValue(size)
        self.material.addParameter(point_size)
        self.entity.addComponent(self.material)
        self.entity.setEnabled(False)

    def update_data(self, points, colors=None):
        if len(points) == 0:
            self.entity.setEnabled(False)
            return
        self.vbuf.setData(points.tobytes())
        self.pos_attr.setCount(len(points))
        if colors is not None:
            self.cbuf.setData(colors.tobytes())
            self.col_attr.setCount(len(colors))
        self.entity.setEnabled(True)

class LineEntity:
    def __init__(self, root, color, offset=None):
        self.entity = Qt3DCore.QEntity(root)
        self.transform = Qt3DCore.QTransform()
        if offset: self.transform.setTranslation(QtGui.QVector3D(*offset))
        self.entity.addComponent(self.transform)
        self.geometry = Qt3DCore.QGeometry(self.entity)
        
        self.vbuf = Qt3DCore.QBuffer(self.geometry)
        self.pos_attr = Qt3DCore.QAttribute(self.geometry)
        self.pos_attr.setName(Qt3DCore.QAttribute.defaultPositionAttributeName())
        self.pos_attr.setVertexBaseType(Qt3DCore.QAttribute.Float)
        self.pos_attr.setVertexSize(3)
        self.pos_attr.setAttributeType(Qt3DCore.QAttribute.VertexAttribute)
        self.pos_attr.setBuffer(self.vbuf)
        self.geometry.addAttribute(self.pos_attr)
        
        self.ibuf = Qt3DCore.QBuffer(self.geometry)
        self.idx_attr = Qt3DCore.QAttribute(self.geometry)
        self.idx_attr.setVertexBaseType(Qt3DCore.QAttribute.UnsignedInt)
        self.idx_attr.setAttributeType(Qt3DCore.QAttribute.IndexAttribute)
        self.idx_attr.setBuffer(self.ibuf)
        self.geometry.addAttribute(self.idx_attr)
        
        self.renderer = Qt3DRender.QGeometryRenderer(self.entity)
        self.renderer.setPrimitiveType(Qt3DRender.QGeometryRenderer.Lines)
        self.renderer.setGeometry(self.geometry)
        self.entity.addComponent(self.renderer)
        
        self.material = Qt3DExtras.QDiffuseSpecularMaterial(self.entity)
        self.material.setAmbient(color)
        self.material.setDiffuse(QtGui.QColor(0, 0, 0, 0))
        self.entity.addComponent(self.material)
        self.entity.setEnabled(False)

    def update_data(self, vertices, indices):
        if len(vertices) == 0 or len(indices) == 0:
            self.entity.setEnabled(False)
            return
        self.vbuf.setData(vertices.tobytes())
        self.pos_attr.setCount(len(vertices))
        self.ibuf.setData(indices.tobytes())
        self.idx_attr.setCount(len(indices))
        self.entity.setEnabled(True)

# ==========================================
# MESH WORKER
# ==========================================

class MeshWorker(QtCore.QObject):
    finished = QtCore.Signal(object, object, float, object, object, float, object, object, object, float, object, object, object, object, object)
    error = QtCore.Signal(str)

    def __init__(self, shape, custom_expr, N, isovalue, max_error):
        super().__init__()
        self.shape, self.custom_expr = shape, custom_expr
        self.N, self.isovalue, self.max_error = N, isovalue, max_error

    def run(self):
        try:
            field, grad = self.generate_field()
            mc_v, mc_f, mc_time = np.array([]), np.array([]), 0.0
            if HAS_SKIMAGE:
                t0 = time.perf_counter()
                mc_v, mc_f = marching_cubes_algorithm(field, self.isovalue)
                mc_time = time.perf_counter() - t0
            
            t0 = time.perf_counter()
            if HAS_NUMBA:
                active_voxels = find_active_voxels_numba(field, self.isovalue)
                sn_v, dc_v, errs, faces, ints, norms, voxels = run_dual_algorithms(field, grad, active_voxels, self.N, self.isovalue, self.max_error)
            else:
                raise RuntimeError("Numba is required for this application. Please run: pip install numba")
            dc_time = time.perf_counter() - t0
            
            self.finished.emit(mc_v, mc_f, mc_time, sn_v, faces, dc_time, dc_v, faces, dc_time, ints, norms, voxels, errs, field, grad)
        except Exception as e:
            self.error.emit(str(e))

    def generate_field(self):
        N = self.N
        x, y, z = np.mgrid[0:N, 0:N, 0:N] - N/2
        if self.shape == "Custom":
            if not self.custom_expr.strip(): raise ValueError("Custom expression is empty.")
            allowed = {"x": x, "y": y, "z": z, "np": np, "sin": np.sin, "cos": np.cos, "sqrt": np.sqrt, "abs": np.abs, "exp": np.exp}
            field = eval(self.custom_expr, {"__builtins__": {}}, allowed)
            grad = np.stack(np.gradient(field), axis=-1)
        elif self.shape == "Sphere":
            field = np.sqrt(x**2 + y**2 + z**2) - (N * 0.35)
            grad = np.stack([x, y, z], axis=-1)
        elif self.shape == "Cube":
            q = np.abs(np.stack([x, y, z], axis=-1)) - (N * 0.3)
            field = np.linalg.norm(np.maximum(q, 0), axis=-1) + np.minimum(np.max(q, axis=-1), 0)
            grad = np.sign(np.stack([x, y, z], axis=-1))
        elif self.shape == "Octahedron":
            field = np.abs(x) + np.abs(y) + np.abs(z) - (N * 0.35)
            grad = np.sign(np.stack([x, y, z], axis=-1))
        elif self.shape == "Torus":
            R, r = N * 0.25, N * 0.1
            q = np.sqrt(x**2 + z**2) - R
            field = np.sqrt(q**2 + y**2) - r
            grad = np.stack([x*q/np.sqrt(q**2+y**2), y/np.sqrt(q**2+y**2), z*q/np.sqrt(q**2+y**2)], axis=-1)
        elif self.shape == "Cube - Sphere (CSG)":
            q_c = np.abs(np.stack([x, y, z], axis=-1)) - (N * 0.35)
            cube_sdf = np.linalg.norm(np.maximum(q_c, 0), axis=-1) + np.minimum(np.max(q_c, axis=-1), 0)
            sphere_sdf = np.sqrt(x**2 + y**2 + z**2) - (N * 0.25)
            field = np.maximum(cube_sdf, -sphere_sdf)
            grad = np.stack(np.gradient(field), axis=-1)
        else:
            field, grad = np.zeros((N,N,N)), np.zeros((N,N,N,3))

        norm = np.linalg.norm(grad, axis=-1, keepdims=True)
        norm[norm == 0] = 1
        return field.astype(np.float32), (grad / norm).astype(np.float32)

# ==========================================
# 2D SLICE VIEW WIDGET (VECTORIZED)
# ==========================================

class SliceViewWidget(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setMinimumHeight(250)
        self.N = 24
        self.z_slice = 12
        self.field = np.zeros((self.N, self.N, self.N))
        self.voxels = np.array([])
        self.intersections = np.array([])
        self.verts = {'mc': [], 'sn': [], 'dc': []}
        self.errors = []
        self.max_error = 1.0

    def set_data(self, field, voxels, ints, mc_v, sn_v, dc_v, errs, max_err, z_slice):
        self.field = field
        self.voxels = voxels
        self.intersections = ints
        self.verts['mc'] = mc_v
        self.verts['sn'] = sn_v
        self.verts['dc'] = dc_v
        self.errors = errs
        self.max_error = max_err
        self.z_slice = z_slice
        self.update()

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        w, h = self.width(), self.height()
        cell = min(w, h) / self.N
        painter.fillRect(self.rect(), QtGui.QColor(30, 30, 30))
        
        z = int(self.z_slice)
        if 0 <= z < self.N:
            slice_f = self.field[:, :, z]
            f_min, f_max = slice_f.min(), slice_f.max()
            if f_max - f_min > 0:
                norm_f = (slice_f - f_min) / (f_max - f_min)
                vals = (norm_f * 255).astype(np.uint8)
                bgra = np.empty((self.N, self.N, 4), dtype=np.uint8)
                bgra[:, :, 0] = vals # B
                bgra[:, :, 1] = vals # G
                bgra[:, :, 2] = vals # R
                bgra[:, :, 3] = 255  # A
                img = QtGui.QImage(bgra.data, self.N, self.N, self.N * 4, QtGui.QImage.Format_ARGB32).copy()
                painter.drawImage(QtCore.QRectF(0, 0, self.N * cell, self.N * cell), img.mirrored(False, True))
                
        painter.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255, 50), 1))
        for v in self.voxels:
            if v[2] == z:
                painter.drawRect(v[0] * cell, (self.N - 1 - v[1]) * cell, cell, cell)
                
        painter.setBrush(QtGui.QColor(255, 255, 0))
        painter.setPen(QtCore.Qt.NoPen)
        for p in self.intersections:
            if abs(p[2] - z) < 1.0:
                painter.drawEllipse(QtCore.QPointF(p[0] * cell, (self.N - p[1]) * cell), 3, 3)
                
        painter.setPen(QtCore.Qt.NoPen)
        painter.setBrush(QtGui.QColor(0, 255, 255))
        for p in self.verts['mc']:
            if abs(p[2] - z) < 0.5:
                painter.drawEllipse(QtCore.QPointF(p[0] * cell, (self.N - p[1]) * cell), 4, 4)
                
        painter.setBrush(QtGui.QColor(255, 255, 255))
        for p in self.verts['sn']:
            if abs(p[2] - z) < 0.5:
                painter.drawEllipse(QtCore.QPointF(p[0] * cell, (self.N - p[1]) * cell), 5, 5)
                
        for i, p in enumerate(self.verts['dc']):
            if abs(p[2] - z) < 0.5:
                err = self.errors[i] if i < len(self.errors) else 0
                t = min(err / self.max_error, 1.0) if self.max_error > 0 else 0
                r = int(255 * t); b = int(255 * (1 - t))
                painter.setBrush(QtGui.QColor(r, 50, b))
                painter.drawEllipse(QtCore.QPointF(p[0] * cell, (self.N - p[1]) * cell), 6, 6)

# ==========================================
# MAIN APPLICATION GUI
# ==========================================

DARK_QSS = """
QMainWindow, QWidget { background-color: #1e1e1e; color: #d4d4d4; font-family: 'Segoe UI', Arial, sans-serif; }
QGroupBox { border: 1px solid #3c3c3c; border-radius: 6px; margin-top: 10px; padding: 10px; font-weight: bold; color: #4a9eff; }
QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }
QPushButton { background-color: #2d2d30; border: 1px solid #3c3c3c; border-radius: 4px; padding: 6px 12px; color: #d4d4d4; }
QPushButton:hover { background-color: #3c3c3c; border: 1px solid #4a9eff; }
QPushButton:pressed { background-color: #4a9eff; color: #1e1e1e; }
QComboBox { background-color: #2d2d30; border: 1px solid #3c3c3c; border-radius: 4px; padding: 4px 8px; }
QComboBox::drop-down { border: none; }
QComboBox QAbstractItemView { background-color: #2d2d30; border: 1px solid #3c3c3c; selection-background-color: #4a9eff; }
QLineEdit { background-color: #2d2d30; border: 1px solid #3c3c3c; border-radius: 4px; padding: 4px 8px; }
QSlider::groove:horizontal { border: 1px solid #3c3c3c; height: 4px; background: #2d2d30; border-radius: 2px; }
QSlider::handle:horizontal { background: #4a9eff; border: 1px solid #1e1e1e; width: 12px; height: 12px; margin: -5px 0; border-radius: 6px; }
QSlider::handle:horizontal:hover { background: #6ab4ff; }
QTabWidget::pane { border: 1px solid #3c3c3c; border-radius: 4px; top: -1px; }
QTabBar::tab { background: #2d2d30; border: 1px solid #3c3c3c; padding: 6px 12px; border-top-left-radius: 4px; border-top-right-radius: 4px; margin-right: 2px; }
QTabBar::tab:selected { background: #1e1e1e; border-bottom: none; color: #4a9eff; }
QCheckBox { spacing: 8px; }
QCheckBox::indicator { width: 14px; height: 14px; border: 1px solid #3c3c3c; border-radius: 3px; background: #2d2d30; }
QCheckBox::indicator:checked { background: #4a9eff; border: 1px solid #4a9eff; }
QLabel { background: transparent; }
"""

class VizWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Algorithm Lab - Ultimate Geometry Inspector")
        self.resize(1300, 900)
        
        # Enable 4x MSAA for perfectly smooth 3D edges
        fmt = QtGui.QSurfaceFormat()
        fmt.setSamples(4)
        QtGui.QSurfaceFormat.setDefaultFormat(fmt)
        
        self.N = 32
        self.isovalue = 0.0
        self.shape = "Cube - Sphere (CSG)"
        self.custom_expr = "np.sin(x*0.5) + np.sin(y*0.5) + np.sin(z*0.5) - 1.0"
        self.max_error = 1.0
        self.cut_val = 1.0
        self.cut_axis = 2 # 0=X, 1=Y, 2=Z
        self.normal_scale = 0.5
        self.auto_rotate = False
        self.rotate_angle = 0.0
        self.jit_warmed = False
        self.is_loading_preset = False
        
        self.saved_shapes_file = "custom_shapes.json"
        self.saved_shapes = self.load_saved_shapes()
        
        self.setup_ui()
        
        self.tab_widget.setCurrentIndex(0)
        self.load_preset(0)

    def load_saved_shapes(self):
        if os.path.exists(self.saved_shapes_file):
            try:
                with open(self.saved_shapes_file, 'r') as f:
                    return json.load(f)
            except: return {}
        return {}

    def save_saved_shapes(self):
        with open(self.saved_shapes_file, 'w') as f:
            json.dump(self.saved_shapes, f, indent=4)

    def setup_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)
        top_layout = QtWidgets.QHBoxLayout()
        layout.addLayout(top_layout, 3)
        
        panel = QtWidgets.QWidget()
        panel.setFixedWidth(380)
        panel_layout = QtWidgets.QVBoxLayout(panel)
        top_layout.addWidget(panel)
        
        title = QtWidgets.QLabel("Algorithm Comparison Lab")
        title.setStyleSheet("font-size: 18px; font-weight: bold; color: #4a9eff;")
        panel_layout.addWidget(title)
        
        self.num_label = QtWidgets.QLabel(f"Numba JIT: {'ENABLED' if HAS_NUMBA else 'DISABLED (Install numba)'}")
        self.num_label.setStyleSheet(f"color: {'#4CAF50' if HAS_NUMBA else '#ff6666'}; font-weight: bold;")
        panel_layout.addWidget(self.num_label)
        
        # Tabs
        self.tab_widget = QtWidgets.QTabWidget()
        self.tab_widget.addTab(QtWidgets.QWidget(), "Medical (MC)")
        self.tab_widget.addTab(QtWidgets.QWidget(), "Terrain (MC)")
        self.tab_widget.addTab(QtWidgets.QWidget(), "Mechanical (DC)")
        self.tab_widget.addTab(QtWidgets.QWidget(), "CSG Arch (DC)")
        self.tab_widget.currentChanged.connect(self.load_preset)
        panel_layout.addWidget(self.tab_widget)
        
        self.preset_info_label = QtWidgets.QLabel("Select an example tab to see a real-world use case.")
        self.preset_info_label.setWordWrap(True)
        self.preset_info_label.setStyleSheet("color: #aaa; font-style: italic; padding: 8px; background: #252526; border-radius: 4px; border-left: 3px solid #4a9eff;")
        panel_layout.addWidget(self.preset_info_label)
        
        # Shape Group
        shape_group = QtWidgets.QGroupBox("Geometry Configuration")
        shape_layout = QtWidgets.QVBoxLayout(shape_group)
        
        self.shape_combo = QtWidgets.QComboBox()
        self.shape_combo.addItems(["Sphere", "Cube", "Octahedron", "Torus", "Cube - Sphere (CSG)", "Custom"])
        self.shape_combo.setCurrentText(self.shape)
        self.shape_combo.currentTextChanged.connect(self.update_params)
        shape_layout.addWidget(QtWidgets.QLabel("Target Shape:"))
        shape_layout.addWidget(self.shape_combo)
        
        # Custom Shape Loading/Saving
        custom_layout = QtWidgets.QHBoxLayout()
        self.saved_shapes_combo = QtWidgets.QComboBox()
        self.saved_shapes_combo.addItems(list(self.saved_shapes.keys()))
        self.saved_shapes_combo.setCurrentText("")
        self.saved_shapes_combo.currentTextChanged.connect(self.load_saved_shape)
        custom_layout.addWidget(self.saved_shapes_combo, 1)
        
        self.btn_save_shape = QtWidgets.QPushButton("Save")
        self.btn_save_shape.clicked.connect(self.save_custom_shape)
        custom_layout.addWidget(self.btn_save_shape)
        shape_layout.addLayout(custom_layout)
        
        self.expr_edit = QtWidgets.QLineEdit(self.custom_expr)
        self.expr_edit.setEnabled(False)
        self.expr_edit.returnPressed.connect(self.regen_meshes)
        shape_layout.addWidget(QtWidgets.QLabel("Custom Equation f(x,y,z):"))
        shape_layout.addWidget(self.expr_edit)
        
        self.res_label = QtWidgets.QLabel(f"Grid Resolution (N): {self.N}")
        shape_layout.addWidget(self.res_label)
        self.res_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.res_slider.setRange(8, 128 if HAS_NUMBA else 32)
        self.res_slider.setValue(self.N)
        self.res_slider.sliderReleased.connect(self.update_params)
        shape_layout.addWidget(self.res_slider)
        
        self.iso_label = QtWidgets.QLabel(f"Isovalue (Threshold): {self.isovalue:.2f}")
        shape_layout.addWidget(self.iso_label)
        self.iso_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.iso_slider.setRange(-20, 20)
        self.iso_slider.setValue(0)
        self.iso_slider.sliderReleased.connect(self.update_params)
        shape_layout.addWidget(self.iso_slider)
        
        self.err_label = QtWidgets.QLabel(f"DC Max QEF Error: {self.max_error:.2f}")
        shape_layout.addWidget(self.err_label)
        self.err_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.err_slider.setRange(0, 100)
        self.err_slider.setValue(10)
        self.err_slider.valueChanged.connect(self.update_error_threshold)
        shape_layout.addWidget(self.err_slider)
        
        panel_layout.addWidget(shape_group)
        
        # Clipping Group
        clip_group = QtWidgets.QGroupBox("3D Clipping Plane")
        clip_layout = QtWidgets.QVBoxLayout(clip_group)
        
        clip_axis_layout = QtWidgets.QHBoxLayout()
        clip_axis_layout.addWidget(QtWidgets.QLabel("Axis:"))
        self.cut_axis_combo = QtWidgets.QComboBox()
        self.cut_axis_combo.addItems(["X Axis", "Y Axis", "Z Axis"])
        self.cut_axis_combo.setCurrentIndex(2)
        self.cut_axis_combo.currentIndexChanged.connect(self.update_cut_axis)
        clip_axis_layout.addWidget(self.cut_axis_combo, 1)
        clip_layout.addLayout(clip_axis_layout)
        
        self.cut_label = QtWidgets.QLabel(f"Cut Position: {int(self.cut_val*100)}%")
        clip_layout.addWidget(self.cut_label)
        self.cut_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.cut_slider.setRange(0, 100)
        self.cut_slider.setValue(100)
        self.cut_slider.valueChanged.connect(self.update_cut_label)
        self.cut_slider.sliderReleased.connect(self.update_cut_val)
        clip_layout.addWidget(self.cut_slider)
        
        panel_layout.addWidget(clip_group)
        
        # View Group
        view_group = QtWidgets.QGroupBox("Visualization")
        view_layout = QtWidgets.QVBoxLayout(view_group)
        
        self.view_combo = QtWidgets.QComboBox()
        self.view_combo.addItems(["Marching Cubes", "Surface Nets", "Dual Contouring", "Side-by-Side (MC vs DC)", "Overlay (MC vs DC)"])
        self.view_combo.currentTextChanged.connect(self.update_view_mode)
        view_layout.addWidget(self.view_combo)
        
        self.wireframe_check = QtWidgets.QCheckBox("Show Mesh Wireframe")
        self.wireframe_check.setChecked(True)
        self.wireframe_check.stateChanged.connect(self.update_view_mode)
        view_layout.addWidget(self.wireframe_check)
        
        self.vertices_check = QtWidgets.QCheckBox("Show Mesh Vertices")
        self.vertices_check.stateChanged.connect(self.update_view_mode)
        view_layout.addWidget(self.vertices_check)
        
        self.intersections_check = QtWidgets.QCheckBox("Show 3D Intersections")
        self.intersections_check.stateChanged.connect(self.update_view_mode)
        view_layout.addWidget(self.intersections_check)
        
        self.normals_check = QtWidgets.QCheckBox("Show Vertex Normals (3D)")
        self.normals_check.stateChanged.connect(self.update_view_mode)
        view_layout.addWidget(self.normals_check)
        
        ns_layout = QtWidgets.QHBoxLayout()
        ns_layout.addWidget(QtWidgets.QLabel("Normal Length:"))
        self.ns_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.ns_slider.setRange(0, 20)
        self.ns_slider.setValue(5)
        self.ns_slider.valueChanged.connect(self.update_normal_scale)
        ns_layout.addWidget(self.ns_slider)
        view_layout.addLayout(ns_layout)
        
        self.heatmap_check = QtWidgets.QCheckBox("DC QEF Error Heatmap")
        self.heatmap_check.stateChanged.connect(self.update_view_mode)
        view_layout.addWidget(self.heatmap_check)
        
        self.voxels_check = QtWidgets.QCheckBox("Show Active Voxels")
        self.voxels_check.stateChanged.connect(self.update_view_mode)
        view_layout.addWidget(self.voxels_check)
        
        self.bbox_check = QtWidgets.QCheckBox("Show Bounding Box")
        self.bbox_check.setChecked(True)
        self.bbox_check.stateChanged.connect(self.update_view_mode)
        view_layout.addWidget(self.bbox_check)
        
        self.rotate_check = QtWidgets.QCheckBox("Auto-Rotate Camera")
        self.rotate_check.stateChanged.connect(self.toggle_auto_rotate)
        view_layout.addWidget(self.rotate_check)
        
        panel_layout.addWidget(view_group)
        
        # Export Buttons
        btn_layout = QtWidgets.QHBoxLayout()
        self.btn_gen = QtWidgets.QPushButton("Generate")
        self.btn_gen.setStyleSheet("background-color: #4a9eff; color: #1e1e1e; font-weight: bold; padding: 8px;")
        self.btn_gen.clicked.connect(self.regen_meshes)
        btn_layout.addWidget(self.btn_gen)
        
        self.btn_export = QtWidgets.QPushButton("Export")
        self.btn_export.clicked.connect(self.export_menu)
        btn_layout.addWidget(self.btn_export)
        panel_layout.addLayout(btn_layout)
        
        panel_layout.addStretch()
        
        # 3D View Setup
        self.view = Qt3DExtras.Qt3DWindow()
        self.view.defaultFrameGraph().setClearColor(QtGui.QColor(25, 25, 25))
        container = QtWidgets.QWidget.createWindowContainer(self.view)
        top_layout.addWidget(container, 1)
        
        self.root_entity = Qt3DCore.QEntity()
        self.cam = self.view.camera()
        self.cam.lens().setPerspectiveProjection(45.0, 16.0/9.0, 0.1, 1000.0)
        self.update_camera_position()
        cam_controller = Qt3DExtras.QOrbitCameraController(self.root_entity)
        cam_controller.setCamera(self.cam)
        
        light_entity = Qt3DCore.QEntity(self.root_entity)
        light = Qt3DRender.QDirectionalLight(light_entity)
        light.setWorldDirection(QtGui.QVector3D(1, -1, -1).normalized())
        light.setIntensity(0.8)
        light_entity.addComponent(light)
        fill_light_entity = Qt3DCore.QEntity(self.root_entity)
        fill_light = Qt3DRender.QDirectionalLight(fill_light_entity)
        fill_light.setWorldDirection(QtGui.QVector3D(-1, 0.5, 1).normalized())
        fill_light.setIntensity(0.4)
        fill_light_entity.addComponent(fill_light)
        self.view.setRootEntity(self.root_entity)
        
        # Bottom Panel (2D Slice + Stats)
        bottom_layout = QtWidgets.QHBoxLayout()
        layout.addLayout(bottom_layout, 1)
        
        self.slice_view = SliceViewWidget()
        bottom_layout.addWidget(self.slice_view, 3)
        
        stats_widget = QtWidgets.QWidget()
        stats_layout = QtWidgets.QVBoxLayout(stats_widget)
        stats_title = QtWidgets.QLabel("Quantitative Metrics")
        stats_title.setStyleSheet("font-weight: bold; color: #4a9eff; font-size: 14px;")
        stats_layout.addWidget(stats_title)
        
        self.stats_grid = QtWidgets.QGridLayout()
        self.stats_labels = {}
        metrics = ["Verts", "Tris", "Area", "Volume"]
        algorithms = ["MC", "SN", "DC"]
        
        self.stats_grid.addWidget(QtWidgets.QLabel(""), 0, 0)
        for c, algo in enumerate(algorithms):
            lbl = QtWidgets.QLabel(algo)
            lbl.setStyleSheet("font-weight: bold; color: #d4d4d4; padding-left: 10px;")
            self.stats_grid.addWidget(lbl, 0, c + 1)
            
        for r, m in enumerate(metrics):
            lbl = QtWidgets.QLabel(m)
            lbl.setStyleSheet("font-weight: bold; color: #aaa;")
            self.stats_grid.addWidget(lbl, r + 1, 0)
            for c, algo in enumerate(algorithms):
                val_lbl = QtWidgets.QLabel("0")
                val_lbl.setStyleSheet("color: #d4d4d4; padding-left: 10px;")
                self.stats_grid.addWidget(val_lbl, r + 1, c + 1)
                self.stats_labels[f"{algo}_{m}"] = val_lbl
                
        stats_layout.addLayout(self.stats_grid)
        stats_layout.addStretch()
        
        slice_legend = QtWidgets.QLabel("2D Z-Slice View Legend\n\nWhite Sq: Active Voxel\nYellow Dot: Intersection\nCyan: MC Vert\nWhite: SN Vert\nBlue/Red: DC Vert (Red=High QEF Err)")
        slice_legend.setStyleSheet("color: #aaa; padding: 10px; background: #252526; border-radius: 4px;")
        stats_layout.addWidget(slice_legend)
        
        bottom_layout.addWidget(stats_widget, 1)

        # Pre-allocate Dynamic Entities
        self.ent_mc_solid = DynamicEntity(self.root_entity, Qt3DRender.QGeometryRenderer.Triangles, QtGui.QColor(100, 200, 255))
        self.ent_mc_wire = LineEntity(self.root_entity, QtGui.QColor(150, 220, 255))
        self.ent_mc_pts = PointEntity(self.root_entity, QtGui.QColor(0, 255, 255), 5.0)
        self.ent_mc_norms = LineEntity(self.root_entity, QtGui.QColor(100, 255, 100))
        
        self.ent_sn_solid = DynamicEntity(self.root_entity, Qt3DRender.QGeometryRenderer.Triangles, QtGui.QColor(180, 180, 180))
        self.ent_sn_wire = LineEntity(self.root_entity, QtGui.QColor(220, 220, 220))
        self.ent_sn_pts = PointEntity(self.root_entity, QtGui.QColor(255, 255, 255), 6.0)
        self.ent_sn_norms = LineEntity(self.root_entity, QtGui.QColor(100, 255, 100))
        
        self.ent_dc_solid = DynamicEntity(self.root_entity, Qt3DRender.QGeometryRenderer.Triangles, QtGui.QColor(255, 150, 100))
        self.ent_dc_wire = LineEntity(self.root_entity, QtGui.QColor(255, 200, 150))
        self.ent_dc_pts = PointEntity(self.root_entity, QtGui.QColor(255, 100, 0), 6.0)
        self.ent_dc_norms = LineEntity(self.root_entity, QtGui.QColor(100, 255, 100))
        
        self.ent_ints = PointEntity(self.root_entity, QtGui.QColor(255, 255, 0), 3.0)
        self.ent_voxels = LineEntity(self.root_entity, QtGui.QColor(80, 80, 80, 150))
        self.bbox_entity = None
        
        # Cached Mesh Data
        self.mc_norms, self.mc_lines, self.mc_norm_lines = np.array([]), np.array([]), np.array([])
        self.sn_norms, self.sn_lines, self.sn_norm_lines = np.array([]), np.array([]), np.array([])
        self.dc_norms, self.dc_lines, self.dc_norm_lines, self.dc_colors = np.array([]), np.array([]), np.array([]), None
        self.int_points, self.voxel_verts, self.voxel_idx = np.array([]), np.array([]), np.array([])
        
        self.rotate_timer = QtCore.QTimer(self)
        self.rotate_timer.timeout.connect(self.on_rotate_tick)
        self.update_ui_context()
        self.rebuild_static_entities()

    def load_preset(self, index):
        self.is_loading_preset = True
        
        if index == 0: # Medical Imaging (MC)
            self.preset_info_label.setText("Medical Imaging (MRI/CT):\nMarching Cubes effortlessly extracts smooth, organic surfaces from dense voxel data. It is the industry standard for medical scans because it does not require sharp features.")
            self.shape_combo.setCurrentText("Custom")
            self.expr_edit.setText("np.sin(x*0.5)*np.cos(y*0.5) + np.sin(y*0.5)*np.cos(z*0.5) + np.sin(z*0.5)*np.cos(x*0.5) - 1.0")
            self.res_slider.setValue(32)
            self.iso_slider.setValue(0)
            self.view_combo.setCurrentText("Marching Cubes")
            self.wireframe_check.setChecked(True)
            self.voxels_check.setChecked(False)
            self.vertices_check.setChecked(False)
            self.intersections_check.setChecked(False)
            self.normals_check.setChecked(False)
            self.heatmap_check.setChecked(False)
            self.cut_slider.setValue(100)
            
        elif index == 1: # Terrain Generation (MC)
            self.preset_info_label.setText("Terrain Generation:\nMarching Cubes smoothly renders rolling hills and valleys from 3D noise fields. It is the backbone of procedural game terrain, providing organic shapes with low computational cost.")
            self.shape_combo.setCurrentText("Custom")
            self.expr_edit.setText("(np.sin(x*0.3) + np.cos(y*0.2) + np.sin((x+y)*0.1)) * 2.0 + z - 5")
            self.res_slider.setValue(48)
            self.iso_slider.setValue(0)
            self.view_combo.setCurrentText("Marching Cubes")
            self.wireframe_check.setChecked(True)
            self.voxels_check.setChecked(False)
            self.vertices_check.setChecked(False)
            self.intersections_check.setChecked(False)
            self.normals_check.setChecked(False)
            self.heatmap_check.setChecked(False)
            self.cut_slider.setValue(100)
            
        elif index == 2: # Mechanical Part (DC)
            self.preset_info_label.setText("Mechanical CAD:\nDual Contouring preserves razor-sharp 90-degree edges on mechanical parts where Marching Cubes would create undesirable bevels. Vertex Normals are enabled to visually prove how DC handles conflicting vectors at the corners.")
            self.shape_combo.setCurrentText("Cube")
            self.res_slider.setValue(16)
            self.iso_slider.setValue(0)
            self.view_combo.setCurrentText("Dual Contouring")
            self.wireframe_check.setChecked(True)
            self.voxels_check.setChecked(True)
            self.vertices_check.setChecked(True)
            self.intersections_check.setChecked(True)
            self.normals_check.setChecked(True)
            self.heatmap_check.setChecked(False)
            self.cut_slider.setValue(50)
            self.cut_axis_combo.setCurrentIndex(2) # Z axis
            
        elif index == 3: # CSG Architecture (DC)
            self.preset_info_label.setText("CSG Architecture (Boolean Ops):\nDual Contouring accurately reconstructs sharp intersection seams from Boolean operations (e.g., Cube minus Sphere). The QEF Heatmap shows the algorithm handling conflicting normals at the seam.")
            self.shape_combo.setCurrentText("Cube - Sphere (CSG)")
            self.res_slider.setValue(32)
            self.iso_slider.setValue(0)
            self.view_combo.setCurrentText("Dual Contouring")
            self.wireframe_check.setChecked(True)
            self.voxels_check.setChecked(False)
            self.vertices_check.setChecked(False)
            self.intersections_check.setChecked(False)
            self.normals_check.setChecked(False)
            self.heatmap_check.setChecked(True)
            self.cut_slider.setValue(50)
            self.cut_axis_combo.setCurrentIndex(0) # X axis

        self.is_loading_preset = False
        self.regen_meshes()
        self.update_camera_position()

    def save_custom_shape(self):
        if self.shape_combo.currentText() != "Custom":
            return
        name, ok = QtWidgets.QInputDialog.getText(self, "Save Custom Shape", "Enter a name for this shape:")
        if ok and name:
            self.saved_shapes[name] = self.expr_edit.text()
            self.save_saved_shapes()
            self.saved_shapes_combo.blockSignals(True)
            self.saved_shapes_combo.clear()
            self.saved_shapes_combo.addItems(list(self.saved_shapes.keys()))
            self.saved_shapes_combo.setCurrentText(name)
            self.saved_shapes_combo.blockSignals(False)

    def load_saved_shape(self, name):
        if name and name in self.saved_shapes:
            self.shape_combo.setCurrentText("Custom")
            self.expr_edit.setText(self.saved_shapes[name])
            self.regen_meshes()

    def update_ui_context(self):
        mode = self.view_combo.currentText()
        is_dual_mode = mode in ["Surface Nets", "Dual Contouring", "Side-by-Side (MC vs DC)", "Overlay (MC vs DC)"]
        self.voxels_check.setEnabled(is_dual_mode)
        self.heatmap_check.setEnabled(mode == "Dual Contouring")

    def update_camera_position(self):
        dist = self.N * 2.0
        self.cam.setPosition(QtGui.QVector3D(dist, dist * 0.8, dist))
        self.cam.setViewCenter(QtGui.QVector3D(self.N/2 - 0.5, self.N/2 - 0.5, self.N/2 - 0.5))

    def toggle_auto_rotate(self, state):
        self.auto_rotate = (state == QtCore.Qt.Checked)
        if self.auto_rotate: self.rotate_timer.start(16)
        else: self.rotate_timer.stop()

    def on_rotate_tick(self):
        self.rotate_angle += 0.5
        dist = self.N * 2.0
        rad = np.deg2rad(self.rotate_angle)
        x = dist * np.sin(rad) + self.N/2 - 0.5
        z = dist * np.cos(rad) + self.N/2 - 0.5
        y = dist * 0.8 + self.N/2 - 0.5
        self.cam.setPosition(QtGui.QVector3D(x, y, z))

    def update_params(self):
        if self.is_loading_preset: return
        old_N = self.N
        self.shape = self.shape_combo.currentText()
        self.N = self.res_slider.value()
        self.isovalue = self.iso_slider.value() / 10.0
        self.expr_edit.setEnabled(self.shape == "Custom")
        if self.shape == "Custom": self.custom_expr = self.expr_edit.text()
        self.res_label.setText(f"Grid Resolution (N): {self.N}")
        self.iso_label.setText(f"Isovalue (Threshold): {self.isovalue:.2f}")
        if self.N != old_N:
            self.rebuild_static_entities()
        if self.sender() in [self.res_slider, self.iso_slider, self.shape_combo]:
            self.regen_meshes()
            self.update_camera_position()

    def update_error_threshold(self, val):
        self.max_error = val / 10.0
        self.err_label.setText(f"DC Max QEF Error: {self.max_error:.2f}")
        self.regen_meshes()

    def update_cut_axis(self, idx):
        self.cut_axis = idx
        self.update_view_mode()

    def update_cut_label(self, val):
        self.cut_label.setText(f"Cut Position: {val}%")
        # For 2D slice, we still only slice Z
        z_slice = int(self.N * (val / 100.0))
        self.slice_view.set_data(self.field, self.dc_voxels, self.int_points, self.mc_verts, self.sn_verts, self.dc_verts, self.dc_errors, self.max_error, z_slice)

    def update_cut_val(self):
        self.cut_val = self.cut_slider.value() / 100.0
        self.update_view_mode()
        
    def update_normal_scale(self, val):
        self.normal_scale = val / 10.0
        self.regen_meshes()

    def rebuild_static_entities(self):
        if self.bbox_entity: self.bbox_entity.setParent(None)
        self.bbox_entity = None
        if self.bbox_check.isChecked():
            self.bbox_entity = self.create_bbox_entity(self.N)

    def get_cut_faces(self, verts, faces):
        if self.cut_val >= 1.0 or len(verts) == 0 or len(faces) == 0:
            return faces
        max_val = self.N * self.cut_val
        v0 = verts[faces[:, 0], self.cut_axis]
        v1 = verts[faces[:, 1], self.cut_axis]
        v2 = verts[faces[:, 2], self.cut_axis]
        mask = (v0 <= max_val) & (v1 <= max_val) & (v2 <= max_val)
        return faces[mask]

    def faces_to_lines(self, faces):
        if len(faces) == 0: return np.array([], dtype=np.uint32)
        if HAS_NUMBA: return faces_to_lines_numba(faces)
        line_indices = np.empty((len(faces), 6), dtype=np.uint32)
        line_indices[:, 0::2] = faces[:, [0, 1, 2]]
        line_indices[:, 1::2] = faces[:, [1, 2, 0]]
        return line_indices.reshape(-1)

    def create_voxel_lines(self, voxels):
        if len(voxels) == 0: return np.array([]), np.array([])
        corners = np.array([[0,0,0], [1,0,0], [0,1,0], [1,1,0], [0,0,1], [1,0,1], [0,1,1], [1,1,1]], dtype=np.float32)
        edges = np.array([(0,1), (2,3), (4,5), (6,7), (0,2), (1,3), (4,6), (5,7), (0,4), (1,5), (2,6), (3,7)], dtype=np.int32)
        num_vox = len(voxels)
        line_verts = np.empty((num_vox * 12 * 2, 3), dtype=np.float32)
        p1 = voxels[:, None, :] + corners[edges[:, 0]]
        p2 = voxels[:, None, :] + corners[edges[:, 1]]
        line_verts[0::2] = p1.reshape(-1, 3)
        line_verts[1::2] = p2.reshape(-1, 3)
        line_indices = np.arange(len(line_verts), dtype=np.uint32)
        return line_verts, line_indices

    def create_normal_lines(self, verts, norms, scale):
        if len(verts) == 0 or scale == 0: return np.array([]), np.array([])
        end_points = verts + norms * scale
        all_pts = np.vstack([verts, end_points]).astype(np.float32)
        idx = np.arange(len(verts))
        line_idx = np.empty(len(idx) * 2, dtype=np.uint32)
        line_idx[0::2] = idx
        line_idx[1::2] = idx + len(verts)
        return all_pts, line_idx

    def update_view_mode(self):
        self.update_ui_context()
        mode = self.view_combo.currentText()
        show_wire = self.wireframe_check.isChecked()
        show_verts = self.vertices_check.isChecked()
        show_voxels = self.voxels_check.isChecked()
        show_heatmap = self.heatmap_check.isChecked()
        show_norms = self.normals_check.isChecked()
        show_ints = self.intersections_check.isChecked()
        
        if show_ints and len(self.int_points) > 0:
            self.ent_ints.update_data(self.int_points)
        else:
            self.ent_ints.entity.setEnabled(False)

        if show_voxels and len(self.voxel_verts) > 0:
            self.ent_voxels.update_data(self.voxel_verts, self.voxel_idx)
        else:
            self.ent_voxels.entity.setEnabled(False)

        mc_f_cut = self.get_cut_faces(self.mc_verts, self.mc_faces)
        sn_f_cut = self.get_cut_faces(self.sn_verts, self.sn_faces)
        dc_f_cut = self.get_cut_faces(self.dc_verts, self.dc_faces)
        
        mc_visible = mode in ["Marching Cubes", "Side-by-Side (MC vs DC)", "Overlay (MC vs DC)"]
        self.ent_mc_solid.set_visible(mc_visible)
        self.ent_mc_wire.entity.setEnabled(mc_visible and show_wire and mode != "Overlay (MC vs DC)")
        self.ent_mc_pts.entity.setEnabled(mc_visible and show_verts)
        self.ent_mc_norms.entity.setEnabled(mc_visible and show_norms)
        
        if mc_visible:
            self.ent_mc_solid.update_vertex_data(self.mc_verts, self.mc_norms)
            self.ent_mc_solid.update_index_data(mc_f_cut.flatten())
            if show_wire and mode != "Overlay (MC vs DC)":
                mc_lines = self.mc_lines if self.cut_val >= 1.0 else self.faces_to_lines(mc_f_cut)
                self.ent_mc_wire.update_data(self.mc_verts, mc_lines)
            if show_verts:
                self.ent_mc_pts.update_data(self.mc_verts)
            if show_norms:
                self.ent_mc_norms.update_data(self.mc_verts, self.mc_norm_lines)
                
        sn_visible = mode == "Surface Nets"
        self.ent_sn_solid.set_visible(sn_visible)
        self.ent_sn_wire.entity.setEnabled(sn_visible and show_wire)
        self.ent_sn_pts.entity.setEnabled(sn_visible and show_verts)
        self.ent_sn_norms.entity.setEnabled(sn_visible and show_norms)
        
        if sn_visible:
            self.ent_sn_solid.update_vertex_data(self.sn_verts, self.sn_norms)
            self.ent_sn_solid.update_index_data(sn_f_cut.flatten())
            if show_wire:
                sn_lines = self.sn_lines if self.cut_val >= 1.0 else self.faces_to_lines(sn_f_cut)
                self.ent_sn_wire.update_data(self.sn_verts, sn_lines)
            if show_verts:
                self.ent_sn_pts.update_data(self.sn_verts)
            if show_norms:
                self.ent_sn_norms.update_data(self.sn_verts, self.sn_norm_lines)

        dc_visible = mode in ["Dual Contouring", "Side-by-Side (MC vs DC)", "Overlay (MC vs DC)"]
        self.ent_dc_solid.set_visible(dc_visible)
        self.ent_dc_wire.entity.setEnabled(dc_visible and show_wire)
        self.ent_dc_pts.entity.setEnabled(dc_visible and show_verts)
        self.ent_dc_norms.entity.setEnabled(dc_visible and show_norms)
        
        if dc_visible:
            colors = self.dc_colors if show_heatmap else None
            self.ent_dc_solid.update_vertex_data(self.dc_verts, self.dc_norms, colors)
            self.ent_dc_solid.update_index_data(dc_f_cut.flatten())
            if show_wire:
                dc_lines = self.dc_lines if self.cut_val >= 1.0 else self.faces_to_lines(dc_f_cut)
                self.ent_dc_wire.update_data(self.dc_verts, dc_lines)
            if show_verts:
                self.ent_dc_pts.update_data(self.dc_verts, colors)
            if show_norms:
                self.ent_dc_norms.update_data(self.dc_verts, self.dc_norm_lines)

    def create_bbox_entity(self, N):
        entity = Qt3DCore.QEntity(self.root_entity)
        mesh = Qt3DExtras.QCuboidMesh(entity)
        mesh.setXExtent(N); mesh.setYExtent(N); mesh.setZExtent(N)
        transform = Qt3DCore.QTransform()
        transform.setTranslation(QtGui.QVector3D(N/2 - 0.5, N/2 - 0.5, N/2 - 0.5))
        entity.addComponent(transform)
        material = Qt3DExtras.QDiffuseSpecularMaterial(entity)
        material.setAmbient(QtGui.QColor(80, 80, 80, 255))
        entity.addComponent(mesh); entity.addComponent(material)
        return entity

    def regen_meshes(self):
        self.btn_gen.setEnabled(False)
        self.preset_info_label.setText("Calculating...")
        self.preset_info_label.setStyleSheet("color: #aaa; font-style: italic; padding: 8px; background: #252526; border-radius: 4px; border-left: 3px solid #ffcc00;")
        QtWidgets.QApplication.processEvents()

        self.N = self.res_slider.value()
        self.shape = self.shape_combo.currentText()
        self.custom_expr = self.expr_edit.text()
        self.isovalue = self.iso_slider.value() / 10.0
        self.max_error = self.err_slider.value() / 10.0

        self.thread = QtCore.QThread()
        self.worker = MeshWorker(self.shape, self.expr_edit.text(), self.N, self.isovalue, self.max_error)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.on_mesh_ready)
        self.worker.error.connect(self.on_mesh_error)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.start()

    def on_mesh_ready(self, mc_v, mc_f, mc_t, sn_v, sn_f, sn_t, dc_v, dc_f, dc_t, ints, norms, voxels, errors, field, grad):
        self.jit_warmed = True
        self.mc_verts, self.mc_faces = mc_v, mc_f
        self.sn_verts, self.sn_faces = sn_v, sn_f
        self.dc_verts, self.dc_faces = dc_v, dc_f
        self.int_points, self.dc_voxels, self.dc_errors = ints, voxels, errors
        self.field, self.grad = field, grad
        
        self.mc_norms = compute_vertex_normals(self.mc_verts, self.mc_faces)
        self.mc_lines = self.faces_to_lines(self.mc_faces)
        self.mc_norm_lines = self.create_normal_lines(self.mc_verts, self.mc_norms, self.normal_scale)
        
        self.sn_norms = compute_vertex_normals(self.sn_verts, self.sn_faces)
        self.sn_lines = self.faces_to_lines(self.sn_faces)
        self.sn_norm_lines = self.create_normal_lines(self.sn_verts, self.sn_norms, self.normal_scale)
        
        self.dc_norms = compute_vertex_normals(self.dc_verts, self.dc_faces)
        self.dc_lines = self.faces_to_lines(self.dc_faces)
        self.dc_norm_lines = self.create_normal_lines(self.dc_verts, self.dc_norms, self.normal_scale)
        
        if HAS_NUMBA:
            self.dc_colors = get_heatmap_colors_numba(self.dc_verts, self.dc_errors, self.max_error)
        else:
            self.dc_colors = None
            
        if len(self.dc_voxels) > 0:
            self.voxel_verts, self.voxel_idx = self.create_voxel_lines(self.dc_voxels)
        else:
            self.voxel_verts, self.voxel_idx = np.array([]), np.array([])
            
        mc_area, mc_vol = calculate_metrics(mc_v, mc_f)
        sn_area, sn_vol = calculate_metrics(sn_v, sn_f)
        dc_area, dc_vol = calculate_metrics(dc_v, dc_f)
        
        self.stats_labels["MC_Verts"].setText(f"{len(mc_v)}")
        self.stats_labels["MC_Tris"].setText(f"{len(mc_f)}")
        self.stats_labels["MC_Area"].setText(f"{mc_area:.1f}")
        self.stats_labels["MC_Volume"].setText(f"{mc_vol:.1f}")
        
        self.stats_labels["SN_Verts"].setText(f"{len(sn_v)}")
        self.stats_labels["SN_Tris"].setText(f"{len(sn_f)}")
        self.stats_labels["SN_Area"].setText(f"{sn_area:.1f}")
        self.stats_labels["SN_Volume"].setText(f"{sn_vol:.1f}")
        
        self.stats_labels["DC_Verts"].setText(f"{len(dc_v)}")
        self.stats_labels["DC_Tris"].setText(f"{len(dc_f)}")
        self.stats_labels["DC_Area"].setText(f"{dc_area:.1f}")
        self.stats_labels["DC_Volume"].setText(f"{dc_vol:.1f}")
        
        self.preset_info_label.setText("Ready.")
        self.preset_info_label.setStyleSheet("color: #aaa; font-style: italic; padding: 8px; background: #252526; border-radius: 4px; border-left: 3px solid #4CAF50;")
        
        self.update_view_mode()
        self.update_cut_label(int(self.cut_val * 100))
        self.btn_gen.setEnabled(True)

    def on_mesh_error(self, err_msg):
        for key in self.stats_labels:
            self.stats_labels[key].setText("ERR")
        self.preset_info_label.setText(f"Math Error:\n{err_msg}")
        self.preset_info_label.setStyleSheet("color: #ff6666; padding: 8px; background: #252526; border-radius: 4px; border-left: 3px solid #ff6666;")
        self.btn_gen.setEnabled(True)

    def export_menu(self):
        if len(self.mc_verts) == 0 and len(self.sn_verts) == 0 and len(self.dc_verts) == 0:
            QtWidgets.QMessageBox.warning(self, "Export Error", "No meshes to export.")
            return
            
        fname, ffilter = QtWidgets.QFileDialog.getSaveFileName(self, "Save Mesh", "", "OBJ Files (*.obj);;Binary STL Files (*.stl)")
        if not fname: return
        
        if ".stl" in fname:
            self.export_stl(fname)
        else:
            self.export_obj(fname)
            
        QtWidgets.QMessageBox.information(self, "Export Complete", f"Meshes saved to {fname}")

    def export_obj(self, fname):
        with open(fname, 'w') as f:
            f.write("# Exported from Algorithm Viz\n")
            offset = 1
            if len(self.mc_verts) > 0:
                f.write("o Marching_Cubes\n")
                for v in self.mc_verts: f.write(f"v {v[0]} {v[1]} {v[2]}\n")
                for face in self.mc_faces: f.write(f"f {face[0]+offset} {face[1]+offset} {face[2]+offset}\n")
                offset += len(self.mc_verts)
            if len(self.sn_verts) > 0:
                f.write("o Surface_Nets\n")
                for v in self.sn_verts: f.write(f"v {v[0]} {v[1]} {v[2]}\n")
                for face in self.sn_faces: f.write(f"f {face[0]+offset} {face[1]+offset} {face[2]+offset}\n")
                offset += len(self.sn_verts)
            if len(self.dc_verts) > 0:
                f.write("o Dual_Contouring\n")
                for v in self.dc_verts: f.write(f"v {v[0]} {v[1]} {v[2]}\n")
                for face in self.dc_faces: f.write(f"f {face[0]+offset} {face[1]+offset} {face[2]+offset}\n")

    def export_stl(self, fname):
        # Export only Dual Contouring (or MC if DC is empty) as a single solid for 3D printing
        verts, faces = self.dc_verts, self.dc_faces
        if len(verts) == 0:
            verts, faces = self.mc_verts, self.mc_faces
        if len(verts) == 0: return
            
        with open(fname, 'wb') as f:
            f.write(b'Binary STL Export\x00' * 20) # 80 byte header
            f.write(struct.pack('<I', len(faces)))
            for face in faces:
                p0, p1, p2 = verts[face[0]], verts[face[1]], verts[face[2]]
                v1 = p1 - p0
                v2 = p2 - p0
                normal = np.cross(v1, v2)
                norm = np.linalg.norm(normal)
                if norm > 0: normal = normal / norm
                f.write(struct.pack('<3f', *normal))
                f.write(struct.pack('<3f', *p0))
                f.write(struct.pack('<3f', *p1))
                f.write(struct.pack('<3f', *p2))
                f.write(b'\x00\x00') # attribute byte count

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    app.setStyleSheet(DARK_QSS)
    window = VizWindow()
    window.show()
    sys.exit(app.exec())