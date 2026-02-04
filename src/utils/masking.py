
import os
import torch
import numpy as np
import torch.nn.functional as F
import cv2
from src.renderer.util import vertex_normals, face_vertices 
from src.FLAME.lbs import vertices2landmarks


def load_probabilities_per_FLAME_triangle():
    """
    FLAME_masks_triangles.npy contains for each face area the indices of the triangles that belong to that area.
    Using that, we can assign a probability to each triangle based on the area it belongs to, and then sample for masking.
    """
    flame_masks_triangles = np.load('assets/FLAME_masks/FLAME_masks_triangles.npy', allow_pickle=True).item()

    area_weights = {
        'neck': 0.0,
        'right_eyeball': 0.0,
        'right_ear': 0.0,
        'lips': 0.5,
        'nose': 0.5,
        'left_ear': 0.0,
        'eye_region': 1.0,
        'forehead':1.0, 
        'left_eye_region': 1.0, 
        'right_eye_region': 1.0, 
        'face_clean': 1.0,
        'cleaner_lips': 1.0
    }

    face_probabilities = torch.zeros(9976)

    for area in area_weights.keys():
        face_probabilities[flame_masks_triangles[area]] = area_weights[area]

    return face_probabilities


def triangle_area(vertices):
    # Using the Shoelace formula to calculate the area of triangles in the xy plane
    # vertices is expected to be of shape (..., 3, 2) where the last dimension holds x and y coordinates.
    x1, y1 = vertices[..., 0, 0], vertices[..., 0, 1]
    x2, y2 = vertices[..., 1, 0], vertices[..., 1, 1]
    x3, y3 = vertices[..., 2, 0], vertices[..., 2, 1]

    # Shoelace formula for the area of a triangle given by coordinates (x1, y1), (x2, y2), (x3, y3)
    area = 0.5 * torch.abs(x1 * y2 + x2 * y3 + x3 * y1 - x2 * y1 - x3 * y2 - x1 * y3)
    return area



def random_barycentric(num=1):
    # Generate two random numbers for each set
    u, v = torch.rand(num), torch.rand(num)
    
    # Adjust the random numbers if they are outside the triangle
    outside_triangle = u + v > 1
    u[outside_triangle], v[outside_triangle] = 1 - u[outside_triangle], 1 - v[outside_triangle]
    
    # Calculate the barycentric coordinates
    alpha = 1 - (u + v)
    beta = u
    gamma = v
    
    # Combine the coordinates into a single tensor
    return torch.stack((alpha, beta, gamma), dim=1)


def masking(img, mask, extra_points, wr=15, rendered_mask=None, extra_noise=True, random_mask=0.01):
    # img: B x C x H x W
    # mask: B x 1 x H x W
    
    B, C, H, W = img.size()
    
    # dilate face mask, drawn from convex hull of face landmarks / 膨胀人脸掩码，该掩码由人脸关键点的凸包生成。
    mask = 1-F.max_pool2d(1-mask, 2 * wr + 1, stride=1, padding=wr)
    
    # optionally remove the rendered mask / 可选地移除渲染掩码,避免重复遮挡。
    if rendered_mask is not None:
        mask = mask * (1 - rendered_mask) 

    # 将图像与掩码逐元素相乘，得到初步遮挡后的图像:
    masked_img = img * mask
    
    # add noise to extra in-face points / 将噪声乘到 extra_points 上，使采样点的像素值有轻微扰动，增加随机性和鲁棒性:
    if extra_noise:
        # normal around 1 with std 0.1
        noise_mult = torch.randn(extra_points.shape).to(img.device) * 0.05 + 1
        extra_points = extra_points * noise_mult

    # select random_mask percentage of pixels as centers to crop out 11x11 patches / 随机选择一定比例的像素作为遮挡中心
    if random_mask > 0:
        random_mask = torch.bernoulli(torch.ones((B, 1, H, W)) * random_mask).to(img.device)
        # dilate the mask to have 11x11 patches / 使用池化膨胀成 11×11 的小块遮挡区域。
        random_mask = 1 - F.max_pool2d(random_mask, 11, stride=1, padding=5)
        extra_points = extra_points * random_mask # 将这些区域应用到extra_points，进一步随机化采样点分布。

    # 在遮挡图像中，把 extra_points 的非零位置写入，替换原有像素值。这样采样点的颜色信息被保留在最终图像中：
    masked_img[extra_points > 0] = extra_points[extra_points > 0]

    # 将结果张量从计算图中分离（不参与梯度回传），返回最终的遮挡图像：
    masked_img = masked_img.detach()
    return masked_img


# 将归一化坐标点转换为图像像素索引：
def point2ind(npoints, H):
    
    npoints = npoints * (H // 2) + H // 2
    npoints = npoints.long()
    npoints[...,1] = torch.clamp(npoints[..., 1], 0, H-1)
    npoints[...,0] = torch.clamp(npoints[..., 0], 0, H-1)
    
    return npoints

# 在三角形面片内部采样点（通过随机重心坐标），这些点的颜色要从原图提取并放到一个新的张量中。这样就能得到稀疏点云的颜色信息，用于训练或可视化。
def transfer_pixels(img, points1, points2, rbound=None):

    B, C, H, W = img.size()
    retained_pixels = torch.zeros_like(img).to(img.device)

    if rbound is not None:
        for bi in range(B):
            retained_pixels[bi, :, points2[bi, :rbound[bi], 1], points2[bi, :rbound[bi], 0]] = \
            img[bi, :, points1[bi, :rbound[bi], 1], points1[bi, :rbound[bi], 0]]
    else:
        retained_pixels[torch.arange(B).unsqueeze(-1), :, points2[..., 1], points2[..., 0]] = \
        img[torch.arange(B).unsqueeze(-1), :, points1[..., 1], points1[..., 0]]

    return retained_pixels

# 输入：FLAME 网格顶点、面片索引、面片采样概率、掩码比例、可选的坐标、图像大小。 输出：采样点的图像坐标，以及采样时的面片索引与重心坐标。功能：根据面片概率和掩码比例，从 FLAME 网格中采样点，用于生成掩码。
def mesh_based_mask_uniform_faces(
    flame_trans_verts,
    flame_faces,
    face_probabilities=None,
    mask_ratio=0.1,
    coords=None,
    IMAGE_SIZE=224,
    sampled_face_indices=None   # ⭐ TEST‑37 定向采样
):
    """
    支持三种模式：
    1) coords 复用模式：       直接复用上一次的 sampled_faces_indices + barycentric_coords
    2) TEST‑37 定向采样模式：  由 sampled_face_indices 显式指定采样哪些三角形
    3) 概率 + mask_ratio 模式：按 face_probabilities 和 mask_ratio 采样
    """

    batch_size = flame_trans_verts.size(0)
    DEVICE = flame_trans_verts.device

    # -----------------------------
    # CASE 1: coords 复用模式
    # -----------------------------
    if isinstance(coords, dict):
        # 约定 coords 是一个 dict
        sampled_faces_indices = coords["sampled_faces_indices"].to(DEVICE)
        barycentric_coords    = coords["barycentric_coords"].to(DEVICE)

        # 推出采样点数量
        num_points_to_sample = sampled_faces_indices.shape[1]

    # -----------------------------
    # CASE 2: TEST‑37 定向采样模式
    # -----------------------------
    elif sampled_face_indices is not None:
        # 自动 squeeze，避免 3D、4D 等异常情况
        sampled_face_indices = sampled_face_indices.squeeze()

        # 现在 dim 只可能是 1 或 2
        if sampled_face_indices.dim() == 1:
            # (K,) → (B,K)
            sampled_faces_indices = (
                sampled_face_indices.unsqueeze(0)
                .repeat(batch_size, 1)
                .to(DEVICE)
                .contiguous()
            )

        elif sampled_face_indices.dim() == 2:
            # (B,K)
            sampled_faces_indices = sampled_face_indices.to(DEVICE).contiguous()

        else:
            raise ValueError(
                f"sampled_face_indices 必须是 1D 或 2D，当前形状为 {sampled_face_indices.shape}"
            )

        num_points_to_sample = sampled_faces_indices.shape[1]

        # 生成重心坐标
        barycentric_coords = random_barycentric(
            num=batch_size * num_points_to_sample
        ).to(DEVICE)
        barycentric_coords = barycentric_coords.view(batch_size, num_points_to_sample, 3)


    # -----------------------------
    # CASE 3: 概率 + mask_ratio 模式
    # -----------------------------
    else:
        # 每个样本采样多少点
        num_points_to_sample = int(mask_ratio * IMAGE_SIZE * IMAGE_SIZE)

        # 扩展 faces 到 batch
        flame_faces_expanded = flame_faces.expand(batch_size, -1, -1)

        # 计算法线
        transformed_normals = vertex_normals(flame_trans_verts, flame_faces_expanded)
        transformed_face_normals = face_vertices(transformed_normals, flame_faces_expanded)
        transformed_face_normals = transformed_face_normals[:, :, :, 2].mean(dim=-1)

        # 如果 face_probabilities 为空，使用均匀分布
        if face_probabilities is None:
            face_probabilities = torch.ones(
                flame_faces.shape[0], device=DEVICE
            )

        # 扩展概率
        face_probabilities = face_probabilities.to(DEVICE).unsqueeze(0).repeat(batch_size, 1)

        # 法线过滤
        face_probabilities = torch.where(
            transformed_face_normals < 0.05,
            face_probabilities,
            torch.zeros_like(transformed_face_normals, device=DEVICE)
        )

        # 面积加权
        fv = face_vertices(flame_trans_verts, flame_faces_expanded)  # (B,F,3,3)
        xy_area = triangle_area(fv[..., :2])  # 只用 xy 平面面积
        face_probabilities = face_probabilities * xy_area

        # 避免全 0 概率
        face_probabilities = face_probabilities + 1e-8

        # 按概率采样面片
        sampled_faces_indices = torch.multinomial(
            face_probabilities,
            num_samples=num_points_to_sample,
            replacement=True
        ).to(DEVICE)  # (B,K)

        # 随机重心坐标
        barycentric_coords = random_barycentric(
            num=batch_size * num_points_to_sample
        ).to(DEVICE)
        barycentric_coords = barycentric_coords.view(batch_size, num_points_to_sample, 3)

    # -----------------------------
    # 统一：根据重心坐标插值采样点
    # -----------------------------
    npoints = vertices2landmarks(
        flame_trans_verts,
        flame_faces,
        sampled_faces_indices,
        barycentric_coords
    )  # (B,K,2) in [-1,1]

    # 映射到图像坐标
    npoints = 0.5 * (1 + npoints) * IMAGE_SIZE
    npoints = npoints.long()
    npoints[..., 1] = torch.clamp(npoints[..., 1], 0, IMAGE_SIZE - 1)
    npoints[..., 0] = torch.clamp(npoints[..., 0], 0, IMAGE_SIZE - 1)

    meta = {
        "sampled_faces_indices": sampled_faces_indices,
        "barycentric_coords": barycentric_coords,
    }

    return npoints, meta
