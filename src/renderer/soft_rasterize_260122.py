### LCX20260122，把SMIRK的基于CUDA的硬光栅化器，修改为兼容CPU/GPU的软光栅化器函数。
### 保持输入输出的接口不变。BY COPILOT。20260122.

import torch
import torch.nn as nn

def rasterize(
    self,
    vertices,
    faces,
    attributes=None,
    h=None,
    w=None,
    faces_per_pixel=8,
    sigma=1e-2,
    gamma=1e-2,
):
    """
    软光栅化器（CPU/GPU 兼容版）

    参数：
        vertices:  [B, V, 3]   已经在“屏幕空间 / NDC 空间”的顶点坐标
        faces:     [B, F, 3]   每个 batch 的三角形顶点索引
        attributes:[B, F, 3, D] 每个三角形三个顶点的属性（如 albedo+normal）
        h, w:      输出图像的高宽；若为 None，则使用 self.image_size
        faces_per_pixel: 每个像素保留的候选三角形数量 K
        sigma:     控制边界软化的“空间尺度”（像素距离）
        gamma:     控制深度 softmax 的“深度尺度”

    返回：
        pixel_vals: [B, D+1, H, W]
            - 前 D 通道：插值后的属性（如颜色+法线）
            - 最后 1 通道：soft vismask（0~1）
    """

    device = vertices.device
    B, V, _ = vertices.shape
    _, F, _ = faces.shape

    # -----------------------------
    # 1. 处理输出分辨率 & 坐标系修正
    # -----------------------------
    if h is None and w is None:
        h = w = self.image_size

    # 复制一份顶点，避免原地修改
    fixed_vertices = vertices.clone()

    # SMIRK 原版：翻转 x,y 以对齐屏幕坐标系
    fixed_vertices[..., :2] = -fixed_vertices[..., :2]

    # 长宽比修正：保持几何比例
    if h > w:
        fixed_vertices[..., 1] = fixed_vertices[..., 1] * h / w
    else:
        fixed_vertices[..., 0] = fixed_vertices[..., 0] * w / h

    # 现在 fixed_vertices 视为“归一化屏幕空间”，范围大致在 [-1,1]
    # 我们需要把它映射到像素坐标系 [0, W-1], [0, H-1]
    # 假设 x,y in [-1,1] → u = (x+1)/2 * (W-1), v = (y+1)/2 * (H-1)
    xs = fixed_vertices[..., 0]
    ys = fixed_vertices[..., 1]
    zs = fixed_vertices[..., 2]

    us = (xs + 1.0) * 0.5 * (w - 1)
    vs = (ys + 1.0) * 0.5 * (h - 1)

    # -----------------------------
    # 2. 构造像素网格 [H, W, 2]
    # -----------------------------
    # 像素中心坐标 (x+0.5, y+0.5)
    yy, xx = torch.meshgrid(
        torch.arange(h, device=device, dtype=torch.float32),
        torch.arange(w, device=device, dtype=torch.float32),
        indexing="ij",
    )
    # [H, W, 2]
    pixel_coords = torch.stack([xx + 0.5, yy + 0.5], dim=-1)  # (u,v)

    # -----------------------------
    # 3. 对每个 batch 独立处理（为了代码清晰，使用 Python for 循环）
    # -----------------------------
    all_pixel_vals = []
    all_vismask = []

    for b in range(B):
        # 当前 batch 的顶点 & faces
        verts_u = us[b]      # [V]
        verts_v = vs[b]      # [V]
        verts_z = zs[b]      # [V]
        faces_b = faces[b]   # [F, 3]

        # 当前 batch 的属性 [F, 3, D]
        if attributes is not None:
            attrs_b = attributes[b]  # [F, 3, D]
            D = attrs_b.shape[-1]
        else:
            # 若没有属性，就只输出 vismask
            D = 0
            attrs_b = None

        # 取出每个三角形三个顶点的 (u,v,z)
        # faces_b: [F, 3]
        v0 = faces_b[:, 0]  # [F]
        v1 = faces_b[:, 1]
        v2 = faces_b[:, 2]

        p0 = torch.stack([verts_u[v0], verts_v[v0]], dim=-1)  # [F, 2]
        p1 = torch.stack([verts_u[v1], verts_v[v1]], dim=-1)  # [F, 2]
        p2 = torch.stack([verts_u[v2], verts_v[v2]], dim=-1)  # [F, 2]

        z0 = verts_z[v0]  # [F]
        z1 = verts_z[v1]
        z2 = verts_z[v2]

        # -----------------------------
        # 4. 计算每个像素相对于每个三角形的重心坐标 & inside mask
        # -----------------------------
        # pixel_coords: [H, W, 2] → [H*W, 2]
        P = pixel_coords.view(-1, 2)  # [Npix, 2], Npix = H*W

        # 为了向量化，扩展维度：
        # P: [Npix, 1, 2]
        # p0,p1,p2: [1, F, 2]
        P_exp = P[:, None, :]          # [Npix, 1, 2]
        p0_exp = p0[None, :, :]        # [1, F, 2]
        p1_exp = p1[None, :, :]
        p2_exp = p2[None, :, :]

        # 使用重心坐标公式：
        # 参考标准三角形重心坐标推导
        v0v1 = p1_exp - p0_exp  # [1, F, 2]
        v0v2 = p2_exp - p0_exp  # [1, F, 2]
        v0p  = P_exp - p0_exp   # [Npix, F, 2]

        # 计算 2x2 矩阵的行列式（用于重心坐标分母）
        # denom = (v0v1.x * v0v2.y - v0v1.y * v0v2.x)
        denom = v0v1[..., 0] * v0v2[..., 1] - v0v1[..., 1] * v0v2[..., 0]  # [1, F]

        # 避免除零
        eps = 1e-8
        denom = denom + (denom.abs() < eps).float() * eps

        # 计算重心坐标 (b1, b2, b0 = 1 - b1 - b2)
        # b1 = (v0p.x * v0v2.y - v0p.y * v0v2.x) / denom
        # b2 = (v0v1.x * v0p.y - v0v1.y * v0p.x) / denom
        b1 = (v0p[..., 0] * v0v2[..., 1] - v0p[..., 1] * v0v2[..., 0]) / denom  # [Npix, F]
        b2 = (v0v1[..., 0] * v0p[..., 1] - v0v1[..., 1] * v0p[..., 0]) / denom  # [Npix, F]
        b0 = 1.0 - b1 - b2                                                    # [Npix, F]

        bary = torch.stack([b0, b1, b2], dim=-1)  # [Npix, F, 3]

        # inside mask：所有重心坐标 >= 0 即在三角形内部
        inside = (bary >= 0.0).all(dim=-1)  # [Npix, F]

        # -----------------------------
        # 5. 计算“到三角形边界的距离”用于 soft 边界
        #    这里用一个简单近似：对每条边的有符号距离取负部分的最大值
        # -----------------------------
        # 边向量
        e0 = p1_exp - p0_exp  # [1, F, 2]
        e1 = p2_exp - p1_exp
        e2 = p0_exp - p2_exp

        # 法向量（2D 中，(x,y) 的法向量可以取 (y,-x) 或 (-y,x)）
        n0 = torch.stack([e0[..., 1], -e0[..., 0]], dim=-1)  # [1, F, 2]
        n1 = torch.stack([e1[..., 1], -e1[..., 0]], dim=-1)
        n2 = torch.stack([e2[..., 1], -e2[..., 0]], dim=-1)

        # 点到边的有符号距离： (P - p0)·n / ||n||
        def signed_dist(P, p, n):
            num = (P - p).mul(n).sum(dim=-1)  # [Npix, F]
            den = n.norm(dim=-1)             # [1, F]
            den = den + (den.abs() < eps).float() * eps
            return num / den

        d0 = signed_dist(P_exp, p0_exp, n0)  # [Npix, F]
        d1 = signed_dist(P_exp, p1_exp, n1)
        d2 = signed_dist(P_exp, p2_exp, n2)

        # 对于在三角形内部的点，三个距离都应为 >=0
        # 对于 soft 边界，我们关心“离边界有多远”，取三个距离的最小值
        d_min = torch.min(torch.min(d0, d1), d2)  # [Npix, F]

        # -----------------------------
        # 6. 计算 soft 覆盖权重（空间维度）
        # -----------------------------
        # 对于在三角形内部的点，d_min >= 0；在外部则 < 0
        # 我们可以用一个高斯核来平滑边界：
        # w_space = exp( - max(0, -d_min)^2 / (2*sigma^2) )
        # 也可以简单地对 d_min 做一个平滑 ReLU，这里用高斯形式
        neg_part = torch.clamp(-d_min, min=0.0)  # [Npix, F]
        w_space = torch.exp(- (neg_part ** 2) / (2.0 * sigma * sigma))  # [Npix, F]

        # 同时乘上 inside mask，让外部区域权重更小
        w_space = w_space * inside.float()  # [Npix, F]

        # -----------------------------
        # 7. 计算每个像素对每个三角形的深度（用重心插值 z）
        # -----------------------------
        z0_exp = z0[None, :]  # [1, F]
        z1_exp = z1[None, :]
        z2_exp = z2[None, :]

        z_face = (
            b0 * z0_exp +
            b1 * z1_exp +
            b2 * z2_exp
        )  # [Npix, F]

        # -----------------------------
        # 8. 选取每个像素的 top-K 三角形（按空间权重 * 深度排序）
        # -----------------------------
        # 先用 w_space 过滤掉几乎为 0 的三角形
        # 再按 z_face 做 soft 排序
        # 这里我们先简单按 z_face（越小越前）排序，然后取 top-K
        K = min(faces_per_pixel, F)

        # 为了避免无效三角形干扰，把 w_space 非零的地方才参与排序
        # 但排序本身不支持 mask，这里简单地把 w_space 很小的 z_face 设为一个大值
        big_val = 1e6
        z_for_sort = z_face.clone()
        z_for_sort[w_space < 1e-6] = big_val

        # argsort 得到从小到大的索引
        _, idx_sort = torch.sort(z_for_sort, dim=-1)  # [Npix, F]
        idx_topk = idx_sort[:, :K]                    # [Npix, K]

        # 取出 top-K 的各种量
        # [Npix, K]
        z_topk = torch.gather(z_face, dim=-1, index=idx_topk)
        w_space_topk = torch.gather(w_space, dim=-1, index=idx_topk)

        # 重心坐标 [Npix, F, 3] → [Npix, K, 3]
        bary_topk = torch.gather(
            bary,
            dim=1,
            index=idx_topk[..., None].expand(-1, -1, 3)
        )  # [Npix, K, 3]

        # -----------------------------
        # 9. 深度 softmax（soft z-buffer）
        # -----------------------------
        # w_depth = softmax( -z / gamma )
        w_depth = torch.softmax(-z_topk / gamma, dim=-1)  # [Npix, K]

        # 总权重 = 空间权重 * 深度权重
        w_total = w_space_topk * w_depth  # [Npix, K]

        # -----------------------------
        # 10. 插值属性并做 soft blending
        # -----------------------------
        if D > 0:
            # attrs_b: [F, 3, D]
            # 先取出 top-K 三角形的属性 [Npix, K, 3, D]
            attrs_exp = attrs_b[None, :, :, :]  # [1, F, 3, D]
            attrs_topk = torch.gather(
                attrs_exp.expand(P.shape[0], -1, -1, -1),  # [Npix, F, 3, D]
                dim=1,
                index=idx_topk[..., None, None].expand(-1, -1, 3, D)
            )  # [Npix, K, 3, D]

            # 用重心坐标对每个三角形内部插值属性
            # bary_topk: [Npix, K, 3]
            # → [Npix, K, 3, 1]
            bary_exp = bary_topk[..., None]  # [Npix, K, 3, 1]
            attr_interp = (bary_exp * attrs_topk).sum(dim=2)  # [Npix, K, D]

            # 对 K 个三角形做 soft blending
            # w_total: [Npix, K] → [Npix, K, 1]
            w_exp = w_total[..., None]  # [Npix, K, 1]
            pixel_attr = (w_exp * attr_interp).sum(dim=1)  # [Npix, D]
        else:
            pixel_attr = None

        # -----------------------------
        # 11. soft vismask
        # -----------------------------
        # vismask = sum_k w_total
        vis = w_total.sum(dim=-1)  # [Npix]

        # -----------------------------
        # 12. reshape 回 [H, W, ...]
        # -----------------------------
        if D > 0:
            pixel_attr = pixel_attr.view(h, w, D)  # [H, W, D]
            pixel_attr = pixel_attr.permute(2, 0, 1)  # [D, H, W]
        vis = vis.view(h, w)  # [H, W]
        vis = vis[None, :, :]  # [1, H, W]

        if D > 0:
            out_b = torch.cat([pixel_attr, vis], dim=0)  # [D+1, H, W]
        else:
            out_b = vis  # [1, H, W]

        all_pixel_vals.append(out_b)
        all_vismask.append(vis)

    # -----------------------------
    # 13. 拼回 batch 维度 [B, D+1, H, W]
    # -----------------------------
    pixel_vals = torch.stack(all_pixel_vals, dim=0)  # [B, D+1, H, W]
    return pixel_vals
