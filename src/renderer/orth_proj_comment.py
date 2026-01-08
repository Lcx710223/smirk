def batch_orth_proj(X, camera):
    ''' orthgraphic projection
        X:  3d vertices, [bz, n_point, 3]
           输入的 3D 顶点坐标（FLAME 输出）
           bz = batch size
           n_point = 顶点数量
           每个点是 (x, y, z)

        camera: scale and translation, [bz, 3]
           弱透视相机参数 [s, tx, ty]
           s  = 缩放
           tx = x 方向平移
           ty = y 方向平移
    '''

    # 将 camera 复制一份，避免原地修改
    # 并 reshape 成 [bz, 1, 3]，方便与所有顶点广播相加
    camera = camera.clone().view(-1, 1, 3)

    # 对 X 的前两个维度 (x, y) 做平移：
    # X[:, :, :2] 是所有顶点的 (x, y)
    # camera[:, :, 1:] 是 (tx, ty)
    # 结果是：x' = x + tx,  y' = y + ty
    X_trans = X[:, :, :2] + camera[:, :, 1:]

    # 把平移后的 (x', y') 与原来的 z 维度拼接回去
    # 得到新的 3D 顶点 (x', y', z)
    X_trans = torch.cat([X_trans, X[:, :, 2:]], dim=2)

    # 最后一步：对所有坐标乘以 scale（弱透视缩放）
    # Xn = s * (x', y', z)
    # 注意：弱透视模型不会改变 z 的相对深度，只是整体缩放
    Xn = camera[:, :, 0:1] * X_trans

    # 返回投影后的 3D 顶点（但 z 仍然保留，用于 rasterizer 深度排序）
    return Xn
