import torch
import numpy as np
from pytorch3d.ops import ball_query, knn_points


def disjoint_set_unioin_find(N, pairs):
    def find(x, parent):
        if parent[x] != x:
            parent[x] = find(parent[x], parent)
        return parent[x]
    def union(a, b, parent, rank):
        rootA = find(a, parent)
        rootB = find(b, parent)
        if rootA != rootB:
            if rank[rootA] > rank[rootB]:
                parent[rootB] = rootA
            elif rank[rootA] < rank[rootB]:
                parent[rootA] = rootB
            else:
                parent[rootB] = rootA
                rank[rootA] += 1
    parent = list(range(N))
    rank = [0] * N
    for a, b in pairs:
        if a >= 0 and b >= 0:
            union(a, b, parent, rank)
    for i in range(N):
        find(i, parent)
    root_ids = np.unique(parent)
    num_joints = len(root_ids)
    joints_index_map = {root_id: idx for idx, root_id in enumerate(root_ids)}
    joints_idx = [joints_index_map[parent[i]] for i in range(N)]
    return num_joints, joints_idx


def threshold_grouping(joints, parents, joints_conf=None, parents_conf=None, conf_threshold=0.1, threshold=1/32):
    '''
    Cluster joints and parents with confidence based on the threshold
    Input:
        joints:          torch.Tensor [N, 3] coordinates of predicted joints
        parents:         torch.Tensor [N, 3] coordinates of predicted parents
        joints_conf:     torch.Tensor [N, 1] confidence of predicted joints
        parents_conf:    torch.Tensor [N, 1] confidence of predicted parents
        conf_threshold:  confidence threshold to filter joints and parents
        threshold:       distance threshold to group joints and parents
    Output:
        grouped_joints:  [M, 3] coordinates of grouped joints
        grouped_parents: [M]    indexes of grouped parents
    '''
    assert joints.shape == parents.shape, "joints and parents must have the same shape"
    # Confidence filtering
    joints_conf = torch.ones_like(joints[:, 0:1]) if joints_conf is None else joints_conf
    parents_conf = torch.ones_like(parents[:, 0:1]) if parents_conf is None else parents_conf
    conf_mask = ((joints_conf >= conf_threshold) & (parents_conf >= conf_threshold)).squeeze()
    if not torch.any(conf_mask):
        return joints.new_empty((0, 3)), joints.new_empty((0, 1), dtype=torch.long)
    joints, parents, joints_conf, parents_conf = joints[conf_mask], parents[conf_mask], joints_conf[conf_mask], parents_conf[conf_mask]
    # Threshold grouping
    _, ball_query_idx, _ = ball_query(joints.unsqueeze(0), joints.unsqueeze(0), K=9, radius=threshold)
    ball_query_idx = ball_query_idx[0, :, 1:]
    pairs = [[idx, neighbor.item()] for idx, neighbors in enumerate(ball_query_idx) for neighbor in neighbors]
    Nj, joints_idx = disjoint_set_unioin_find(joints.shape[0], pairs)
    joints_idx = torch.tensor(joints_idx, device=joints.device).long()
    # Compute grouped joints and parents
    grouped_joints_weighted = torch.scatter_add(torch.zeros((Nj, 3), device=joints.device), 0, joints_idx.unsqueeze(1).expand(-1, 3), joints * joints_conf)
    grouped_joints_conf_sum = torch.scatter_add(torch.zeros((Nj, 1), device=joints.device), 0, joints_idx.unsqueeze(1),               joints_conf)
    grouped_joints = grouped_joints_weighted / (grouped_joints_conf_sum + 1e-8)
    grouped_parents_weighted = torch.scatter_add(torch.zeros((Nj, 3), device=joints.device), 0, joints_idx.unsqueeze(1).expand(-1, 3), parents * parents_conf)
    grouped_parents_conf_sum = torch.scatter_add(torch.zeros((Nj, 1), device=joints.device), 0, joints_idx.unsqueeze(1),               parents_conf)
    grouped_parents = grouped_parents_weighted / (grouped_parents_conf_sum + 1e-8)
    # Determine the parents index
    parents_idx = knn_points(grouped_parents.unsqueeze(0), grouped_joints.unsqueeze(0), K=1).idx[0, :, 0]
    parents_idx[parents_idx == torch.arange(parents_idx.shape[0], device=parents_idx.device)] = -1
    return grouped_joints, parents_idx


def mean_shift_grouping(joints, parents, joints_conf=None, parents_conf=None, conf_threshold=0.5, threshold=1/100, cluster_size_threshold=5):
    '''
    Cluster joints and parents with confidence based on a mean shift procedure
    Input:
        joints:          torch.Tensor [N, 3] coordinates of predicted joints
        parents:         torch.Tensor [N, 3] coordinates of predicted parents
        joints_conf:     torch.Tensor [N, 1] confidence of predicted joints
        parents_conf:    torch.Tensor [N, 1] confidence of predicted parents
        conf_threshold:  confidence threshold to filter joints and parents
        threshold:       bandwidth parameter for mean shift (radius of local window)
    Output:
        grouped_joints:  [M, 3] coordinates of grouped joints
        grouped_parents: [M]    indexes of grouped parents
    '''
    assert joints.shape == parents.shape, "joints and parents must have the same shape"
    joints_conf = torch.ones_like(joints[:, 0:1]) if joints_conf is None else joints_conf
    parents_conf = torch.ones_like(parents[:, 0:1]) if parents_conf is None else parents_conf

    conf_mask = ((joints_conf >= conf_threshold) & (parents_conf >= conf_threshold)).squeeze()
    if not torch.any(conf_mask):
        return torch.zeros_like(joints[:1]), torch.zeros([1], device=joints.device, dtype=torch.long)

    joints = joints[conf_mask]
    parents = parents[conf_mask]
    joints_conf = joints_conf[conf_mask]
    parents_conf = parents_conf[conf_mask]

    bandwidth = max(float(threshold), 1e-4)
    tol = bandwidth * 0.1
    max_iters = 30
    max_neighbors = 32
    inv_two_sigma_sq = 0.5 / (bandwidth * bandwidth)

    # Mean shift iterations (vectorized over all points)
    shifted = joints.clone()
    weights = joints_conf
    for _ in range(max_iters):
        neighbor_dist, neighbor_idx, _ = ball_query(shifted.unsqueeze(0), shifted.unsqueeze(0), K=max_neighbors+1, radius=bandwidth)
        neighbor_dist, neighbor_idx = neighbor_dist[0, :, 1:], neighbor_idx[0, :, 1:]
        valid_mask = (neighbor_idx >= 0)
        safe_idx = neighbor_idx.clamp(min=0)
        neighbor_points = shifted[safe_idx]
        neighbor_weights = weights[safe_idx]
        neighbor_weights = neighbor_weights * valid_mask[..., None].float()

        kernel = torch.exp(-neighbor_dist.unsqueeze(-1) * inv_two_sigma_sq) * valid_mask[..., None].float()
        neighbor_weights = neighbor_weights * kernel

        weight_sum = neighbor_weights.sum(dim=1)
        weighted_sum = (neighbor_points * neighbor_weights).sum(dim=1)
        updated = torch.where(weight_sum > 0, weighted_sum / (weight_sum + 1e-8), shifted)
        max_shift = (updated - shifted).norm(dim=1).max()
        shifted = updated
        if max_shift <= tol:
            break

    # Merge converged points into discrete clusters
    Nc = shifted.shape[0]
    cluster_indices = torch.full((Nc,), -1, dtype=torch.long, device=joints.device)
    merge_radius = bandwidth * 0.5
    cluster_count = 0
    for i in range(Nc):
        if cluster_indices[i] >= 0:
            continue
        center = shifted[i]
        distances = torch.norm(shifted - center, dim=1)
        same_cluster = distances <= merge_radius
        cluster_indices[same_cluster] = cluster_count
        cluster_count += 1

    Nj = cluster_count
    joints_idx = cluster_indices

    cluster_sizes = torch.bincount(joints_idx, minlength=Nj).float()
    if cluster_size_threshold > 0:
        keep_mask = cluster_sizes >= cluster_size_threshold
    else:
        keep_mask = torch.ones((Nj,), dtype=torch.bool, device=joints.device)
    if not torch.any(keep_mask):
        keep_mask = torch.ones((Nj,), dtype=torch.bool, device=joints.device)

    grouped_joints_weighted = torch.scatter_add(torch.zeros((Nj, 3), device=joints.device), 0, joints_idx.unsqueeze(1).expand(-1, 3), joints * joints_conf)
    grouped_joints_conf_sum = torch.scatter_add(torch.zeros((Nj, 1), device=joints.device), 0, joints_idx.unsqueeze(1), joints_conf)
    grouped_joints = grouped_joints_weighted / (grouped_joints_conf_sum + 1e-8)

    grouped_parents_weighted = torch.scatter_add( torch.zeros((Nj, 3), device=joints.device), 0, joints_idx.unsqueeze(1).expand(-1, 3), parents * parents_conf)
    grouped_parents_conf_sum = torch.scatter_add( torch.zeros((Nj, 1), device=joints.device), 0, joints_idx.unsqueeze(1), parents_conf)
    grouped_parents = grouped_parents_weighted / (grouped_parents_conf_sum + 1e-8)

    grouped_joints = grouped_joints[keep_mask]
    grouped_parents = grouped_parents[keep_mask]

    if grouped_joints.numel() == 0:
        return torch.zeros_like(joints[:1]), torch.zeros([1], device=joints.device, dtype=torch.long)

    parents_idx = knn_points(grouped_parents.unsqueeze(0), grouped_joints.unsqueeze(0), K=1).idx[0, :, 0]
    parents_idx[parents_idx == torch.arange(parents_idx.shape[0], device=parents_idx.device)] = -1
    return grouped_joints, parents_idx


GROUPING_STRATEGIES = {
    "threshold": threshold_grouping,
    "mean_shift": mean_shift_grouping,
}

