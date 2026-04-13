import torch
import numpy as np
from sklearn.decomposition import PCA


def transform_np(data, pca, bounds):
    x = data.reshape([-1, data.shape[-1]])
    x_pca = pca.transform(x)[..., -3:]
    x_bd = (x_pca - bounds[0]) / (bounds[1] - bounds[0])
    x_bd = np.clip(x_bd, 0., 1.)
    x_bd = x_bd.reshape([*data.shape[:-1], x_bd.shape[-1]])
    return x_bd


def get_transform_np(data):
    x = data.reshape([-1, data.shape[-1]])
    pca = PCA(n_components=3)
    pca.fit(x)
    x_pca = pca.transform(x)[..., -3:]
    bounds = np.stack([x_pca.min(axis=0), x_pca.max(axis=0)])
    trans_func = lambda a: transform_np(a, pca=pca, bounds=bounds)
    return trans_func


def get_transform(data):
    x = data.reshape([-1, data.shape[-1]])
    m, n = x.shape
    if n < 3:
        # Use normalization function if PCA cannot be applied
        bmax, bmin = x.max(dim=0).values, x.min(dim=0).values
        pad = 3 - n
        trans_func = lambda a: torch.cat([((a - bmin) / (bmax - bmin)).clamp(0., 1.), torch.ones((*a.shape[:-1], pad), device=a.device)], dim=-1)
        return trans_func
    else:
        U, _, V = torch.pca_lowrank(x, q=3)
        bmax, bmin = U.max(dim=0).values, U.min(dim=0).values
        trans_func = lambda a: ((torch.matmul(a, V) - bmin) / (bmax - bmin)).clamp(0., 1.)
        return trans_func

def _find_parent_cycles(parents: np.ndarray):
    """Return list of cycles, each as an ordered list of node indices.

    parents is a length-N array of parent pointers. Values may be -1 (root) or self-parent.
    """
    parents = np.asarray(parents).astype(np.int64, copy=False).reshape(-1)
    n = parents.shape[0]
    visited_global = np.zeros(n, dtype=bool)
    cycles = []

    for start in range(n):
        if visited_global[start]:
            continue

        order = []
        index_in_path = {}
        cur = int(start)

        while 0 <= cur < n and not visited_global[cur]:
            if cur in index_in_path:
                cycle = order[index_in_path[cur]:]
                if len(cycle) >= 2:
                    cycles.append(cycle)
                break

            index_in_path[cur] = len(order)
            order.append(cur)

            p = int(parents[cur])
            if p == -1 or p == cur:
                break
            cur = p

        for v in order:
            visited_global[v] = True

    return cycles

def _format_cycle(cycle, joints=None):
    loop = cycle + [cycle[0]]
    s = " -> ".join(str(i) for i in loop)
    if joints is None:
        return s
    j = np.asarray(joints)
    try:
        coords = [j[i].tolist() for i in cycle]
        return f"{s} | joints={coords}"
    except Exception:
        return s

def repair_skeleton_parents(joints, parents, verbose: bool = True, max_iters: int | None = None):
    """Repair cyclic/invalid parent pointers into an acyclic forest.

    Strategy (sensible + minimal change):
    - First sanitize invalid parent indices to -1.
    - Then iteratively detect cycles.
    - For each cycle, break it by reparenting ONE joint in the cycle to the nearest joint
      outside the cycle (using Euclidean distance in joint space). If no outside joint exists,
      make that joint a root (parent = -1).

    This avoids merging joints (which would require updating skin weights), while achieving a
    tree/forest that Blender and GLB viewers can load reliably.
    """
    parents = np.asarray(parents).astype(np.int64, copy=True).reshape(-1)
    n = parents.shape[0]
    if n == 0:
        return parents

    if max_iters is None:
        max_iters = max(4, n)

    # Sanitize: out-of-range parents become roots.
    invalid = ~((parents == -1) | ((parents >= 0) & (parents < n)) | (parents == np.arange(n)))
    if np.any(invalid):
        bad = np.where(invalid)[0]
        if verbose:
            print(f"Warning: invalid parent indices at {bad[:20].tolist()} (showing up to 20); setting them to -1.")
        parents[invalid] = -1

    joints_np = None
    if joints is not None:
        joints_np = np.asarray(joints, dtype=np.float32)
        if joints_np.ndim != 2 or joints_np.shape[0] != n or joints_np.shape[1] < 3:
            joints_np = None

    for it in range(int(max_iters)):
        cycles = _find_parent_cycles(parents)
        if not cycles:
            return parents

        if verbose:
            print(f"Warning: detected {len(cycles)} skeleton cycle(s); repairing (iter {it+1}/{max_iters}).")

        all_nodes = np.arange(n, dtype=np.int64)
        for cycle in cycles:
            if verbose:
                print("  Cycle:", _format_cycle(cycle, joints=joints_np))

            cyc = np.asarray(cycle, dtype=np.int64)
            in_cycle = np.zeros(n, dtype=bool)
            in_cycle[cyc] = True
            outside = all_nodes[~in_cycle]

            if joints_np is None or outside.size == 0:
                # Fallback: make the first node in the cycle a root.
                cut_node = int(cyc[0])
                if verbose:
                    print(f"  Fix: set parent[{cut_node}] = -1")
                parents[cut_node] = -1
                continue

            # Pick the closest (cycle_node, outside_node) pair to break the cycle with minimal distortion.
            cyc_pos = joints_np[cyc, :3]  # (C,3)
            out_pos = joints_np[outside, :3]  # (O,3)
            # squared distances
            d2 = ((cyc_pos[:, None, :] - out_pos[None, :, :]) ** 2).sum(axis=-1)
            flat = int(np.argmin(d2))
            ci, oi = divmod(flat, d2.shape[1])
            cut_node = int(cyc[ci])
            new_parent = int(outside[oi])
            if verbose:
                print(f"  Fix: set parent[{cut_node}] = {new_parent} (nearest outside)")
            parents[cut_node] = new_parent

    # If we get here, we failed to fully repair cycles within the limit.
    remaining = _find_parent_cycles(parents)
    if remaining and verbose:
        print(f"Error: remaining cycles after repair attempts: {len(remaining)}")
        for cycle in remaining[:10]:
            print("  Remaining cycle:", _format_cycle(cycle, joints=joints_np))
    return parents

def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    x = x - np.max(x, axis=axis, keepdims=True)
    np.exp(x, out=x)
    s = np.sum(x, axis=axis, keepdims=True)
    s = np.maximum(s, 1e-12)
    return x / s

def _joint_graph_all_pairs_hops(parents: np.ndarray) -> np.ndarray:
    """All-pairs shortest path length in hop-count on skeleton graph.

    Skeleton treated as undirected graph with edges (i, parent[i]) for valid parents.
    Returns an int32 matrix [M, M], where unreachable pairs are filled with a large value.
    """
    parents = np.asarray(parents, dtype=np.int64).reshape(-1)
    m = int(parents.shape[0])

    # Build undirected adjacency.
    adj: list[list[int]] = [[] for _ in range(m)]
    for i in range(m):
        p = int(parents[i])
        if p == -1 or p == i:
            continue
        if not (0 <= p < m):
            continue
        adj[i].append(p)
        adj[p].append(i)

    inf_hop = np.int32(1 << 29)
    hops = np.full((m, m), inf_hop, dtype=np.int32)

    for src in range(m):
        d = hops[src]
        d[src] = 0
        queue = [src]
        q_head = 0
        while q_head < len(queue):
            u = queue[q_head]
            q_head += 1
            nd = int(d[u]) + 1
            for v in adj[u]:
                if nd < int(d[v]):
                    d[v] = nd
                    queue.append(v)

    return hops

def filter_skinning_weights(
    mesh,
    skin_weights: np.ndarray,
    joints: np.ndarray,
    parents: np.ndarray,
    chunk_size: int = 8192,
    eps: float = 1e-8,
    anchor_mode: str = 'skin', # euclidean or skin
) -> np.ndarray:
    """Prune/renormalize skinning weights with a 2-hop skeleton constraint.

    For each vertex v:
    - anchor joint j* = argmax(skin_weights[v, :])
    - for joints k with hop_dist(j*, k) > 2 on the skeleton graph, set weight to 0
    - renormalize weights(v, :) to sum to 1
    """

    if mesh is None or not hasattr(mesh, 'vertices'):
        raise ValueError('mesh must have vertices')

    verts = np.asarray(mesh.vertices, dtype=np.float32)
    skin_weights = np.asarray(skin_weights, dtype=np.float32)
    joints = np.asarray(joints, dtype=np.float32)
    parents = np.asarray(parents, dtype=np.int64).reshape(-1)

    num_verts = int(verts.shape[0])
    num_joints = int(joints.shape[0])
    if num_verts == 0 or num_joints == 0:
        return np.zeros((num_verts, num_joints), dtype=np.float32)

    if skin_weights.ndim != 2 or skin_weights.shape[0] != num_verts or skin_weights.shape[1] != num_joints:
        raise ValueError(
            f"skin_weights has wrong shape: expected ({num_verts}, {num_joints}), got {tuple(skin_weights.shape)}"
        )

    # Ensure parents are valid/acyclic for graph construction.
    parents = repair_skeleton_parents(joints=joints, parents=parents, verbose=False)
    joint_hops = _joint_graph_all_pairs_hops(parents=parents)  # [M, M]
    out = np.empty_like(skin_weights, dtype=np.float32)
    max_hops = 2 * max(1, int(joints.shape[0] // 10))

    for start in range(0, num_verts, int(chunk_size)):
        end = min(num_verts, start + int(chunk_size))

        chunk_w = skin_weights[start:end, :].copy()  # [B, M]

        if anchor_mode == 'euclidean':
            d2 = ((verts[start:end, None, :] - joints[None, :, :]) ** 2).sum(axis=-1)  # [B, M]
            anchor = np.argmin(d2, axis=1).astype(np.int64)  # [B]
        elif anchor_mode == 'skin':
            anchor = np.argmax(chunk_w, axis=1).astype(np.int64)  # [B]

        # Keep only joints within max_hops from each anchor joint.
        keep_mask = joint_hops[anchor, :] <= max_hops  # [B, M]
        chunk_w[~keep_mask] = 0.0

        # Renormalize per vertex.
        sums = np.sum(chunk_w, axis=1, keepdims=True)
        bad = sums[:, 0] <= eps
        if np.any(bad):
            # Fallback to one-hot on anchor if row degenerates.
            chunk_w[bad, :] = 0.0
            chunk_w[bad, anchor[bad]] = 1.0
            sums = np.sum(chunk_w, axis=1, keepdims=True)
        out[start:end, :] = chunk_w / np.maximum(sums, eps)

    return out

def smooth_skin_weights_on_mesh(mesh, skin_weights, iterations=10, alpha=0.5):
    """Smooth per-vertex skin weights over the mesh surface.

    Uses iterative neighbor averaging over the undirected edge graph.
    Keeps weights non-negative and renormalized per vertex.
    """
    if iterations is None or iterations <= 0 or alpha is None or alpha <= 0:
        return skin_weights

    if not hasattr(mesh, "edges_unique") or mesh.edges_unique is None:
        return skin_weights

    w = np.asarray(skin_weights, dtype=np.float32)
    if w.ndim != 2:
        return skin_weights

    num_verts = mesh.vertices.shape[0]
    if w.shape[0] != num_verts:
        return skin_weights

    edges = np.asarray(mesh.edges_unique)
    if edges.size == 0:
        return skin_weights

    edges = edges.astype(np.int64, copy=False)
    i = edges[:, 0]
    j = edges[:, 1]

    # Clamp alpha to a sane range; alpha=1 becomes pure neighbor averaging.
    alpha = float(alpha)
    if alpha > 1.0:
        alpha = 1.0

    for _ in range(int(iterations)):
        neighbor_sum = np.zeros_like(w)
        np.add.at(neighbor_sum, i, w[j])
        np.add.at(neighbor_sum, j, w[i])

        degree = np.zeros((num_verts, 1), dtype=np.float32)
        np.add.at(degree, i, 1.0)
        np.add.at(degree, j, 1.0)

        avg = neighbor_sum / np.maximum(degree, 1.0)
        w = (1.0 - alpha) * w + alpha * avg

        # Keep weights valid.
        np.clip(w, 0.0, None, out=w)
        weight_sums = np.sum(w, axis=1, keepdims=True)
        weight_sums[weight_sums == 0] = 1.0
        w = w / weight_sums

    return w

