import sys
import struct
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict

# Third-party imports
try:
    from pygltflib import GLTF2, BufferFormat, Accessor, BufferView
    import networkx as nx
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Please install required packages:")
    print("  pip install pygltflib networkx numpy")
    sys.exit(1)

try:
    import open3d as o3d
except:
    print("Missing dependency: open3d")
    pass

# Splitting parameters
GEODESIC_DISTANCE_THRESHOLD = 3  # Max allowed geodesic distance between bones
WEIGHT_THRESHOLD = 0.15  # Minimum weight to consider a joint as "influential"
SPREAD_WEIGHT_THRESHOLD = 0.1  # Minimum weight for spread detection
EDGE_SMOOTH_ITERATIONS = 5  # Iterations for smoothing edge vertex weights
EDGE_SMOOTH_ALPHA = 0.7  # Smoothing strength for edge vertices


# =============================================================================
# glTF Data Access Utilities (reused from texture_transfer.py)
# =============================================================================

# Component type mapping for glTF accessors
COMPONENT_TYPES = {
    5120: ('b', 1, np.int8),      # BYTE
    5121: ('B', 1, np.uint8),     # UNSIGNED_BYTE
    5122: ('h', 2, np.int16),     # SHORT
    5123: ('H', 2, np.uint16),    # UNSIGNED_SHORT
    5125: ('I', 4, np.uint32),    # UNSIGNED_INT
    5126: ('f', 4, np.float32),   # FLOAT
}

# Type to component count mapping
TYPE_COUNTS = {
    'SCALAR': 1, 'VEC2': 2, 'VEC3': 3, 'VEC4': 4,
    'MAT2': 4, 'MAT3': 9, 'MAT4': 16
}


def get_binary_blob(gltf: GLTF2) -> bytes:
    """Get the binary blob from a GLTF2 object."""
    binary_blob = gltf.binary_blob
    if callable(binary_blob):
        binary_blob = binary_blob()
    return binary_blob if isinstance(binary_blob, bytes) else b''


def set_binary_blob(gltf: GLTF2, data: bytes) -> None:
    """Set the binary blob on a GLTF2 object."""
    if hasattr(gltf, 'set_binary_blob') and callable(gltf.set_binary_blob):
        gltf.set_binary_blob(data)
    else:
        gltf.binary_blob = data
    
    if gltf.buffers:
        gltf.buffers[0].byteLength = len(data)


def get_buffer_data(gltf: GLTF2) -> bytes:
    """Get the binary buffer data from a GLB file."""
    gltf.convert_buffers(BufferFormat.BINARYBLOB)
    binary_blob = get_binary_blob(gltf)
    
    if binary_blob is not None and len(binary_blob) > 0:
        return binary_blob
    
    if hasattr(gltf, '_glb_data') and gltf._glb_data is not None:
        return gltf._glb_data
    
    return b''


def read_accessor_data(gltf: GLTF2, accessor_index: int, 
                       buffer_data: Optional[bytes] = None) -> np.ndarray:
    """Read data from a glTF accessor into a numpy array."""
    if buffer_data is None:
        buffer_data = get_buffer_data(gltf)
    
    accessor = gltf.accessors[accessor_index]
    buffer_view = gltf.bufferViews[accessor.bufferView]
    
    fmt_char, component_size, np_dtype = COMPONENT_TYPES[accessor.componentType]
    component_count = TYPE_COUNTS[accessor.type]
    element_size = component_size * component_count
    
    byte_offset = (buffer_view.byteOffset or 0) + (accessor.byteOffset or 0)
    stride = buffer_view.byteStride or element_size
    
    values = []
    for i in range(accessor.count):
        offset = byte_offset + i * stride
        for j in range(component_count):
            value = struct.unpack_from(fmt_char, buffer_data, offset + j * component_size)[0]
            values.append(value)
    
    arr = np.array(values, dtype=np_dtype)
    if accessor.type == 'SCALAR':
        return arr
    elif accessor.type in ('VEC2', 'VEC3', 'VEC4'):
        return arr.reshape(-1, component_count)
    elif accessor.type == 'MAT4':
        return arr.reshape(-1, 4, 4)
    
    return arr


def read_mesh_data(gltf: GLTF2, mesh_index: int = 0,
                   buffer_data: Optional[bytes] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Read vertex positions and face indices from a mesh.
    
    Returns:
        Tuple of (vertices, faces) where:
        - vertices: (N, 3) array of vertex positions
        - faces: (M, 3) array of triangle indices
    """
    if buffer_data is None:
        buffer_data = get_buffer_data(gltf)
    
    mesh = gltf.meshes[mesh_index]
    all_vertices = []
    all_faces = []
    vertex_offset = 0
    
    for primitive in mesh.primitives:
        # Read vertices
        pos_accessor_idx = primitive.attributes.POSITION
        vertices = read_accessor_data(gltf, pos_accessor_idx, buffer_data)
        all_vertices.append(vertices)
        
        # Read indices
        if primitive.indices is not None:
            indices = read_accessor_data(gltf, primitive.indices, buffer_data)
            # Reshape to triangles and offset by vertex count
            faces = indices.reshape(-1, 3) + vertex_offset
            all_faces.append(faces)
        
        vertex_offset += len(vertices)
    
    vertices = np.vstack(all_vertices) if len(all_vertices) > 1 else all_vertices[0]
    faces = np.vstack(all_faces) if len(all_faces) > 1 else all_faces[0] if all_faces else np.array([])
    
    return vertices.astype(np.float32), faces.astype(np.int32)


def read_skinning_data(gltf: GLTF2, mesh_index: int = 0,
                       buffer_data: Optional[bytes] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Read JOINTS_0 and WEIGHTS_0 from a mesh.
    
    Returns:
        Tuple of (joints, weights) arrays, each with shape (num_vertices, 4)
    """
    if buffer_data is None:
        buffer_data = get_buffer_data(gltf)
    
    mesh = gltf.meshes[mesh_index]
    all_joints = []
    all_weights = []
    
    for primitive in mesh.primitives:
        attrs = primitive.attributes
        
        if hasattr(attrs, 'JOINTS_0') and attrs.JOINTS_0 is not None:
            joints = read_accessor_data(gltf, attrs.JOINTS_0, buffer_data)
            all_joints.append(joints.reshape(-1, 4))
        
        if hasattr(attrs, 'WEIGHTS_0') and attrs.WEIGHTS_0 is not None:
            weights = read_accessor_data(gltf, attrs.WEIGHTS_0, buffer_data)
            all_weights.append(weights.reshape(-1, 4))
    
    if not all_joints or not all_weights:
        return None, None
    
    joints = np.vstack(all_joints) if len(all_joints) > 1 else all_joints[0]
    weights = np.vstack(all_weights) if len(all_weights) > 1 else all_weights[0]
    
    return joints.astype(np.int32), weights.astype(np.float32)


# =============================================================================
# Skeleton Graph Construction
# =============================================================================

def build_skeleton_graph(gltf: GLTF2) -> Tuple[nx.Graph, List[int]]:
    """
    Build an undirected graph representing the skeleton structure.
    
    Args:
        gltf: The GLTF2 object containing the skeleton
    
    Returns:
        Tuple of (graph, joint_indices) where:
        - graph: NetworkX graph with joints as nodes and bones as edges
        - joint_indices: List of node indices that are joints
    """
    if not gltf.skins:
        raise ValueError("GLB file has no skin/skeleton data")
    
    skin = gltf.skins[0]
    joint_indices = skin.joints
    
    # Create graph
    G = nx.Graph()
    
    # Add all joints as nodes
    for joint_idx in joint_indices:
        node = gltf.nodes[joint_idx]
        G.add_node(joint_idx, name=node.name or f"Joint_{joint_idx}")
    
    # Add edges based on parent-child relationships
    for joint_idx in joint_indices:
        node = gltf.nodes[joint_idx]
        if node.children:
            for child_idx in node.children:
                if child_idx in joint_indices:
                    G.add_edge(joint_idx, child_idx, weight=1)
    
    return G, joint_indices


def compute_geodesic_distances(G: nx.Graph, joint_indices: List[int]) -> Dict[Tuple[int, int], int]:
    """
    Compute geodesic distances (shortest path lengths) between all pairs of joints.
    
    Args:
        G: The skeleton graph
        joint_indices: List of joint node indices
    
    Returns:
        Dictionary mapping (joint_i, joint_j) -> distance
    """
    # Compute all pairs shortest paths
    distances = dict(nx.all_pairs_shortest_path_length(G))
    
    # Build distance dictionary
    dist_dict = {}
    for i in joint_indices:
        for j in joint_indices:
            if i in distances and j in distances[i]:
                dist_dict[(i, j)] = distances[i][j]
            else:
                # If no path exists, set to infinity
                dist_dict[(i, j)] = float('inf')
    
    return dist_dict


def get_primary_joint(joints: np.ndarray, weights: np.ndarray, vertex_idx: int) -> int:
    """
    Get the primary (highest weight) joint for a vertex.
    
    Args:
        joints: Joint indices array (N, 4)
        weights: Weight values array (N, 4)
        vertex_idx: Index of the vertex
    
    Returns:
        Joint index with highest weight
    """
    vertex_joints = joints[vertex_idx]
    vertex_weights = weights[vertex_idx]
    
    max_idx = np.argmax(vertex_weights)
    return int(vertex_joints[max_idx])


def get_influential_joints(joints: np.ndarray, weights: np.ndarray, 
                           vertex_idx: int, weight_threshold: float = WEIGHT_THRESHOLD) -> List[Tuple[int, float]]:
    """
    Get all joints with weight above threshold for a vertex.
    
    Args:
        joints: Joint indices array (N, 4)
        weights: Weight values array (N, 4)
        vertex_idx: Index of the vertex
        weight_threshold: Minimum weight to consider
    
    Returns:
        List of (joint_index, weight) tuples for influential joints
    """
    vertex_joints = joints[vertex_idx]
    vertex_weights = weights[vertex_idx]
    
    influential = []
    for i in range(4):
        if vertex_weights[i] >= weight_threshold:
            influential.append((int(vertex_joints[i]), float(vertex_weights[i])))
    
    return influential


def is_vertex_spread_to_distant_bones(joints: np.ndarray, weights: np.ndarray,
                                       vertex_idx: int, joint_indices: List[int],
                                       geodesic_distances: Dict[Tuple[int, int], int],
                                       distance_threshold: int = GEODESIC_DISTANCE_THRESHOLD,
                                       weight_threshold: float = SPREAD_WEIGHT_THRESHOLD) -> bool:
    """
    Check if a vertex has its weights spread across distant bones.
    
    A vertex is problematic if it has significant weights (>= weight_threshold)
    on multiple bones that are far apart in the skeleton graph.
    
    Args:
        joints: Joint indices array (N, 4)
        weights: Weight values array (N, 4)
        vertex_idx: Index of the vertex
        joint_indices: List of valid joint node indices
        geodesic_distances: Precomputed geodesic distances
        distance_threshold: Maximum allowed geodesic distance
        weight_threshold: Minimum weight to consider as "significant"
    
    Returns:
        True if vertex has weights spread to distant bones
    """
    vertex_joints = joints[vertex_idx]
    vertex_weights = weights[vertex_idx]
    
    # Create mapping from skin joint index to node index
    skin_to_node = {i: joint_indices[i] for i in range(len(joint_indices))}
    
    # Find all joints with significant weight
    significant_joints = []
    for i in range(4):
        if vertex_weights[i] >= weight_threshold:
            skin_idx = int(vertex_joints[i])
            if skin_idx < len(joint_indices):
                node_idx = skin_to_node[skin_idx]
                significant_joints.append((node_idx, float(vertex_weights[i])))
    
    # If only one or zero significant joints, not spread
    if len(significant_joints) < 2:
        return False
    
    # Check geodesic distance between all pairs of significant joints
    for i in range(len(significant_joints)):
        for j in range(i + 1, len(significant_joints)):
            joint_i, weight_i = significant_joints[i]
            joint_j, weight_j = significant_joints[j]
            
            dist = geodesic_distances.get((joint_i, joint_j), float('inf'))
            
            # If two joints are far apart and both have significant weight
            if dist > distance_threshold:
                # Additional check: the combined weight should be substantial
                # This catches cases where weights are "balanced" between distant bones
                combined_weight = weight_i + weight_j
                if combined_weight >= 0.3:  # At least 30% of weight on distant bones
                    return True
    
    return False


# =============================================================================
# Edge Vertex Weight Cleaning
# =============================================================================

def find_edge_vertices(faces: np.ndarray, 
                       triangles_to_remove: Set[int]) -> Tuple[Set[int], Set[int]]:
    """
    Find vertices that are on the edge of removed regions.
    
    Edge vertices are those that:
    - Belong to at least one removed triangle
    - Also belong to at least one remaining triangle
    
    Args:
        faces: Triangle indices (M, 3)
        triangles_to_remove: Set of triangle indices to remove
    
    Returns:
        Tuple of (edge_vertices, removed_only_vertices)
    """
    removed_vertices = set()
    remaining_vertices = set()
    
    for face_idx, face in enumerate(faces):
        if face_idx in triangles_to_remove:
            for v in face:
                removed_vertices.add(int(v))
        else:
            for v in face:
                remaining_vertices.add(int(v))
    
    # Edge vertices: in both removed and remaining triangles
    edge_vertices = removed_vertices & remaining_vertices
    
    # Vertices only in removed triangles (will become orphaned)
    removed_only_vertices = removed_vertices - remaining_vertices
    
    return edge_vertices, removed_only_vertices


def clean_edge_vertex_weights(joints: np.ndarray, weights: np.ndarray,
                               edge_vertices: Set[int],
                               joint_indices: List[int],
                               geodesic_distances: Dict[Tuple[int, int], int],
                               distance_threshold: int = GEODESIC_DISTANCE_THRESHOLD) -> np.ndarray:
    """
    Clean weights of edge vertices by removing influence from distant bones.
    
    For each edge vertex, find its primary joint and remove weights from
    joints that are too far away in the skeleton graph.
    
    Args:
        joints: Joint indices array (N, 4)
        weights: Weight values array (N, 4)
        edge_vertices: Set of edge vertex indices
        joint_indices: List of valid joint node indices
        geodesic_distances: Precomputed geodesic distances
        distance_threshold: Maximum allowed geodesic distance
    
    Returns:
        Modified weights array
    """
    new_weights = weights.copy()
    skin_to_node = {i: joint_indices[i] for i in range(len(joint_indices))}
    
    cleaned_count = 0
    
    for vertex_idx in edge_vertices:
        vertex_joints = joints[vertex_idx]
        vertex_weights = new_weights[vertex_idx]
        
        # Find primary joint (highest weight)
        primary_idx = np.argmax(vertex_weights)
        primary_skin_idx = int(vertex_joints[primary_idx])
        
        if primary_skin_idx >= len(joint_indices):
            continue
        
        primary_node_idx = skin_to_node[primary_skin_idx]
        
        # Check each joint and zero out distant ones
        modified = False
        for i in range(4):
            if i == primary_idx:
                continue
            
            skin_idx = int(vertex_joints[i])
            if skin_idx >= len(joint_indices):
                continue
            
            node_idx = skin_to_node[skin_idx]
            dist = geodesic_distances.get((primary_node_idx, node_idx), float('inf'))
            
            # If this joint is too far from the primary joint, zero its weight
            if dist > distance_threshold:
                vertex_weights[i] = 0.0
                modified = True
        
        if modified:
            # Renormalize weights
            weight_sum = vertex_weights.sum()
            if weight_sum > 0:
                vertex_weights /= weight_sum
            cleaned_count += 1
    
    print(f"    Cleaned weights for {cleaned_count} edge vertices")
    return new_weights


def find_neighbor_vertices(faces: np.ndarray, target_vertices: Set[int], 
                           triangles_to_remove: Set[int], hops: int = 2) -> Set[int]:
    """
    Find vertices within N hops of target vertices on the remaining mesh.
    
    Args:
        faces: Triangle indices (M, 3)
        target_vertices: Set of vertices to expand from
        triangles_to_remove: Triangles to exclude
        hops: Number of hops to expand
    
    Returns:
        Set of vertices within N hops (including original targets)
    """
    # Build adjacency from remaining triangles
    adjacency = defaultdict(set)
    for face_idx, face in enumerate(faces):
        if face_idx in triangles_to_remove:
            continue
        v0, v1, v2 = int(face[0]), int(face[1]), int(face[2])
        adjacency[v0].update([v1, v2])
        adjacency[v1].update([v0, v2])
        adjacency[v2].update([v0, v1])
    
    current = set(target_vertices)
    all_vertices = set(target_vertices)
    
    for _ in range(hops):
        next_ring = set()
        for v in current:
            next_ring.update(adjacency[v])
        next_ring -= all_vertices
        all_vertices.update(next_ring)
        current = next_ring
    
    return all_vertices


def update_weights_in_buffer(gltf: GLTF2, mesh_index: int,
                              new_weights: np.ndarray,
                              buffer_data: bytes) -> bytes:
    """
    Update the WEIGHTS_0 data in the buffer.
    
    Args:
        gltf: The GLTF2 object
        mesh_index: Index of the mesh
        new_weights: New weights array (N, 4)
        buffer_data: Current buffer data
    
    Returns:
        Updated buffer data
    """
    mesh = gltf.meshes[mesh_index]
    buffer_data = bytearray(buffer_data)
    
    vertex_offset = 0
    
    for primitive in mesh.primitives:
        attrs = primitive.attributes
        
        if not hasattr(attrs, 'WEIGHTS_0') or attrs.WEIGHTS_0 is None:
            # Count vertices for offset
            if hasattr(attrs, 'POSITION') and attrs.POSITION is not None:
                accessor = gltf.accessors[attrs.POSITION]
                vertex_offset += accessor.count
            continue
        
        accessor = gltf.accessors[attrs.WEIGHTS_0]
        buffer_view = gltf.bufferViews[accessor.bufferView]
        
        num_verts = accessor.count
        byte_offset = (buffer_view.byteOffset or 0) + (accessor.byteOffset or 0)
        
        # Get stride (default to 16 bytes for VEC4 float)
        stride = buffer_view.byteStride or 16
        
        # Update weights in buffer
        for i in range(num_verts):
            offset = byte_offset + i * stride
            w = new_weights[vertex_offset + i]
            struct.pack_into('ffff', buffer_data, offset,
                           float(w[0]), float(w[1]), float(w[2]), float(w[3]))
        
        vertex_offset += num_verts
    
    return bytes(buffer_data)


# =============================================================================
# Triangle Filtering
# =============================================================================

def identify_problematic_triangles(faces: np.ndarray,
                                   joints: np.ndarray,
                                   weights: np.ndarray,
                                   joint_indices: List[int],
                                   geodesic_distances: Dict[Tuple[int, int], int],
                                   threshold: int = GEODESIC_DISTANCE_THRESHOLD) -> Tuple[Set[int], Set[int]]:
    """
    Identify triangles that should be removed based on two criteria:
    1. Triangles with vertices bound to far-apart bones (original method)
    2. Triangles containing vertices with weights spread to distant bones (new method)
    
    Args:
        faces: Triangle indices (M, 3)
        joints: Joint indices per vertex (N, 4)
        weights: Joint weights per vertex (N, 4)
        joint_indices: List of valid joint indices
        geodesic_distances: Precomputed geodesic distances
        threshold: Maximum allowed geodesic distance
    
    Returns:
        Tuple of (problematic_triangles, problematic_vertices)
    """
    problematic_triangles = set()
    problematic_vertices = set()
    
    # Create mapping from skin joint index to node index
    skin_to_node = {i: joint_indices[i] for i in range(len(joint_indices))}
    
    num_vertices = len(joints)
    
    # First pass: identify vertices with weights spread to distant bones
    print("  Checking vertices for weight spread to distant bones...")
    for vertex_idx in range(num_vertices):
        if is_vertex_spread_to_distant_bones(joints, weights, vertex_idx, 
                                              joint_indices, geodesic_distances, threshold):
            problematic_vertices.add(vertex_idx)
    
    print(f"    Vertices with spread weights: {len(problematic_vertices)} / {num_vertices} ({100*len(problematic_vertices)/num_vertices:.2f}%)")
    
    # Second pass: identify triangles
    print("  Checking triangles...")
    for face_idx, face in enumerate(faces):
        # Method 1: Check if any vertex has spread weights
        for vertex_idx in face:
            if vertex_idx in problematic_vertices:
                problematic_triangles.add(face_idx)
                break
        
        if face_idx in problematic_triangles:
            continue
        
        # Method 2: Check if vertices are bound to far-apart primary bones
        primary_joints = []
        for vertex_idx in face:
            skin_joint_idx = get_primary_joint(joints, weights, vertex_idx)
            if skin_joint_idx < len(joint_indices):
                node_idx = skin_to_node[skin_joint_idx]
                primary_joints.append(node_idx)
            else:
                primary_joints.append(None)
        
        for i in range(3):
            for j in range(i + 1, 3):
                joint_i = primary_joints[i]
                joint_j = primary_joints[j]
                
                if joint_i is None or joint_j is None:
                    continue
                
                dist = geodesic_distances.get((joint_i, joint_j), float('inf'))
                
                if dist > threshold:
                    problematic_triangles.add(face_idx)
                    break
            
            if face_idx in problematic_triangles:
                break
    
    return problematic_triangles, problematic_vertices


# =============================================================================
# Mesh Reconstruction
# =============================================================================

def rebuild_mesh_without_triangles(gltf: GLTF2, 
                                   mesh_index: int,
                                   triangles_to_remove: Set[int],
                                   buffer_data: bytes) -> bytes:
    """
    Rebuild the mesh without the specified triangles.
    
    This function modifies the GLB in place, updating the indices buffer
    to exclude the removed triangles.
    
    Args:
        gltf: The GLTF2 object to modify
        mesh_index: Index of the mesh to modify
        triangles_to_remove: Set of triangle indices to remove
        buffer_data: Original buffer data
    
    Returns:
        Updated buffer data
    """
    mesh = gltf.meshes[mesh_index]
    buffer_data = bytearray(buffer_data)
    
    face_offset = 0
    
    for primitive in mesh.primitives:
        if primitive.indices is None:
            continue
        
        # Read current indices
        indices = read_accessor_data(gltf, primitive.indices, bytes(buffer_data))
        num_triangles = len(indices) // 3
        
        # Filter out problematic triangles
        new_indices = []
        for tri_idx in range(num_triangles):
            global_tri_idx = face_offset + tri_idx
            if global_tri_idx not in triangles_to_remove:
                start = tri_idx * 3
                new_indices.extend(indices[start:start + 3])
        
        face_offset += num_triangles
        
        if len(new_indices) == len(indices):
            # No triangles removed from this primitive
            continue
        
        new_indices = np.array(new_indices, dtype=np.uint32)
        
        # Get accessor and buffer view info
        accessor = gltf.accessors[primitive.indices]
        buffer_view = gltf.bufferViews[accessor.bufferView]
        
        # Determine the component type and size
        _, component_size, _ = COMPONENT_TYPES[accessor.componentType]
        
        # Pack new indices
        if accessor.componentType == 5123:  # UNSIGNED_SHORT
            new_indices_bytes = new_indices.astype(np.uint16).tobytes()
        elif accessor.componentType == 5125:  # UNSIGNED_INT
            new_indices_bytes = new_indices.astype(np.uint32).tobytes()
        else:
            new_indices_bytes = new_indices.astype(np.uint16).tobytes()
        
        # Calculate offset
        byte_offset = (buffer_view.byteOffset or 0) + (accessor.byteOffset or 0)
        
        # Update the buffer with new indices
        # Note: This assumes the new indices are smaller or equal in size
        old_size = accessor.count * component_size
        new_size = len(new_indices) * component_size
        
        if new_size <= old_size:
            # Write new indices in place
            buffer_data[byte_offset:byte_offset + new_size] = new_indices_bytes
            
            # Update accessor count
            accessor.count = len(new_indices)
            
            # Update buffer view length
            buffer_view.byteLength = new_size
        else:
            # This shouldn't happen since we're removing triangles
            print(f"Warning: New indices larger than old, skipping update")
    
    return bytes(buffer_data)


# =============================================================================
# Main Processing Function
# =============================================================================

def split_mesh(input_path: Path, output_path: Path, 
               threshold: int = GEODESIC_DISTANCE_THRESHOLD) -> None:
    """
    Split mesh by removing triangles that span far-apart bones.
    
    Args:
        input_path: Path to input GLB file
        output_path: Path to output GLB file
        threshold: Maximum allowed geodesic distance between bones
    """
    if type(input_path) == str:
        input_path = Path(input_path)
    if type(output_path) == str:
        output_path = Path(output_path)
    
    print(f"Loading: {input_path}")
    
    # Load GLB
    gltf = GLTF2().load(str(input_path))
    buffer_data = get_buffer_data(gltf)
    
    # Read mesh data
    print("Reading mesh data...")
    vertices, faces = read_mesh_data(gltf, 0, buffer_data)
    print(f"  Vertices: {len(vertices)}")
    print(f"  Triangles: {len(faces)}")
    
    # Read skinning data
    print("Reading skinning data...")
    joints, weights = read_skinning_data(gltf, 0, buffer_data)
    
    if joints is None or weights is None:
        print("ERROR: No skinning data found in mesh")
        return
    
    print(f"  Skinning data: {joints.shape}")
    
    # Build skeleton graph
    print("Building skeleton graph...")
    skeleton_graph, joint_indices = build_skeleton_graph(gltf)
    print(f"  Joints: {len(joint_indices)}")
    print(f"  Bones (edges): {skeleton_graph.number_of_edges()}")
    
    # Print skeleton structure
    print("  Skeleton hierarchy:")
    for joint_idx in joint_indices:
        node = gltf.nodes[joint_idx]
        name = node.name or f"Joint_{joint_idx}"
        children = node.children or []
        child_names = [gltf.nodes[c].name or f"Joint_{c}" for c in children if c in joint_indices]
        if child_names:
            print(f"    {name} -> {', '.join(child_names)}")
    
    # Compute geodesic distances
    print("Computing geodesic distances...")
    geodesic_distances = compute_geodesic_distances(skeleton_graph, joint_indices)
    
    # Identify problematic triangles
    print(f"Identifying problematic triangles (threshold={threshold})...")
    problematic, problematic_verts = identify_problematic_triangles(
        faces, joints, weights, joint_indices, geodesic_distances, threshold
    )
    print(f"  Problematic triangles: {len(problematic)} / {len(faces)} ({100*len(problematic)/len(faces):.2f}%)")
    
    # Debug: show some examples of problematic vertices
    if problematic_verts:
        print(f"\n  Sample problematic vertices (showing up to 5):")
        skin_to_node = {i: joint_indices[i] for i in range(len(joint_indices))}
        for idx, vert_idx in enumerate(list(problematic_verts)[:5]):
            vert_joints = joints[vert_idx]
            vert_weights = weights[vert_idx]
            joint_info = []
            for i in range(4):
                if vert_weights[i] > 0.01:
                    skin_idx = int(vert_joints[i])
                    if skin_idx < len(joint_indices):
                        node_idx = skin_to_node[skin_idx]
                        name = gltf.nodes[node_idx].name or f"Joint_{node_idx}"
                        joint_info.append(f"{name}:{vert_weights[i]:.2f}")
            print(f"    Vertex {vert_idx}: {', '.join(joint_info)}")
    
    if len(problematic) == 0:
        print("No problematic triangles found, saving unchanged mesh...")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        gltf.save(str(output_path))
        print(f"Saved to: {output_path}")
        return
    
    # Find edge vertices (on the boundary of removed regions)
    print("Finding edge vertices...")
    edge_vertices, removed_only = find_edge_vertices(faces, problematic)
    print(f"  Edge vertices: {len(edge_vertices)}")
    print(f"  Orphaned vertices (in removed triangles only): {len(removed_only)}")
    
    # Expand to include neighbors for smoother transition
    print("Expanding to neighbor vertices...")
    vertices_to_clean = find_neighbor_vertices(faces, edge_vertices, problematic, hops=2)
    print(f"  Vertices to clean (including 2-hop neighbors): {len(vertices_to_clean)}")
    
    # Clean edge vertex weights
    print("Cleaning edge vertex weights...")
    new_weights = clean_edge_vertex_weights(
        joints, weights, vertices_to_clean, joint_indices, geodesic_distances, threshold
    )
    
    # Update weights in buffer
    print("Updating weights in buffer...")
    buffer_data = update_weights_in_buffer(gltf, 0, new_weights, buffer_data)
    
    # Rebuild mesh without problematic triangles
    print("Rebuilding mesh...")
    new_buffer_data = rebuild_mesh_without_triangles(gltf, 0, problematic, buffer_data)
    
    # Update buffer
    set_binary_blob(gltf, new_buffer_data)
    
    # Save result
    output_path.parent.mkdir(parents=True, exist_ok=True)
    gltf.save(str(output_path))
    print(f"Saved to: {output_path}")
    
    # Summary
    remaining = len(faces) - len(problematic)
    print(f"\nSummary:")
    print(f"  Original triangles: {len(faces)}")
    print(f"  Removed triangles: {len(problematic)}")
    print(f"  Remaining triangles: {remaining}")
