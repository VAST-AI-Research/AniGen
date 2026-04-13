import io
import numpy as np
import torch
import trimesh
from PIL import Image as PILImage
from pygltflib import (
    GLTF2, Scene, Node, Mesh, Primitive, Attributes, Buffer, BufferView, Accessor, Asset, Skin,
    Material, PbrMetallicRoughness, TextureInfo,
    Image as GltfImage, Texture as GltfTexture, Sampler as GltfSampler,
    FLOAT, UNSIGNED_SHORT, UNSIGNED_INT, VEC3, VEC2, VEC4, SCALAR, MAT4, ARRAY_BUFFER, ELEMENT_ARRAY_BUFFER
)
from anigen.utils.skin_utils import repair_skeleton_parents

_ROT_Z_UP_TO_Y_UP = np.array([[-1, 0, 0], [0, 0, 1], [0, 1, 0]], dtype=np.float32)

def visualize_skeleton_as_mesh(joints, parents, joint_radius_ratio=0.02, bone_radius_ratio=0.01):
    """
    Convert a skeleton into a trimesh object with spheres for joints and cones for bones.
    The sizes are adaptive to the skeleton's bounding box.
    """
    if len(joints) == 0:
        return trimesh.Trimesh()

    joints = np.asarray(joints, dtype=np.float32) @ _ROT_Z_UP_TO_Y_UP

    # Calculate adaptive scale
    min_bound = np.min(joints, axis=0)
    max_bound = np.max(joints, axis=0)
    scale = np.linalg.norm(max_bound - min_bound)
    if scale < 1e-5:
        scale = 1.0

    joint_radius = scale * joint_radius_ratio
    bone_radius = scale * bone_radius_ratio

    meshes = []

    # Create joints
    for i, joint in enumerate(joints):
        sphere = trimesh.creation.icosphere(radius=joint_radius, subdivisions=2)
        sphere.apply_translation(joint)
        # Optional: add vertex colors for joints
        sphere.visual.vertex_colors = [230, 180, 80, 255]
        meshes.append(sphere)

    # Create bones
    for i, parent_idx in enumerate(parents):
        if parent_idx < 0 or parent_idx == i:
            continue
        
        child = joints[i]
        parent = joints[parent_idx]
        
        vec = child - parent
        length = np.linalg.norm(vec)
        if length < 1e-5:
            continue
            
        # Create a cone pointing along Z axis by default
        cone = trimesh.creation.cone(radius=bone_radius, height=length)
        
        # Align the cone's Z axis to the vector
        z_axis = np.array([0, 0, 1])
        direction = vec / length
        
        # Calculate rotation matrix from Z axis to direction
        rot_mat = trimesh.geometry.align_vectors(z_axis, direction)
        
        # Apply rotation
        cone.apply_transform(rot_mat)
        
        # Translate to parent
        cone.apply_translation(parent)
        
        # Optional: add vertex colors for bones
        cone.visual.vertex_colors = [160, 180, 200, 255]
        meshes.append(cone)

    if not meshes:
        return trimesh.Trimesh()

    # Combine all meshes
    skeleton_mesh = trimesh.util.concatenate(meshes)
    return skeleton_mesh

def convert_to_glb_from_data(mesh, joints, parents, skin_weights, output_path, vertex_colors=None, texture_image=None):
    joints = np.asarray(joints, dtype=np.float32) @ _ROT_Z_UP_TO_Y_UP
    num_joints = joints.shape[0]
    num_verts = mesh.vertices.shape[0]

    # Basic sanity checks to avoid writing malformed GLB that can hang Blender.
    if hasattr(mesh, 'faces'):
        faces = np.asarray(mesh.faces)
        if faces.size > 0:
            if faces.min() < 0 or faces.max() >= num_verts:
                raise ValueError(f"Mesh faces contain out-of-range indices (min={faces.min()}, max={faces.max()}, num_verts={num_verts}).")

    for name, arr in (
        ("mesh.vertices", getattr(mesh, 'vertices', None)),
        ("joints", joints),
        ("skin_weights", skin_weights),
    ):
        if arr is None:
            continue
        arr_np = np.asarray(arr)
        if arr_np.size and not np.isfinite(arr_np).all():
            raise ValueError(f"Non-finite values detected in {name}; refusing to write GLB.")

    parents = np.asarray(parents).astype(np.int64, copy=False)
    if parents.shape != (num_joints,):
        parents = parents.reshape(-1)
    if parents.shape[0] != num_joints:
        raise ValueError(f"parents has wrong shape: expected ({num_joints},) got {parents.shape}")

    # Repair invalid/cyclic skeletons instead of exporting a GLB that hangs viewers.
    parents = repair_skeleton_parents(joints=joints, parents=parents, verbose=True)
    
    if skin_weights.shape[0] != num_verts:
        print(f"Warning: Mismatch in vertex count. Mesh: {num_verts}, Skin: {skin_weights.shape[0]}")
        return

    # Prepare Binary Data
    binary_data = bytearray()
    
    def align_to_4bytes():
        padding = len(binary_data) % 4
        if padding > 0:
            binary_data.extend(b'\x00' * (4 - padding))

    accessors = []
    buffer_views = []

    def add_buffer_view(data_bytes, target=None):
        align_to_4bytes()
        byte_offset = len(binary_data)
        binary_data.extend(data_bytes)
        byte_length = len(data_bytes)
        bv = BufferView(buffer=0, byteOffset=byte_offset, byteLength=byte_length, target=target)
        buffer_views.append(bv)
        return len(buffer_views) - 1

    def add_accessor(buffer_view_idx, component_type, count, type_str, min_val=None, max_val=None):
        acc = Accessor(
            bufferView=buffer_view_idx,
            componentType=component_type,
            count=count,
            type=type_str,
            min=min_val,
            max=max_val
        )
        accessors.append(acc)
        return len(accessors) - 1

    # 1. Indices
    if hasattr(mesh, 'faces'):
        indices = mesh.faces.flatten().astype(np.uint32)
        if indices.max() <= 65535:
            indices = indices.astype(np.uint16)
            component_type = UNSIGNED_SHORT
        else:
            component_type = UNSIGNED_INT
        
        indices_bytes = indices.tobytes()
        bv_idx = add_buffer_view(indices_bytes, target=ELEMENT_ARRAY_BUFFER)
        indices_acc_idx = add_accessor(bv_idx, component_type, len(indices), SCALAR, min_val=[int(indices.min())], max_val=[int(indices.max())])
    
    # 2. Positions
    positions = mesh.vertices.astype(np.float32) @ _ROT_Z_UP_TO_Y_UP
    # GLTF requires min/max for POSITION
    min_pos = positions.min(axis=0).tolist()
    max_pos = positions.max(axis=0).tolist()
    
    pos_bytes = positions.tobytes()
    bv_idx = add_buffer_view(pos_bytes, target=ARRAY_BUFFER)
    pos_acc_idx = add_accessor(bv_idx, FLOAT, len(positions), VEC3, min_val=min_pos, max_val=max_pos)

    # 3. Normals
    norm_acc_idx = None
    if hasattr(mesh, 'vertex_normals'):
        normals = mesh.vertex_normals.astype(np.float32) @ _ROT_Z_UP_TO_Y_UP
        norm_bytes = normals.tobytes()
        bv_idx = add_buffer_view(norm_bytes, target=ARRAY_BUFFER)
        norm_acc_idx = add_accessor(bv_idx, FLOAT, len(normals), VEC3)

    # 4. UVs
    tex_acc_idx = None
    if hasattr(mesh.visual, 'uv') and mesh.visual.uv is not None:
        uvs = mesh.visual.uv.astype(np.float32)
        uvs[:, 1] = 1.0 - uvs[:, 1]  # OpenGL -> glTF: flip V
        uv_bytes = uvs.tobytes()
        bv_idx = add_buffer_view(uv_bytes, target=ARRAY_BUFFER)
        tex_acc_idx = add_accessor(bv_idx, FLOAT, len(uvs), VEC2)

    # 4.5 Vertex Colors (skipped when a texture image is provided, since glTF
    #     multiplies vertex colors with the texture which is not desired here)
    color_acc_idx = None
    if vertex_colors is not None and texture_image is None:
        colors = np.asarray(vertex_colors)
        if colors.shape[0] != num_verts:
            print(f"Warning: Mismatch in vertex color count. Mesh: {num_verts}, Colors: {colors.shape[0]}")
        else:
            if colors.ndim != 2 or colors.shape[1] not in (3, 4):
                print(f"Warning: Unexpected vertex_colors shape: {colors.shape}. Expected (N,3) or (N,4).")
            else:
                colors = colors.astype(np.float32, copy=False)
                # Expect colors in [0,1]; clamp to be safe.
                np.clip(colors, 0.0, 1.0, out=colors)
                color_bytes = colors.tobytes()
                bv_idx = add_buffer_view(color_bytes, target=ARRAY_BUFFER)
                color_acc_idx = add_accessor(bv_idx, FLOAT, len(colors), VEC4 if colors.shape[1] == 4 else VEC3)

    # 4.6 Texture Image (embedded as PNG in the binary blob)
    gltf_images = []
    gltf_textures = []
    gltf_samplers = []
    if texture_image is not None:
        tex_arr = np.asarray(texture_image)
        img_pil = PILImage.fromarray(tex_arr)
        buf = io.BytesIO()
        img_pil.save(buf, format='PNG')
        png_bytes = buf.getvalue()

        align_to_4bytes()
        img_bv_offset = len(binary_data)
        binary_data.extend(png_bytes)
        img_bv_len = len(png_bytes)
        img_bv = BufferView(buffer=0, byteOffset=img_bv_offset, byteLength=img_bv_len)
        buffer_views.append(img_bv)
        img_bv_idx = len(buffer_views) - 1

        gltf_images.append(GltfImage(bufferView=img_bv_idx, mimeType='image/png'))
        gltf_samplers.append(GltfSampler(magFilter=9729, minFilter=9987, wrapS=10497, wrapT=10497))
        gltf_textures.append(GltfTexture(source=0, sampler=0))

    # 5. Joints & Weights
    if num_joints >= 4:
        top_indices = np.argsort(skin_weights, axis=1)[:, -4:] 
        top_indices = np.flip(top_indices, axis=1)
        top_weights = np.take_along_axis(skin_weights, top_indices, axis=1)
    else:
        top_indices = np.argsort(skin_weights, axis=1)
        top_indices = np.flip(top_indices, axis=1)
        top_weights = np.take_along_axis(skin_weights, top_indices, axis=1)
        
        pad_width = 4 - num_joints
        top_indices = np.pad(top_indices, ((0,0), (0, pad_width)), mode='constant', constant_values=0)
        top_weights = np.pad(top_weights, ((0,0), (0, pad_width)), mode='constant', constant_values=0.0)

    weight_sums = np.sum(top_weights, axis=1, keepdims=True)
    weight_sums[weight_sums == 0] = 1.0
    top_weights = top_weights / weight_sums

    joints_0 = top_indices.astype(np.uint16)
    weights_0 = top_weights.astype(np.float32)

    joints_bytes = joints_0.tobytes()
    bv_idx = add_buffer_view(joints_bytes, target=ARRAY_BUFFER)
    joints_acc_idx = add_accessor(bv_idx, UNSIGNED_SHORT, len(joints_0), VEC4)

    weights_bytes = weights_0.tobytes()
    bv_idx = add_buffer_view(weights_bytes, target=ARRAY_BUFFER)
    weights_acc_idx = add_accessor(bv_idx, FLOAT, len(weights_0), VEC4)

    # 6. Inverse Bind Matrices
    ibms_list = []
    for i in range(num_joints):
        mat = np.eye(4, dtype=np.float32)
        mat[:3, 3] = joints[i]
        inv_mat = np.linalg.inv(mat)
        ibms_list.append(inv_mat.flatten('F'))
    
    ibms_data = np.concatenate(ibms_list)
    ibms_bytes = ibms_data.tobytes()
    bv_idx = add_buffer_view(ibms_bytes)
    ibms_acc_idx = add_accessor(bv_idx, FLOAT, num_joints, MAT4)

    # Nodes
    nodes = []
    
    # Mesh Node (Node 0)
    mesh_node = Node(name="Mesh", mesh=0, skin=0)
    nodes.append(mesh_node)

    # Skeleton Nodes (Node 1 to M)
    joint_nodes = []
    for i in range(num_joints):
        parent_idx = parents[i]
        pos = joints[i]
        
        if parent_idx == -1 or parent_idx == i:
            local_pos = pos
        else:
            parent_pos = joints[parent_idx]
            local_pos = pos - parent_pos
            
        node = Node(name=f"joint_{i}", translation=local_pos.tolist())
        joint_nodes.append(node)
    
    # Set children for skeleton nodes
    for i in range(num_joints):
        parent_idx = parents[i]
        if parent_idx != -1 and parent_idx != i:
            parent_node = joint_nodes[parent_idx]
            if parent_node.children is None:
                parent_node.children = []
            parent_node.children.append(i + 1)
            
    nodes.extend(joint_nodes)

    # Find root joint index
    root_joint_indices = np.where(parents == -1)[0]
    if len(root_joint_indices) == 0:
        root_joint_indices = np.where(parents == np.arange(num_joints))[0]

    if len(root_joint_indices) == 0:
        print("Error: No root joint found.")
        return

    if len(root_joint_indices) > 1:
        print(f"Warning: Found {len(root_joint_indices)} root joints. Creating a virtual root.")
        virtual_root_node_idx = len(nodes)
        virtual_root_node = Node(name="VirtualRoot", children=[int(i) + 1 for i in root_joint_indices])
        nodes.append(virtual_root_node)
        root_node_idx = virtual_root_node_idx
    else:
        root_joint_idx = root_joint_indices[0]
        root_node_idx = int(root_joint_idx) + 1

    # Skin
    skin = Skin(
        inverseBindMatrices=ibms_acc_idx,
        joints=[i + 1 for i in range(num_joints)],
        skeleton=root_node_idx
    )

    # Mesh
    attributes_dict = {
        "POSITION": pos_acc_idx,
        "JOINTS_0": joints_acc_idx,
        "WEIGHTS_0": weights_acc_idx
    }
    if norm_acc_idx is not None:
        attributes_dict["NORMAL"] = norm_acc_idx
    if tex_acc_idx is not None:
        attributes_dict["TEXCOORD_0"] = tex_acc_idx
    if color_acc_idx is not None:
        attributes_dict["COLOR_0"] = color_acc_idx

    primitive = Primitive(
        attributes=Attributes(**attributes_dict),
        indices=indices_acc_idx,
        material=0,
    )
    mesh_obj = Mesh(primitives=[primitive])

    # Scene
    scene = Scene(nodes=[0, root_node_idx])

    # Ensure GLB BIN chunk is 4-byte aligned.
    align_to_4bytes()

    # Buffer
    buffer = Buffer(byteLength=len(binary_data))
    
    # Material — attach texture when available, otherwise plain white
    if gltf_textures:
        pbr = PbrMetallicRoughness(
            baseColorTexture=TextureInfo(index=0),
            metallicFactor=0.0,
            roughnessFactor=1.0,
        )
    else:
        pbr = PbrMetallicRoughness(
            baseColorFactor=[1.0, 1.0, 1.0, 1.0],
            metallicFactor=0.0,
            roughnessFactor=1.0,
        )
    material = Material(pbrMetallicRoughness=pbr)

    gltf = GLTF2(
        asset=Asset(version="2.0"),
        scene=0,
        scenes=[scene],
        nodes=nodes,
        meshes=[mesh_obj],
        materials=[material],
        skins=[skin],
        accessors=accessors,
        bufferViews=buffer_views,
        buffers=[buffer],
        images=gltf_images if gltf_images else None,
        textures=gltf_textures if gltf_textures else None,
        samplers=gltf_samplers if gltf_samplers else None,
    )
    
    gltf.set_binary_blob(bytes(binary_data))

    # IMPORTANT: `save()` may write JSON `.gltf` even if the filename ends with `.glb`.
    # Many viewers (Meshlab/Open3D) will then report “glb not supported”.
    if str(output_path).lower().endswith('.glb'):
        gltf.save_binary(output_path)
    else:
        gltf.save(output_path)

    # Quick sanity check: a real GLB starts with magic bytes b'glTF'.
    if str(output_path).lower().endswith('.glb'):
        try:
            with open(output_path, 'rb') as f:
                magic = f.read(4)
            if magic != b'glTF':
                print(f"Warning: output does not look like a valid GLB (magic={magic!r}).")
        except Exception as e:
            print(f"Warning: failed to validate GLB header: {e}")
    print(f"Saved GLB to {output_path}")

def save_colored_pcl(array, filename):
    if isinstance(array, torch.Tensor):
        array = array.detach().cpu().numpy()
    points = array[:, :3]
    if array.shape[1] >= 6:
        colors = (array[:, 3:6] * 255).astype(np.uint8)
    else:
        colors = np.full((points.shape[0], 3), 255, dtype=np.uint8)
        
    # Combine points and colors into a single array
    data = np.hstack((points, colors))
    # Define the .ply header
    header = f"""ply\nformat ascii 1.0\nelement vertex {points.shape[0]}\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"""
    # Write to the .ply file
    with open(filename, 'w') as file:
        file.write(header)
        np.savetxt(file, data, fmt='%f %f %f %d %d %d')

def _extract_vertex_rgb(vertex_attrs):
    """Extract Nx3 RGB in [0,1] from vertex_attrs (torch/np), or return None."""
    if vertex_attrs is None:
        return None
    attrs = vertex_attrs
    if isinstance(attrs, torch.Tensor):
        attrs = attrs.detach().cpu().numpy()
    attrs = np.asarray(attrs)
    if attrs.ndim != 2 or attrs.shape[1] < 3:
        return None
    colors = attrs[:, :3].astype(np.float32, copy=False)
    np.clip(colors, 0.0, 1.0, out=colors)
    return colors

def transfer_vertex_colors_nearest(src_vertices, src_colors, dst_vertices):
    """Transfer per-vertex colors by nearest-vertex mapping.

    Args:
        src_vertices: (Ns, 3) float
        src_colors:   (Ns, 3) float in [0,1]
        dst_vertices: (Nd, 3) float
    Returns:
        (Nd, 3) float in [0,1]
    """
    if src_vertices is None or src_colors is None or dst_vertices is None:
        return None
    src_vertices = np.asarray(src_vertices)
    src_colors = np.asarray(src_colors)
    dst_vertices = np.asarray(dst_vertices)
    if src_vertices.ndim != 2 or src_vertices.shape[1] != 3:
        return None
    if dst_vertices.ndim != 2 or dst_vertices.shape[1] != 3:
        return None
    if src_colors.ndim != 2 or src_colors.shape[1] != 3 or src_colors.shape[0] != src_vertices.shape[0]:
        return None

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    src_v = torch.from_numpy(src_vertices).to(device=device, dtype=torch.float32)
    dst_v = torch.from_numpy(dst_vertices).to(device=device, dtype=torch.float32)
    src_c = torch.from_numpy(src_colors).to(device=device, dtype=torch.float32)

    # Prefer pytorch3d for KNN if available.
    idx = None
    try:
        from pytorch3d.ops import knn_points
        _, nn_idx, _ = knn_points(dst_v[None], src_v[None], K=1, return_nn=False)
        idx = nn_idx[0, :, 0]
    except Exception:
        # Fallback: brute-force cdist.
        dist = torch.cdist(dst_v, src_v)  # (Nd, Ns)
        idx = torch.argmin(dist, dim=1)

    out = src_c[idx].clamp_(0.0, 1.0)
    return out.detach().cpu().numpy()
