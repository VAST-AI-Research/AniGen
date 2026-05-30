import torch
import trimesh
import cumesh
import utils


if __name__ == "__main__":
    mesh = utils.get_bunny()
    mesh.export("original.ply")
    vertices = torch.from_numpy(mesh.vertices).float()
    faces = torch.from_numpy(mesh.faces).int()
    print(f"Original mesh: {vertices.shape[0]} vertices, {faces.shape[0]} faces")

    vertices = vertices.cuda()
    faces = faces.cuda()
    
    aabb_max = vertices.max(dim=0)[0]
    aabb_min = vertices.min(dim=0)[0]
    center = (aabb_max + aabb_min) / 2
    scale = (aabb_max - aabb_min).max().item()
    print(f"Center: {center}, Scale: {scale}")

    new_vertices, new_faces = cumesh.remeshing.remesh_narrow_band_dc(
        vertices, faces,
        center = center,
        scale = 1.05 * scale,
        resolution = 128,
        band = 1,
        project_back = 0.9,
        verbose = True
    )

    print(f"Remeshed mesh: {new_vertices.shape[0]} vertices, {new_faces.shape[0]} faces")

    new_mesh = trimesh.Trimesh(vertices=new_vertices.cpu().numpy(), faces=new_faces.cpu().numpy(), process=False)
    new_mesh.export("remeshed.ply")
