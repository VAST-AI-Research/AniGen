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

    target_faces = 10000
    
    mesh = cumesh.CuMesh()
    
    mesh.init(vertices, faces)
    mesh.fill_holes(max_hole_perimeter=0.1)
    new_vertices, new_faces = mesh.read()

    print(f"Hole filled mesh: {new_vertices.shape[0]} vertices, {new_faces.shape[0]} faces")

    new_mesh = trimesh.Trimesh(vertices=new_vertices.cpu().numpy(), faces=new_faces.cpu().numpy(), process=False)
    new_mesh.export("hole_filled.ply")
