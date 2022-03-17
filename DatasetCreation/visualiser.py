import trimesh
import os
import numpy as np
from data_loader import PileLoader, VoxelGridLoader
import skimage.measure as measure
import argparse

def tsdf_to_mesh(tsdf):
    vertices, faces, vertex_normals, _ = \
        measure.marching_cubes_lewiner(tsdf, level=0, gradient_direction='descent')
    mesh = trimesh.Trimesh(
        vertices=vertices, vertex_normals=vertex_normals, faces=faces
    )

    return mesh

def add_floor(scene, mesh):
    """
    TODO: fix offset in visualization
    :param plane_shape:
    :param scene:
    :return:
    """
    min_z = np.min(mesh.vertices,axis=0)[2]
    extents = mesh.extents
    center = np.median(mesh.vertices, axis=0)

    # print('floor added at: ', min_z)

    plane_depth = 0.01
    plane = trimesh.creation.box([extents[0], extents[1], plane_depth])
    plane.apply_translation([
        center[0],
        center[1],
        (plane_depth / 2.) + min_z,
    ])
    plane.visual.face_colors = [[1., 1., 1.]] * len(plane.faces)
    scene.add_geometry(plane)


def show_original_PyBullet_pile():
    loader = PileLoader(path_to_scenes=path_to_scenes, path_to_models=path_to_models, type="train")
    sample = loader[0]
    original_scene = trimesh.Scene()
    for mesh, transform in zip(sample['meshes'], sample['transforms']):
        original_scene.add_geometry(mesh, transform=transform)

    original_scene = original_scene.scaled(64)
    original_scene.show()

def show_random_example(path_to_scenes):

    with open(os.path.join(path_to_scenes, "max_n_objects.txt")) as fp:
        max_n_objects = int(fp.readline())

    v_loader = VoxelGridLoader(path_to_scenes=path_to_scenes, max_n_objects=max_n_objects)
    sample = v_loader[np.random.randint(0,len(v_loader))]

    if 'nbr_of_objects' in sample:
        nbr_of_objects = sample['nbr_of_objects']
        combined_scene = trimesh.Scene()
        concatenated_objects = sample["concatenated_objects"]

        for el in range(nbr_of_objects):
            tsdf = concatenated_objects[el, :]
            mesh = tsdf_to_mesh(tsdf.numpy())
            mesh.visual.face_colors = list(np.random.uniform(0, 255, (3,))) + [255]
            combined_scene.add_geometry(mesh)

        add_floor(combined_scene, combined_scene.dump(concatenate=True))
        combined_scene.add_geometry(trimesh.creation.axis(origin_size=2))
        combined_scene.show()

    else:
        full_scene = sample["scene"]
        full_scene_mesh = tsdf_to_mesh(full_scene[0].numpy())
        gt_scene = trimesh.Scene()
        gt_scene.add_geometry(full_scene_mesh)
        add_floor(gt_scene, full_scene_mesh)
        gt_scene.add_geometry(trimesh.creation.axis(origin_size=2))
        gt_scene.show()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument('--scene_dir', help='destination path to dataset', default="../examples/sample_data/")

    args = parser.parse_args()

    show_random_example(args.scene_dir)
