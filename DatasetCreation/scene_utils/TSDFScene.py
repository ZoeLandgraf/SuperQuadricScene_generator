import numpy as np
import trimesh
import copy

from trimesh.proximity import closest_point
from trimesh.constants import tol
import trimesh.transformations as ttf
import mesh_to_sdf
import skimage.measure as measure

def tsdf_to_mesh(tsdf):
    vertices, faces, vertex_normals, _ = \
        measure.marching_cubes_lewiner(tsdf, level=0, gradient_direction='descent')
    mesh = trimesh.Trimesh(
        vertices=vertices, vertex_normals=vertex_normals, faces=faces
    )

    return mesh


class TSDFScene:

    @classmethod
    def mesh_to_voxels_no_rescaling(cls, mesh, voxel_resolution=64, surface_point_method='scan', sign_method='normal', scan_count=100,
                       scan_resolution=400, sample_point_count=10000000, normal_sample_count=11, pad=False,
                       check_result=False,
                       scene_center = None):
        """
        Creates the voxel sdf grid from a mesh using mesh_to_sdf library
        :param mesh:
        :param voxel_resolution:
        :param surface_point_method:
        :param sign_method:
        :param scan_count:
        :param scan_resolution:
        :param sample_point_count:
        :param normal_sample_count:
        :param pad:
        :param check_result:
        :return:
        """
        # center the mesh
        if scene_center is None:
            center = mesh.bounding_box.centroid
        else:
            center = scene_center

        vertices = mesh.vertices - center

        mesh = trimesh.Trimesh(vertices=vertices, faces=mesh.faces)

        surface_point_cloud = mesh_to_sdf.get_surface_point_cloud(mesh, surface_point_method, 3 ** 0.5, scan_count, scan_resolution,
                                                      sample_point_count, sign_method == 'normal')

        return surface_point_cloud.get_voxels(voxel_resolution, sign_method == 'depth', normal_sample_count, pad,
                                              check_result)

    @classmethod
    def generate_sdf_with_library(cls,
                                  scene,
                                  fixed_floor=None,
                                  scene_center=None,
                                  move_pixel_rows=None,
                                  voxel_res = 64):
        """
        Generates an SDF grid from a joint mesh directly
        The original mesh is scaled by 64! (This is dependant on the voxel resolution)!
        :param scene:
        :return:
        """

        if isinstance(scene, trimesh.Scene):
            joint_mesh = scene.dump(concatenate=True)
        else:
            joint_mesh = scene

        joint_mesh = joint_mesh.apply_scale(2)

        sdf_grid = TSDFScene.mesh_to_voxels_no_rescaling(joint_mesh, voxel_resolution=voxel_res, surface_point_method='scan',
                                                    sign_method='depth',
                                                    scan_count=15, scan_resolution=600, sample_point_count=10000,
                                                    pad=False, check_result=False, scene_center=scene_center)

        sdf_grid = sdf_grid / 2
        if fixed_floor is not None:
            sdf_grid = TSDFScene.move_to_fixed_floor(sdf_grid, fixed_floor)
        elif move_pixel_rows is not None:
            sdf_grid = TSDFScene.move_by_fixed_amount(sdf_grid, move_pixel_rows)
        return sdf_grid



    @classmethod
    def move_by_fixed_amount(cls, sdf_grid, amount):
        """

        """

        remove_rows = amount
        new_sdf = np.ones(sdf_grid.shape) * 10
        if remove_rows > 0:
            max_val = sdf_grid[:, :, remove_rows:].max()
            new_sdf[:, :, 0:sdf_grid.shape[2] - remove_rows] = sdf_grid[:, :, remove_rows:]
            new_sdf[new_sdf == 10] = max_val
        elif remove_rows < 0:
            max_val = sdf_grid[:, :, 0:sdf_grid.shape[2] - abs(remove_rows)].max()
            new_sdf[:, :, abs(remove_rows):] = sdf_grid[:, :, 0:sdf_grid.shape[2] - abs(remove_rows)]
            new_sdf[new_sdf == 10] = max_val

            # assert scene doesn't grow out of bounds
            last_row = new_sdf[:, :, -1]
            if (last_row < 0).any():
                raise ValueError("Scene out of bounds!!!")
        else:
            new_sdf = sdf_grid

        return new_sdf


    @classmethod
    def move_to_fixed_floor(cls, sdf_grid, floor_location=20):
        """
        Moves the scene to a fixed floor location accuracte to 1 voxel.
        It does so by moving the first negative voxel value (the first inside value) to z location floor_location + 1
        :param sdf_grid:
        :return:
        """

        current_floor = 0
        for row in range(sdf_grid.shape[2]):
            _2d_p = sdf_grid[:,:,row]
            if (_2d_p < 0).any():
                current_floor = row
                break

        if current_floor > (sdf_grid.shape[2] / 2):
            raise ValueError("scene start above center")

        remove_rows = current_floor - floor_location


        new_sdf = np.ones(sdf_grid.shape) * 10
        if remove_rows > 0:
            max_val = sdf_grid[:,:,remove_rows:].max()
            new_sdf[:,:,0:sdf_grid.shape[2] - remove_rows] = sdf_grid[:,:,remove_rows:]
            new_sdf[new_sdf == 10] = max_val
        elif remove_rows < 0:
            max_val = sdf_grid[:,:,0:sdf_grid.shape[2]-abs(remove_rows)].max()
            new_sdf[:,:,abs(remove_rows):] = sdf_grid[:,:,0:sdf_grid.shape[2]-abs(remove_rows)]
            new_sdf[new_sdf == 10] = max_val

            #assert scene doesn't grow out of bounds
            last_row = new_sdf[:,:,-1]
            if (last_row < 0).any():
                raise ValueError("Scene out of bounds!!!")
        else:
            new_sdf = sdf_grid

        return new_sdf



def create_tsdf_per_object(scene_dict):
    """
    Generates a TSDF volume for every object in the scene
    """

    scene = scene_dict['scene']

    joint_mesh = scene.dump(concatenate=True)
    scene_center = joint_mesh.bounding_box.centroid

    per_object_tsdfs = []

    lowest_z = 64 # number which is always larger than possible position
    for mesh_t, scale_t, transform_t in zip(scene_dict['meshes'].items(), scene_dict['mesh_scales'].items(),
                                            scene_dict['transforms'].items()):
        ind_scene = trimesh.Scene()
        key, mesh = mesh_t
        mesh = mesh.apply_transform(transform_t[1])

        ind_scene.add_geometry(mesh)

        ind_tsdf = TSDFScene.generate_sdf_with_library(ind_scene,
                                                       fixed_floor=None,
                                                       scene_center=scene_center,
                                                       move_pixel_rows=None)

        ind_mesh = tsdf_to_mesh(ind_tsdf)
        min_mesh_z = ind_mesh.vertices.min(axis=0)[2]
        if min_mesh_z < lowest_z:
            lowest_z = min_mesh_z

        per_object_tsdfs.append(ind_tsdf)

    moved_per_object_tsdfs = []
    for ind_tsdf in per_object_tsdfs:
        moved_ind_tsdf = TSDFScene.move_by_fixed_amount(ind_tsdf, round(lowest_z - 15.0))
        moved_per_object_tsdfs.append(moved_ind_tsdf)
    per_object_tsdfs = moved_per_object_tsdfs

    return per_object_tsdfs