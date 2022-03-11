import numpy as np
import trimesh


class ExtendedScene(trimesh.Scene):

    def __init__(self,
                 geometry=None,
                 base_frame='world',
                 metadata={},
                 graph=None,
                 camera=None,
                 lights=None,
                 camera_transform=None
                 ):
        super().__init__(geometry,
                 base_frame,
                 metadata,
                 graph,
                 camera,
                 lights,
                 camera_transform)

    @classmethod
    def points_to_indices(cls,
                          points,
                          pitch=None,
                          origin=None):
        """
        Convert center points of an (n,m,p) matrix into its indices.
        Parameters
        ----------
        points : (q, 3) float
          Center points of voxel matrix (n,m,p)
        pitch : float
          What pitch was the voxel matrix computed with
        origin : (3,) float
          What is the origin of the voxel matrix
        Returns
        ----------
        indices : (q, 3) int
          List of indices
        """

        points = np.array(points, dtype=np.float64)
        if points.shape != (points.shape[0], 3):
            raise ValueError('shape of points must be (q, 3)')

        if origin is not None:
            origin = np.asanyarray(origin)
            if origin.shape != (3,):
                raise ValueError('shape of origin must be (3,)')
            points -= origin

        if pitch is not None:
            points /= pitch

        origin = np.asanyarray(origin, dtype=np.float64)
        pitch = float(pitch)

        indices = np.round(points).astype(int)

        return indices

    @classmethod
    def points_to_indices_keep_instances(cls,
                                         points,
                                         points_to_instances,
                                         pitch=None,
                                         origin=None):
        """
        Convert center points of an (n,m,p) matrix into its indices.
        Parameters
        ----------
        points : (q, 3) float
          Center points of voxel matrix (n,m,p)
        pitch : float
          What pitch was the voxel matrix computed with
        origin : (3,) float
          What is the origin of the voxel matrix
        Returns
        ----------
        indices : (q, 3) int
          List of indices
        """
        def subtract_origin_and_round(point, origin, pitch):
            new_point = np.array(point) - np.array(origin)
            new_point /= pitch
            new_point = np.round(new_point).astype(int)
            return tuple(new_point)

        points = np.array(points, dtype=np.float64)
        if points.shape != (points.shape[0], 3):
            raise ValueError('shape of points must be (q, 3)')

        if origin is not None:
            origin = np.asanyarray(origin)
            if origin.shape != (3,):
                raise ValueError('shape of origin must be (3,)')
            points -= origin

        if pitch is not None:
            points /= pitch

        origin = np.asanyarray(origin, dtype=np.float64)
        pitch = float(pitch)

        indices = np.round(points).astype(int)
        updated_dict = {subtract_origin_and_round(point, origin, pitch): val for point, val in points_to_instances.items()}

        return indices, updated_dict

    def voxelized(self, pitch, transform_dict):
        """

        :param pitch:
        :param transform_dict:
        :return:
        """

        points = []
        for name, mesh in self.geometry.items():
            transform = transform_dict[name]
            voxel_grid = mesh.voxelized(pitch=pitch,
                                        method='binvox'
                                        ).apply_transform(transform)

            instance_id = int(name)
            new_points = voxel_grid.points
            points.append(new_points)

        if len(points) == 1:
            points = np.squeeze(np.array(points), axis=0)
        else:
            points = np.concatenate(points, axis=0)

        origin = points.min(axis=0)

        indices = ExtendedScene.points_to_indices(
            points,
            pitch=pitch, origin=origin)

        encoding = trimesh.voxel.encoding.SparseBinaryEncoding(indices=indices)
        vg = trimesh.voxel.VoxelGrid(
            encoding, transform=trimesh.transformations.scale_and_translate(pitch, origin))

        return vg


    def voxelized_with_instances(self, pitch, transform_dict, resolution=(64,64,64)):
        """
        Voxelizes the scene and retains a mapping of scene_indices to instance ids
        :param pitch:
        :param transform_dict:
        :return:
        """

        points = []
        points_to_instances = {}
        for name, mesh in self.geometry.items():
            transform = transform_dict[name]
            voxel_grid = mesh.voxelized(pitch=pitch,
                                        method='binvox'
                                        ).apply_transform(transform)

            instance_id = int(name)
            new_points = voxel_grid.points
            new_instance_refs = {tuple(point): instance_id for point in new_points}
            points_to_instances.update(new_instance_refs)
            points.append(new_points)

        if len(points) == 1:
            points = np.expand_dims(np.array(points), axis=0)
        else:
            points = np.concatenate(points, axis=0)

        #center at 0
        joint_mesh = self.dump(concatenate=True)



        new_center_of_mesh = (np.array(resolution) / 2) * pitch  # center of the cube
        # need to move by 0.5 to move to the center of the voxel, not the border of the voxel
        displace_by = -new_center_of_mesh + (0.5*pitch) + joint_mesh.bounding_box.centroid
        origin = displace_by

        indices, indices_to_instances = ExtendedScene.points_to_indices_keep_instances(
            points,
            points_to_instances,
            pitch=pitch, origin=origin)

        if not (indices > 0).all():
            raise ValueError("negative indices for scene")

        encoding = trimesh.voxel.encoding.SparseBinaryEncoding(indices=indices)
        vg = trimesh.voxel.VoxelGrid(
            encoding, transform=trimesh.transformations.scale_and_translate(pitch, origin))

        return vg, indices_to_instances