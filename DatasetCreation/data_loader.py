import json
import os
import re
import trimesh
import copy
from copy import deepcopy
import torch
import path
import numpy as np

def center_scene(meshes, transforms):
    """
    centers meshes around 0
    """
    SQ_scene = trimesh.Scene()
    for mesh, transform in zip(meshes, transforms):
        SQ_scene.add_geometry(mesh, transform=transform)

    joint_mesh = SQ_scene.dump(concatenate=True)
    center_point = joint_mesh.vertices.min(axis=0) + ((joint_mesh.vertices.max(axis=0) - joint_mesh.vertices.min(axis=0)) / 2)
    center_point[2] = 0

    transforms_to_O = []
    for i, val in enumerate(transforms):
        T = val
        t_trans = trimesh.transformations.translation_matrix(-center_point)
        tt = np.matmul(t_trans, T)

        transforms_to_O.append(tt)
    return transforms_to_O

class SuperQuadricModels():

    def __init__(self, data_dir):
        self.data_dir = path.Path(data_dir)

    def get_cad_ids(self):
        cad_ids = []
        for file in sorted((self.data_dir / 'meshes').listdir()):
            if not re.match(r'[0-9]{8}\.obj', file.basename()):
                continue
            cad_ids.append(file.basename().stem)
        return tuple(cad_ids)

    def get_cad_file_from_id(self, cad_id):
        return self.data_dir / 'meshes' / f'{cad_id}.obj'

    def get_parameters_from_id(self, cad_id):
        return self.data_dir / 'parameters' / f'{cad_id}.json'


class PileLoader:
    """
    Loader which loads and can display the original piles
    generated in PyBullet
    """

    class SceneInfo:
        def __init__(self, data):
            self.cad_ids = data['cad_ids']
            self.scales = data['scales']
            self.Ts_cad2cams = data['Ts_cad2cam']
            self.Ts_cam2world = data['T_cam2world']

    def __init__(self, path_to_scenes, path_to_models):

        self.path_to_models = path_to_models
        self.models = SuperQuadricModels(path_to_models)

        # load all folders
        folder_list = []
        folder = path_to_scenes
        for el in os.listdir(folder):
            if not re.match(r'[0-9]', path.Path(el).basename()):
                continue
            folder_list.append(os.path.join(folder, el))

        self.data_path = folder
        self.data = self.init_files(folder_list)

    def __len__(self):
        return len(self.data)

    def extract_scene_info(self, scene_dir):
        """
        extracts the scene info from the file which is generated by PyBullet
        :param filename: scene info filename
        :return: SceneInfo object
        """
        scene_dir = path.Path(scene_dir)
        scene_dir_d = scene_dir / 'pybullet_scene_info'
        npz_file = list(sorted(scene_dir_d.listdir()))[0]
        data = dict(np.load(npz_file), allow_pickle=True)
        return PileLoader.SceneInfo(data)


    def scenedict_from_scene_info(self,
                        scene_info: SceneInfo,
                        cad_id_as_key=True):
        """
        Generates a dictionary with scene info from sceneinfo object
        """

        cad_ids = scene_info.cad_ids
        scales = scene_info.scales
        Ts_cad2cams = scene_info.Ts_cad2cams
        Ts_cam2world = scene_info.Ts_cam2world


        #load exponents from models
        exponents_dict = {}
        for cad_id in cad_ids:
            with open(str((path.Path(self.models.data_dir) / 'parameters' / (cad_id + '.json')))) as json_file:
                parameter_file = json.load(json_file)
                exponents_dict[cad_id] = json.loads(parameter_file['exponents'])

        transforms = {}
        meshes = {}
        unscaled_meshes = {}
        mesh_scales = {}
        exponents = {}
        scene = trimesh.Scene()
        for i, (cad_id, scale, Ts_cad2cam) in enumerate(zip(
                cad_ids, scales, Ts_cad2cams)):
            if cad_id_as_key:
                object_id = str(int(cad_id))
            else:
                object_id = str(int(i + 1))
            transform = Ts_cam2world @ Ts_cad2cam
            transforms[object_id] = transform

            # cad
            cad_file = self.models.get_cad_file_from_id(cad_id)
            cad = trimesh.load_mesh(cad_file, process=False)
            if isinstance(cad, trimesh.Scene):
                cad = cad.dump(concatenate=True)

            unscaled_meshes[object_id] = deepcopy(cad)

            cad.vertices *= scale

            meshes[object_id] = cad
            mesh_scales[object_id] = scale
            exponents[object_id] = exponents_dict[cad_id]

            scene.add_geometry(cad, geom_name=object_id, transform=transform)

        scene_dict = {'scene' : scene,
                      'transforms' : transforms,
                      'meshes' : meshes,
                      'unscaled_meshes' : unscaled_meshes,
                      'mesh_scales' : mesh_scales,
                      'exponents' : exponents}

        return scene_dict

    def init_files(self, scene_list):
        "Creates the list of dataset samples. Every dataset sample is a data item dictionary"
        scenes = []

        for file in scene_list:
            file = path.Path(file)
            if not file.basename().isnumeric():
                continue

            tag = os.path.splitext(os.path.basename(file))[0]

            data_item = {"scene_file": file,
                         "tag": tag
                         }
            scenes.append(data_item)

        return scenes


    def build_scene_from_dict(self, scene_dict):
        gt_meshes = [mesh for key, mesh in scene_dict['meshes'].items()]
        gt_transforms = [transf for key, transf in scene_dict['transforms'].items()]

        meshes = []
        transforms = []
        random_colors = np.random.uniform(50, 250, (len(gt_meshes), 4))
        random_colors[:, 3] = 255

        T_to_center = center_scene(gt_meshes, gt_transforms)
        for nbr, (mesh, transf) in enumerate(zip(gt_meshes, T_to_center)):
            meshc = copy.deepcopy(mesh)
            mesh.apply_scale(64)
            mesh.apply_transform(mesh.principal_inertia_transform)
            meshes.append(mesh)
            meshc = meshc.apply_transform(transf)
            meshc.apply_scale(64)
            mesh.visual.face_colors = random_colors[nbr]
            transform = trimesh.transformations.inverse_matrix(meshc.principal_inertia_transform)
            transforms.append(transform)

        return meshes, transforms


    def __getitem__(self, i):

        scene = self.data[i]
        scene_info = self.extract_scene_info(scene['scene_file'])
        scene_dict = self.scenedict_from_scene_info(scene_info,
                                          cad_id_as_key=False)
        meshes, transforms = self.build_scene_from_dict(scene_dict)

        return {'meshes':meshes, "transforms": transforms}


class VoxelGridLoader:
    """

    """
    def __init__(self, path_to_scenes, max_n_objects):

        # load all folders
        folder_list = []
        folder = path_to_scenes
        for el in os.listdir(folder):
            if not re.match(r'[0-9]', path.Path(el).basename()):
                continue
            folder_list.append(os.path.join(folder, el))

        self.folder_list = folder_list
        self.max_n_objects = max_n_objects

    def __len__(self):
        return len(self.folder_list)

    def __getitem__(self, i):

        folder = self.folder_list[i]

        scene_file = os.path.join(folder, "scene_tsdf.npy")
        if not os.path.exists(scene_file):
            raise IOError(f"scene file {scene_file} doesn't exist")
        # load scene tsdf and create scene occupancy grid
        scene_tsdf = np.load(os.path.join(folder, "scene_tsdf.npy"))
        scene_occ = np.zeros_like(scene_tsdf)
        scene_occ[scene_tsdf < 0] = 1

        tsdfs = []
        occs = []
        # extract number of objects
        nbr_of_objects = 0;

        while os.path.exists(os.path.join(folder, "tsdf" + str(nbr_of_objects) + ".npy")):
            tsdf = np.load(os.path.join(folder, "tsdf" + str(nbr_of_objects) + ".npy"))
            nbr_of_objects += 1
            tsdfs.append(tsdf)
            occ = np.zeros_like(tsdf)
            occ[tsdf < 0] = 1
            occs.append(occ)

        sample = {"scene": torch.from_numpy(scene_tsdf).unsqueeze(0),
                  "scene_occ": torch.from_numpy(scene_occ).unsqueeze(0)}

        if nbr_of_objects > 0:
            # concatenate tsdfs
            concatenated_tsdfs = torch.cat([torch.from_numpy(el).unsqueeze(0) for el in tsdfs], dim=0)

            # add empty scenes if necessary (for datasets with varying number of objects)
            for i in range(self.max_n_objects - concatenated_tsdfs.shape[0]):
                concatenated_tsdfs = torch.cat(
                    [concatenated_tsdfs, torch.zeros_like(concatenated_tsdfs[0].unsqueeze(0))], dim=0)

            # concatenate occs
            concatenated_occs = torch.cat([torch.from_numpy(el).unsqueeze(0) for el in occs], dim=0)

            # add empty scenes if necessary (for datasets with varying number of objects)
            for i in range(self.max_n_objects - concatenated_occs.shape[0]):
                concatenated_occs = torch.cat(
                    [concatenated_occs, torch.zeros_like(concatenated_occs[0].unsqueeze(0))], dim=0)

            sample.update({"concatenated_objects": concatenated_tsdfs,
                      "concatenated_objects_occ": concatenated_occs,
                      "nbr_of_objects": nbr_of_objects})

        return sample