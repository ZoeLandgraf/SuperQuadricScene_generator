import trimesh
import path
import json
import numpy as np
from copy import deepcopy

class SceneInfo:
    def __init__(self, data):
        self.cad_ids = data['cad_ids']
        self.scales = data['scales']
        self.Ts_cad2cams = data['Ts_cad2cam']
        self.Ts_cam2world = data['T_cam2world']


class ObjectPileSceneParser:
    """
    Loader and File processing class which parses scenes files defined in a certain format.
    Also provides some utility functions
    """

    @classmethod
    def extract_sq_info_from_file(cls, scene_dir):
        """
        Reads the scene info from file. SPECIFIC TO PYBULLET SCENES
        :param scene_dir:
        :return:
        """
        scene_dir = path.Path(scene_dir)
        scene_dir_d = scene_dir / 'pybullet_scene_info'
        npz_file = list(sorted(scene_dir_d.listdir()))[0]
        data = dict(np.load(npz_file), allow_pickle=True)
        return SceneInfo(data)

    @classmethod
    def scene_from_file(cls,
                        scene_info: SceneInfo,
                        models,
                        transform_meshes=True,
                        cad_id_as_key=True):

        cad_ids = scene_info.cad_ids
        scales = scene_info.scales
        Ts_cad2cams = scene_info.Ts_cad2cams
        Ts_cam2world = scene_info.Ts_cam2world


        exponents_dict = ObjectPileSceneParser.load_exponents(cad_ids, models.data_dir)
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
            cad_file = models.get_cad_file_from_id(cad_id)
            cad = trimesh.load_mesh(cad_file, process=False)
            if isinstance(cad, trimesh.Scene):
                cad = cad.dump(concatenate=True)

            unscaled_meshes[object_id] = deepcopy(cad)

            cad.vertices *= scale

            meshes[object_id] = cad
            mesh_scales[object_id] = scale
            exponents[object_id] = exponents_dict[cad_id]
            if transform_meshes:
                scene.add_geometry(cad, geom_name=object_id, transform=transform)
            else:
                scene.add_geometry(cad, geom_name=object_id)


        scene_dict = {'scene' : scene,
                      'transforms' : transforms,
                      'meshes' : meshes,
                      'unscaled_meshes' : unscaled_meshes,
                      'mesh_scales' : mesh_scales,
                      'exponents' : exponents}

        return scene_dict

    @classmethod
    def load_exponents(cls, cad_ids, model_dir):
        """
        Loads json file corresponding to cad id and extracts exponents
        :param cad_ids:
        :return:
        """
        exponents = {}
        for cad_id in cad_ids:
            with open(str((model_dir / 'parameters' / (cad_id + '.json')))) as json_file:
                parameter_file = json.load(json_file)
                exponents[cad_id] = json.loads(parameter_file['exponents'])
        return exponents

    @classmethod
    def load_scales(cls, cad_ids, model_dir):
        """
        Loads json file corresponding to cad id and extracts exponents
        :param cad_ids:
        :return:
        """
        scales = {}
        for cad_id in cad_ids:
            with open(str((model_dir / 'parameters' / cad_id + '.json'))) as json_file:
                parameter_file = json.load(json_file)
                scales[cad_id] = json.loads(parameter_file['scales'])
        return scales






