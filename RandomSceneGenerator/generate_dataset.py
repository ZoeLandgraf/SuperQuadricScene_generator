#!/usr/bin/env python

# This is borrowed from Kentaro's script

import argparse
import datetime
import shutil
import os
import numpy as np
import pybullet
import multiprocessing
from helper import extra, plane_type, utils

from model_loaders.SuperQuadricModels import SuperQuadricModels
from model_loaders.YCBModels import YCB_Models

from scene_utils.ObjectPileSceneParser import ObjectPileSceneParser

def generate_a_video(out, random_state, connection_method=None, filter_ids=None):
    out.makedirs_p()
    (out / 'models').mkdir_p()

    models = SuperQuadricModels(filter_cad_ids=filter_ids)
    # models = YCB_Models()

    class_weight = np.zeros((models.n_class), dtype=float)
    class_weight[...] = 1
    class_weight /= class_weight.sum()

    generator = plane_type.PlaneTypeSceneGeneration(
        extents=(0.4, 0.4, 0.3),
        models=models,
        n_object=6,
        random_state=random_state,
        class_weight=class_weight,
        multi_instance=True,
        connection_method=connection_method,
        mesh_scale=((0.1,0.05,0.1),(0.15,0.15,0.15)),
        # mesh_scale=((1,1,1),(1,1,1)),
        n_trial=7,
    )
    pybullet.resetDebugVisualizerCamera(
        cameraDistance=1.5,
        cameraYaw=45,
        cameraPitch=-60,
        cameraTargetPosition=(0, 0, 0),
    )

    """
    IF you want to laod from IDs
    ids_file = "/home/zoe/Research/Object_Decomposition/data_loaders/scripts/train_dataset_cad_ids.txt"
    ids = []
    ids_f = open(ids_file, 'r')
    for line in ids_f:
        ids.append(int(line))
    ids = np.array(ids)
    """

    class_weight = np.zeros((models.n_class), dtype=float)
    class_weight[...] = 1
    class_weight /= class_weight.sum()

    try:
        generator.generate()
    except ValueError:
        extra.pybullet.del_world()
        return


    cad_files = {}
    for ins_id, data in generator._objects.items():
        if 'cad_file' in data:
            dst_file = out / f'models/{ins_id:08d}.obj'
            # print(f'Saved: {dst_file}')
            shutil.copy(data['cad_file'], dst_file)
            cad_files[ins_id] = f'models/{ins_id:08d}.obj'

    Ts_cam2world = generator.random_camera_trajectory(
        n_keypoints=5, n_points=7, distance=(1, 2), elevation=(30, 90)
    )
    camera = extra.trimesh.OpenGLCamera(
        resolution=(640, 480), fovy=45
    )

    for index, T_cam2world in enumerate(Ts_cam2world):
        rgb, depth, instance_label, class_label = generator.render(
            T_cam2world,
            fovy=camera.fov[1],
            height=camera.resolution[1],
            width=camera.resolution[0],
        )
        instance_ids = generator.unique_ids
        cad_ids = generator.unique_ids_to_cad_ids(instance_ids)
        class_ids = generator.unique_ids_to_class_ids(instance_ids)
        scales = generator.unique_ids_to_scales(instance_ids)
        Ts_cad2world = generator.unique_ids_to_poses(instance_ids)
        T_world2cam = np.linalg.inv(T_cam2world)
        Ts_cad2cam = T_world2cam @ Ts_cad2world

        # validation
        n_instance = len(instance_ids)
        assert len(Ts_cad2cam) == n_instance
        assert len(cad_ids) == n_instance
        assert len(class_ids) == n_instance
        assert len(scales) == n_instance

        width, height = camera.resolution
        assert rgb.shape == (height, width, 3)
        assert rgb.dtype == np.uint8
        assert depth.shape == (height, width)
        assert depth.dtype == np.float32
        assert instance_label.shape == (height, width)
        assert instance_label.dtype == np.int32
        assert class_label.shape == (height, width)
        assert class_label.dtype == np.int32

        assert Ts_cad2cam.shape == (n_instance, 4, 4)
        assert Ts_cad2cam.dtype == np.float64
        assert T_cam2world.shape == (4, 4)
        assert T_cam2world.dtype == np.float64

        data = dict(
            rgb=rgb,
            depth=depth,
            instance_label=instance_label,
            class_label=class_label,
            intrinsic_matrix=camera.K,
            T_cam2world=T_cam2world,
            Ts_cad2cam=Ts_cad2cam,
            instance_ids=instance_ids,
            class_ids=class_ids,
            cad_ids=cad_ids,
            scales=scales,
            cad_files=[cad_files.get(i, '') for i in instance_ids],
        )

        npz_file = out / f'{index:08d}.npz'
        # print(f'==> Saved: {npz_file}')
        np.savez_compressed(npz_file, **data)

    extra.pybullet.del_world()


def extract_objects_from_dataset(mode, dataset_path):
    id_list = []
    for folder in os.listdir(dataset_path):
        pf = os.path.join(dataset_path, folder)
        scene_info = ObjectPileSceneParser.extract_sq_info_from_file(pf)
        id_list.extend(scene_info.cad_ids)

    if mode == '2':
        # filter all objects but the ones in dataset
        all_ids = SuperQuadricModels().get_cad_ids()
        ids = set(all_ids) - set(id_list)


    return list(ids)


def main(filter_ids, dataset_p, n_video, n_processes):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--nogui', help='no gui')

    args = parser.parse_args()

    if args.nogui == "True":
        connection_method = pybullet.DIRECT
    else:
        connection_method = pybullet.GUI

    if filter_ids is not None:
        ids = extract_objects_from_dataset(filter_ids, dataset_p)
    else:
        ids = filter_ids

    now = datetime.datetime.utcnow()
    timestamp = now.strftime('%Y%m%d_%H%M%S.%f')

    root_dir = utils.get_data_path(
        dataset_p + "/" + timestamp
    )
    if not os.path.exists(root_dir):
       os.makedirs(root_dir)

    def create(index):
        video_dir = root_dir / f'{index:08d}'
        random_state = np.random.RandomState(index)
        generate_a_video(video_dir, random_state, connection_method, ids)

    for index in range(0, n_video//n_processes):
        start = index * n_processes
        processes = [multiprocessing.Process(target=create, args=(i,)) for i in
                     range(start, start + n_processes)]
        [t.start() for t in processes]
        [t.join() for t in processes]



if __name__ == '__main__':
    filter_ids = None # None - don't apply filter, 1 - only use objects from given dataset 2 - only use objects not in dataset
    dataset_p = "/media/zoe/ExtDrive/_3DObjectDiscovery/Data/sq_scenes/6_objects/test/"
    n_video = 10
    n_processes = 2
    main(filter_ids, dataset_p, n_video, n_processes)