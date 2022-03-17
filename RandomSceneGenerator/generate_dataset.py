#!/usr/bin/env python

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

def generate_data(out, model_dir, random_state, connection_method, min_objects=4, max_objects=8):
    out.makedirs_p()
    (out / 'models').mkdir_p()

    models = SuperQuadricModels(data_dir=model_dir)

    class_weight = np.zeros((models.n_class), dtype=float)
    class_weight[...] = 1
    class_weight /= class_weight.sum()

    generator = plane_type.PlaneTypeSceneGeneration(
        extents=(0.4, 0.4, 0.3),
        models=models,
        min_objects=min_objects,
        max_objects=max_objects,
        random_state=random_state,
        class_weight=class_weight,
        multi_instance=True,
        connection_method=connection_method,
        mesh_scale=((0.1,0.05,0.1),(0.20,0.20,0.20)),
        n_trial=7,
    )
    pybullet.resetDebugVisualizerCamera(
        cameraDistance=1.5,
        cameraYaw=45,
        cameraPitch=-60,
        cameraTargetPosition=(0, 0, 0),
    )

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
            shutil.copy(data['cad_file'], dst_file)
            cad_files[ins_id] = f'models/{ins_id:08d}.obj'

    Ts_cam2world = generator.random_camera_trajectory(
        n_keypoints=5, n_points=7, distance=(1, 2), elevation=(30, 90)
    )
    camera = extra.trimesh.OpenGLCamera(
        resolution=(640, 480), fovy=45
    )

    # save number of objects in the file
    with open(os.path.join(out/"nbr_of_objects.txt"), 'w') as fp:
        fp.write(str(len(generator.unique_ids)))

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
        np.savez_compressed(npz_file, **data)

    extra.pybullet.del_world()


def main(out_dir, model_dir, n_video, n_processes, connection_method, min_objects, max_objects):

    now = datetime.datetime.utcnow()
    timestamp = now.strftime('%Y%m%d_%H%M%S.%f')

    root_dir = utils.get_data_path(
        out_dir + "/" + timestamp
    )
    if not os.path.exists(root_dir):
       os.makedirs(root_dir)

       # save max number of objects as text
       with open(os.path.join(root_dir, "max_n_objects.txt"), 'w') as fp:
           fp.write(str(max_objects))

    def create(index):
        scene_dir = root_dir / f'{index:08d}'
        random_state = np.random.RandomState(index)

        generate_data(scene_dir,
                      model_dir,
                      random_state,
                      connection_method,
                      min_objects=min_objects,
                      max_objects=max_objects)

    for index in range(0, n_video//n_processes):
        start = index * n_processes
        processes = [multiprocessing.Process(target=create, args=(i,)) for i in
                     range(start, start + n_processes)]
        [t.start() for t in processes]
        [t.join() for t in processes]



if __name__ == '__main__':


    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument('out_dir', help='destination path to dataset')
    parser.add_argument('model_dir', help='path to SQ models')
    parser.add_argument('n_scenes', type=int,help='number of scenes to generate')

    parser.add_argument('--min_objects', type=int, help='minimum number of objects in the scene',default=4)
    parser.add_argument('--max_objects', type=int, help='maximum number of objects in the scene', default=8)
    parser.add_argument('--gui', help='gui? True')
    parser.add_argument('--n_processes', type=int, help='multiprocessing: number of processes', default=1)

    args = parser.parse_args()

    if args.gui == "True":
        connection_method = pybullet.GUI
    else:
        connection_method = pybullet.DIRECT

    print(f"Generating {args.n_scenes} scenes at: {args.out_dir} \nUsing models from {args.model_dir} \n")
    main(args.out_dir, args.model_dir, args.n_scenes, args.n_processes, connection_method, args.min_objects, args.max_objects)