"""
Creates a dataset from a set of pregenerated PyBullet scenes
"""
import time
import path
import numpy as np
import os
import argparse

import warnings
import multiprocessing
import shutil
from data_loader import PileLoader
from scene_utils.TSDFScene import TSDFScene, create_tsdf_per_object

warnings.filterwarnings("ignore")

def organise_folders(data_dir):
    total_folders = 0
    for el in os.listdir(data_dir):

        if ".txt" in el:
            continue

        scene_tag = el

        # save models type id
        model_type = 1  # 0: 1500 SQ (without extrq squares), 1: 2000 SQ (with extra squares)
        fn = str(path.Path(data_dir) / scene_tag / "model_type.txt")
        np.savetxt(fn, np.array([model_type]).astype(int))

        pybullet_scene_info_f = str(path.Path(data_dir) / scene_tag / "pybullet_scene_info")

        if  os.path.exists(pybullet_scene_info_f):
            total_folders += 1
            continue
        else:
            os.mkdir(pybullet_scene_info_f)

        scene_info = str(path.Path(data_dir) / scene_tag / '00000000.npz')
        new_scene_info = str(path.Path(pybullet_scene_info_f) / '00000000.npz')

        #delete folder if it has not information, otherwise organise it
        if not (os.path.exists(scene_info) or os.path.exists(new_scene_info)):
            print("This folder doesn't look like it has the right content, skipping")
            continue
        elif not os.path.exists(new_scene_info):
            shutil.move(scene_info, new_scene_info)
            total_folders += 1

        for el in [str(i).zfill(8) + '.npz' for i in range(1, 7)]:
            scene_info = str(path.Path(data_dir) / scene_tag / el)
            if os.path.exists(scene_info):
                os.remove(scene_info)

    print("number of valid folders: ", total_folders)
    time.sleep(2)

def create_data_for_scene(i, per_instance_scene=False):
    tag = data_loader.data[i]["tag"]
    scene_dir = data_loader.data[i]['scene_file']

    # if all folders exist, skip
    all_folders = True
    for j in range(6):
        folder = os.path.join(scene_dir, "tsdf" + str(j) + ".npy")
        if not os.path.exists(folder):
            all_folders = False
            break
    if all_folders == True and os.path.exists(os.path.join(scene_dir, "scene_tsdf.npy")):
        print("skipping scene: ", i)
        return

    print(f"saving scene {i}")

    try:
        scene_info = data_loader.extract_scene_info(scene_dir)
        scene_dict = data_loader.scenedict_from_scene_info(scene_info,
                                                       cad_id_as_key=False)

    except:
        print(f"Issue with PyBullet scene: {os.path.basename(scene_dir)}")
        print("deleting scene: ", tag)
        shutil.rmtree(scene_dir)
        return

    # create scene tsdf
    try:
        tsdf = TSDFScene.generate_sdf_with_library(scene_dict['scene'], fixed_floor=15)
    except:
        print("problem creating full scene tsdf")
        return

    np.save(os.path.join(scene_dir, "scene_tsdf.npy"), tsdf)

    if per_instance_scene:
        # create individual tsdfs
        try:
            per_object_tsdfs = create_tsdf_per_object(scene_dict)
        except:
            print("problem creating individual tsdfs")
            return

        for j in range(len(per_object_tsdfs)):
            tsdf = per_object_tsdfs[j]
            np.save(os.path.join(scene_dir, "tsdf" + str(j) + ".npy"), tsdf)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument('scene_dir', help='destination path to dataset')
    parser.add_argument('model_dir', help='path to SQ models')
    parser.add_argument('--n_processes', type=int, help='multiprocessing: number of processes', default=1)
    parser.add_argument('--per_instance_scene', type=bool, help='create per instance tsdf and voxelgrids')

    args = parser.parse_args()

    data_loader = PileLoader(path_to_scenes=args.scene_dir, path_to_models=args.model_dir)
    organise_folders(data_dir=data_loader.data_path)
    # create processes
    nbr_of_processes = args.n_processes

    data_length = data_loader.__len__()

    a = time.time()
    for batch in range(0,data_length//nbr_of_processes):
        index_start = batch*nbr_of_processes
        processes = [multiprocessing.Process(target=create_data_for_scene, args=(i,args.per_instance_scene)) for i in range(index_start,index_start+nbr_of_processes)]
        [t.start() for t in processes]
        [t.join() for t in processes]

    # complete remaining folders
    if (data_length % nbr_of_processes) != 0:
        start =  data_length - (data_length//nbr_of_processes)*nbr_of_processes + 1
        end = data_length
        processes = [multiprocessing.Process(target=create_data_for_scene, args=(i,)) for i in
                     range(start, end)]
        [t.start() for t in processes]
        [t.join() for t in processes]

    b = time.time()
    print("time taken: " , b - a)
