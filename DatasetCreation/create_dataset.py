"""
Creates a dataset from a set of pregenerated PyBullet scenes
"""
import time
import path
import numpy as np
import os
import warnings
import multiprocessing
import shutil
from data_loader import PileLoader
from scene_utils.TSDFScene import TSDFScene, create_tsdf_per_object

path_to_scenes = "/media/zoe/ExtDrive/_3DObjectDiscovery/Data/sq_scenes/6_objects/test/"
path_to_models = "/media/zoe/ExtDrive/_3DObjectDiscovery/Data/sq_models/2000_convex_models"
data_loader = PileLoader(path_to_scenes=path_to_scenes, path_to_models=path_to_models, type="val")

warnings.filterwarnings("ignore")


def organise_folders(data_dir):
    total_folders = 0
    for el in os.listdir(data_dir):

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
            print("deleting scene: ", scene_tag)
            shutil.rmtree(str(path.Path(data_dir) / scene_tag ))
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

def create_data_for_scene(i):
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
        print("deleting scene: ", scene_tag)
        shutil.rmtree(scene_dir)
        return

    # create scene tsdf
    try:
        tsdf = TSDFScene.generate_sdf_with_library(scene_dict['scene'], fixed_floor=15)
    except:
        print("problem creating full scene tsdf")
        return

    np.save(os.path.join(scene_dir, "scene_tsdf.npy"), tsdf)

    # create individual tsdfs
    # try:
    per_object_tsdfs = create_tsdf_per_object(scene_dict)
    # except:
    #     print("problem creating individual tsdfs")
    #     return

    for j in range(len(per_object_tsdfs)):
        tsdf = per_object_tsdfs[j]
        np.save(os.path.join(scene_dir, "tsdf" + str(j) + ".npy"), tsdf)


organise_folders(data_dir=data_loader.data_path)
# create processes
nbr_of_processes = 2

data_length = data_loader.__len__()

a = time.time()
for batch in range(0,data_length//nbr_of_processes):
    index_start = batch*nbr_of_processes
    processes = [multiprocessing.Process(target=create_data_for_scene, args=(i,)) for i in range(index_start,index_start+nbr_of_processes)]
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
