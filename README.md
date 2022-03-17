# SuperQuadric scene generator
Generates scenes of cluttered SuperQuadric shapes. 
![title](examples/example_image.png)


###Installation
create your virtual conda env from the environment.yml file

###Scene generation
1) A scene is generated inside the PyBullet simulator using a collection of Superquadric shapes.
    
    a) Download the collection of Superquadric shapes from [superquadric_models](https://drive.google.com/file/d/1CMJca_4V_87AYjjEEtqXF05GimTHNqJZ/view?usp=sharing)
    
    b) Run the 'generate_dataset.py' script in RadomSceneGenerator. The script takes the following input:
       
        usage: generate_dataset.py [-h] [--min_objects MIN_OBJECTS]
                           
                            [--max_objects MAX_OBJECTS] [--gui GUI]
                          
                            [--n_processes N_PROCESSES]
                           
    out_dir model_dir n_scenes

2) Each scene can be transformed into a TSDF representation. 

    a) Run the 'create_dataset.py' script in DatasetCreation. 
   
        usage: create_dataset.py [-h] [--n_processes N_PROCESSES]
                         
                            [--per_instance_scene PER_INSTANCE_SCENE]
                            
                            scene_dir model_dir
    Setting per_instance_scene=True 
    in the optional arguments will generate a tsdf grid for every individual Superquadric.
