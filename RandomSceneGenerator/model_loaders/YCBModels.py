import path
from OtherRepos.morefusion.morefusion.datasets.base import ModelsBase
class YCB_Models(ModelsBase):

    def __init__(self):
        self.data_dir = path.Path('/media/zoe/ExtDrive/_3DObjectDiscovery/Data/YCB_models/models_for_InstanceNet')

    @property
    def class_names(self):

        """
        return ('__background__', 'superquadric')
        """
        names = []
        for file in sorted(self.data_dir.listdir()):
            # if not 'obj' in file:
            #     continue
            names.append(str(file.basename().stem))

        return tuple(names)

    def get_cad_ids(self, class_id=None):
        cad_ids = []

        for file in sorted(self.data_dir.listdir()):
            if "convex" in file.basename():
                continue

            cad_ids.append(str(file.basename().stem))

        return tuple(cad_ids)

    def get_cad_file_from_id(self, cad_id):
        return self.data_dir / f'{cad_id}' / 'textured.obj'