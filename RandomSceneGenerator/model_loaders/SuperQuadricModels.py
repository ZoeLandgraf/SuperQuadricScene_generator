import re
import gdown
import path
import morefusion
from morefusion.datasets.base import ModelsBase

class SuperQuadricModels(ModelsBase):

    def __init__(self, filter_cad_ids=None):
        self.data_dir = path.Path('/media/zoe/ExtDrive/SIMstack/Data/sq_models/2000_convex_models')
        self.filter_cad_ids = filter_cad_ids

    @property
    def class_names(self):
        """
        return ('__background__', 'superquadric')
        """
        names = []
        for file in sorted((self.data_dir / 'meshes').listdir()):
            if not re.match(r'[0-9]{8}\.obj', file.basename()):
                continue
            names.append(file.basename().stem)

        return tuple(names)


    def get_cad_ids(self, class_id=None):
        cad_ids = []
        for file in sorted((self.data_dir / 'meshes').listdir()):
            if not re.match(r'[0-9]{8}\.obj', file.basename()):
                continue
            if self.filter_cad_ids is not None:
                if file.basename() in self.filter_cad_ids:
                    continue
            if class_id is None:
                cad_ids.append(file.basename().stem)
            else:
                if re.match(f"{class_id:08d}", file.basename()):
                    cad_ids.append(file.basename().stem)
        return tuple(cad_ids)

    def get_cad_file_from_id(self, cad_id):
        return self.data_dir / 'meshes' / f'{cad_id}.obj'


    def get_parameters_from_id(self, cad_id):
        return self.data_dir / 'parameters' / f'{cad_id}.json'