"""
The Software used in this folder is taken from the following project (https://github.com/wkentaro/morefusion)

MoreFusion is an object-level reconstruction system that builds a map with known-shaped objects,
exploiting volumetric reconstruction of detected objects in a real-time, incremental scene
reconstruction senario. It is based on the techniques described in the following publication:

    • Kentaro Wada, Edgar Sucar, Stephen James, Daniel Lenton, Andrew J. Davison.
MoreFusion: Multi-object Reasoning for 6D Pose Estimation from Volumetric Fusion,
IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2020
"""

import path
import typing

class ModelsBase:

    _root_dir: typing.Optional[str] = None

    @property
    def root_dir(self) -> path.Path:
        if self._root_dir is None:
            raise ValueError("self._root_dir is not set")
        if type(self._root_dir) is not path.Path:
            self._root_dir = path.Path(self._root_dir)
        return self._root_dir

    @property
    def class_names(self):
        raise NotImplementedError

    @property
    def n_class(self):
        return len(self.class_names)

    def get_cad_ids(self, class_id):
        raise NotImplementedError

    def get_cad_file_from_id(self, cad_id):
        return NotImplementedError