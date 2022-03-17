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