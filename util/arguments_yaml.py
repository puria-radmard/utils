from __future__ import annotations

import yaml


node_types = [int, float, str, bool, type(None)]
iter_types = [list, tuple]


class ConfigNamepace:

    def __init__(self, config: dict) -> None:
        self.dict = {}
        for k, v in config.items():
            assert isinstance(k, str)
            self.dict[k] = ConfigNamepace.from_obj(v)

    @classmethod
    def from_obj(cls, obj):
        obj_type = type(obj)
        if obj_type in node_types:
            return obj
        elif obj_type in iter_types:
            return [cls.from_obj(item) for item in obj]
        elif obj_type == dict:
            return cls(obj)
        else:
            raise TypeError(obj_type)
    
    @classmethod
    def from_yaml_path(cls, yaml_path) -> ConfigNamepace:
        with open(yaml_path) as f:
            content = yaml.safe_load(f)
        assert isinstance(content, dict)
        return cls.from_obj(content)
    
    def update(self, other: ConfigNamepace):
        for k, v in other.dict.items():
            v_type = type(v)
            if v_type in node_types:
                self.dict[k] = v
            elif v_type in iter_types:
                if k in self.dict:
                    if len(v) != len(self.dict[k]):
                        print(f"NB: updating config parameter '{k}' from an array of size {len(self.dict[k])} to one of size {len(v)}")
                self.dict[k] = v
            elif v_type == ConfigNamepace:
                if k in self.dict:
                    self.dict[k].update(v)
                else:
                    self.dict[k] = v
            else:
                raise TypeError(v_type)

    @staticmethod
    def to_dict_inner(obj):
        obj_type = type(obj)
        if obj_type in node_types:
            return obj
        elif obj_type in iter_types:
            return [ConfigNamepace.to_dict_inner(item) for item in obj]
        elif obj_type == ConfigNamepace:
            return obj_type.to_dict()
        else:
            raise TypeError(obj_type)
   
    def to_dict(self):
        return {k: self.to_dict_inner(v) for k, v in self.dict.items()}

    def write_to_yaml(self, destination_path):
        output_dictionary = self.to_dict()
        with open(destination_path, 'w') as f:
            yaml.dump(output_dictionary, f)

    def __contains__(self, key):
        return key in self.dict

    def __getattr__(self, __name: str):
        ret = self.dict.get(__name)
        return ret
