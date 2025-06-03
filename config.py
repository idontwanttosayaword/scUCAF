from pathlib import Path

import numpy as np
import yaml


class Config:
    def __init__(self, config_path):
        if config_path is None:
            
            current_directory = Path(__file__).parent
            config_file_path = current_directory / 'config.yaml'
        else:
            
            config_file_path = Path(config_path)

        with config_file_path.open('r', encoding='utf-8') as file:
            config_data = yaml.safe_load(file)

        self._load(config_data)

    def __getattr__(self, name):
        
        return self.__dict__.get(name)

    def __setattr__(self, name, value):
        
        self.__dict__[name] = value

    def _load(self, entries):
        for key, value in entries.items():
            if isinstance(value, dict):
                value = Config.create_from_dict(value)
            setattr(self, key, value)

    @classmethod
    def create_from_dict(cls, data):
        instance = cls.__new__(cls)
        instance._load(data)
        return instance

    def __repr__(self):
        return f'{self.__class__.__name__}({self.__dict__})'

    def to_dict(self):
        
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Config):
                result[key] = value.to_dict()
            else:
                
                if isinstance(value, np.generic):
                    value = value.item()
                
                elif isinstance(value, np.ndarray):
                    value = value.tolist()
                result[key] = value
        return result

    def save(self, run_folder):
        
        run_folder = Path(run_folder)
        save_path = run_folder / 'config.yaml'

        
        run_folder.mkdir(parents=True, exist_ok=True)

        config_dict = self.to_dict()
        with save_path.open('w', encoding='utf-8') as file:
            yaml.safe_dump(config_dict, file, allow_unicode=True)
