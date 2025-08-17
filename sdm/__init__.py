# my_package/__init__.py
import configparser
from pathlib import Path

# Initialize the config when the package is imported
config = configparser.ConfigParser()
default_config_path = Path(__file__).parent / "config.ini"
config.read(default_config_path)

try:
    path_data = config['path']['data']
except KeyError:
    path_data = None

try:
    path_repo = config['path']['repo']
except KeyError:
    path_repo = None