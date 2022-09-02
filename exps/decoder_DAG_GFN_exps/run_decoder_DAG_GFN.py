import sys
import pathlib
import ruamel.yaml as yaml
from tqdm import tqdm

sys.path.append("..")
sys.path.append("../..")
sys.path.append("../../modules")
sys.path.append("../../models")

configs = yaml.safe_load((pathlib.Path("../..") / "configs.yaml").read_text())
