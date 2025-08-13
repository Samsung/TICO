"""setup.py for TICO"""

import os
import re
from pathlib import Path

from setuptools import find_packages, setup
from version import VERSION

############################################################
### Set dev(nightly) version                             ###
############################################################
if nightly_release_version := os.environ.get("NIGHTLY_VERSION"):
    VERSION = f"{VERSION}.dev{nightly_release_version}"

############################################################
### Update __version__ in __init__.py                    ###
############################################################
with open("tico/__init__.py", "r") as init_file:
    init_file_update = re.sub(
        "__version__ =.*", f'__version__ = "{VERSION}"', init_file.read()
    )
with open("tico/__init__.py", "w") as init_file:
    init_file.write(init_file_update)

############################################################
### Prepare long_description                             ###
############################################################
readme = Path(__file__).with_name("README.md").read_text(encoding="utf-8")

############################################################
### Run setup                                            ###
############################################################
setup(
    name="tico",
    python_requires=">=3.10.0",
    version=VERSION,
    description="Convert exported Torch module to circle",
    long_description=readme,
    long_description_content_type="text/markdown",
    license_files=("LICENSE",),
    packages=find_packages(include=["tico*"]),
    entry_points={"console_scripts": ["pt2-to-circle = tico.pt2_to_circle:main"]},
    install_requires=["circle-schema", "packaging", "cffi", "torch", "pyyaml", "tqdm"],
)
