from pathlib import Path
from setuptools import setup, find_packages
import re


# Package meta-data.
NAME = "Elmundo"
DESCRIPTION = "Assessment Work"
URL = "https://github.com/mmoulton1/Elmundo"
EMAIL = "mmoulton@nrel.gov"
AUTHOR = "matthew"
REQUIRES_PYTHON = ">=3.9.0"
VERSION = 0.0
ROOT = Path(__file__).parent

# Get package data
base_path = Path("elmundo")
package_data_files = []

package_data = {
    "elmundo": [],
    "ProFAST":[]
    # "elmundo": [str(file.relative_to(base_path)) for file in package_data_files],
}

setup(
    name=NAME,
    version=VERSION,
    url=URL,
    description=DESCRIPTION,
    license='Apache Version 2.0',
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    packages=find_packages(),
    package_data=package_data,
    include_package_data=True,
    install_requires=(base_path.parent / "requirements.txt").read_text().splitlines(),
    # tests_require=['pytest', 'pytest-subtests', 'responses']
)
