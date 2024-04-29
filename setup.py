import os
import numpy
from setuptools import find_packages, setup
from distutils.extension import Extension
from Cython.Build import cythonize

# Get README
with open("README.md", "r") as fh:
    long_description = fh.read()

# Execute version.py and recover globals
version_globals = {}
with open("stateye/version.py") as fp:
    exec(fp.read(), version_globals)

with open("requirements.txt") as fr:
    requirements = fr.read().splitlines()

cython_files = [
    ("xiaolin_wu", ("stateye", "xiaolin_wu.pyx")),
    ("ideal_cdr", ("stateye", "ideal_cdr.pyx")),
    ("measurements.utilities", ("stateye", "measurements", "utilities.pyx")),
]
extensions = []
for name, source in cython_files:
    extensions.append(
        Extension(
            f"stateye.{name}",
            [os.path.join(*source)],
            include_dirs=[numpy.get_include()],
            language="c++",
            extra_compile_args=["-ffast-math"],
        )
    )

# Package data
setup(
    name="stateye",
    version=version_globals["__version__"],
    packages=find_packages(),
    include_package_data=True,
    author="Ayar Labs",
    description="Project for analyzing time-series data from link models or "
    "measurements. Generates eye diagrams, extracts statistics, "
    "and contains useful plotting utilities.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="http://noether.ayarlabs.com/link-model/stateye",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=requirements,
    python_requires=">=3.6",
    ext_modules=cythonize(
        extensions, annotate=True
    ),  # annotate can be set to False to suppress cython HTMLs
    include_dirs=[numpy.get_include()],
    zip_safe=False,
)
