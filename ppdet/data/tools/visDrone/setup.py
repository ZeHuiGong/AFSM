import numpy
from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

extensions = [
    Extension(
        "_visdrone_eval",
        ["_visdrone_eval.pyx"],
        extra_compile_args=["-Wno-cpp", "-Wno-unused-function"]
    )
]

setup(
    name="_visdrone_eval",
    ext_modules=cythonize(extensions),
    include_dirs=[numpy.get_include()]
)