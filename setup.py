import os.path

from Cython.Build import cythonize
from setuptools import Extension
from setuptools import setup


py_defines = []
include_dirs = []
try:
    import numpy
    py_defines.append(('NUMPY', True))
    include_dirs.append(numpy.get_include())
except ImportError:
    py_defines.append(('NUMPY', False))

defs_path = 'src/defs.pyx'
defs = '\n'.join(['DEF ' + n + ' = ' + str(v) for n, v in py_defines])
old_defs = open(defs_path, 'r').read() if os.path.isfile(defs_path) else ''
if defs != old_defs:
    with open('src/defs.pyx', 'w') as f:
        f.write(defs)

setup(
    name='embedded_jubatus',
    version='0.1',
    ext_modules=cythonize([
        Extension(
            'embedded_jubatus',
            [
                'src/embedded_jubatus.pyx',
                'src/_wrapper.cpp',
                'src/_model.cpp'
            ],
            include_dirs=include_dirs,
            libraries=['jubatus_core', 'jubaserv_common'],
            language='c++')
    ]),
    install_requires=['jubatus'],
    test_suite='tests',
)
