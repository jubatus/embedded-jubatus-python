from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

setup(
    ext_modules = cythonize([
        Extension(
            'embedded_jubatus',
            [
                'src/embedded_jubatus.pyx',
                'src/_wrapper.cpp',
                'src/_model.cpp'
            ],
            libraries=['jubatus_core', 'jubaserv_common'],
            language='c++')
    ])
)
