from Cython.Build import cythonize
from setuptools import Extension
from setuptools import setup


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
            libraries=['jubatus_core', 'jubaserv_common'],
            language='c++')
    ]),
    install_requires=['jubatus'],
    test_suite='tests',
)
