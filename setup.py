import unittest

from Cython.Build import cythonize
from distutils.core import Command
from distutils.core import setup
from distutils.extension import Extension


class TestCmd(Command):
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        runner = unittest.TextTestRunner(verbosity=2)
        runner.run(unittest.defaultTestLoader.discover('tests'))

setup(
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
    cmdclass={'test': TestCmd},
)
