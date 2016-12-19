import os.path
import platform

from Cython.Build import cythonize
from setuptools import Extension
from setuptools import setup

def read(name):
    with open(os.path.join(os.path.dirname(__file__), name)) as f:
        return f.read()

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

if platform.system() == 'Darwin':
    extra_compile_args = ['-std=c++11']
else:
    extra_compile_args = []

setup(
    name='embedded_jubatus',
    version=read('VERSION').rstrip(),
    description='embedded-jubatus-python is a Python bridge to call Jubatus Core library.',
    long_description=read('README.rst'),
    author='PFN & NTT',
    author_email='jubatus-team@googlegroups.com',
    url='http://jubat.us',
    download_url='http://pypi.python.org/pypi/embedded_jubatus',
    license='LGPLv2',
    classifiers=[
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'License :: OSI Approved :: GNU Lesser General Public License v2 (LGPLv2)',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 3',
    ],
    ext_modules=cythonize([
        Extension(
            'embedded_jubatus',
            [
                'src/embedded_jubatus.pyx',
                'src/_wrapper.cpp',
                'src/_model.cpp'
            ],
            include_dirs=include_dirs,
            libraries=['jubatus_core', 'jubatus_util_text'],
            language='c++',
            extra_compile_args=extra_compile_args)
    ]),
    install_requires=['jubatus'],
    test_suite='tests',
)
