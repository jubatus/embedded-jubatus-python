import os.path
import platform

from setuptools import Extension
from setuptools import setup


def read(name):
    with open(os.path.join(os.path.dirname(__file__), name)) as f:
        return f.read()


def _setup():
    include_dirs = []
    install_requires = read('requirements.txt').split('\n')
    setup_requires = ['Cython'] + [
        row for row in install_requires if row.startswith('numpy')]

    try:
        from Cython.Build import cythonize
    except ImportError:
        cythonize = None
    try:
        import numpy
        include_dirs.append(numpy.get_include())
    except ImportError:
        pass

    if platform.system() == 'Darwin':
        extra_compile_args = ['-std=c++11']
    else:
        extra_compile_args = []

    kwargs = dict(
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
        setup_requires=setup_requires,
        install_requires=install_requires,
        test_suite='tests',
    )
    kwargs['ext_modules'] = [Extension(
        'embedded_jubatus', [
            'src/embedded_jubatus.{}'.format(
                'pyx' if cythonize else 'cpp'),
            'src/_wrapper.cpp',
            'src/_model.cpp'
        ],
        include_dirs=include_dirs,
        libraries=['jubatus_core', 'jubatus_util_text'],
        language='c++',
        extra_compile_args=extra_compile_args)
    ]
    if cythonize:
        kwargs['ext_modules'] = cythonize(kwargs['ext_modules'])
    setup(**kwargs)


if __name__ == '__main__':
    _setup()
