#!/bin/bash
set -e -x

function build_wheels() {
    if [ -z "${JUBATUS_VERSION}" ]; then
        JUBATUS_VERSION=${TRAVIS_TAG:-master}
    fi

    rpm -Uvh http://download.jubat.us/yum/rhel/6/stable/x86_64/jubatus-release-6-2.el6.x86_64.rpm
    yum install -y msgpack-devel oniguruma-devel

    curl -OL https://github.com/jubatus/jubatus_core/archive/${JUBATUS_VERSION}.tar.gz
    tar xf ${JUBATUS_VERSION}.tar.gz
    cd jubatus_core-${JUBATUS_VERSION}
    ./waf configure --prefix=/usr
    ./waf build 
    ./waf install
    cd /io

    for PYBIN in /opt/python/*m/bin; do
        "${PYBIN}/pip" install cython
        "${PYBIN}/pip" install -r /io/requirements.txt
        "${PYBIN}/pip" wheel /io/ -w wheelhouse/
    done

    for whl in wheelhouse/*.whl; do
        auditwheel repair "$whl" --plat manylinux2010_x86_64 -w /io/wheelhouse/
    done

    /opt/python/cp37-cp37m/bin/python ./setup.py sdist

    for PYBIN in /opt/python/*m/bin; do
        V=${PYBIN%/bin}
        V=${V#/opt/python/}
        "${PYBIN}/pip" install /io/wheelhouse/embedded_jubatus-*-${V}-manylinux2010_x86_64.whl scipy
        "${PYBIN}/python" ./setup.py test
    done
}

if [ "$1" = "build_wheels" ]; then
    build_wheels
else
    docker pull quay.io/pypa/manylinux2010_x86_64
    docker run --rm \
           -e JUBATUS_VERSION=${JUBATUS_VERSION} \
           -e TRAVIS_TAG=${TRAVIS_TAG} \
           -v $(pwd):/io \
           quay.io/pypa/manylinux2010_x86_64 \
           /io/build-wheels.sh build_wheels
fi
