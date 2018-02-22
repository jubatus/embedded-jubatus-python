|Travis|_ |PyPi|_

.. |Travis| image:: https://api.travis-ci.org/jubatus/embedded-jubatus-python.svg?branch=master
.. _Travis: https://travis-ci.org/jubatus/embedded-jubatus-python

.. |PyPi| image:: https://badge.fury.io/py/embedded_jubatus.svg
.. _PyPi: https://pypi.python.org/pypi/embedded_jubatus

embedded-jubatus-python
=======================

embedded-jubatus-python is a Python bridge to call `Jubatus Core <https://github.com/jubatus/jubatus_core>`_ C++ library.

The interface of embedded-jubatus-python is the same as that of RPC client classes.
See the `API Reference <http://jubat.us/en/api/>`_ for details.

Install
-------

::

  pip install Cython
  pip install embedded_jubatus

Requirements
------------

* Python 2.7, 3.3, 3.4 or 3.5.
* `Jubatus <http://jubat.us/en/quickstart.html>`_ needs to be installed.

Limitations
-----------

* The following methods are currently unavailable: ``get_status``, ``get_proxy_status``, ``do_mix``, ``get_name``, ``set_name`` and ``get_client``.
* ``save`` method saves the model file as ``/tmp/127.0.0.1_0_${type}_${id}.jubatus``, where ``${type}`` is a name of the service (``classifier``, ``recommender``, etc.) and ``${id}`` is the value specified as an argument to the ``save`` method.
  If you need to save the model into a different location, call ``save_bytes`` method, which returns the model data as a binary, then save the returned bytes to the preferred location.
  The same rule applies to ``load`` / ``load_bytes`` methods.

License
-------

LGPL 2.1
