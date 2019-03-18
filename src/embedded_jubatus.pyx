from libc.stdint cimport uint64_t
from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp.pair cimport pair
from libcpp.map cimport map
from cython.operator cimport dereference
from cython.operator cimport preincrement

cimport numpy as c_np

from _wrapper cimport _Anomaly
from _wrapper cimport _Bandit
from _wrapper cimport _Burst
from _wrapper cimport _Classifier
from _wrapper cimport _Clustering
from _wrapper cimport _NearestNeighbor
from _wrapper cimport _Recommender
from _wrapper cimport _Regression
from _wrapper cimport _Stat
from _wrapper cimport _Weight
from _wrapper cimport _Graph
from _wrapper cimport arm_info
from _wrapper cimport classify_result_elem
from _wrapper cimport datum
from _wrapper cimport keyword_params
from _wrapper cimport keyword_with_params
from _wrapper cimport lexical_cast
from _wrapper cimport sfv_t
from _wrapper cimport status_t
from _wrapper cimport prop_t
from _wrapper cimport node_id_t
from _wrapper cimport edge_id_t
from _wrapper cimport node_info
from _wrapper cimport edge_info
from _wrapper cimport preset_query
from _wrapper cimport indexed_point

from jubatus.anomaly.types import IdWithScore as AnomalyIdWithScore
from jubatus.bandit.types import ArmInfo
from jubatus.burst.types import Batch
from jubatus.burst.types import Document
from jubatus.burst.types import KeywordWithParams
from jubatus.burst.types import Window
from jubatus.classifier.types import EstimateResult
from jubatus.classifier.types import LabeledDatum
from jubatus.clustering.types import WeightedDatum
from jubatus.clustering.types import WeightedIndex
from jubatus.common.datum import Datum
from jubatus.nearest_neighbor.types import IdWithScore as NNIdWithScore
from jubatus.recommender.types import IdWithScore as RecommenderIdWithScore
from jubatus.regression.types import ScoredDatum
from jubatus.weight.types import Feature
from jubatus.graph.types import Edge
from jubatus.graph.types import Node
from jubatus.graph.types import PresetQuery
from jubatus.graph.types import Query
from jubatus.graph.types import ShortestPathQuery


cdef class _JubatusBase:
    cdef object _type
    cdef int _model_ver

    def __init__(self, config):
        import json
        if isinstance(config, str):
            # loads config from file
            config = open(config, 'rb').read().decode('utf8')
            # JSON parse test
            json.loads(config)
        else:
            config = json.dumps(config, sort_keys=True, indent=4)
        self._init(config.encode('utf8'))

    def _get_model_path(self, id_):
        host, port, typ = '127.0.0.1', 0, self._type
        if str != bytes and isinstance(typ, bytes):
            typ = typ.decode('ascii')
        path = '/tmp/{host}_{port}_{type}_{id}.jubatus'.format(
            host=host, port=port, type=typ, id=id_)
        return (path, '{host}_{port}'.format(host=host, port=port))

    def load(self, id_):
        path, name = self._get_model_path(id_)
        with open(path, 'rb') as f:
            self.load_bytes(f.read())
        return True

    def save(self, id_):
        path, name = self._get_model_path(id_)
        with open(path, 'wb') as f:
            f.write(self.save_bytes())
        return {name: path}

    def do_mix(self):
        return True

    def get_proxy_status(self):
        raise RuntimeError

    def get_name(self):
        return ''

    def set_name(self, new_name):
        raise RuntimeError('Unsupported')

    def get_client(self):
        raise RuntimeError

    def __getstate__(self):
        return (self.get_config(), self.save_bytes())

    def __setstate__(self, state):
        import json
        cfg, model = state
        self.__init__(json.loads(cfg))
        self.load_bytes(model)


include 'types.pyx'
include 'anomaly.pyx'
include 'bandit.pyx'
include 'burst.pyx'
include 'classifier.pyx'
include 'clustering.pyx'
include 'nearest_neighbor.pyx'
include 'recommender.pyx'
include 'regression.pyx'
include 'stat.pyx'
include 'weight.pyx'
include 'graph.pyx'
