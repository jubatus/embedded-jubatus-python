from libc.stdint cimport uint64_t
from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp.pair cimport pair

class _JubatusBase(object):
    def __init__(self, config):
        import json
        if isinstance(config, str):
            # JSON parse test
            json.loads(config)
        else:
            config = json.dumps(config, sort_keys=True, indent=4)
        self.get_config, self.dump, self.load, self.clear = (
            self._init(config.encode('utf8')))

    def get_status(self):
        raise RuntimeError

    def do_mix(self):
        raise RuntimeError

    def get_proxy_status(self):
        raise RuntimeError

    def get_name(self):
        raise RuntimeError

    def set_name(self, new_name):
        raise RuntimeError

    def get_client(self):
        raise RuntimeError

include 'types.pyx'
include 'anomaly.pyx'
include 'classifier.pyx'
include 'nearest_neighbor.pyx'
include 'recommender.pyx'
include 'regression.pyx'

class Anomaly(_JubatusBase, _AnomalyWrapper):
    pass

class Classifier(_JubatusBase, _ClassifierWrapper):
    pass

class NearestNeighbor(_JubatusBase, _NearestNeighborWrapper):
    pass

class Recommender(_JubatusBase, _RecommenderWrapper):
    pass

class Regression(_JubatusBase, _RegressionWrapper):
    pass
