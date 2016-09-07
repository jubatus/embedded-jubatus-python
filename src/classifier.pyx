from _wrapper cimport _Classifier
from _wrapper cimport classify_result_elem

from jubatus.classifier.types import EstimateResult
from jubatus.classifier.types import LabeledDatum

cdef class _ClassifierWrapper:
    cdef _Classifier *_handle

    def __cinit__(self):
        self._handle = NULL

    def __dealloc__(self):
        if self._handle != NULL:
            del self._handle

    def _init(self, config):
        self._handle = new _Classifier(config)
        typ, ver = b'classifier', 1
        return (
            lambda: self._handle.get_config().decode('utf8'),
            lambda: self._handle.dump(typ, ver),
            lambda x: self._handle.load(x, typ, ver),
            lambda: self._handle.clear(),
        )

    def train(self, data):
        cdef string l
        cdef datum d
        if isinstance(data, list):
            if len(data) == 0:
                return 0
            if isinstance(data[0], LabeledDatum):
                for r in data:
                    label, datum = r.label, r.data
                    l = label.encode('utf8')
                    datum_py2native(datum, d)
                    self._handle.train(l, d)
            elif isinstance(data[0], (tuple, list)) and len(data[0]) == 2:
                for label, datum in data:
                    l = label.encode('utf8')
                    datum_py2native(datum, d)
                    self._handle.train(l, d)
            else:
                raise ValueError
            return len(data)
        else:
            raise ValueError

    def classify(self, data):
        cdef datum d
        cdef vector[classify_result_elem] r
        if isinstance(data, list):
            ret = []
            for x in data:
                datum_py2native(x, d)
                r = self._handle.classify(d)
                ret.append([
                    EstimateResult(r[i].label.decode('utf8'), r[i].score)
                    for i in range(r.size())
                ])
            return ret
        else:
            raise ValueError

    def get_labels(self):
        cdef vector[pair[string, uint64_t]] ret = self._handle.get_labels()
        return {ret[i].first.decode('utf8'): ret[i].second for i in range(ret.size())}

    def set_label(self, new_label):
        return self._handle.set_label(new_label.encode('utf8'))

    def delete_label(self, target_label):
        return self._handle.delete_label(target_label.encode('utf8'))
