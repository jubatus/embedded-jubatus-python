import json
import unittest

from embedded_jubatus import Burst
from jubatus.burst.types import Batch
from jubatus.burst.types import Document
from jubatus.burst.types import KeywordWithParams
from jubatus.burst.types import Window


CONFIG = {
    'parameter': {
        'window_batch_size': 5,
        'batch_interval': 10,
        'max_reuse_batch_num': 5,
        'costcut_threshold': -1,
        'result_window_rotate_size': 5
    },
    'method': 'burst',
}


class TestBurst(unittest.TestCase):
    def test_invalid_configs(self):
        self.assertRaises(TypeError, Burst)
        self.assertRaises(ValueError, Burst, {})
        self.assertRaises(RuntimeError, Burst, {'method': 'hoge'})
        invalid_config = dict(CONFIG)
        invalid_config['parameter'] = {'hoge': 0.1}
        self.assertRaises(RuntimeError, Burst, invalid_config)

    def test(self):
        x = Burst(CONFIG)
        balse_kwp = KeywordWithParams('balse', 1.001, 0.1)
        self.assertTrue(x.add_keyword(balse_kwp))
        k = x.get_all_keywords()
        self.assertEqual(1, len(k))
        self.assertTrue(isinstance(k[0], KeywordWithParams))
        self.assertEqual('balse', k[0].keyword)
        self.assertTrue(x.remove_keyword('balse'))
        self.assertFalse(x.remove_keyword('hoge'))
        self.assertEqual([], x.get_all_keywords())
        x.add_keyword(balse_kwp)
        self.assertTrue(x.remove_all_keywords())
        self.assertEqual([], x.get_all_keywords())
        x.add_keyword(balse_kwp)

        def add(pos, burst_count, nonburst_count):
            pos = float(pos)
            x.add_documents(
                [Document(pos, 'balse')] * burst_count +
                [Document(pos, 'jubatus')] * nonburst_count
            )

        # Time   Burst  Non-Burst
        add(1,       5,      30)
        add(11,     15,      50)
        add(21,    500,      10)
        add(31,   2000,      10)
        add(41,  22222,      40)
        add(51,     10,      10)
        add(61,      5,      25)

        ret = x.get_result('balse')
        self.assertTrue(isinstance(ret, Window))
        self.assertTrue(isinstance(ret.batches[0], Batch))
        ret = x.get_result_at('balse', 41)
        self.assertTrue(isinstance(ret, Window))
        self.assertTrue(isinstance(ret.batches[0], Batch))

        ret = x.get_all_bursted_results_at(41)
        self.assertTrue(ret, dict)
        self.assertEqual(['balse'], list(ret.keys()))
        self.assertTrue(isinstance(ret['balse'], Window))
        self.assertTrue(isinstance(ret['balse'].batches[0], Batch))

        model = x.save_bytes()
        x = Burst(CONFIG)
        x.load_bytes(model)
        self.assertEqual(CONFIG, json.loads(x.get_config()))
        self.assertEqual(1, len(x.get_all_keywords()))
        ret = x.get_result_at('balse', 41)
        self.assertTrue(isinstance(ret, Window))
        self.assertTrue(isinstance(ret.batches[0], Batch))

        st = x.get_status()
        self.assertTrue(isinstance(st, dict))
        self.assertEqual(len(st), 1)
        self.assertEqual(list(st.keys())[0], 'embedded')
        self.assertTrue(isinstance(st['embedded'], dict))
