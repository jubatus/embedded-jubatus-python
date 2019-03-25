import json
import math
import random
import unittest

from embedded_jubatus import Stat


class TestStat(unittest.TestCase):
    def test_invalid_configs(self):
        self.assertRaises(TypeError, Stat)
        self.assertRaises(RuntimeError, Stat, {})
        self.assertRaises(RuntimeError, Stat, {'window_size': 'hoge'})

    def test(self):
        window_size = 2000
        config = {
            'window_size': window_size
        }
        s = Stat(config)

        x_avg, x_stddev = 0.0, 1.0
        y_avg, y_stddev = 10.0, 10.0
        x_list, y_list = [], []

        for _ in range(window_size // 2):
            xv = random.gauss(x_avg, x_stddev)
            yv = random.gauss(y_avg, y_stddev)
            self.assertTrue(s.push('x', xv))
            self.assertTrue(s.push('y', yv))
            x_list.append(xv)
            y_list.append(yv)

        x_sum, y_sum = sum(x_list), sum(y_list)
        x_avg, y_avg = x_sum / len(x_list), y_sum / len(y_list)
        self.assertAlmostEqual(x_sum, s.sum('x'))
        self.assertAlmostEqual(y_sum, s.sum('y'))
        self.assertAlmostEqual(
            math.sqrt(sum([(x - x_avg)**2 for x in x_list]) / len(x_list)),
            s.stddev('x'))
        self.assertAlmostEqual(
            math.sqrt(sum([(y - y_avg)**2 for y in y_list]) / len(y_list)),
            s.stddev('y'))
        self.assertAlmostEqual(max(x_list), s.max('x'))
        self.assertAlmostEqual(max(y_list), s.max('y'))
        self.assertAlmostEqual(min(x_list), s.min('x'))
        self.assertAlmostEqual(min(y_list), s.min('y'))
        self.assertTrue(isinstance(s.entropy(''), float))
        self.assertTrue(isinstance(s.moment('x', 1, 1.0), float))

        model = s.save_bytes()
        s = Stat(config)
        s.load_bytes(model)
        self.assertEqual(config, json.loads(s.get_config()))
        self.assertAlmostEqual(x_sum, s.sum('x'))

        st = s.get_status()
        self.assertTrue(isinstance(st, dict))
        self.assertEqual(len(st), 1)
        self.assertEqual(list(st.keys())[0], 'embedded')
        self.assertTrue(isinstance(st['embedded'], dict))
