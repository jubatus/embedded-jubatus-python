import json
import unittest

from embedded_jubatus import Graph
from jubatus.graph.types import Edge
from jubatus.graph.types import Node
from jubatus.graph.types import PresetQuery
from jubatus.graph.types import ShortestPathQuery


CONFIG = {
    "parameter": {
        "damping_factor": 0.9,
        "landmark_num": 5
    },
    "method": "graph_wo_index"
}


class TestGraph(unittest.TestCase):
    def test(self):
        g = Graph(CONFIG)
        self.assertTrue(g.add_shortest_path_query(PresetQuery([], [])))

        n0 = g.create_node()
        n1 = g.create_node()
        self.assertTrue(g.remove_node(g.create_node()))
        self.assertEqual('0', n0)
        self.assertEqual('1', n1)
        self.assertTrue(g.update_node(n0, {'hoge': 'piyo'}))

        e0 = g.create_edge(n0, Edge({'foo': 'bar'}, n0, n1))
        self.assertEqual(e0, 3)

        ni = g.get_node(n0)
        self.assertTrue(isinstance(ni, Node))
        self.assertEqual({'hoge': 'piyo'}, ni.property)
        self.assertEqual([], ni.in_edges)
        self.assertEqual([e0], ni.out_edges)
        ei = g.get_edge(n0, e0)
        self.assertTrue(isinstance(ei, Edge))
        self.assertEqual({'foo': 'bar'}, ei.property)
        self.assertEqual(n0, ei.source)
        self.assertEqual(n1, ei.target)
        self.assertTrue(g.update_edge(n0, e0, Edge({'foo': 'bar2'}, n0, n1)))
        ei = g.get_edge(n0, e0)
        self.assertEqual({'foo': 'bar2'}, ei.property)

        self.assertTrue(g.add_centrality_query(PresetQuery([], [])))
        self.assertTrue(g.update_index())
        self.assertTrue(isinstance(
            g.get_centrality(n0, 0, PresetQuery([], [])), float))
        self.assertTrue(g.remove_centrality_query(PresetQuery([], [])))

        self.assertEqual([n0, n1], g.get_shortest_path(
            ShortestPathQuery(n0, n1, 100, PresetQuery([], []))))
        model = g.save_bytes()
        self.assertTrue(g.remove_shortest_path_query(PresetQuery([], [])))

        g = Graph(CONFIG)
        g.load_bytes(model)
        self.assertEqual(CONFIG, json.loads(g.get_config()))
        self.assertEqual([n0, n1], g.get_shortest_path(
            ShortestPathQuery(n0, n1, 100, PresetQuery([], []))))
        self.assertEqual('4', g.create_node())

        st = g.get_status()
        self.assertTrue(isinstance(st, dict))
        self.assertEqual(len(st), 1)
        self.assertEqual(list(st.keys())[0], 'embedded')
        self.assertTrue(isinstance(st['embedded'], dict))
