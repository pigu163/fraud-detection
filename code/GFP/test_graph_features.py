"""Small NetworkX reference check for the vectorized graph features."""

import unittest

import networkx as nx
import pandas as pd

from graph_features import FEATURE_COLUMNS, build_gfp_features


class GraphFeatureTests(unittest.TestCase):
    def test_prior_hours_match_networkx_and_preserve_order(self):
        rows = pd.DataFrame(
            {
                "hours_elapsed": [3, 1, 2, 1, 2, 3, 1],
                "initiating_customer": ["A", "A", "A", "B", "B", "B", "A"],
                "recipient_customer": ["B", "B", "C", "C", "C", "A", "B"],
                "is_fraud": [1, 0, 0, 0, 0, 0, 0],
            },
            index=[10, 20, 30, 40, 50, 60, 70],
        )
        actual = build_gfp_features(rows)
        self.assertTrue(actual.index.equals(rows.index))
        self.assertEqual(list(actual.columns), list(FEATURE_COLUMNS))
        for index, row in rows.iterrows():
            graph = nx.MultiDiGraph()
            for _, old in rows[rows.hours_elapsed.lt(row.hours_elapsed)].iterrows():
                graph.add_edge(old.initiating_customer, old.recipient_customer)
            source = row.initiating_customer
            target = row.recipient_customer
            outgoing = list(graph.out_edges(source))
            incoming = list(graph.in_edges(target))
            expected = [
                len(outgoing),
                len({v for _, v in outgoing}),
                len(incoming),
                len({u for u, _ in incoming}),
                sum(u == source and v == target for u, v in graph.edges()),
            ]
            self.assertEqual(actual.loc[index].tolist(), expected)

    def test_future_and_label_changes_do_not_change_earlier_features(self):
        rows = pd.DataFrame(
            {
                "hours_elapsed": [1, 2],
                "initiating_customer": ["A", "A"],
                "recipient_customer": ["B", "B"],
                "is_fraud": [0, 1],
            }
        )
        first = build_gfp_features(rows)
        changed = rows.copy()
        changed.loc[1, "recipient_customer"] = "C"
        changed["is_fraud"] = 1 - changed["is_fraud"]
        second = build_gfp_features(changed)
        self.assertEqual(first.iloc[0].tolist(), second.iloc[0].tolist())


if __name__ == "__main__":
    unittest.main()
