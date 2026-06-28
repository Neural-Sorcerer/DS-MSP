"""Covisibility graph: connected components + min-weight shortest path."""

from ds_msp.rig.graph import (connected_components, covis_weights, shortest_path)


def test_components_match_connectivity():
    # two clusters: {0,1,2} and {3,4}
    counts = {(0, 1): 5, (1, 2): 3, (3, 4): 2}
    w = covis_weights(counts)
    comps = connected_components([0, 1, 2, 3, 4], w)
    comps = sorted(comps, key=lambda c: c[0])
    assert comps == [[0, 1, 2], [3, 4]]
    # each component sorted -> min id first (reference choice)
    assert comps[0][0] == 0 and comps[1][0] == 3


def test_shortest_path_prefers_more_coobservations():
    # 0-2 direct edge weak (1 obs -> weight 1.0); 0-1-2 strong (10 obs each -> 0.2 total)
    counts = {(0, 2): 1, (0, 1): 10, (1, 2): 10}
    w = covis_weights(counts)
    assert shortest_path(0, 2, w) == [0, 1, 2]


def test_isolated_node_is_own_component():
    w = covis_weights({(0, 1): 4})
    comps = connected_components([0, 1, 2], w)
    assert [2] in comps
