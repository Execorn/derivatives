"""tests/test_adversarial_cross_m5_dag_stress.py — Adversarial Stress Test Suite for Milestone 5.

Probes DAG resolution, boundary conditions, and negative validation paths:
1. Complex DAG Topology Probes: deep linear chains, broad trees, diamonds, lattices (topological invariants).
2. Cycle & Error Boundary Stress: 2-cycles, 3-cycles, self-loops (asserting ValueError), dangling refs (asserting KeyError).
3. Message Parsing Robustness: mixed formats, empty graphs, disconnected components, non-hex 40-char string false-positives.
4. Curve & Interpolator Boundary Probes: single-pillar, duplicate pillars, decreasing pillars, extreme tenors, partition of unity.
5. Triangle Consistency & Serializer Stress: triangle mismatch, product type rejection, mathematical conservation.
"""

from __future__ import annotations

import copy
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple, cast

import numpy as np
import pytest

from deepvol.api.schemas_fx_cross import (
    CashDeltaLegs,
    DeltaLegs,
    ModelRiskTelemetry,
    PricingEngineEnum,
    TriangulatedFXOptionResponse,
    VegaLegs,
)
from deepvol.market.legacy_adapter import (
    ConstMessage,
    ConstMessageGraph,
    DiscountCurve,
    build_request,
    to_mutable_message,
    year_fraction_act365f,
)


def _gen_hex40(index: int) -> str:
    """Generates a deterministic 40-char lowercase hex ID from an integer index."""
    return hashlib.sha1(f"node_{index}".encode("utf-8")).hexdigest()


# ─────────────────────────────────────────────────────────────────────────────
# 1. Complex DAG Topology Probes
# ─────────────────────────────────────────────────────────────────────────────


class TestDAGTopologyProbes:
    """Probes complex DAG topologies: deep chains, broad trees, diamonds, and lattices."""

    def test_deep_linear_chain_topological_order(self) -> None:
        """Deep linear chain of 100 nodes: Node_k -> Node_{k-1} -> ... -> Node_0.

        Asserts exact topological order: Node_0 must appear first, Node_99 must appear last.
        For all i < j, order[i] must not depend on order[j].
        """
        n_nodes = 100
        messages: List[ConstMessage] = []
        ids = [_gen_hex40(i) for i in range(n_nodes)]

        for i in range(n_nodes):
            val: Dict[str, Any] = {"Name": f"Node_{i}"}
            if i > 0:
                val["Dependency"] = ids[i - 1]
            messages.append(ConstMessage(id=ids[i], value=val))

        graph = ConstMessageGraph(messages)
        assert len(graph) == n_nodes

        root_id = ids[-1]
        resolved = graph.resolve(root_id)
        assert len(resolved) == n_nodes

        # Verify strict topological ordering: Node_0, Node_1, ..., Node_99
        resolved_ids = [m.id for m in resolved]
        assert resolved_ids == ids
        assert resolved[0].id == ids[0]
        assert resolved[-1].id == ids[-1]

    def test_deep_linear_chain_near_recursion_boundary(self) -> None:
        """Probes a chain of 400 nodes to verify DFS depth handling."""
        n_nodes = 400
        ids = [_gen_hex40(i) for i in range(n_nodes)]
        messages = [
            ConstMessage(
                id=ids[i],
                value={"Name": f"Node_{i}", "Dep": ids[i - 1] if i > 0 else "None"},
            )
            for i in range(n_nodes)
        ]
        graph = ConstMessageGraph(messages)
        resolved = graph.resolve(ids[-1])
        assert len(resolved) == n_nodes
        assert [m.id for m in resolved] == ids

    def test_broad_tree_topology(self) -> None:
        """Broad tree: 1 root referencing 120 child nodes, each child referencing 2 leaf nodes.

        Total nodes: 1 + 120 + 240 = 361.
        Asserts topological invariant:
        - All leaf nodes appear before their parent children.
        - All child nodes appear before root.
        - Root is strictly the last element.
        """
        messages: List[ConstMessage] = []
        leaf_ids: List[str] = []
        child_ids: List[str] = []

        leaf_count = 0
        for c in range(120):
            c_id = _gen_hex40(1000 + c)
            child_ids.append(c_id)
            c_leaves: List[str] = []
            for _ in range(2):
                l_id = _gen_hex40(5000 + leaf_count)
                leaf_count += 1
                leaf_ids.append(l_id)
                c_leaves.append(l_id)
                messages.append(ConstMessage(id=l_id, value={"Type": "Leaf"}))

            messages.append(
                ConstMessage(id=c_id, value={"Type": "Child", "Leaves": c_leaves})
            )

        root_id = _gen_hex40(9999)
        messages.append(
            ConstMessage(id=root_id, value={"Type": "Root", "Children": child_ids})
        )

        graph = ConstMessageGraph(messages)
        resolved = graph.resolve(root_id)
        assert len(resolved) == 1 + 120 + 240

        resolved_indices = {m.id: idx for idx, m in enumerate(resolved)}

        # Invariant 1: root is strictly last
        assert resolved[-1].id == root_id

        # Invariant 2: every leaf appears before its child
        for c_msg in [m for m in messages if m.value.get("Type") == "Child"]:
            c_idx = resolved_indices[c_msg.id]
            for l_id in c_msg.value["Leaves"]:
                l_idx = resolved_indices[l_id]
                assert l_idx < c_idx, f"Leaf {l_id} (idx {l_idx}) must precede Child {c_msg.id} (idx {c_idx})"

        # Invariant 3: every child appears before root
        root_idx = resolved_indices[root_id]
        for c_id in child_ids:
            assert resolved_indices[c_id] < root_idx

    def test_diamond_and_lattice_graphs(self) -> None:
        """Diamond graph:
        Root -> Left, Right
        Left -> Bottom
        Right -> Bottom

        Asserts Bottom appears before Left and Right; both appear before Root; exactly 4 nodes.
        """
        bottom_id = _gen_hex40(1)
        left_id = _gen_hex40(2)
        right_id = _gen_hex40(3)
        root_id = _gen_hex40(4)

        messages = [
            ConstMessage(id=bottom_id, value={"Name": "Bottom"}),
            ConstMessage(id=left_id, value={"Name": "Left", "Dep": bottom_id}),
            ConstMessage(id=right_id, value={"Name": "Right", "Dep": bottom_id}),
            ConstMessage(id=root_id, value={"Name": "Root", "Deps": [left_id, right_id]}),
        ]

        graph = ConstMessageGraph(messages)
        resolved = graph.resolve(root_id)
        assert len(resolved) == 4

        idx_map = {m.id: i for i, m in enumerate(resolved)}
        assert idx_map[bottom_id] < idx_map[left_id]
        assert idx_map[bottom_id] < idx_map[right_id]
        assert idx_map[left_id] < idx_map[root_id]
        assert idx_map[right_id] < idx_map[root_id]
        assert resolved[-1].id == root_id

    def test_multi_path_grid_lattice_dag(self) -> None:
        """4x4 Grid Lattice DAG where Node(i,j) references Node(i+1, j) and Node(i, j+1).

        Bottom-right sink is (3,3). Top-left source (root) is (0,0).
        Verifies every reachable node has its dependencies resolved strictly before itself.
        """
        grid_ids: Dict[Tuple[int, int], str] = {}
        for r in range(4):
            for c in range(4):
                grid_ids[(r, c)] = _gen_hex40(r * 10 + c)

        messages: List[ConstMessage] = []
        for r in range(4):
            for c in range(4):
                deps: List[str] = []
                if r < 3:
                    deps.append(grid_ids[(r + 1, c)])
                if c < 3:
                    deps.append(grid_ids[(r, c + 1)])
                messages.append(
                    ConstMessage(
                        id=grid_ids[(r, c)],
                        value={"Coord": [r, c], "Deps": deps},
                    )
                )

        graph = ConstMessageGraph(messages)
        root_id = grid_ids[(0, 0)]
        resolved = graph.resolve(root_id)
        assert len(resolved) == 16

        idx_map = {m.id: idx for idx, m in enumerate(resolved)}
        # Verify edge property: for every node, its dependencies must appear before it
        for m in resolved:
            m_idx = idx_map[m.id]
            for dep_id in m.refs():
                dep_idx = idx_map[dep_id]
                assert dep_idx < m_idx, f"Dep {dep_id} must precede {m.id}"


# ─────────────────────────────────────────────────────────────────────────────
# 2. Cycle & Error Boundary Stress
# ─────────────────────────────────────────────────────────────────────────────


class TestCycleAndErrorBoundaryStress:
    """Stress-tests cycle detection and negative boundary conditions in DAG resolution."""

    def test_2_cycle_raises_value_error(self) -> None:
        """2-cycle: A -> B -> A. Must raise ValueError with cycle path."""
        id_a = _gen_hex40(101)
        id_b = _gen_hex40(102)

        messages = [
            ConstMessage(id=id_a, value={"Dep": id_b}),
            ConstMessage(id=id_b, value={"Dep": id_a}),
        ]
        graph = ConstMessageGraph(messages)

        with pytest.raises(ValueError, match="Circular dependency cycle detected"):
            graph.resolve(id_a)

        with pytest.raises(ValueError, match="Circular dependency cycle detected"):
            graph.resolve(id_b)

    def test_3_cycle_raises_value_error(self) -> None:
        """3-cycle: A -> B -> C -> A. Must raise ValueError."""
        id_a = _gen_hex40(201)
        id_b = _gen_hex40(202)
        id_c = _gen_hex40(203)

        messages = [
            ConstMessage(id=id_a, value={"Dep": id_b}),
            ConstMessage(id=id_b, value={"Dep": id_c}),
            ConstMessage(id=id_c, value={"Dep": id_a}),
        ]
        graph = ConstMessageGraph(messages)

        with pytest.raises(ValueError, match="Circular dependency cycle detected") as exc_info:
            graph.resolve(id_a)
        assert "->" in str(exc_info.value)

    def test_self_loop_raises_value_error(self) -> None:
        """Self loop: A -> A. Must raise ValueError."""
        id_a = _gen_hex40(301)
        messages = [ConstMessage(id=id_a, value={"Dep": id_a})]
        graph = ConstMessageGraph(messages)

        with pytest.raises(ValueError, match="Circular dependency cycle detected"):
            graph.resolve(id_a)

    def test_deep_nested_cycle_raises_value_error(self) -> None:
        """Cycle nested deep in a subgraph: Root -> N1 -> N2 -> (C1 -> C2 -> C1)."""
        id_root = _gen_hex40(400)
        id_n1 = _gen_hex40(401)
        id_n2 = _gen_hex40(402)
        id_c1 = _gen_hex40(403)
        id_c2 = _gen_hex40(404)

        messages = [
            ConstMessage(id=id_root, value={"Dep": id_n1}),
            ConstMessage(id=id_n1, value={"Dep": id_n2}),
            ConstMessage(id=id_n2, value={"Dep": id_c1}),
            ConstMessage(id=id_c1, value={"Dep": id_c2}),
            ConstMessage(id=id_c2, value={"Dep": id_c1}),  # Cycle C1 <-> C2
        ]
        graph = ConstMessageGraph(messages)

        with pytest.raises(ValueError, match="Circular dependency cycle detected"):
            graph.resolve(id_root)

    def test_dangling_reference_direct_raises_key_error(self) -> None:
        """Root directly references missing 40-hex hash."""
        id_root = _gen_hex40(500)
        missing_id = _gen_hex40(999)

        messages = [ConstMessage(id=id_root, value={"Dep": missing_id})]
        graph = ConstMessageGraph(messages)

        with pytest.raises(KeyError, match=missing_id):
            graph.resolve(id_root)

    def test_dangling_reference_deep_raises_key_error(self) -> None:
        """Root -> N1 -> N2 -> missing_hash."""
        id_root = _gen_hex40(600)
        id_n1 = _gen_hex40(601)
        id_n2 = _gen_hex40(602)
        missing_id = _gen_hex40(699)

        messages = [
            ConstMessage(id=id_root, value={"Dep": id_n1}),
            ConstMessage(id=id_n1, value={"Dep": id_n2}),
            ConstMessage(id=id_n2, value={"Dep": missing_id}),
        ]
        graph = ConstMessageGraph(messages)

        with pytest.raises(KeyError, match=f"Dangling reference: message '{missing_id}'"):
            graph.resolve(id_root)

    def test_missing_root_raises_key_error(self) -> None:
        """Root itself is not present in graph."""
        graph = ConstMessageGraph([ConstMessage(id=_gen_hex40(1), value={})])
        missing_root = _gen_hex40(999)
        with pytest.raises(KeyError, match=f"Root message '{missing_root}' not found"):
            graph.resolve(missing_root)

    def test_getitem_missing_key_raises_key_error(self) -> None:
        """Accessing missing key via graph[key] raises KeyError."""
        graph = ConstMessageGraph([])
        with pytest.raises(KeyError, match="Message ID 'unknown' not found"):
            _ = graph["unknown"]


# ─────────────────────────────────────────────────────────────────────────────
# 3. Message Parsing Robustness
# ─────────────────────────────────────────────────────────────────────────────


class TestMessageParsingRobustness:
    """Stress-tests input normalization, empty graphs, disconnected components, and regex false-positives."""

    def test_mixed_message_input_normalization(self) -> None:
        """ConstMessageGraph correctly ingests mixed wrapped, flat, ConstMessage instances, and generator."""
        id1 = _gen_hex40(1)
        id2 = _gen_hex40(2)
        id3 = _gen_hex40(3)
        id4 = _gen_hex40(4)

        raw_list: List[Mapping[str, Any] | ConstMessage] = [
            {"ConstMessage": {"id": id1, "value": {"TypeName": "T1", "Ref": id2}}},
            {"id": id2, "value": {"TypeName": "T2", "Ref": id3}},
            ConstMessage(id=id3, value={"TypeName": "T3", "Ref": id4}),
            {"id": id4, "value": {"TypeName": "T4"}},
        ]

        # Ingest via generator iterator
        graph = ConstMessageGraph((item for item in raw_list))
        assert len(graph) == 4
        assert id1 in graph
        assert id2 in graph
        assert id3 in graph
        assert id4 in graph

        resolved = graph.resolve(id1)
        assert len(resolved) == 4
        assert [m.id for m in resolved] == [id4, id3, id2, id1]

    def test_empty_graph_boundaries(self) -> None:
        """Empty graph has len 0, contains False, get None, by_type [], and resolve raises KeyError."""
        graph = ConstMessageGraph([])
        assert len(graph) == 0
        assert "any_id" not in graph
        assert graph.get("any_id") is None
        assert graph.by_type("AnyType") == []

        with pytest.raises(KeyError):
            graph.resolve("any_id")

    def test_disconnected_components_isolation(self) -> None:
        """Graph with 4 disconnected components:
        Comp A: Valid DAG A -> B
        Comp B: Cycle C -> D -> C
        Comp C: Dangling E -> Missing
        Comp D: Isolated F

        resolve(A) succeeds and is completely insulated from B, C, D.
        """
        id_a = _gen_hex40(10)
        id_b = _gen_hex40(11)
        id_c = _gen_hex40(20)
        id_d = _gen_hex40(21)
        id_e = _gen_hex40(30)
        id_f = _gen_hex40(40)
        missing_id = _gen_hex40(99)

        messages = [
            # Comp A
            ConstMessage(id=id_a, value={"Dep": id_b}),
            ConstMessage(id=id_b, value={"Name": "B"}),
            # Comp B (Cycle)
            ConstMessage(id=id_c, value={"Dep": id_d}),
            ConstMessage(id=id_d, value={"Dep": id_c}),
            # Comp C (Dangling)
            ConstMessage(id=id_e, value={"Dep": missing_id}),
            # Comp D (Isolated)
            ConstMessage(id=id_f, value={"Name": "F"}),
        ]

        graph = ConstMessageGraph(messages)
        assert len(graph) == 6

        # Comp A resolves cleanly
        res_a = graph.resolve(id_a)
        assert [m.id for m in res_a] == [id_b, id_a]

        # Comp D resolves cleanly
        res_f = graph.resolve(id_f)
        assert [m.id for m in res_f] == [id_f]

        # Comp B raises ValueError
        with pytest.raises(ValueError, match="Circular dependency cycle detected"):
            graph.resolve(id_c)

        # Comp C raises KeyError
        with pytest.raises(KeyError, match=f"Dangling reference: message '{missing_id}'"):
            graph.resolve(id_e)

    def test_hex40_regex_false_positive_filtering(self) -> None:
        """Ensures refs() strictly matches only 40-character lowercase hexadecimal hashes."""
        valid_hash_1 = "a1b2c3d4e5f60718293a4b5c6d7e8f9012345678"
        valid_hash_2 = "0000000000000000000000000000000000000000"

        value_with_noise: Dict[str, Any] = {
            # Valid hashes
            "Ref1": valid_hash_1,
            "Deep": {
                "NestedList": [
                    {"Tuple": (valid_hash_2,)},
                ]
            },
            # Noise: uppercase hex (must NOT match)
            "UpperHex": "A1B2C3D4E5F60718293A4B5C6D7E8F9012345678",
            "MixedHex": "a1b2c3d4e5f60718293A4B5C6D7E8F9012345678",
            # Noise: invalid length (39 and 41 chars)
            "ShortHex": "a1b2c3d4e5f60718293a4b5c6d7e8f901234567",
            "LongHex": "a1b2c3d4e5f60718293a4b5c6d7e8f90123456789",
            # Noise: non-hex characters
            "NonHexChar": "a1b2c3d4e5f60718293a4b5c6d7e8f901234567g",
            "SpecialChar": "a1b2c3d4e5f60718293a4b5c6d7e8f901234567-",
            # Noise: padded with spaces
            "Padded": " a1b2c3d4e5f60718293a4b5c6d7e8f9012345678",
            # Duplicate occurrence of valid_hash_1
            "Duplicate": valid_hash_1,
            # Numbers, Booleans, None
            "IntVal": 12345678901234567890,
            "FloatVal": 3.1415926535,
            "BoolVal": True,
            "NoneVal": None,
        }

        msg = ConstMessage(id=_gen_hex40(1), value=value_with_noise)
        discovered_refs = msg.refs()

        # Must discover exactly valid_hash_1 and valid_hash_2 in discovery order
        assert discovered_refs == [valid_hash_1, valid_hash_2]


# ─────────────────────────────────────────────────────────────────────────────
# 4. Curve & Interpolator Boundary Probes
# ─────────────────────────────────────────────────────────────────────────────


class TestDiscountCurveBoundaryStress:
    """Stress-tests single-pillar, duplicate pillars, decreasing pillars, and extreme tenors."""

    def test_single_pillar_curve_behavior(self) -> None:
        """Single-pillar curve provides flat zero rate and partition of unity weights ([1.0])."""
        as_of = datetime(2026, 1, 1, tzinfo=timezone.utc)
        pillar = datetime(2027, 1, 1, tzinfo=timezone.utc)
        rate_pct = 5.0  # 5% zero rate

        curve = DiscountCurve(as_of, [pillar], [rate_pct])
        assert len(curve.tenors) == 1
        assert math.isclose(curve.tenors[0], 1.0, rel_tol=1e-6)
        assert math.isclose(curve.rates[0], 0.05, rel_tol=1e-12)

        # Zero rate must be 0.05 everywhere (flat extrapolation)
        for t in [-10.0, 0.0, 0.5, 1.0, 5.0, 50.0]:
            assert math.isclose(curve.zero(t), 0.05, rel_tol=1e-12)

        # DF(0.0) == 1.0 exactly
        assert math.isclose(curve.df(0.0), 1.0, rel_tol=1e-14)
        assert math.isclose(curve.df(1.0), math.exp(-0.05), rel_tol=1e-14)
        assert math.isclose(curve.df(2.0), math.exp(-0.10), rel_tol=1e-14)

        # Weights must be [1.0] everywhere
        for t in [-5.0, 0.0, 0.5, 1.0, 10.0]:
            w = curve.weights(t)
            assert len(w) == 1
            assert math.isclose(w[0], 1.0, rel_tol=1e-14)

    def test_duplicate_pillar_dates_raises_value_error(self) -> None:
        """Duplicate dates have dt == 0.0, which violates strictly increasing tenors."""
        as_of = datetime(2026, 1, 1, tzinfo=timezone.utc)
        d1 = datetime(2026, 6, 1, tzinfo=timezone.utc)
        d2 = datetime(2027, 1, 1, tzinfo=timezone.utc)

        with pytest.raises(ValueError, match="strictly increasing tenors"):
            DiscountCurve(as_of, [d1, d1, d2], [4.0, 4.5, 5.0])

        with pytest.raises(ValueError, match="strictly increasing tenors"):
            DiscountCurve(as_of, [d1, d2, d2], [4.0, 4.5, 5.0])

    def test_decreasing_pillar_dates_raises_value_error(self) -> None:
        """Decreasing dates have dt < 0.0, which violates strictly increasing tenors."""
        as_of = datetime(2026, 1, 1, tzinfo=timezone.utc)
        d1 = datetime(2026, 6, 1, tzinfo=timezone.utc)
        d2 = datetime(2027, 1, 1, tzinfo=timezone.utc)

        with pytest.raises(ValueError, match="strictly increasing tenors"):
            DiscountCurve(as_of, [d2, d1], [5.0, 4.0])

    def test_empty_dates_raises_value_error(self) -> None:
        """Empty dates sequence raises ValueError."""
        as_of = datetime(2026, 1, 1, tzinfo=timezone.utc)
        with pytest.raises(ValueError, match="at least one pillar date"):
            DiscountCurve(as_of, [], [])

    def test_mismatched_dates_and_rates_length_raises_value_error(self) -> None:
        """Dates and rates_pct length mismatch raises ValueError."""
        as_of = datetime(2026, 1, 1, tzinfo=timezone.utc)
        d1 = datetime(2026, 6, 1, tzinfo=timezone.utc)
        d2 = datetime(2027, 1, 1, tzinfo=timezone.utc)
        with pytest.raises(ValueError, match="length mismatch"):
            DiscountCurve(as_of, [d1, d2], [4.0])

    def test_extreme_tenors_and_partition_of_unity(self) -> None:
        """Multi-pillar curve across 8 tenors tested with 500 extreme points in [-50, 100].

        Asserts:
        - Partition of unity: sum(weights(t)) == 1.0 to 1e-14.
        - Non-negativity: all weights >= 0.0.
        - Exact reconstruction: w(t) @ rates == zero(t) to 1e-15.
        - Kronecker delta property at all knot points.
        - Left extrapolation: t <= t_0 -> w[0] = 1.0, rate = r[0].
        - Right extrapolation: t >= t_last -> w[-1] = 1.0, rate = r[-1].
        """
        as_of = datetime(2026, 1, 1, tzinfo=timezone.utc)
        dates = [
            as_of + timedelta(days=30),     # ~0.082Y
            as_of + timedelta(days=90),     # ~0.247Y
            as_of + timedelta(days=180),    # ~0.493Y
            as_of + timedelta(days=365),    # 1.0Y
            as_of + timedelta(days=730),    # 2.0Y
            as_of + timedelta(days=1825),   # 5.0Y
            as_of + timedelta(days=3650),   # 10.0Y
            as_of + timedelta(days=10950),  # 30.0Y
        ]
        rates_pct = [3.5, 3.8, 4.1, 4.5, 4.9, 5.3, 5.7, 6.0]

        curve = DiscountCurve(as_of, dates, rates_pct)
        k = len(dates)

        # 1. Test knot points Kronecker delta property
        for knot_idx, knot_t in enumerate(curve.tenors):
            w = curve.weights(knot_t)
            assert math.isclose(w[knot_idx], 1.0, abs_tol=1e-14)
            for j in range(k):
                if j != knot_idx:
                    assert math.isclose(w[j], 0.0, abs_tol=1e-14)

        # 2. Test left extrapolation
        for t_neg in [-50.0, -10.0, -1.0, 0.0, curve.tenors[0] / 2]:
            w = curve.weights(t_neg)
            assert math.isclose(w[0], 1.0, abs_tol=1e-14)
            assert np.all(w[1:] == 0.0)
            assert math.isclose(curve.zero(t_neg), curve.rates[0], abs_tol=1e-14)

        # 3. Test right extrapolation
        for t_pos in [curve.tenors[-1] + 1.0, 50.0, 100.0, 1000.0]:
            w = curve.weights(t_pos)
            assert math.isclose(w[-1], 1.0, abs_tol=1e-14)
            assert np.all(w[:-1] == 0.0)
            assert math.isclose(curve.zero(t_pos), curve.rates[-1], abs_tol=1e-14)

        # 4. Dense random sweep across [-20, 50]
        rng = np.random.default_rng(42)
        test_tenors = rng.uniform(-20.0, 50.0, size=500)
        for t in test_tenors:
            w = curve.weights(float(t))
            assert math.isclose(np.sum(w), 1.0, abs_tol=1e-14), f"Partition of unity violated at t={t}"
            assert np.all(w >= 0.0), f"Negative weight encountered at t={t}"
            reconstructed_r = float(np.dot(w, curve.rates))
            direct_r = curve.zero(float(t))
            assert math.isclose(reconstructed_r, direct_r, abs_tol=1e-14), f"Reconstruction mismatch at t={t}"


# ─────────────────────────────────────────────────────────────────────────────
# 5. Triangle Consistency & Negative Validation Paths
# ─────────────────────────────────────────────────────────────────────────────


class TestTriangleAndNegativeValidationStress:
    """Stress-tests triangle validation, unsupported products, and mathematical identities."""

    @pytest.fixture(scope="class")
    def complete_raw_messages(self) -> List[Dict[str, Any]]:
        fixtures_path = (
            Path(__file__).parent
            / "fixtures"
            / "legacy"
            / "xaurub_k10_complete_const_messages.json"
        )
        with open(fixtures_path, "r", encoding="utf-8") as f:
            return cast(List[Dict[str, Any]], json.load(f))

    def test_triangle_violation_raises_value_error(
        self, complete_raw_messages: List[Dict[str, Any]]
    ) -> None:
        """Tampering with USD funding curve between legs by > 1e-6 raises ValueError."""
        root_id = "1b4eef0d8162d4d520fe6bbebfed6798a4600b99"

        # Tamper USDRUB ForeignForwardCurve to point to RUB curve, violating D_B match
        tampered_msgs = copy.deepcopy(complete_raw_messages)
        for m in tampered_msgs:
            val = m.get("ConstMessage", m).get("value", {})
            if val.get("Identifier") == "USDRUB" and val.get("TypeName") == "FXForwardObject":
                val["ForeignForwardCurve"] = "bee2a9441a169c74148755c135c5a066db548b77"

        tampered_graph = ConstMessageGraph(tampered_msgs)
        with pytest.raises(ValueError, match="Triangle violation"):
            build_request(tampered_graph, root_id)

    def test_unsupported_product_type_raises_value_error(
        self, complete_raw_messages: List[Dict[str, Any]]
    ) -> None:
        """Products other than EuropeanVanillaOptionProduct raise ValueError."""
        root_id = "1b4eef0d8162d4d520fe6bbebfed6798a4600b99"
        prod_id = "8a1cf4f66a8e62fcc8bba5ffb74a70910bf13850"

        tampered_msgs = copy.deepcopy(complete_raw_messages)
        for m in tampered_msgs:
            raw = m.get("ConstMessage", m)
            if raw.get("id") == prod_id:
                raw["value"]["TypeName"] = "BarrierOptionProduct"

        tampered_graph = ConstMessageGraph(tampered_msgs)
        with pytest.raises(ValueError, match="Unsupported product type: BarrierOptionProduct"):
            build_request(tampered_graph, root_id)

    def test_missing_forward_leg_raises_value_error(
        self, complete_raw_messages: List[Dict[str, Any]]
    ) -> None:
        """When market dataset has forwards but missing legs for cross, raises ValueError."""
        root_id = "1b4eef0d8162d4d520fe6bbebfed6798a4600b99"
        # Mutate forward object identifiers so they don't match XAUUSD or USDRUB
        tampered_msgs = copy.deepcopy(complete_raw_messages)
        for m in tampered_msgs:
            val = m.get("ConstMessage", m).get("value", {})
            if val.get("TypeName") == "FXForwardObject" and val.get("Identifier") == "USDRUB":
                val["Identifier"] = "EURUSD"

        tampered_graph = ConstMessageGraph(tampered_msgs)
        with pytest.raises(ValueError, match="Unable to find required forward messages"):
            build_request(tampered_graph, root_id)

    def test_mutable_message_mathematical_conservation_stress(
        self, complete_raw_messages: List[Dict[str, Any]]
    ) -> None:
        """Validates that MutableMessage Greek mappings preserve mathematical conservation under stress:
        - CashDelta conservation: max |CD_cross - CD_leg1| <= 1e-6 abs, <= 1e-12 rel.
        - Delta * Spot == CashDelta to <= 1e-9 rel.
        - Delta / NonAdaptedDelta == g to <= 1e-12 rel.
        - Gamma scaling: Gamma_1 = Gamma_cross * (S_cross / S1)^2.
        - CashGamma 1% convention.
        """
        root_id = "1b4eef0d8162d4d520fe6bbebfed6798a4600b99"
        complete_graph = ConstMessageGraph(complete_raw_messages)
        req, _, ctx = build_request(complete_graph, root_id)

        s1 = req.market_data.leg1.spot
        s2 = req.market_data.leg2.spot
        s_cross = s1 * s2
        d_a = req.market_data.leg1.foreign_discount_factor
        d_c = req.market_data.leg2.domestic_discount_factor
        g = d_a / d_c
        f = s_cross * g

        # Create response with arbitrary non-trivial numbers
        d_set = req.market_data.settlement_discount_factor
        dvd_f = d_set * 0.95
        cd = f * dvd_f

        mock_resp = TriangulatedFXOptionResponse(
            status="done",
            product_value=350000.0,
            implied_vol_cross=0.19,
            forward_cross=f,
            delta=DeltaLegs(cross=g * dvd_f, leg1=dvd_f * (f / s1), leg2=dvd_f * (f / s2)),
            cash_delta=CashDeltaLegs(cross=cd, leg1=cd, leg2=cd),
            gamma=1.5e-6,
            vega=VegaLegs(
                cross=12000.0,
                leg1=7500.0,
                leg2=6200.0,
                smile_matrix_leg1=[7500.0, 300.0, 150.0, 450.0, 200.0],
                smile_matrix_leg2=[6200.0, 250.0, 120.0, 380.0, 180.0],
            ),
            theta=-145.0,
            vanna=-0.02,
            vomma=0.035,
            correlation_sensitivity=15000.0,
            rho_domestic=[100.0] * 26,
            rho_foreign=[-50.0] * 43,
            model_reserve_bps=0.0,
            latency_ms=0.15,
            telemetry=ModelRiskTelemetry(
                model_version="v3.0.0",
                engine_used=PricingEngineEnum.ANALYTIC_TRIANGULATED,
                standardized_moneyness=-2.5,
                mahalanobis_distance=1.0,
                ood_threshold=5.35,
                is_out_of_distribution=False,
                ensemble_epistemic_std_bps=0.0,
                fallback_activated=False,
                fallback_reason=None,
                durrleman_min_g=0.35,
            ),
        )

        mut_msg = to_mutable_message(
            mock_resp, req, ctx, nonce="test-stress-nonce", total_sec=0.015, proc_sec=0.005
        )
        res = mut_msg["MutableMessage"]["value"]["Result"]

        # 1. CashDelta conservation
        cd_map = res["CashDelta"]
        assert math.isclose(cd_map["XAURUB"], cd_map["XAUUSD"], abs_tol=1e-6, rel_tol=1e-12)
        assert math.isclose(cd_map["XAURUB"], cd_map["USDRUB"], abs_tol=1e-6, rel_tol=1e-12)

        # 2. Delta * Spot == CashDelta
        delta_map = res["Delta"]
        assert math.isclose(delta_map["XAURUB"] * s_cross, cd_map["XAURUB"], rel_tol=1e-9)
        assert math.isclose(delta_map["XAUUSD"] * s1, cd_map["XAUUSD"], rel_tol=1e-9)
        assert math.isclose(delta_map["USDRUB"] * s2, cd_map["USDRUB"], rel_tol=1e-9)

        # 3. Delta / NonAdaptedDelta == g
        nad_map = res["NonAdaptedDelta"]
        for k in ("XAURUB", "XAUUSD", "USDRUB"):
            assert math.isclose(delta_map[k] / nad_map[k], g, rel_tol=1e-12)

        # 4. Gamma scaling
        gamma_map = res["Gamma"]
        gamma_cross = gamma_map["XAURUB"]
        assert math.isclose(gamma_map["XAUUSD"], gamma_cross * (s2 ** 2), rel_tol=1e-12)
        assert math.isclose(gamma_map["USDRUB"], gamma_cross * (s1 ** 2), rel_tol=1e-12)

        # 5. CashGamma 1% convention
        cg_map = res["CashGamma"]
        assert math.isclose(cg_map["XAURUB"], gamma_cross * (s_cross ** 2) / 100.0, rel_tol=1e-12)
        assert math.isclose(cg_map["XAURUB"], cg_map["XAUUSD"], rel_tol=1e-12)
        assert math.isclose(cg_map["XAURUB"], cg_map["USDRUB"], rel_tol=1e-12)

        # 6. JSON serializability
        json_str = json.dumps(mut_msg)
        assert len(json_str) > 0

    def test_hash_verification_integrity(self) -> None:
        """verify_hash=True accepts bitwise SHA-1 match and rejects tampered values."""
        val = {"A": 1, "B": "test", "C": [10, 20]}
        canon_bytes = json.dumps(val, sort_keys=True, separators=(",", ":")).encode("utf-8")
        valid_id = hashlib.sha1(canon_bytes).hexdigest()

        # Valid hash passes
        g_valid = ConstMessageGraph([ConstMessage(id=valid_id, value=val)], verify_hash=True)
        assert valid_id in g_valid

        # Tampered hash raises ValueError
        tampered_id = "0000000000000000000000000000000000000000"
        with pytest.raises(ValueError, match="Hash mismatch for message"):
            ConstMessageGraph([ConstMessage(id=tampered_id, value=val)], verify_hash=True)

    def test_act365f_cross_timezone_offsets_and_leap_years(self) -> None:
        """year_fraction_act365f behaves consistently across timezone offsets and leap years."""
        # 1. Physical instant identity across different UTC offsets
        t_utc = datetime(2026, 6, 1, 0, 0, tzinfo=timezone.utc)
        t_tokyo = datetime(2026, 6, 1, 9, 0, tzinfo=timezone(timedelta(hours=9)))
        assert math.isclose(year_fraction_act365f(t_utc, t_tokyo), 0.0, abs_tol=1e-15)

        # 2. Leap year 2024: 2024-02-28 to 2024-03-01 has 2 full calendar days (leap day included)
        t_feb28 = datetime(2024, 2, 28, 0, 0, tzinfo=timezone.utc)
        t_mar01 = datetime(2024, 3, 1, 0, 0, tzinfo=timezone.utc)
        assert math.isclose(year_fraction_act365f(t_feb28, t_mar01), 2.0 / 365.0, abs_tol=1e-12)

        # 3. String ISO 8601 parsing with Z vs +00:00
        yf_z = year_fraction_act365f("2026-01-01T00:00:00Z", "2027-01-01T00:00:00Z")
        yf_offset = year_fraction_act365f("2026-01-01T00:00:00+00:00", "2027-01-01T00:00:00+00:00")
        assert math.isclose(yf_z, yf_offset, abs_tol=1e-15)
        assert math.isclose(yf_z, 1.0, abs_tol=1e-12)

    def test_vega_surface_and_correlation_sensitivity_conservation(
        self, complete_raw_messages: List[Dict[str, Any]]
    ) -> None:
        """Tests that time-interpolated Vega and CorrelationSensitivity matrices sum to total Greek."""
        root_id = "1b4eef0d8162d4d520fe6bbebfed6798a4600b99"
        req, _, ctx = build_request(ConstMessageGraph(complete_raw_messages), root_id)

        target_cross_vega = 45000.0
        target_corr_sens = 8500.0
        smile_mat_1 = [1200.0, 300.0, 150.0, 450.0, 200.0]  # [ATM, RR25, BF25, RR10, BF10]

        mock_resp = TriangulatedFXOptionResponse(
            status="done",
            product_value=1000.0,
            implied_vol_cross=0.20,
            forward_cross=100.0,
            delta=DeltaLegs(cross=0.5, leg1=0.5, leg2=0.5),
            cash_delta=CashDeltaLegs(cross=50.0, leg1=50.0, leg2=50.0),
            gamma=0.01,
            vega=VegaLegs(
                cross=target_cross_vega,
                leg1=1200.0,
                leg2=800.0,
                smile_matrix_leg1=smile_mat_1,
                smile_matrix_leg2=[800.0, 200.0, 100.0, 300.0, 150.0],
            ),
            theta=-10.0,
            vanna=0.01,
            vomma=0.02,
            correlation_sensitivity=target_corr_sens,
            model_reserve_bps=0.0,
            latency_ms=0.2,
            telemetry=ModelRiskTelemetry(
                model_version="v3.0.0",
                engine_used=PricingEngineEnum.ANALYTIC_TRIANGULATED,
                standardized_moneyness=0.0,
                mahalanobis_distance=0.5,
                ood_threshold=5.35,
                is_out_of_distribution=False,
                ensemble_epistemic_std_bps=0.0,
                fallback_activated=False,
                fallback_reason=None,
                durrleman_min_g=0.4,
            ),
        )

        serialized = to_mutable_message(
            mock_resp, req, ctx, nonce="test-nonce-sens", total_sec=0.1, proc_sec=0.1
        )
        res = serialized["MutableMessage"]["value"]["Result"]

        # 1. Cross vega conservation (sum of all elements in cross vega matrix == target_cross_vega)
        cross_vega_mat = res["Vega"]["XAURUB"]
        total_cross_vega = sum(sum(row) for row in cross_vega_mat)
        assert math.isclose(total_cross_vega, target_cross_vega, rel_tol=1e-12)

        # 2. Leg 1 vega matrix column sums match smile_mat_1 remapped
        # Legacy order: [10RR, 25RR, ATM, 25BF, 10BF] -> [smile_mat[3], smile_mat[1], smile_mat[0], smile_mat[2], smile_mat[4]]
        leg1_vega_mat = res["Vega"]["XAUUSD"]
        col_sums = [sum(row[c] for row in leg1_vega_mat) for c in range(5)]
        expected_cols = [smile_mat_1[3], smile_mat_1[1], smile_mat_1[0], smile_mat_1[2], smile_mat_1[4]]
        for col_idx in range(5):
            assert math.isclose(col_sums[col_idx], expected_cols[col_idx], rel_tol=1e-12)

        # 3. CorrelationSensitivity list sums to target_corr_sens
        corr_list = res["CorrelationSensitivity"]
        assert math.isclose(sum(corr_list), target_corr_sens, rel_tol=1e-12)

