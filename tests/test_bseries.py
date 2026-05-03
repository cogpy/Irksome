"""Tests for Butcher B-series rooted trees and elementary weights."""

import numpy as np
import pytest
from numpy import allclose

from irksome.bseries import (
    RootedTree,
    all_trees,
    check_order_conditions,
    elementary_weight,
    elementary_weights,
    order_violations,
)


# ---------------------------------------------------------------------------
# Minimal Butcher tableau helper (avoids FIAT dependency in tests)
# The B-series functions only access bt.A, bt.b and bt.num_stages.
# ---------------------------------------------------------------------------

class _BT:
    """Minimal Butcher tableau for testing (no FIAT required)."""
    def __init__(self, A, b, c):
        self.A = np.asarray(A, dtype=float)
        self.b = np.asarray(b, dtype=float)
        self.c = np.asarray(c, dtype=float)

    @property
    def num_stages(self):
        return len(self.b)


# ---------------------------------------------------------------------------
# Canonical trees used throughout the tests
# ---------------------------------------------------------------------------

tau = RootedTree()                # τ, order 1
t2 = RootedTree([tau])            # [τ], order 2
t3a = RootedTree([t2])            # [[τ]], order 3 (chain)
t3b = RootedTree([tau, tau])      # [τ, τ], order 3 (cherry)
t4a = RootedTree([t3a])           # [[[τ]]], order 4
t4b = RootedTree([t3b])           # [[τ, τ]], order 4
t4c = RootedTree([t2, tau])       # [[τ], τ], order 4
t4d = RootedTree([tau, tau, tau])  # [τ, τ, τ], order 4

# Order 5: 9 trees (descending all_trees() sort order within each order, matching orders 3–4 above)
t5a = RootedTree([t4a])                    # [[[[τ]]]], order 5
t5b = RootedTree([t4b])                    # [[[τ, τ]]], order 5
t5c = RootedTree([t4c])                    # [[[τ], τ]], order 5
t5d = RootedTree([t4d])                    # [[τ, τ, τ]], order 5
t5e = RootedTree([t2, t2])                 # [[τ], [τ]], order 5
t5f = RootedTree([tau, t3a])               # [τ, [[τ]]], order 5
t5g = RootedTree([tau, t3b])               # [τ, [τ, τ]], order 5
t5h = RootedTree([tau, tau, t2])           # [τ, τ, [τ]], order 5
t5i = RootedTree([tau, tau, tau, tau])     # [τ, τ, τ, τ], order 5

# Order 6: 20 trees
t6a = RootedTree([t5a])                          # [[[[[τ]]]]], order 6
t6b = RootedTree([t5b])                          # [[[[τ, τ]]]], order 6
t6c = RootedTree([t5c])                          # [[[[τ], τ]]], order 6
t6d = RootedTree([t5d])                          # [[[τ, τ, τ]]], order 6
t6e = RootedTree([t5e])                          # [[[τ], [τ]]], order 6
t6f = RootedTree([t5f])                          # [[τ, [[τ]]]], order 6
t6g = RootedTree([t5g])                          # [[τ, [τ, τ]]], order 6
t6h = RootedTree([t5h])                          # [[τ, τ, [τ]]], order 6
t6i = RootedTree([t5i])                          # [[τ, τ, τ, τ]], order 6
t6j = RootedTree([t2, t3a])                      # [[τ], [[τ]]], order 6
t6k = RootedTree([t2, t3b])                      # [[τ], [τ, τ]], order 6
t6l = RootedTree([tau, t4a])                     # [τ, [[[τ]]]], order 6
t6m = RootedTree([tau, t4b])                     # [τ, [[τ, τ]]], order 6
t6n = RootedTree([tau, t4c])                     # [τ, [[τ], τ]], order 6
t6o = RootedTree([tau, t4d])                     # [τ, [τ, τ, τ]], order 6
t6p = RootedTree([tau, t2, t2])                  # [τ, [τ], [τ]], order 6
t6q = RootedTree([tau, tau, t3a])                # [τ, τ, [[τ]]], order 6
t6r = RootedTree([tau, tau, t3b])                # [τ, τ, [τ, τ]], order 6
t6s = RootedTree([tau, tau, tau, t2])            # [τ, τ, τ, [τ]], order 6
t6t = RootedTree([tau, tau, tau, tau, tau])      # [τ, τ, τ, τ, τ], order 6

# Order 7: 48 trees
t7a = RootedTree([t6a])                               # [[[[[[τ]]]]]], order 7
t7b = RootedTree([t6b])                               # [[[[[τ, τ]]]]], order 7
t7c = RootedTree([t6c])                               # [[[[[τ], τ]]]], order 7
t7d = RootedTree([t6d])                               # [[[[τ, τ, τ]]]], order 7
t7e = RootedTree([t6e])                               # [[[[τ], [τ]]]], order 7
t7f = RootedTree([t6f])                               # [[[τ, [[τ]]]]], order 7
t7g = RootedTree([t6g])                               # [[[τ, [τ, τ]]]], order 7
t7h = RootedTree([t6h])                               # [[[τ, τ, [τ]]]], order 7
t7i = RootedTree([t6i])                               # [[[τ, τ, τ, τ]]], order 7
t7j = RootedTree([t6j])                               # [[[τ], [[τ]]]], order 7
t7k = RootedTree([t6k])                               # [[[τ], [τ, τ]]], order 7
t7l = RootedTree([t6l])                               # [[τ, [[[τ]]]]], order 7
t7m = RootedTree([t6m])                               # [[τ, [[τ, τ]]]], order 7
t7n = RootedTree([t6n])                               # [[τ, [[τ], τ]]], order 7
t7o = RootedTree([t6o])                               # [[τ, [τ, τ, τ]]], order 7
t7p = RootedTree([t6p])                               # [[τ, [τ], [τ]]], order 7
t7q = RootedTree([t6q])                               # [[τ, τ, [[τ]]]], order 7
t7r = RootedTree([t6r])                               # [[τ, τ, [τ, τ]]], order 7
t7s = RootedTree([t6s])                               # [[τ, τ, τ, [τ]]], order 7
t7t = RootedTree([t6t])                               # [[τ, τ, τ, τ, τ]], order 7
t7u = RootedTree([t3a, t3a])                          # [[[τ]], [[τ]]], order 7
t7v = RootedTree([t3b, t3a])                          # [[τ, τ], [[τ]]], order 7
t7w = RootedTree([t3b, t3b])                          # [[τ, τ], [τ, τ]], order 7
t7x = RootedTree([t2, t4a])                           # [[τ], [[[τ]]]], order 7
t7y = RootedTree([t2, t4b])                           # [[τ], [[τ, τ]]], order 7
t7z = RootedTree([t2, t4c])                           # [[τ], [[τ], τ]], order 7
t7aa = RootedTree([t2, t4d])                           # [[τ], [τ, τ, τ]], order 7
t7ab = RootedTree([t2, t2, t2])                        # [[τ], [τ], [τ]], order 7
t7ac = RootedTree([tau, t5a])                          # [τ, [[[[τ]]]]], order 7
t7ad = RootedTree([tau, t5b])                          # [τ, [[[τ, τ]]]], order 7
t7ae = RootedTree([tau, t5c])                          # [τ, [[[τ], τ]]], order 7
t7af = RootedTree([tau, t5d])                          # [τ, [[τ, τ, τ]]], order 7
t7ag = RootedTree([tau, t5e])                          # [τ, [[τ], [τ]]], order 7
t7ah = RootedTree([tau, t5f])                          # [τ, [τ, [[τ]]]], order 7
t7ai = RootedTree([tau, t5g])                          # [τ, [τ, [τ, τ]]], order 7
t7aj = RootedTree([tau, t5h])                          # [τ, [τ, τ, [τ]]], order 7
t7ak = RootedTree([tau, t5i])                          # [τ, [τ, τ, τ, τ]], order 7
t7al = RootedTree([tau, t2, t3a])                      # [τ, [τ], [[τ]]], order 7
t7am = RootedTree([tau, t2, t3b])                      # [τ, [τ], [τ, τ]], order 7
t7an = RootedTree([tau, tau, t4a])                     # [τ, τ, [[[τ]]]], order 7
t7ao = RootedTree([tau, tau, t4b])                     # [τ, τ, [[τ, τ]]], order 7
t7ap = RootedTree([tau, tau, t4c])                     # [τ, τ, [[τ], τ]], order 7
t7aq = RootedTree([tau, tau, t4d])                     # [τ, τ, [τ, τ, τ]], order 7
t7ar = RootedTree([tau, tau, t2, t2])                  # [τ, τ, [τ], [τ]], order 7
t7as = RootedTree([tau, tau, tau, t3a])                # [τ, τ, τ, [[τ]]], order 7
t7at = RootedTree([tau, tau, tau, t3b])                # [τ, τ, τ, [τ, τ]], order 7
t7au = RootedTree([tau, tau, tau, tau, t2])            # [τ, τ, τ, τ, [τ]], order 7
t7av = RootedTree([tau, tau, tau, tau, tau, tau])      # [τ, τ, τ, τ, τ, τ], order 7

# ---------------------------------------------------------------------------
# Helper tableaux (defined without FIAT/Firedrake)
# ---------------------------------------------------------------------------


def _euler():
    """Forward/Backward Euler (1 stage, order 1)."""
    A = np.array([[0.0]])
    b = np.array([1.0])
    c = np.array([0.0])
    return _BT(A, b, c)


def _midpoint():
    """Explicit midpoint rule (2 stages, order 2)."""
    A = np.array([[0.0, 0.0], [0.5, 0.0]])
    b = np.array([0.0, 1.0])
    c = np.array([0.0, 0.5])
    return _BT(A, b, c)


def _rk4():
    """Classical 4-stage Runge-Kutta (order 4)."""
    A = np.array([[0, 0, 0, 0],
                  [0.5, 0, 0, 0],
                  [0, 0.5, 0, 0],
                  [0, 0, 1, 0]], dtype=float)
    b = np.array([1/6, 1/3, 1/3, 1/6])
    c = np.array([0, 0.5, 0.5, 1.0])
    return _BT(A, b, c)


def _implicit_midpoint():
    """Implicit midpoint rule (1 stage Gauss-Legendre, order 2)."""
    A = np.array([[0.5]])
    b = np.array([1.0])
    c = np.array([0.5])
    return _BT(A, b, c)


def _radau_iia_2():
    """RadauIIA with 2 stages (order 3)."""
    A = np.array([[5/12, -1/12], [3/4, 1/4]])
    b = np.array([3/4, 1/4])
    c = np.array([1/3, 1.0])
    return _BT(A, b, c)


# ===========================================================================
# RootedTree property tests
# ===========================================================================

class TestRootedTreeProperties:

    def test_order(self):
        assert tau.order == 1
        assert t2.order == 2
        assert t3a.order == 3
        assert t3b.order == 3
        assert t4a.order == 4
        assert t4b.order == 4
        assert t4c.order == 4
        assert t4d.order == 4
        for t in [t5a, t5b, t5c, t5d, t5e, t5f, t5g, t5h, t5i]:
            assert t.order == 5
        for t in [t6a, t6b, t6c, t6d, t6e, t6f, t6g, t6h, t6i, t6j,
                  t6k, t6l, t6m, t6n, t6o, t6p, t6q, t6r, t6s, t6t]:
            assert t.order == 6
        for t in [t7a, t7b, t7c, t7d, t7e, t7f, t7g, t7h, t7i, t7j,
                  t7k, t7l, t7m, t7n, t7o, t7p, t7q, t7r, t7s, t7t,
                  t7u, t7v, t7w, t7x, t7y, t7z, t7aa, t7ab, t7ac, t7ad,
                  t7ae, t7af, t7ag, t7ah, t7ai, t7aj, t7ak, t7al, t7am,
                  t7an, t7ao, t7ap, t7aq, t7ar, t7as, t7at, t7au, t7av]:
            assert t.order == 7

    def test_symmetry(self):
        assert tau.symmetry == 1
        assert t2.symmetry == 1      # one child, appears once
        assert t3a.symmetry == 1
        assert t3b.symmetry == 2     # two identical children τ → 2!
        assert t4a.symmetry == 1
        assert t4b.symmetry == 2     # one child [τ,τ] with σ=2 → 1! * 2^1 = 2
        assert t4c.symmetry == 1     # children [τ] and τ are distinct
        assert t4d.symmetry == 6     # three identical children τ → 3!
        # order 5
        assert t5a.symmetry == 1
        assert t5b.symmetry == 2     # child [τ,τ] has σ=2
        assert t5c.symmetry == 1
        assert t5d.symmetry == 6     # child [τ,τ,τ] has σ=6
        assert t5e.symmetry == 2     # two identical children [τ] → 2!
        assert t5f.symmetry == 1
        assert t5g.symmetry == 2     # child [τ,τ] has σ=2
        assert t5h.symmetry == 2     # two identical children τ → 2!
        assert t5i.symmetry == 24    # four identical children τ → 4!

    def test_density(self):
        assert tau.density == 1
        assert t2.density == 2
        assert t3a.density == 6      # 3 * γ([τ]) = 3*2
        assert t3b.density == 3      # 3 * γ(τ)^2 = 3*1
        assert t4a.density == 24     # 4 * 6
        assert t4b.density == 12     # 4 * 3
        assert t4c.density == 8      # 4 * 2 * 1
        assert t4d.density == 4      # 4 * 1 * 1 * 1
        # order 5
        assert t5a.density == 120    # 5 * 24
        assert t5b.density == 60     # 5 * 12
        assert t5c.density == 40     # 5 * 8
        assert t5d.density == 20     # 5 * 4
        assert t5e.density == 20     # 5 * 2 * 2
        assert t5f.density == 30     # 5 * 6
        assert t5g.density == 15     # 5 * 3
        assert t5h.density == 10     # 5 * 2
        assert t5i.density == 5      # 5 * 1 * 1 * 1 * 1

    def test_tree_factorial(self):
        # t! = σ(t) * γ(t)
        assert tau.tree_factorial == 1
        assert t2.tree_factorial == 2
        assert t3a.tree_factorial == 6
        assert t3b.tree_factorial == 6   # σ=2, γ=3
        assert t4a.tree_factorial == 24
        assert t4b.tree_factorial == 24  # σ=2, γ=12
        assert t4c.tree_factorial == 8
        assert t4d.tree_factorial == 24  # σ=6, γ=4
        # order 5
        assert t5a.tree_factorial == 120  # σ=1, γ=120
        assert t5b.tree_factorial == 120  # σ=2, γ=60
        assert t5c.tree_factorial == 40   # σ=1, γ=40
        assert t5d.tree_factorial == 120  # σ=6, γ=20
        assert t5e.tree_factorial == 40   # σ=2, γ=20
        assert t5f.tree_factorial == 30   # σ=1, γ=30
        assert t5g.tree_factorial == 30   # σ=2, γ=15
        assert t5h.tree_factorial == 20   # σ=2, γ=10
        assert t5i.tree_factorial == 120  # σ=24, γ=5

    def test_equality(self):
        assert tau == RootedTree()
        assert t2 == RootedTree([RootedTree()])
        assert t3b == RootedTree([tau, tau])
        assert tau != t2
        assert t3a != t3b

    def test_canonical_children_sorting(self):
        # Children should be sorted regardless of input order.
        assert RootedTree([t2, tau]) == RootedTree([tau, t2])

    def test_hash_consistency(self):
        s = {tau, t2, t3a, t3b, t4a, t4b, t4c, t4d,
             t5a, t5b, t5c, t5d, t5e, t5f, t5g, t5h, t5i}
        assert len(s) == 17
        assert tau in s
        assert t5i in s

    def test_repr(self):
        assert repr(tau) == "τ"
        assert repr(t2) == "[τ]"
        assert repr(t3a) == "[[τ]]"
        assert repr(t3b) == "[τ, τ]"

    def test_ordering(self):
        trees = [t4a, tau, t3b, t2, t4c, t3a, t4d, t4b]
        ordered = sorted(trees)
        # Must start with the smallest (tau, order 1) and end with an
        # order-4 tree; order-1 < order-2 < order-3 < order-4.
        assert ordered[0] == tau
        assert ordered[1] == t2
        assert set(ordered[2:4]) == {t3a, t3b}
        assert set(ordered[4:]) == {t4a, t4b, t4c, t4d}

        # Order-5 trees must all sort after order-4 trees.
        order5 = [t5a, t5b, t5c, t5d, t5e, t5f, t5g, t5h, t5i]
        mixed = sorted(trees + order5)
        assert mixed[0] == tau
        assert set(mixed[8:]) == set(order5)


# ===========================================================================
# Tree enumeration tests
# ===========================================================================

class TestAllTrees:
    # Number of rooted trees by order (orders 1–7): 1, 1, 2, 4, 9, 20, 48
    @pytest.mark.parametrize("n,count", [(1, 1), (2, 1), (3, 2), (4, 4), (5, 9), (6, 20), (7, 48)])
    def test_trees_at_each_order(self, n, count):
        trees_at_n = [t for t in all_trees(n) if t.order == n]
        assert len(trees_at_n) == count

    @pytest.mark.parametrize("n,total", [(1, 1), (2, 2), (3, 4), (4, 8), (5, 17), (6, 37), (7, 85)])
    def test_cumulative_count(self, n, total):
        assert len(all_trees(n)) == total

    def test_known_trees_present(self):
        trees4 = set(all_trees(4))
        for t in [tau, t2, t3a, t3b, t4a, t4b, t4c, t4d]:
            assert t in trees4
        trees5 = set(all_trees(5))
        for t in [t5a, t5b, t5c, t5d, t5e, t5f, t5g, t5h, t5i]:
            assert t in trees5

    def test_order_bounds(self):
        for t in all_trees(7):
            assert 1 <= t.order <= 7

    def test_empty_returns_empty(self):
        assert all_trees(0) == []


# ===========================================================================
# Higher-order canonical tree property tests (orders 6 and 7)
# ===========================================================================

# (sigma, gamma) = (symmetry coefficient, density) pairs for order-6 trees a..t
_ORDER6_SIGMA_GAMMA = [
    (1, 720), (2, 360), (1, 240), (6, 120), (2, 120),
    (1, 180), (2, 90), (2, 60), (24, 30), (1, 72),
    (2, 36), (1, 144), (2, 72), (1, 48), (6, 24),
    (2, 24), (2, 36), (4, 18), (6, 12), (120, 6),
]
_ORDER6_TREES = [
    t6a, t6b, t6c, t6d, t6e, t6f, t6g, t6h, t6i, t6j,
    t6k, t6l, t6m, t6n, t6o, t6p, t6q, t6r, t6s, t6t,
]

# (sigma, gamma) = (symmetry coefficient, density) pairs for order-7 trees a..av
_ORDER7_SIGMA_GAMMA = [
    (1, 5040), (2, 2520), (1, 1680), (6, 840), (2, 840),
    (1, 1260), (2, 630), (2, 420), (24, 210), (1, 504),
    (2, 252), (1, 1008), (2, 504), (1, 336), (6, 168),
    (2, 168), (2, 252), (4, 126), (6, 84), (120, 42),
    (2, 252), (2, 126), (8, 63), (1, 336), (2, 168),
    (1, 112), (6, 56), (6, 56), (1, 840), (2, 420),
    (1, 280), (6, 140), (2, 140), (1, 210), (2, 105),
    (2, 70), (24, 35), (1, 84), (2, 42), (2, 168),
    (4, 84), (2, 56), (12, 28), (4, 28), (6, 42),
    (12, 21), (24, 14), (720, 7),
]
_ORDER7_TREES = [
    t7a, t7b, t7c, t7d, t7e, t7f, t7g, t7h, t7i, t7j,
    t7k, t7l, t7m, t7n, t7o, t7p, t7q, t7r, t7s, t7t,
    t7u, t7v, t7w, t7x, t7y, t7z, t7aa, t7ab, t7ac, t7ad,
    t7ae, t7af, t7ag, t7ah, t7ai, t7aj, t7ak, t7al, t7am,
    t7an, t7ao, t7ap, t7aq, t7ar, t7as, t7at, t7au, t7av,
]


class TestHigherOrderTreeProperties:

    @pytest.mark.parametrize("t,sigma,gamma", [
        (tree, sg[0], sg[1])
        for tree, sg in zip(_ORDER6_TREES, _ORDER6_SIGMA_GAMMA)
    ])
    def test_order6_properties(self, t, sigma, gamma):
        assert t.order == 6
        assert t.symmetry == sigma
        assert t.density == gamma
        assert t.tree_factorial == sigma * gamma

    @pytest.mark.parametrize("t,sigma,gamma", [
        (tree, sg[0], sg[1])
        for tree, sg in zip(_ORDER7_TREES, _ORDER7_SIGMA_GAMMA)
    ])
    def test_order7_properties(self, t, sigma, gamma):
        assert t.order == 7
        assert t.symmetry == sigma
        assert t.density == gamma
        assert t.tree_factorial == sigma * gamma

    def test_order6_all_distinct(self):
        assert len(set(_ORDER6_TREES)) == 20

    def test_order7_all_distinct(self):
        assert len(set(_ORDER7_TREES)) == 48

    def test_order6_in_all_trees(self):
        trees6 = set(all_trees(6))
        for t in _ORDER6_TREES:
            assert t in trees6

    def test_order7_in_all_trees(self):
        trees7 = set(all_trees(7))
        for t in _ORDER7_TREES:
            assert t in trees7


# ===========================================================================
# Elementary weight tests
# ===========================================================================

class TestElementaryWeight:

    # --- consistency (order-1) condition holds for every method ---
    @pytest.mark.parametrize("bt", [_euler(), _midpoint(), _rk4(),
                                    _implicit_midpoint(), _radau_iia_2()])
    def test_order1_always_one(self, bt):
        assert allclose(elementary_weight(bt, tau), 1.0)

    # --- explicit Euler (order 1) ---
    def test_euler_fails_order2(self):
        bt = _euler()
        # b^T c = 1*0 = 0 ≠ 1/2
        assert allclose(elementary_weight(bt, t2), 0.0)

    # --- explicit midpoint (order 2) ---
    def test_midpoint_order2_condition(self):
        bt = _midpoint()
        # Σ bᵢ cᵢ = 0*0 + 1*(1/2) = 1/2
        assert allclose(elementary_weight(bt, t2), 0.5)

    def test_midpoint_fails_order3_chain(self):
        bt = _midpoint()
        # b^T Ac = 0, not 1/6
        assert allclose(elementary_weight(bt, t3a), 0.0)

    def test_midpoint_fails_order3_cherry(self):
        bt = _midpoint()
        # Σ bᵢ cᵢ² = 1*(1/2)² = 1/4, not 1/3
        assert allclose(elementary_weight(bt, t3b), 0.25)

    # --- classical RK4 (order 4): all conditions should pass ---
    def test_rk4_order1(self):
        assert allclose(elementary_weight(_rk4(), tau), 1.0)    # 1/γ=1

    def test_rk4_order2(self):
        assert allclose(elementary_weight(_rk4(), t2), 0.5)     # 1/γ=1/2

    def test_rk4_order3_chain(self):
        assert allclose(elementary_weight(_rk4(), t3a), 1/6)    # 1/γ=1/6

    def test_rk4_order3_cherry(self):
        assert allclose(elementary_weight(_rk4(), t3b), 1/3)    # 1/γ=1/3

    def test_rk4_order4_trees(self):
        bt = _rk4()
        assert allclose(elementary_weight(bt, t4a), 1/24)       # 1/γ=1/24
        assert allclose(elementary_weight(bt, t4b), 1/12)       # 1/γ=1/12
        assert allclose(elementary_weight(bt, t4c), 1/8)        # 1/γ=1/8
        assert allclose(elementary_weight(bt, t4d), 1/4)        # 1/γ=1/4

    # --- implicit midpoint (order 2) ---
    def test_implicit_midpoint_order2(self):
        bt = _implicit_midpoint()
        assert allclose(elementary_weight(bt, t2), 0.5)

    # --- RadauIIA(2) (order 3) ---
    def test_radau2_order3(self):
        bt = _radau_iia_2()
        assert allclose(elementary_weight(bt, tau), 1.0)
        assert allclose(elementary_weight(bt, t2), 0.5)
        assert allclose(elementary_weight(bt, t3a), 1/6)
        assert allclose(elementary_weight(bt, t3b), 1/3)

    def test_radau2_fails_order4(self):
        bt = _radau_iia_2()
        # At least one order-4 condition must fail
        order4_weights = [elementary_weight(bt, t) for t in [t4a, t4b, t4c, t4d]]
        required = [1/24, 1/12, 1/8, 1/4]
        assert not all(allclose(w, r) for w, r in zip(order4_weights, required))


class TestElementaryWeightsDict:

    def test_returns_all_trees(self):
        bt = _rk4()
        ew = elementary_weights(bt, 4)
        assert set(ew.keys()) == set(all_trees(4))

    def test_values_match_individual(self):
        bt = _midpoint()
        ew = elementary_weights(bt, 3)
        for t in all_trees(3):
            assert allclose(ew[t], elementary_weight(bt, t))


# ===========================================================================
# check_order_conditions tests
# ===========================================================================

class TestCheckOrderConditions:

    # Euler passes order 1 but not order 2
    def test_euler_order1(self):
        assert check_order_conditions(_euler(), 1)

    def test_euler_fails_order2(self):
        assert not check_order_conditions(_euler(), 2)

    # Explicit midpoint passes order 2, fails order 3
    def test_midpoint_order2(self):
        assert check_order_conditions(_midpoint(), 2)

    def test_midpoint_fails_order3(self):
        assert not check_order_conditions(_midpoint(), 3)

    # Implicit midpoint (1-stage GL) also has order 2
    def test_implicit_midpoint_order2(self):
        assert check_order_conditions(_implicit_midpoint(), 2)

    # Classical RK4 passes orders 1-4, fails 5
    @pytest.mark.parametrize("p", [1, 2, 3, 4])
    def test_rk4_satisfies(self, p):
        assert check_order_conditions(_rk4(), p)

    def test_rk4_fails_order5(self):
        assert not check_order_conditions(_rk4(), 5)

    # RadauIIA(2) passes 1-3, fails 4
    @pytest.mark.parametrize("p", [1, 2, 3])
    def test_radau2_satisfies(self, p):
        assert check_order_conditions(_radau_iia_2(), p)

    def test_radau2_fails_order4(self):
        assert not check_order_conditions(_radau_iia_2(), 4)


# ===========================================================================
# order_violations tests
# ===========================================================================

class TestOrderViolations:

    def test_rk4_no_violations_up_to_order4(self):
        assert len(order_violations(_rk4(), 4)) == 0

    def test_rk4_has_violations_at_order5(self):
        assert len(order_violations(_rk4(), 5)) > 0

    def test_midpoint_no_violations_order2(self):
        assert len(order_violations(_midpoint(), 2)) == 0

    def test_midpoint_violations_at_order3(self):
        viols = order_violations(_midpoint(), 3)
        assert len(viols) > 0
        # Both order-3 trees should fail
        assert t3a in viols
        assert t3b in viols

    def test_violation_format(self):
        viols = order_violations(_euler(), 2)
        assert t2 in viols
        phi, required = viols[t2]
        assert allclose(phi, 0.0)        # Euler: b^T c = 1*0 = 0
        assert allclose(required, 0.5)   # 1/γ([τ]) = 1/2

    def test_radau2_no_violations_order3(self):
        assert len(order_violations(_radau_iia_2(), 3)) == 0

    def test_radau2_violations_at_order4(self):
        assert len(order_violations(_radau_iia_2(), 4)) > 0
