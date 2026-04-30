"""Tests for Butcher B-series rooted trees and elementary weights."""

import math

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
t4d = RootedTree([tau, tau, tau]) # [τ, τ, τ], order 4

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

    def test_symmetry(self):
        assert tau.symmetry == 1
        assert t2.symmetry == 1      # one child, appears once
        assert t3a.symmetry == 1
        assert t3b.symmetry == 2     # two identical children τ → 2!
        assert t4a.symmetry == 1
        assert t4b.symmetry == 2     # one child [τ,τ] with σ=2 → 1! * 2^1 = 2
        assert t4c.symmetry == 1     # children [τ] and τ are distinct
        assert t4d.symmetry == 6     # three identical children τ → 3!

    def test_density(self):
        assert tau.density == 1
        assert t2.density == 2
        assert t3a.density == 6      # 3 * γ([τ]) = 3*2
        assert t3b.density == 3      # 3 * γ(τ)^2 = 3*1
        assert t4a.density == 24     # 4 * 6
        assert t4b.density == 12     # 4 * 3
        assert t4c.density == 8      # 4 * 2 * 1
        assert t4d.density == 4      # 4 * 1 * 1 * 1

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
        s = {tau, t2, t3a, t3b, t4a, t4b, t4c, t4d}
        assert len(s) == 8
        assert tau in s
        assert t4c in s

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


# ===========================================================================
# Tree enumeration tests
# ===========================================================================

class TestAllTrees:
    # Number of rooted trees by order: 1, 1, 2, 4, 9, 20, …
    @pytest.mark.parametrize("n,count", [(1, 1), (2, 1), (3, 2), (4, 4), (5, 9)])
    def test_trees_at_each_order(self, n, count):
        trees_at_n = [t for t in all_trees(n) if t.order == n]
        assert len(trees_at_n) == count

    @pytest.mark.parametrize("n,total", [(1, 1), (2, 2), (3, 4), (4, 8), (5, 17)])
    def test_cumulative_count(self, n, total):
        assert len(all_trees(n)) == total

    def test_known_trees_present(self):
        trees = set(all_trees(4))
        for t in [tau, t2, t3a, t3b, t4a, t4b, t4c, t4d]:
            assert t in trees

    def test_order_bounds(self):
        for t in all_trees(5):
            assert 1 <= t.order <= 5

    def test_empty_returns_empty(self):
        assert all_trees(0) == []


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
