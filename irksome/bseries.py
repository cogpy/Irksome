"""Butcher's B-series theory: rooted trees, elementary differentials,
and elementary weights for Runge-Kutta methods.

Given a Runge-Kutta method with Butcher tableau (A, b, c), the numerical
solution of y' = f(y) has a *B-series* expansion whose terms are indexed by
rooted trees.  For each rooted tree t the *elementary weight* Φ(t) is a
scalar that depends only on the tableau.  The method has order p if and only
if the order condition

    Φ(t) = 1 / γ(t)

holds for every rooted tree t with |t| ≤ p, where γ(t) is the *density* of t.

References
----------
* Butcher, J.C. (2016). *Numerical Methods for Ordinary Differential
  Equations*, 3rd ed. Wiley.
* Hairer, E., Nørsett, S.P., Wanner, G. (1993). *Solving Ordinary
  Differential Equations I: Nonstiff Problems*, 2nd ed. Springer.
"""

import math
from collections import Counter

import numpy

__all__ = [
    "RootedTree",
    "all_trees",
    "elementary_weight",
    "elementary_weights",
    "check_order_conditions",
    "order_violations",
]


class RootedTree:
    """A rooted tree in Butcher's B-series theory.

    Rooted trees index the terms of the B-series expansion of the numerical
    solution produced by a Runge-Kutta method applied to y' = f(y).

    A rooted tree is stored as a canonically sorted tuple of child trees.
    The *elementary tree* τ (a single node with no children) is represented
    by ``RootedTree()``.

    :arg children: a sequence of :class:`RootedTree` objects forming the
        children of the root vertex.  Defaults to the empty tuple (τ).

    Examples::

        tau = RootedTree()           # τ: single node, order 1
        t2  = RootedTree([tau])      # [τ]: order-2 tree
        t3a = RootedTree([t2])       # [[τ]]: order-3 chain
        t3b = RootedTree([tau, tau]) # [τ, τ]: order-3 cherry
    """

    def __init__(self, children=()):
        self.children = tuple(sorted(children))

    # ------------------------------------------------------------------
    # Intrinsic properties
    # ------------------------------------------------------------------

    @property
    def order(self):
        """The order |t|: total number of nodes in the tree."""
        return 1 + sum(c.order for c in self.children)

    @property
    def symmetry(self):
        """The symmetry coefficient σ(t).

        Defined recursively: σ(τ) = 1, and for a tree whose children
        group into distinct types with multiplicities m_l,

            σ(t) = ∏_l  m_l! · σ(t_l)^{m_l}.
        """
        if not self.children:
            return 1
        result = 1
        for child, m in Counter(self.children).items():
            result *= math.factorial(m) * (child.symmetry ** m)
        return result

    @property
    def density(self):
        """The density γ(t).

        Defined recursively: γ(τ) = 1, and

            γ(t) = |t| · ∏_{children t_l} γ(t_l).

        The *order condition* for tree t in a Runge-Kutta method is
        Φ(t) = 1/γ(t).
        """
        result = self.order
        for c in self.children:
            result *= c.density
        return result

    @property
    def tree_factorial(self):
        """The tree factorial t! = σ(t) · γ(t)."""
        return self.symmetry * self.density

    # ------------------------------------------------------------------
    # Python special methods
    # ------------------------------------------------------------------

    def __eq__(self, other):
        return isinstance(other, RootedTree) and self.children == other.children

    def __lt__(self, other):
        if self.order != other.order:
            return self.order < other.order
        return self.children < other.children

    def __le__(self, other):
        return self == other or self < other

    def __hash__(self):
        return hash(self.children)

    def __repr__(self):
        if not self.children:
            return "τ"
        return "[%s]" % ", ".join(repr(c) for c in self.children)


# ---------------------------------------------------------------------------
# Tree enumeration
# ---------------------------------------------------------------------------

_trees_cache: dict = {}


def _trees_of_order(n):
    """Return all rooted trees with exactly *n* nodes (cached)."""
    if n not in _trees_cache:
        if n == 1:
            _trees_cache[n] = [RootedTree()]
        else:
            _trees_cache[n] = [
                RootedTree(ch) for ch in _multisets_of_trees(n - 1)
            ]
    return _trees_cache[n]


def _multisets_of_trees(total, min_tree=None):
    """Yield sorted tuples of rooted trees whose orders sum to *total*.

    :arg total: required sum of tree orders.
    :arg min_tree: lower bound for the smallest tree (enforces canonical
        non-decreasing order so each multiset is generated only once).
    """
    if total == 0:
        yield ()
        return
    for k in range(1, total + 1):
        for tree in _trees_of_order(k):
            if min_tree is not None and tree < min_tree:
                continue
            for rest in _multisets_of_trees(total - k, min_tree=tree):
                yield (tree,) + rest


def all_trees(max_order):
    """Return all rooted trees up to the given order.

    :arg max_order: maximum order (number of nodes) to include.
    :returns: a list of :class:`RootedTree` objects with |t| ≤ max_order,
        sorted first by increasing order and then lexicographically within
        each order.
    """
    result = []
    for n in range(1, max_order + 1):
        result.extend(_trees_of_order(n))
    return result


# ---------------------------------------------------------------------------
# Elementary weights
# ---------------------------------------------------------------------------

def _stage_weight_vectors(bt, t):
    """Compute the stage weight vector φ(s) for every subtree *s* of *t*.

    The stage weight vector φ(s) ∈ ℝˢ (one entry per RK stage) satisfies
    the recursion

        φᵢ(s) = Σⱼ Aᵢⱼ · ∏_l φⱼ(s_l)

    where s₁, …, sₖ are the children of s and the empty product equals 1.
    For the elementary tree τ (no children) this reduces to

        φᵢ(τ) = Σⱼ Aᵢⱼ = cᵢ.

    :arg bt: a :class:`~irksome.tableaux.ButcherTableaux.ButcherTableau`.
    :arg t: a :class:`RootedTree`.
    :returns: a dict mapping each subtree of *t* to its numpy stage weight
        vector of length ``bt.num_stages``.
    """
    cache = {}

    def compute(s):
        if s in cache:
            return cache[s]
        # child_product[j] = ∏_l φⱼ(children of s)
        child_product = numpy.ones(bt.num_stages)
        for c in s.children:
            child_product = child_product * compute(c)
        # φ(s) = A · child_product  (matrix-vector product over stages)
        cache[s] = bt.A @ child_product
        return cache[s]

    compute(t)
    return cache


def elementary_weight(bt, t):
    """Return the elementary weight Φ(t) for Butcher tableau *bt* and tree *t*.

    The elementary weight is defined by

        Φ(t) = Σᵢ bᵢ · ∏_l φᵢ(t_l)

    where t₁, …, tₖ are the *children* of t and φᵢ is the stage weight
    at stage i (see :func:`_stage_weight_vectors`).  The B-series order
    condition for tree t is Φ(t) = 1/γ(t).

    :arg bt: a :class:`~irksome.tableaux.ButcherTableaux.ButcherTableau`.
    :arg t: a :class:`RootedTree`.
    :returns: Φ(t) as a float.
    """
    sw = _stage_weight_vectors(bt, t)
    # child_product[i] = ∏_l φᵢ(children of t)
    child_product = numpy.ones(bt.num_stages)
    for c in t.children:
        child_product = child_product * sw[c]
    return float(bt.b @ child_product)


def elementary_weights(bt, max_order):
    """Return a dict of elementary weights Φ(t) for all trees up to *max_order*.

    :arg bt: a :class:`~irksome.tableaux.ButcherTableaux.ButcherTableau`.
    :arg max_order: maximum tree order to consider.
    :returns: a dict mapping each :class:`RootedTree` t to its elementary
        weight Φ(t).
    """
    return {t: elementary_weight(bt, t) for t in all_trees(max_order)}


def check_order_conditions(bt, order):
    """Return ``True`` if *bt* satisfies the B-series order conditions up to *order*.

    The order condition for tree t requires Φ(t) = 1/γ(t).

    :arg bt: a :class:`~irksome.tableaux.ButcherTableaux.ButcherTableau`.
    :arg order: the order to verify (checks all trees with ≤ *order* nodes).
    :returns: ``True`` if every order condition is satisfied to floating-point
        tolerance, ``False`` otherwise.
    """
    return all(
        numpy.isclose(elementary_weight(bt, t), 1.0 / t.density)
        for t in all_trees(order)
    )


def order_violations(bt, order):
    """Return trees where the order conditions fail for *bt*.

    :arg bt: a :class:`~irksome.tableaux.ButcherTableaux.ButcherTableau`.
    :arg order: the order to check.
    :returns: a dict mapping each :class:`RootedTree` t where Φ(t) ≠ 1/γ(t)
        to the tuple ``(Φ(t), 1/γ(t))``.
    """
    violations = {}
    for t in all_trees(order):
        phi = elementary_weight(bt, t)
        required = 1.0 / t.density
        if not numpy.isclose(phi, required):
            violations[t] = (phi, required)
    return violations
