"""Run with: python3 -m unittest discover -s tests -v.

Solver tests require Docplex; the integration test additionally requires CPLEX.
"""
import contextlib
import importlib.util
import io
import itertools
import math
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from src.BB_node import BB_node
from src.Link import Link
from src.Network import Network
from src.Node import Node
from src.Params import Params
from src.PASList import PASList
from src.Timing import SolveTimeLimit
from src.Zone import Zone

HAS_DOCPLEX = importlib.util.find_spec("docplex") is not None
HAS_CPLEX = importlib.util.find_spec("cplex") is not None
if HAS_DOCPLEX:
    from docplex.mp.model import Model
    from src.NewOA_DNDP_withtimelimit import OA_DNDP_CG


def parallel_network():
    n = Network.__new__(Network)
    r, s = Zone(1), Zone(2)
    r.addDemand(s, 1)
    r.addBikeDemand(s, 0.1)
    n.nodes = n.zones = [r, s]
    n.origins = [r]
    n.links = [Link(0, r, s, 20, 1, 100, 0.15, 4, 0),
               Link(1, r, s, 24, 3, 100, 0.15, 4, 1)]
    n.links2 = []
    n.params = Params()
    n.allPAS = PASList()
    n.type = "UE"
    n.TD, n.TD_bike = 1, 0.1
    return n


class NetworkRegressionTests(unittest.TestCase):
    def test_trace_rejects_unreachable_and_cyclic_predecessors(self):
        n = Network.__new__(Network)
        r, s, t = Node(1), Node(2), Node(3)
        with self.assertRaisesRegex(ValueError, "No valid path"):
            n.trace(r, s)
        with self.assertRaisesRegex(ValueError, "No valid tree path"):
            n.traceTree({}, r, s)
        a = Link(0, t, s, 1, 1, 1, 0.15, 4, 0)
        b = Link(1, s, t, 1, 1, 1, 0.15, 4, 0)
        s.pred, t.pred = a, b
        with self.assertRaises(ValueError):
            n.trace(r, s)
        with self.assertRaises(ValueError):
            n.traceTree({s: a, t: b}, r, s)
        s.pred = Link(2, r, s, 1, 1, 1, 0.15, 4, 0)
        self.assertEqual(n.trace(r, s).links, {s.pred})
        self.assertEqual(n.traceTree({s: s.pred}, r, s), [s.pred])
        self.assertEqual(n.trace(r, r).links, set())

    def test_shortest_path_can_restart_after_deadline_interrupt(self):
        n = parallel_network()
        r, s = n.nodes
        with patch("src.Network.check_deadline", side_effect=[None, None, SolveTimeLimit()]):
            with self.assertRaises(SolveTimeLimit):
                n.dijkstras(r, "UE", deadline=100)
        n.dijkstras(r, "UE")
        self.assertEqual(n.trace(r, s).links, {n.links[0]})

    def test_tapas_checks_deadline_before_starting(self):
        n = parallel_network()
        with self.assertRaises(SolveTimeLimit):
            n.tapas("UE", None, deadline=time.monotonic() - 1)
        self.assertIsNone(n.origins[0].bush)


@unittest.skipUnless(HAS_DOCPLEX, "Docplex is not installed")
class SolverRegressionTests(unittest.TestCase):
    def setUp(self):
        self.output = contextlib.redirect_stdout(io.StringIO())
        self.output.__enter__()
        self.addCleanup(self.output.__exit__, None, None, None)
        self.n = parallel_network()
        self.s = OA_DNDP_CG(self.n, 1, budget_percentage=100)
        self.s.best_ub = self.s.inf
        self.s.best_y = None
        self.b = self.n.links[1]
        self.s.initRMP()
        self.addCleanup(self.s.rmp.end)

    def test_fixed_design_uses_shortest_bike_route_and_caches_complete_flows(self):
        # The initial RMP contains only the 100-minute bike route. Building b
        # makes its initially omitted route cost 72 minutes.
        self.s.TAP = Mock(side_effect=lambda bike, y: (
            {a: 0 for a in self.n.links}, self.s.calcTSTT_bike(bike, y)))
        self.s.solveRMP = Mock(side_effect=AssertionError("Fixed node must not solve RMP"))
        node = BB_node({self.b: 1}, 1, 0, math.inf, None, 0, True)
        status, y = self.s.solveNode(node, 1)
        self.assertEqual(status, "fixed")
        self.assertAlmostEqual(self.s.best_ub, 7.2)
        self.assertAlmostEqual(self.s.best_x_bike[self.b], 0.1)
        self.assertEqual(self.s.best_x_bike[self.n.links[0]], 0)
        stale_bike = {self.n.links[0]: 0.1, self.b: 0}
        self.assertAlmostEqual(self.s.getUB(stale_bike, y), 7.2)
        self.s.TAP.assert_called_once()

    def test_integral_relaxation_does_not_discard_other_designs(self):
        old_model = self.s.rmp
        self.s.initRMP = Mock()  # Keep the small model, without an optimizer call.
        zero_flows = {a: 0 for a in self.n.links}
        self.s.CG = Mock(return_value=("solved", 100, zero_flows, zero_flows, {self.b: 0}))
        self.s.addOAcuts = Mock()
        visited = []

        def evaluate(_bike, y):
            value = round(y[self.b])
            visited.append(value)
            obj = 150 if value == 0 else 120
            if obj < self.s.best_ub:
                self.s.best_ub = obj
                self.s.best_y = {self.b: value}
            return obj

        self.s.getUB = evaluate
        result = self.s.solve()
        self.assertIn(1, visited)
        self.assertEqual(result[0], 120)
        self.assertEqual(result[6], 0)
        self.assertEqual(len(result), 12)
        self.assertIs(self.s.rmp, old_model)

    def test_infeasible_restricted_master_keeps_unfixed_designs(self):
        self.s.CG = Mock(return_value=("infeasible", math.inf, {}, {}, {}))
        node = BB_node({}, 0, 5, math.inf, None, 0, True)
        status, y = self.s.solveNode(node, 1)
        self.assertEqual(status, "unresolved")
        self.assertEqual(node.lb, 5)
        self.assertEqual(y[self.b], 0.5)

    def test_beckmann_cut_touches_anchor_and_underestimates_nearby_points(self):
        self.b.C = 100
        self.b.cap_per_bikelane = 25
        for yhat in (0, 0.5, 1):
            self.s.addOACutLink(self.b, 100, yhat)
            cut = list(self.s.rmp.iter_constraints())[-1]
            for x in (70, 100, 130):
                for y in (0, 0.5, 1):
                    solution = self.s.rmp.new_solution()
                    solution.add_var_value(self.s.rmp.x_car[self.b], x)
                    solution.add_var_value(self.s.rmp.u[self.b], -25 * y)
                    rhs = solution.get_value(cut.rhs)
                    exact = self.b.getPrimitiveTravelTimeC(x, y)
                    self.assertLessEqual(rhs, exact + 1e-9)
                    if x == 100 and y == yhat:
                        self.assertAlmostEqual(rhs, exact)

    def test_timeout_during_initialization_preserves_incumbent_and_return_shape(self):
        clock = [0.0]
        self.s.timelimit = 1
        self.s.use_ub_initialization = True
        self.s.initRMP = Mock()

        def initialize():
            self.s.best_ub = 50
            self.s.best_y = {self.b: 0}
            clock[0] = 2
            self.s._check_time_limit()

        self.s.initializeUpperBound = initialize
        with patch("time.monotonic", side_effect=lambda: clock[0]):
            result = self.s.solve()
        self.assertEqual(self.s.solve_status, "TimeLimit")
        self.assertEqual(result[0], 50)
        self.assertEqual(result[1], 2)
        self.assertEqual(len(result), 12)
        self.s.initRMP.assert_not_called()
        self.assertIsNone(self.s._deadline)

    def test_timeout_preserves_interrupted_node_lower_bound(self):
        self.s.initRMP = Mock()

        def interrupted(node, _iterations):
            self.s.best_ub = 100
            self.s.best_y = {self.b: 0}
            node.lb = 20
            raise SolveTimeLimit()

        self.s.solveNode = interrupted
        result = self.s.solve()
        self.assertEqual(self.s.lower_bound, 20)
        self.assertEqual(result[6], 4)
        self.assertEqual(len(result), 12)

    def test_tree_exhaustion_without_incumbent_keeps_return_contract(self):
        self.s.initRMP = Mock()
        self.s.solveNode = Mock(return_value=("infeasible", None))
        result = self.s.solve()
        self.assertEqual(self.s.solve_status, "TreeExhausted")
        self.assertEqual(len(result), 12)
        self.assertTrue(math.isinf(result[0]))
        self.assertTrue(math.isinf(result[6]))
        self.assertEqual(result[9:11], ("N/A", []))

    def test_tap_restores_lane_state_on_timeout(self):
        self.b.bikelanes = 0
        self.n.tapas = Mock(side_effect=SolveTimeLimit())
        with self.assertRaises(SolveTimeLimit):
            self.s.TAP({a: 0 for a in self.n.links}, {self.b: 1})
        self.assertEqual(self.b.bikelanes, 0)
        self.assertGreaterEqual(self.s.rt_TAP, 0)

    def test_rmp_accumulates_time_and_passes_remaining_deadline(self):
        clock = [0.0]
        limits = []

        def solve(**_kwargs):
            clock[0] += 2
            return object()

        fake = SimpleNamespace(solve=solve, set_time_limit=limits.append,
                               solve_details=SimpleNamespace(status="optimal"), objective_value=5)
        self.s.rmp = fake
        self.s.network = SimpleNamespace(links=[])
        self.s.dem_cons_car = self.s.dem_cons_bike = {}
        self.s._deadline = 10
        with patch("time.monotonic", side_effect=lambda: clock[0]):
            self.s.solveRMP()
            self.s.solveRMP()
        self.assertEqual(self.s.rt_RMP, 4)
        self.assertEqual(limits, [10, 8])
        fake.solve_details.status = "time limit exceeded"
        with self.assertRaises(SolveTimeLimit):
            self.s._deadline = None
            self.s.solveRMP()

    def test_value_function_cut_dedup_checks_both_directions(self):
        self.s.addVFCut = Mock()
        self.s.vf_cuts = [{a: 100 for a in self.n.links}]
        self.s.addVFCuts({a: 50 for a in self.n.links})
        self.assertEqual(len(self.s.vf_cuts), 2)
        self.s.addVFCut.assert_called_once()

    @unittest.skipUnless(HAS_CPLEX, "CPLEX is not installed")
    def test_real_solver_handles_zero_budget_and_full_budget(self):
        for percentage, expected_y in ((0, 0), (100, 1)):
            with self.subTest(budget_percentage=percentage):
                s = OA_DNDP_CG(parallel_network(), 1, budget_percentage=percentage)
                s.w = 5
                s.timelimit = 5
                result = s.solve()
                self.addCleanup(s.rmp.end)
                self.assertEqual(len(result), 12)
                self.assertEqual(s.best_y[s.varlinks[0]], expected_y)
                self.assertEqual(s.solve_status, "Converged")
                self.assertAlmostEqual(result[0], result[7] + result[8])

    @unittest.skipUnless(HAS_CPLEX, "CPLEX is not installed")
    def test_real_solver_matches_enumeration_on_harker_friesz(self):
        root = Path(__file__).resolve().parents[1]
        # Use three of the supplied network's candidates to keep this check
        # within the CPLEX Community Edition model size limit.
        with contextlib.chdir(root):
            n = Network("HarkerFriesz", "HF_CNDP_1", 0.5, 1, 1, 0.25)
        for a in n.links[3:]:
            a.cost = 0
        n.params.min_gap = 1e-7
        n.params.tapas_max_iter = 500
        s = OA_DNDP_CG(n, 1)
        s.best_ub = s.inf
        s.params.min_NDP_gap = 1e-5
        feasible = []
        for values in itertools.product((0, 1), repeat=len(s.varlinks)):
            y = dict(zip(s.varlinks, values))
            if s.checkBudgetFeasible(y):
                feasible.append(s.evaluateSolution(y))
        s.timelimit = 10
        result = s.solve()
        self.addCleanup(s.rmp.end)
        self.assertEqual(len(result), 12)
        self.assertEqual(s.solve_status, "Converged")
        self.assertAlmostEqual(result[0], min(feasible), delta=1e-3)
        self.assertTrue(s.checkBudgetFeasible(s.best_y))
        self.assertAlmostEqual(result[0], result[7] + result[8])
        self.assertLess(result[6], s.params.min_NDP_gap)


if __name__ == "__main__":
    unittest.main()
