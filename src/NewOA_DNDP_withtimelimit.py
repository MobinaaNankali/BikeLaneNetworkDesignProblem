import math
import time
import random
from src import Params
from src import BB_node
from src import YDict
from docplex.mp.model import Model
from src.Timing import SolveTimeLimit, check_deadline

class OA_DNDP_CG:
    
    def __init__(self, network, inflate_costs, use_ub_initialization=False, budget_percentage=50.0):
        self.network = network
        self.M = self.network.TD 
        self.CG_tol = 1e-4
        self.inf = math.inf

        self.best_solution_bikeshare = 0.0  
        self.best_solution_carshare = 0.0   
        self.bikeshare = 0.0
        self.carshare = 0.0
        
        self.gap = 100
        self.w = 1
        self.tstt = 10000000000
        self.last_xhat = None
        self.last_obj_f = 1000000000


        self.rt_TAP = 0.0
        self.rt_RMP = 0.0
        self.rt_pricing = 0.0

        # Bike lane bonus: percentage reduction in travel time (0.2 = 20% reduction)
        self.bikelane_bonus = 0.4
        
        # UB initialization flag
        self.use_ub_initialization = use_ub_initialization
        
        # Budget percentage parameter (default 50%)
        self.budget_percentage = budget_percentage
        
        # Track UB history for each BB node iteration
        self.ub_history = []  # List of (iteration, ub_value) tuples
        
        
        self.car_paths = {r:{s:[] for s in self.network.zones} for r in self.network.origins}
        self.bike_paths = {r:{s:[] for s in self.network.zones} for r in self.network.origins}
        
        self.y_fix_cons = list()
        self.tstt_cuts = dict()
        self.vf_cuts = list()
        
        
        for a in self.network.links:
           self.tstt_cuts[a] = list()

        for a in self.network.links2:
            a.bikelanes = 0
            
        self.varlinks = []
        
        for a in self.network.links:
            #print(a, a.cost)
            if a.cost > 1e-6:
                self.varlinks.append(a)
                
        #self.budget = 30
        total_cost = 0.0
        for a in self.varlinks:
            print(a.length, a.cost)
            total_cost +=  a.cost * a.length
        
        print("total_cost is ", total_cost)
        print(f"Budget is {self.budget_percentage}% of the total cost")
        self.budget = (self.budget_percentage / 100.0) * total_cost
        
        self.timelimit = 3600 * 4
        
        self.ydict = YDict.YDict()
                
        print("yvars", len(self.varlinks))
        

        self.link_cons = dict()
        self.dem_cons = dict()
        
        
        self.params = Params.Params()
        self._deadline = None
        self._solution_cache = {}
    
    
    def initializeUpperBound(self):
        """
        Initialize a good upper bound using heuristics before starting B&B.
        This helps prune the search tree early and speeds up convergence.
        """
        print("\n" + "="*50)
        print("INITIALIZING UPPER BOUND WITH HEURISTICS")
        print("="*50)
        
        best_heuristic_ub = self.inf
        best_heuristic_y = None
        
        # Strategy 1: Baseline - no bike lanes (y=0 for all)
        print("Strategy 1: Baseline (no bike lanes)...")
        y_baseline = {a: 0 for a in self.varlinks}
        ub_baseline = self.evaluateSolution(y_baseline)
        print(f"  Baseline UB: {ub_baseline:.4f}")
        
        if ub_baseline < best_heuristic_ub:
            best_heuristic_ub = ub_baseline
            best_heuristic_y = y_baseline.copy()
        
        # Strategy 2: Greedy by potential benefit/cost ratio
        print("Strategy 2: Greedy by benefit/cost ratio...")
        y_greedy = self.greedyHeuristic()
        if y_greedy is not None:
            ub_greedy = self.evaluateSolution(y_greedy)
            print(f"  Greedy UB: {ub_greedy:.4f}")
            if ub_greedy < best_heuristic_ub:
                best_heuristic_ub = ub_greedy
                best_heuristic_y = y_greedy.copy()
        
        # Strategy 3: Random feasible solutions (try a few)
        print("Strategy 3: Random feasible solutions...")
        num_random = min(10, max(3, len(self.varlinks) // 10))
        for i in range(num_random):
            y_random = self.randomFeasibleSolution()
            if y_random is not None:
                ub_random = self.evaluateSolution(y_random)
                if ub_random < best_heuristic_ub:
                    best_heuristic_ub = ub_random
                    best_heuristic_y = y_random.copy()
                    print(f"  Random {i+1} UB: {ub_random:.4f} (new best!)")
        
        # Strategy 4: Greedy by link length
        print("Strategy 4: Greedy by link length...")
        y_length = self.greedyByLength()
        if y_length is not None:
            ub_length = self.evaluateSolution(y_length)
            print(f"  Length-based UB: {ub_length:.4f}")
            if ub_length < best_heuristic_ub:
                best_heuristic_ub = ub_length
                best_heuristic_y = y_length.copy()
        
        # Update best upper bound
        if best_heuristic_ub < self.inf:
            self.best_ub = best_heuristic_ub
            self.best_y = best_heuristic_y
            print(f"\nInitial Upper Bound: {self.best_ub:.4f}")
            print(f"Selected links: {sum(best_heuristic_y.values())}/{len(self.varlinks)}")
        else:
            print("Warning: Could not find a good initial upper bound")
        
        print("="*50 + "\n")
        
        return self.best_ub
    
    def greedyHeuristic(self):
        y = {a: 0 for a in self.varlinks}
        link_scores = []
        for a in self.varlinks:
            # Exclude zero-length connector links
            if a.length > 1e-6:
                link_cost = a.cost * a.length
                if link_cost > 1e-6:
                    benefit = a.getTTBike() * self.bikelane_bonus * a.length
                    score = benefit / link_cost
                    link_scores.append((a, score, link_cost))
        link_scores.sort(key=lambda x: x[1], reverse=True)
        used_budget = 0
        for a, score, cost in link_scores:
            if used_budget + cost <= self.budget:
                y[a] = 1
                used_budget += cost
        return y
    
    def greedyByLength(self):
        y = {a: 0 for a in self.varlinks}
        sorted_links = sorted(self.varlinks, key=lambda a: a.length, reverse=True)
        used_budget = 0
        for a in sorted_links:
            cost = a.cost * a.length
            if used_budget + cost <= self.budget:
                y[a] = 1
                used_budget += cost
        return y
    
    def randomFeasibleSolution(self):
        y = {a: 0 for a in self.varlinks}
        shuffled = list(self.varlinks)
        random.shuffle(shuffled)
        used_budget = 0
        for a in shuffled:
            cost = a.cost * a.length
            if random.random() < 0.5 and used_budget + cost <= self.budget:
                y[a] = 1
                used_budget += cost
        return y
    
    def _check_time_limit(self):
        check_deadline(self._deadline)

    def evaluateSolution(self, y):
        return self._evaluate_design(y, add_cuts=False)

    def _evaluate_design(self, y, add_cuts):
        """Evaluate a binary design independently of the current RMP path set."""
        self._check_time_limit()
        if not self.isInteger(y):
            raise ValueError("Upper bounds require a binary lane design")
        y = self.roundY(y)
        if not self.checkBudgetFeasible(y):
            return self.inf

        key = tuple(y[a] for a in self.varlinks)
        if key not in self._solution_cache:
            x_bike = self.computeBikeFlows(y)
            x_car, _ = self.TAP(x_bike, y)
            self._solution_cache[key] = (x_car.copy(), x_bike.copy())
        x_car, x_bike = self._solution_cache[key]
        obj = self.calcOFV(x_car, x_bike, y)
        self.ydict.insertUE(y.copy(), obj)
        if obj < self.best_ub:
            self.best_ub = obj
            self.best_y = y.copy()
            self.best_x_car = x_car.copy()
            self.best_x_bike = x_bike.copy()
            self.best_solution_bikeshare = self.bikeshare
            self.best_solution_carshare = self.carshare * self.w
        if add_cuts:
            self.addVFCuts(x_car)
            self.addOAcuts(x_car, y)
        return obj

    def computeBikeFlows(self, y):
        # Set travel time for bike routing
        for a in self.network.links:
            if a in self.varlinks and y[a] == 1:
                a.dual = a.getTTBike() * (1 - self.bikelane_bonus)
            else:
                a.dual = a.getTTBike()
        x_bike = {a: 0 for a in self.network.links}
        for r in self.network.origins:
            if r.totaldemand_bike > 0:
                self.network.dijkstras(r, 'RC', deadline=self._deadline)
                for s in self.network.zones:
                    demand = r.getDemandBike(s)
                    if demand > 0:
                        p = self.network.trace(r, s)
                        for a in p.links:
                            x_bike[a] += demand
        return x_bike

    @staticmethod
    def _relative_gap(upper, lower):
        if not math.isfinite(upper):
            return float("inf")
        if upper <= lower:
            return 0.0
        if lower <= 0:
            return float("inf")
        return (upper - lower) / lower

    def solve(self):
        start_time = time.monotonic()
        self._deadline = start_time + self.timelimit
        self.best_ub = self.inf
        self.best_y = None
        self.best_x_bike = None
        self.best_x_car = None
        self.best_solution_bikeshare = self.best_solution_carshare = 0.0
        self.rt_TAP = self.rt_RMP = self.rt_pricing = 0.0
        self.ub_history = []
        self._solution_cache = {}
        self.ydict = YDict.YDict()
        self.y_fix_cons = []
        self.vf_cuts = []
        self.tstt_cuts = {a: [] for a in self.network.links}
        self.car_paths = {r: {} for r in self.network.origins}
        self.bike_paths = {r: {} for r in self.network.origins}

        root = BB_node.BB_node({}, 0, 0, self.best_ub, None, 0, True)
        bbnodes = [root]
        active_node = None
        iteration = 0
        lb = 0
        status = "TreeExhausted"

        try:
            self._check_time_limit()
            if self.use_ub_initialization:
                self.initializeUpperBound()
            self._check_time_limit()
            self.initRMP()
            print("iteration", "lb", "best ub", "node ub", "gap", "time")

            while bbnodes:
                self._check_time_limit()
                active_node = min(
                    bbnodes, key=lambda n: (n.lb, -len(n.y_fix), -n.num_one)
                )
                bbnodes.remove(active_node)
                if active_node.lb >= self.best_ub:
                    active_node = None
                    continue

                iteration += 1
                node_status, y = self.solveNode(active_node, 10 if iteration == 1 else 1)
                if node_status not in ("infeasible", "fixed", "pruned"):
                    open_links = [a for a in self.varlinks if a not in active_node.y_fix]
                    # An integral OA point need not solve this node's original
                    # problem. Partition the remaining designs if its bound
                    # still cannot prune the node, even when y is integral.
                    branch_link = max(open_links, key=lambda a: abs(y[a] - round(y[a])))
                    for value in (0, 1):
                        y_fix = active_node.y_fix.copy()
                        y_fix[branch_link] = value
                        bbnodes.append(BB_node.BB_node(
                            y_fix, active_node.num_one + value, active_node.lb,
                            active_node.ub, active_node, iteration, True
                        ))

                node_ub = active_node.ub
                active_node = None
                lb = min((n.lb for n in bbnodes), default=self.best_ub)
                gap = self._relative_gap(self.best_ub, lb)
                self.ub_history.append((iteration, self.best_ub))
                print(iteration, lb, self.best_ub, node_ub, gap,
                      time.monotonic() - start_time)
                if gap < self.params.min_NDP_gap:
                    status = "Converged"
                    break
        except SolveTimeLimit:
            status = "TimeLimit"
            # A node interrupted during CG/TAP must remain in the global bound.
            bounds = [n.lb for n in bbnodes]
            if active_node is not None:
                bounds.append(active_node.lb)
            lb = min(bounds, default=self.best_ub)
        finally:
            self._deadline = None

        if not bbnodes and active_node is None and status != "TimeLimit":
            lb = self.best_ub
        elapsed = time.monotonic() - start_time
        self.solve_status = status
        self.lower_bound = lb
        self.gap = self._relative_gap(self.best_ub, lb)
        if not self.ub_history or self.ub_history[-1] != (iteration, self.best_ub):
            self.ub_history.append((iteration, self.best_ub))
        selected = [] if self.best_y is None else [
            str(a) for a in self.varlinks if self.best_y[a] == 1
        ]
        selected_str = "N/A" if self.best_y is None else f"{len(selected)}/{len(self.varlinks)}"
        print(status, "best ub:", self.best_ub, "lb:", lb, "gap:", self.gap)
        print("Selected links:", selected_str, "Total time:", elapsed)
        return (
            self.best_ub, elapsed, self.rt_TAP, self.rt_pricing, self.rt_RMP,
            iteration, self.gap, self.best_solution_bikeshare,
            self.best_solution_carshare, selected_str, selected, self.ub_history
        )

    def addVFCut(self, x):

        if self.params.useValueFunctionCuts1 == True:
            xi = {a:self.rmp.continuous_var(lb=0) for a in self.network.links}
            
            vf_eps = 0.01
            
            self.rmp.add_constraint(sum(self.rmp.beta[a] for a in self.network.links) <= sum(xi[a] for a in self.network.links) + vf_eps)
            for a in self.network.links:
                
                
                int_tt_no_bike = a.getPrimitiveTravelTimeC(x[a], 0)
                
                if a in self.varlinks:
                    int_tt_bike = a.getPrimitiveTravelTimeC(x[a], 1)
                    
                    self.rmp.add_constraint(xi[a] <= int_tt_bike)
                    M = int_tt_bike - int_tt_no_bike
                    
                    self.rmp.add_constraint(xi[a] <= int_tt_no_bike + M * self.rmp.y[a]) 
                    
                else:
                    self.rmp.add_constraint(xi[a] <= int_tt_no_bike)
                
        
    
    def initRMP(self):
        
        for r in self.network.origins:
            self._check_time_limit()
            if r.totaldemand > 0:
                self.network.dijkstras(r, "UE", deadline=self._deadline)

                self.car_paths[r] = dict()

                for s in self.network.zones:
                    if r.getDemand(s) > 0:
                        p = self.network.trace(r, s)
                        self.car_paths[r][s] = list()
                        self.car_paths[r][s].append(p)
                        
        for r in self.network.origins:
            self._check_time_limit()
            if r.totaldemand_bike > 0:
                self.network.dijkstras(r, "bike", deadline=self._deadline)
            
                self.bike_paths[r] = dict()

                for s in self.network.zones:
                    if r.getDemandBike(s) > 0:  # Fixed: use getDemandBike instead of getDemand
                        p = self.network.trace(r, s)
                        self.bike_paths[r][s] = list()
                        self.bike_paths[r][s].append(p)
                        
    
        self.rmp = Model()
        self.rmp.parameters.read.scale = -1
        
        self.rmp.M = dict()
        
        for a in self.network.links:
            self._check_time_limit()
            self.rmp.M[a] = a.getTTBike() * self.bikelane_bonus * self.network.TD_bike
        
        
        self.link_cons_car = dict()
        self.link_cons_bike = dict()
        self.dem_cons_car = dict()
        self.dem_cons_bike = dict()
        
        self.bike_obj = dict()
        
        self.rmp.zeta = {a:self.rmp.continuous_var(lb=0) for a in self.network.links} # obj OA
        self.rmp.beta = {a:self.rmp.continuous_var(lb=0) for a in self.network.links} # beckmann OA
        self.rmp.zeta_bike = {a:self.rmp.continuous_var(lb=0) for a in self.network.links} # bike obj
        self.rmp.y = {a:self.rmp.continuous_var(lb=0, ub=1) for a in self.varlinks}
        self.rmp.u = {a:self.rmp.continuous_var(lb=-a.C, ub=0) for a in self.varlinks}

        self.rmp.w = {a:self.rmp.continuous_var(lb=0) for a in self.varlinks}


        self.rmp.x_bike = {a:self.rmp.continuous_var(lb=0, ub=self.network.TD) for a in self.network.links}
        self.rmp.x_car = {a:self.rmp.continuous_var(lb=0, ub=self.network.TD) for a in self.network.links}
        self.rmp.h_bike = {p:self.rmp.continuous_var(lb=0) for p in self.getBikePaths()}
        self.rmp.h_car = {p:self.rmp.continuous_var(lb=0) for p in self.getCarPaths()}
        
        for a in self.network.links:
            self._check_time_limit()
            self.link_cons_car[a] = self.rmp.add_constraint(self.rmp.x_car[a] - sum(self.rmp.h_car[p] for p in self.getCarPaths() if a in p.links) >= 0)
            self.link_cons_bike[a] = self.rmp.add_constraint(self.rmp.x_bike[a] - sum(self.rmp.h_bike[p] for p in self.getBikePaths() if a in p.links) >= 0)
            
        for a in self.varlinks:
            self._check_time_limit()
            self.rmp.add_constraint(self.rmp.u[a] == -self.rmp.y[a] * a.cap_per_bikelane)
            self.rmp.add_constraint(self.rmp.w[a] <= self.rmp.y[a] * self.M)
            self.rmp.add_constraint(self.rmp.w[a] <= self.rmp.x_bike[a])
            self.rmp.add_constraint(self.rmp.w[a] >= self.rmp.x_bike[a] - self.M * (1 - self.rmp.y[a]))



        for r in self.network.origins:
            self._check_time_limit()
            for s in self.network.zones:
                if r.getDemand(s) > 0:
                    self.dem_cons_car[(r,s)] = self.rmp.add_constraint(sum(self.rmp.h_car[p] for p in self.car_paths[r][s]) >= r.getDemand(s))
                if r.getDemandBike(s) > 0:
                    self.dem_cons_bike[(r,s)] = self.rmp.add_constraint(sum(self.rmp.h_bike[p] for p in self.bike_paths[r][s]) >= r.getDemandBike(s))  # Fixed: use getDemandBike
        
        # budget
        self.rmp.add_constraint(sum(self.rmp.y[a] * a.cost * a.length for a in self.varlinks) <= self.budget)
      
        for a in self.network.links:
            self._check_time_limit()
            self.rmp.add_constraint(self.rmp.zeta[a] >= a.t_ff * self.rmp.x_car[a])
            if a in self.varlinks:
                self.tstt_cuts[a].append( (0, 0))
            else:
                self.tstt_cuts[a].append(0)
        #
        #for a in self.network.links:
        #    if a in self.varlinks:
        #        self.bike_obj[a] = self.rmp.add_constraint(self.rmp.zeta_bike[a] >= self.rmp.x_bike[a] * a.getTTBike() * (1 - self.bikelane_bonus * self.rmp.y[a].ub))  
        #        self.rmp.add_constraint(self.rmp.zeta_bike[a] >= self.rmp.x_bike[a] * a.getTTBike() - self.rmp.M[a] * self.rmp.y[a])
        #    else:
        #        self.bike_obj[a] = self.rmp.add_constraint(self.rmp.zeta_bike[a] == self.rmp.x_bike[a] * a.getTTBike())

        self.rmp.tstt = sum(self.rmp.zeta[a] for a in self.network.links)
        self.rmp.tstt_bike = (
            sum(self.rmp.x_bike[a] * a.getTTBike() for a in self.network.links) -
            sum(self.rmp.w[a] * a.getTTBike() * self.bikelane_bonus for a in self.varlinks)
        )
        self.rmp.minimize((self.w *self.rmp.tstt) + self.rmp.tstt_bike)        
    
    
    def solveNode(self, bbnode, max_iter):
        self._check_time_limit()
        for cons in self.y_fix_cons:
            self.rmp.remove(cons)
        self.y_fix_cons = []
        y = bbnode.y_fix.copy()
        used_budget = sum(y[a] * a.cost * a.length for a in y)
        if used_budget > self.budget + 1e-6:
            return "infeasible", None

        open_links = [a for a in self.varlinks if a not in y]
        cheapest = min((a.cost * a.length for a in open_links), default=math.inf)
        if self.budget - used_budget + 1e-6 < cheapest:
            y.update({a: 0 for a in open_links})
        fixed = len(y) == len(self.varlinks)
        for a, value in y.items():
            self.y_fix_cons.append(self.rmp.add_constraint(self.rmp.y[a] == value))

        if fixed:
            # A single design needs UE car assignment and shortest bike routes,
            # not a solve over whatever columns happen to be in the RMP.
            obj = self.getUB(None, y)
            bbnode.lb = bbnode.ub = obj
            bbnode.bikeshareBB = self.bikeshare
            bbnode.carshareBB = self.carshare
            return "fixed", y

        for _ in range(max_iter):
            self._check_time_limit()
            cg_status, obj_l, x_car, x_bike, y = self.CG()
            if cg_status == "infeasible":
                # Missing columns can make the restricted model infeasible.
                # Preserve the inherited bound and explore the unfixed designs.
                return "unresolved", {a: bbnode.y_fix.get(a, 0.5) for a in self.varlinks}
            bbnode.lb = max(bbnode.lb, obj_l)
            if bbnode.lb >= self.best_ub:
                return "pruned", y
            self.addOAcuts(x_car, y)
            if self.isInteger(y):
                obj_f = self.getUB(x_bike, y)
                bbnode.ub = min(bbnode.ub, obj_f)
                if bbnode.lb >= self.best_ub:
                    return "pruned", y
        return "solved", y

    def checkBudgetFeasible(self, y):
        used_budget = sum(y[a] * a.cost * a.length for a in self.varlinks)
        return used_budget <= self.budget + 1e-6

    def getUB(self, x_l_bike, y):
        # Keep the call signature for existing callers; RMP bike flows may omit
        # the optimal route, so evaluate the binary design independently.
        return self._evaluate_design(y, add_cuts=True)

    def isInteger(self, y):
        for a in self.varlinks:
            if abs(y[a] - round(y[a])) > 0.001:
                return False
                
        return True
        
    def pricing(self, link_duals_car, link_duals_bike, dem_duals_car, dem_duals_bike):
        self._check_time_limit()
        rt_pricing = time.monotonic()
        try:
            return self._price_paths(link_duals_car, link_duals_bike, dem_duals_car, dem_duals_bike)
        finally:
            self.rt_pricing += time.monotonic() - rt_pricing

    def _price_paths(self, link_duals_car, link_duals_bike, dem_duals_car, dem_duals_bike):
        new = 0
        minrc = self.inf
        
        #print("len check", len(link_duals))
        for a in self.network.links:
            a.dual = link_duals_car[a]
        
        for r in self.network.origins:
            self.network.dijkstras(r, 'RC', deadline=self._deadline)
            
            for s in self.network.zones:
                
                if r.getDemand(s) > 0:
                    
                
                    rc = - dem_duals_car[(r,s)] + s.cost                
                    
                    if rc < - self.CG_tol:
                        p = self.network.trace(r,s)
                        self.car_paths[r][s].append(p) #---is it needed to store paths if directly adding to RMP?
                        
                        #---add new path var to RMP                                                
                        self.rmp.h_car[p] = self.rmp.continuous_var(lb=0)
                        
                        #---update RMP constraints
                        self.dem_cons_car[(r, s)].lhs.add_term(self.rmp.h_car[p], 1)
                        
                        for a in p.links:
                            self.link_cons_car[a].lhs.add_term(self.rmp.h_car[p], -1)
                        
                        new += 1
                    
                        if rc < minrc:
                            minrc = rc
                            
        for a in self.network.links:
            a.dual = link_duals_bike[a]
        
        for r in self.network.origins:
            self.network.dijkstras(r, 'RC', deadline=self._deadline)
            
            for s in self.network.zones:
                
                if r.getDemandBike(s) > 0:
                
                    rc = - dem_duals_bike[(r,s)] + s.cost                    
                    
                    if rc < - self.CG_tol:
                        p = self.network.trace(r,s)
                        self.bike_paths[r][s].append(p) #---is it needed to store paths if directly adding to RMP?
                        
                        #---add new path var to RMP                                                
                        self.rmp.h_bike[p] = self.rmp.continuous_var(lb=0)
                        
                        #---update RMP constraints
                        self.dem_cons_bike[(r, s)].lhs.add_term(self.rmp.h_bike[p], 1)
                        
                        for a in p.links:
                            self.link_cons_bike[a].lhs.add_term(self.rmp.h_bike[p], -1)
                        
                        new += 1
                    
                        if rc < minrc:
                            minrc = rc

        return minrc
        
    def solveRMP(self):
        self._check_time_limit()
        if self._deadline is not None:
            self.rmp.set_time_limit(max(1e-6, self._deadline - time.monotonic()))
        t_solve = time.monotonic()
        try:
            solution = self.rmp.solve(log_output=False)
        finally:
            self.rt_RMP += time.monotonic() - t_solve

        status = self.rmp.solve_details.status
        if "time limit" in status.lower():
            raise SolveTimeLimit("RMP time limit reached")
        self._check_time_limit()
        if status in ("infeasible", "integer infeasible"):
            return "infeasible", self.inf, {}, {}, {}, {}
        # Pricing needs optimal LP duals; a partial solve is not a node bound.
        if solution is None or "optimal" not in status.lower():
            raise RuntimeError(f"RMP did not solve to optimality: {status}")

        link_duals_car = {a: self.link_cons_car[a].dual_value for a in self.network.links}
        link_duals_bike = {a: self.link_cons_bike[a].dual_value for a in self.network.links}
        dem_duals_car = {od: cons.dual_value for od, cons in self.dem_cons_car.items()}
        dem_duals_bike = {od: cons.dual_value for od, cons in self.dem_cons_bike.items()}
        return (status, self.rmp.objective_value, link_duals_car, link_duals_bike,
                dem_duals_car, dem_duals_bike)

    def CG(self):
        conv = False
        nCG = 0
        OFV = self.inf
        
        starttime = time.monotonic()
        
        
        while conv == False:
            self._check_time_limit()
            RMP_status, OFV, link_duals_car, link_duals_bike, dem_duals_car, dem_duals_bike = self.solveRMP()
            #print("solved?", RMP_status)
            
            
            if RMP_status == 'infeasible':
                CG_status = 'infeasible'
                
                return CG_status, self.inf, dict(), dict(), dict()
            else:
                CG_status = "solved"
                
            minrc = self.pricing(link_duals_car, link_duals_bike, dem_duals_car, dem_duals_bike)

            elapsed = time.monotonic() - starttime
            
            if self.params.PRINT_CG_INFO:
                npaths = len(self.getCarPaths()) + len(self.getBikePaths())
                print('CG: %d\t%d\t%.1f\t%.2f\t%.2f' % (nCG,npaths,OFV,minrc, elapsed))
                
            if minrc > -self.CG_tol:
                conv = True
                
            nCG += 1
        
        yhat = {a:self.rmp.y[a].solution_value for a in self.varlinks}
        print("yhat", yhat)
        x_l_car = {a:self.rmp.x_car[a].solution_value for a in self.network.links}
        x_l_bike = {a:self.rmp.x_bike[a].solution_value for a in self.network.links}
        
        return CG_status, OFV, x_l_car, x_l_bike, yhat
    
    
    def checkBudget(self, y):
        used_budget = 0
        
        for a in y:
            used_budget += y[a] * a.cost * a.length
        
        return self.budget > used_budget
    
    def roundY(self, y):
        y_output = dict()
        
        for a in y:
            y_output[a] = round(y[a])
    
        return y_output
        
    def addVFCuts(self, xhat_car):
        
        match_any = False
        for x_save in self.vf_cuts:
            match_one = True
            for a in self.network.links:
                if xhat_car[a] < self.params.VF_TOL_X:
                    if x_save[a] > self.params.VF_TOL_X:
                        match_one = False
                        break
                elif abs((xhat_car[a] - x_save[a]) / xhat_car[a]) > self.params.VF_TOL_X:
                    match_one = False
                    break
            
            if match_one:
                match_any = True
                break
        
        
        if not match_any:
            self.vf_cuts.append(xhat_car)
            self.addVFCut(xhat_car)
            
        
    def addOAcuts(self, xhat_car, yhat):
        
        for a in self.network.links:
            if xhat_car[a] < self.params.OA_TOL_X:
                continue
                
            if a in self.varlinks:
                found = False
                for xy in self.tstt_cuts[a]:
                    x = xy[0]
                    y = xy[1]
                    
                    if abs((x - xhat_car[a])/xhat_car[a]) < self.params.OA_TOL_X:
                        if yhat[a] == 0:
                            if y < self.params.OA_TOL_Y:
                                found = True
                                break
                        elif abs((y - yhat[a])/yhat[a]) < self.params.OA_TOL_Y:
                            found = True
                            break
                 
                if not found:
                    self.addOACutLink(a, xhat_car[a], yhat[a])
            else:
                found = False
                for x in self.tstt_cuts[a]:
                    if abs((x - xhat_car[a])/xhat_car[a]) < self.params.OA_TOL_X:
                        found = True
                        break
                
                if not found:
                    self.addOACutLink(a, xhat_car[a], 0)
                
        
    def addOACutLink(self, a, xhat, yhat):
        if a in self.varlinks:
            # this is TSTT cut
            firstterm = xhat * a.getTravelTimeC(xhat, yhat, "UE")
            secondterm = a.getTravelTimeC(xhat, yhat, "UE") + xhat * a.getDerivativeTravelTimeCx(xhat, yhat)
            thirdterm = xhat * a.getDerivativeTravelTimeCy(xhat, yhat)
            
            uhat = -yhat * a.cap_per_bikelane

            self.rmp.add_constraint(self.rmp.zeta[a] >= firstterm + secondterm * (self.rmp.x_car[a] - xhat) + thirdterm * (self.rmp.u[a] - uhat))
            
            self.tstt_cuts[a].append( (xhat, yhat))
            
            # this is Beckmann cut
            
            uhat = yhat * a.cap_per_bikelane
            xterm = (self.rmp.x_car[a] - xhat) * a.getTravelTimeC(xhat, yhat, "UE")
            # u is negative removed capacity; intdtdu uses positive reduction.
            yterm = (-self.rmp.u[a] - uhat) * a.intdtdu(xhat, uhat) 

            B = a.getPrimitiveTravelTimeC(xhat, yhat)
            
            self.rmp.add_constraint(self.rmp.beta[a] >= B + yterm + xterm)
            
            
            '''
            for i in range(0, 3):
                y = i * 0.5
                u = y * a.cap_per_bikelane
                xterm = (xhat - xhat) * a.getTravelTimeC(xhat, yhat, "UE")
                yterm = (u - uhat) * a.intdtdu(xhat, uhat) 
                B = a.getPrimitiveTravelTimeC(xhat, yhat)
                
                print("check", a, y, B + xterm + yterm <= a.getPrimitiveTravelTimeC(xhat, y), B + xterm + yterm, a.getPrimitiveTravelTimeC(xhat, y), xhat, y, yhat)
            '''    
                
        else:
            # this is TSTT cut
            firstterm = xhat * a.getTravelTimeC(xhat, 0, "UE")
            secondterm = a.getTravelTimeC(xhat, 0, "UE") + xhat * a.getDerivativeTravelTimeCx(xhat, 0)

            self.rmp.add_constraint(self.rmp.zeta[a] >= firstterm + secondterm * (self.rmp.x_car[a] - xhat))

            self.tstt_cuts[a].append(xhat)
            
            # this is Beckmann cut
            xterm = (self.rmp.x_car[a] - xhat) * a.getTravelTimeC(xhat, 0, "UE")

            B = a.getPrimitiveTravelTimeC(xhat, 0)
            self.rmp.add_constraint(self.rmp.beta[a] >= B + xterm)
                
                
                
    def TAP(self, xhat_bike, y):
        self._check_time_limit()
        previous_lanes = {a: a.bikelanes for a in self.varlinks}
        t0_TAP = time.monotonic()
        try:
            for a in self.varlinks:
                a.bikelanes = round(y[a])
            self.network.tapas("UE", None, deadline=self._deadline)
            self._check_time_limit()
            xhat_car = {a: a.x for a in self.network.links}
            obj_f = self.calcOFV(xhat_car, xhat_bike, y)
            return xhat_car, obj_f
        finally:
            self.rt_TAP += time.monotonic() - t0_TAP
            for a, lanes in previous_lanes.items():
                a.bikelanes = lanes

    def getBeckmanOFV(self, x_car, y):
        total = 0
        
        for a in self.network.links:
            if a in self.varlinks:
                total += a.getPrimitiveTravelTimeC(x_car[a], y[a])
            else:
                total += a.getPrimitiveTravelTimeC(x_car[a], 0)
        
        return total
        
    def calcTSTT_car(self, x_car, y):
        tstt = 0
        
        for a in self.network.links:
            if a in self.varlinks:
                tstt += x_car[a] * a.getTravelTimeC(x_car[a], y[a], "UE")
                #print(a, x_car[a] * a.getTravelTimeC(x_car[a], y[a], "UE"), self.rmp.zeta[a].solution_value)
            else:
                tstt += x_car[a] * a.getTravelTimeC(x_car[a], 0, "UE")
                #print(a, x_car[a] * a.getTravelTimeC(x_car[a], 0, "UE"), self.rmp.zeta[a].solution_value)
        
        return tstt
        
    def calcTSTT_bike(self, x_bike, y):
        biketstt = 0
        
        for a in self.network.links:
            if a in self.varlinks:
                biketstt += x_bike[a] * a.getTTBike() * (1 - self.bikelane_bonus * y[a])
            else:
                biketstt += x_bike[a] * a.getTTBike()
        
        return biketstt
            
    def calcOFV(self, x_car, x_bike, y):
        TSTT_car = self.calcTSTT_car(x_car, y)
        TSTT_bike = self.calcTSTT_bike(x_bike, y)
        print("This is OFV value separetly car, bike",self.w * TSTT_car,TSTT_bike)
        self.carshare = TSTT_car
        self.bikeshare = TSTT_bike
        
        return (self.w *TSTT_car) + TSTT_bike
        
        
  
    def getBikePaths(self):
        all_paths = []
        for r in self.network.origins:
            for s in self.bike_paths[r].keys():
                for p in self.bike_paths[r][s]:
                    all_paths.append(p)
        return all_paths    
            
    def getCarPaths(self):
        all_paths = []
        for r in self.network.origins:
            for s in self.car_paths[r].keys():
                for p in self.car_paths[r][s]:
                    all_paths.append(p)
        return all_paths
