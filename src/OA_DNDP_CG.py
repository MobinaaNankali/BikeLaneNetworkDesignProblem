import time
import random
from src import Params
from src import BB_node
from src import YDict
from docplex.mp.model import Model
import time

class OA_DNDP_CG:
    
    def __init__(self, network, inflate_costs):
        self.network = network
        self.M = self.network.TD 
        self.CG_tol = 1e-4
        self.inf = 1e+9

        self.best_solution_bikeshare = 0.0  
        self.best_solution_carshare = 0.0   
        self.bikeshare = 0.0
        self.carshare = 0.0
        
        self.gap = 100
        self.w = 6
        self.tstt = 10000000000
        self.last_xhat = None
        self.last_obj_f = 1000000000


        self.rt_TAP = 0.0
        self.rt_RMP = 0.0
        self.rt_pricing = 0.0

        self.bikelane_bonus = 0.2
        
        
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
        print("Budget is always half of the total cost")
        self.budget = 0.5 * total_cost
        
        self.timelimit = 3600
        
        self.ydict = YDict.YDict()
                
        print("yvars", len(self.varlinks))
        

        self.link_cons = dict()
        self.dem_cons = dict()
        
        
        self.params = Params.Params()
    
    
    def initializeUpperBound(self):
        """
        Initialize a good upper bound using heuristics before starting B&B.
        This helps prune the search tree early and speeds up convergence.
        
        Strategy:
        1. Baseline: y=0 for all links (no bike lanes) - always feasible
        2. Greedy by benefit/cost ratio
        3. Random feasible solutions
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
        
        # Strategy 4: Greedy by link length (longer links may benefit more cyclists)
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
        """
        Greedy heuristic: select links with best benefit/cost ratio while respecting budget.
        """
        y = {a: 0 for a in self.varlinks}
        
        link_scores = []
        for a in self.varlinks:
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
        """
        Greedy heuristic: select longest links first (within budget).
        """
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
        """
        Generate a random feasible solution that respects the budget constraint.
        """
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
    
    def evaluateSolution(self, y):
        """
        Evaluate the objective function value for a given integer solution y.
        This solves TAP and computes the true TSTT.
        """
        for a in self.varlinks:
            a.bikelanes = y[a]
        
        self.network.tapas("UE", None)
        x_car = {a: a.x for a in self.network.links}
        
        x_bike = self.computeBikeFlows(y)
        
        obj = self.calcOFV(x_car, x_bike, y)
        
        if obj < self.best_ub:
            self.best_ub = obj
            self.best_y = y.copy()
            self.best_x_car = x_car
            self.best_x_bike = x_bike
            self.best_solution_bikeshare = self.bikeshare
            self.best_solution_carshare = self.carshare * self.w
        
        for a in self.varlinks:
            a.bikelanes = 0
        
        return obj
    
    def computeBikeFlows(self, y):
        """
        Compute bike flows using shortest path assignment based on current y.
        """
        for a in self.network.links:
            if a in self.varlinks and y[a] == 1:
                a.dual = a.getTTBike() - self.bikelane_bonus
            else:
                a.dual = a.getTTBike()
        
        x_bike = {a: 0 for a in self.network.links}
        
        for r in self.network.origins:
            if r.totaldemand_bike > 0:
                self.network.dijkstras(r, 'RC')
                for s in self.network.zones:
                    demand = r.getDemandBike(s)
                    if demand > 0:
                        p = self.network.trace(r, s)
                        for a in p.links:
                            x_bike[a] += demand
        
        return x_bike

    def solve(self):
        
        self.best_ub = self.inf
        lb = 0
        
        self.best_y = None
        self.best_x_bike = None
        self.best_x_car = None
        
        # Initialize upper bound with heuristics before starting B&B
        # This is crucial for faster convergence on large networks
        self.initializeUpperBound()
        
        root = BB_node.BB_node(dict(), 0, lb, self.best_ub, None, 0, True)
        
        self.initRMP()
        
        max_node_iter = 10
        max_iter = 1000000
        iteration = 1
        
        bbnodes = list()
        
        bbnodes.append(root)
        
        start_time = time.time()
        
        print("iteration", "lb", "best ub", "node ub", "gap", "time")
        
        while len(bbnodes) > 0 and iteration <= max_iter:
            
            
            bbnode = None
            best_node_lb = self.inf
            
            for n in bbnodes:
                if n.lb < best_node_lb:
                    best_node_lb = n.lb
                    bbnode = n
                elif n.lb == best_node_lb:
                    if len(n.y_fix) > len(bbnode.y_fix):
                        bbnode = n
                    elif len(n.y_fix) == len(bbnode.y_fix): 
                        if n.num_one > bbnode.num_one:
                            bbnode = n
            
            
            bbnodes.remove(bbnode)
            
            if bbnode.lb > self.best_ub:
                continue
                
            status, y = self.solveNode(bbnode, max_node_iter)
            print("node done", status, bbnode.lb, y)
            
            
            # no branching if node was infeasible, probably exceeded budget
            if status == "infeasible":
                continue
            
            #print("\t", y)
            lb = bbnode.lb
            
            for n in bbnodes:
                lb = min(lb, n.lb)
                
            
            elapsed_time = time.time() - start_time
            
            
            gap = (self.best_ub - lb)/lb
            
            print(iteration, lb, self.best_ub, bbnode.ub, gap, elapsed_time)
            
            if elapsed_time > self.timelimit:
                break
            
            if gap < self.params.min_NDP_gap:
                selected_count = 0
                total_candidates = 0
                selected_links_str = "N/A"
                selected_link_list = []
                if hasattr(self, 'varlinks') and self.best_y is not None: 
                    total_candidates = len(self.varlinks)
                    selected_count = sum(1 for a in self.varlinks if self.best_y[a] == 1)
                    selected_link_list = [str(a) for a in self.varlinks if self.best_y[a] == 1]
                    selected_links_str = f"{selected_count}/{total_candidates}"
                print("\n" + "=" * 20 + " Final Result " + "=" * 20)
                print(f"Gap tol reached: {gap:.9g} < {self.params.min_NDP_gap}")
                print(f"Best y: {self.best_y}")
                print(f"Number of branch and bound tree:", iteration)
                print("best ub:", self.best_ub, "lb", lb)
                print(f"Selected Links: {selected_links_str}")
                print("Bike share of objective function", self.best_solution_bikeshare)
                print("Car Sahre of objective function", self.best_solution_carshare)
                print("Total time:", elapsed_time)
                print("TAP time:", self.rt_TAP)
                print("RMP time:", self.rt_RMP)
                print("Pricing time:", self.rt_pricing)
                print("=" * 50)

                return (
                self.best_ub, 
                elapsed_time, 
                self.rt_TAP, 
                self.rt_pricing, 
                self.rt_RMP, 
                iteration,
                gap, 
                self.best_solution_bikeshare, 
                self.best_solution_carshare,
                selected_links_str,
                selected_link_list
            )
                break
            
            
            # no branching if node was fixed
            if status == "fixed":
                continue
                
                
            branch_link = None
            diff = 0
            
            for a in self.varlinks:
                diff_a = abs(y[a] - round(y[a]))
                if diff_a > diff:
                    diff = diff_a
                    branch_link = a
            
            if branch_link is not None:
                
                new_lb = bbnode.lb
                
                #print("lb check", old_lb, new_lb)
                
                if branch_link in bbnode.y_fix:
                    print("branching on fixed link")
                    exit(1)
                    
                    
                left_y_fix = bbnode.y_fix.copy()
                right_y_fix = bbnode.y_fix.copy()
                left_y_fix[branch_link] = 0
                solved0 = True
                right_y_fix[branch_link] = 1
                solved1 = True
                
                print("yfix", bbnode.y_fix)
                print("\tbranch", branch_link)
                
                left = BB_node.BB_node(left_y_fix, bbnode.num_one, bbnode.lb, bbnode.ub, bbnode, iteration, solved0)
                right = BB_node.BB_node(right_y_fix, bbnode.num_one + 1, bbnode.lb, bbnode.ub, bbnode, iteration, solved1)
                if solved0 == True:
                    bbnodes.append(left)
                if solved1 == True:
                    bbnodes.append(right)

                #print("added", left.lb, right.lb)
                
            iteration += 1
            max_node_iter = 1
        
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
            if r.totaldemand > 0:
                self.network.dijkstras(r, "UE")

                self.car_paths[r] = dict()

                for s in self.network.zones:
                    if r.getDemand(s) > 0:
                        p = self.network.trace(r, s)
                        self.car_paths[r][s] = list()
                        self.car_paths[r][s].append(p)
                        
        for r in self.network.origins:
            if r.totaldemand_bike > 0:
                self.network.dijkstras(r, "bike")
            
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
            self.link_cons_car[a] = self.rmp.add_constraint(self.rmp.x_car[a] - sum(self.rmp.h_car[p] for p in self.getCarPaths() if a in p.links) >= 0)
            self.link_cons_bike[a] = self.rmp.add_constraint(self.rmp.x_bike[a] - sum(self.rmp.h_bike[p] for p in self.getBikePaths() if a in p.links) >= 0)
            
        for a in self.varlinks:
            self.rmp.add_constraint(self.rmp.u[a] == -self.rmp.y[a] * a.cap_per_bikelane)
            self.rmp.add_constraint(self.rmp.w[a] <= self.rmp.y[a] * self.M)
            self.rmp.add_constraint(self.rmp.w[a] <= self.rmp.x_bike[a])
            self.rmp.add_constraint(self.rmp.w[a] >= self.rmp.x_bike[a] - self.M * (1 - self.rmp.y[a]))



        for r in self.network.origins:
            for s in self.network.zones:
                if r.getDemand(s) > 0:
                    self.dem_cons_car[(r,s)] = self.rmp.add_constraint(sum(self.rmp.h_car[p] for p in self.car_paths[r][s]) >= r.getDemand(s))
                if r.getDemandBike(s) > 0:
                    self.dem_cons_bike[(r,s)] = self.rmp.add_constraint(sum(self.rmp.h_bike[p] for p in self.bike_paths[r][s]) >= r.getDemandBike(s))  # Fixed: use getDemandBike
        
        # budget
        self.rmp.add_constraint(sum(self.rmp.y[a] * a.cost * a.length for a in self.varlinks) <= self.budget)
      
        for a in self.network.links:
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
            sum(self.rmp.w[a] * self.bikelane_bonus for a in self.varlinks)
        )
        self.rmp.minimize((self.w *self.rmp.tstt) + self.rmp.tstt_bike)        
    
    
    '''
    def solveNode(self, bbnode, max_iter):
        for cons in self.y_fix_cons:
            self.rmp.remove(cons)
                
        self.y_fix_cons = list()
        used_budget = 0
        open_budget = self.inf
        open_links = 0
        y = dict()
        
        for a in self.varlinks:
            if a in bbnode.y_fix:
                self.y_fix_cons.append(self.rmp.add_constraint(self.rmp.zeta_bike[a] == self.rmp.x_bike[a] * a.getTTBike() * (1 - self.bikelane_bonus * bbnode.y_fix[a])))
                self.y_fix_cons.append(self.rmp.add_constraint(self.rmp.y[a] == bbnode.y_fix[a]))
                used_budget += bbnode.y_fix[a] * a.cost * a.length
                y[a] = bbnode.y_fix[a]
            else:
                open_budget = min(a.cost * a.length, open_budget)
                open_links += 1
        
        fixed = False
        
        
        if self.budget < used_budget:
            if self.params.PRINT_BB_INFO:
                print("terminate node exceeded budget")
            return "infeasible", None
            
        #print("budget", self.budget - used_budget, open_budget)
        
        if self.budget - used_budget < open_budget or open_links == 0:
            print("we are getting fixed point because of not enough budget")
            # node is fixed
            fixed = True
            
            for a in self.varlinks:
                if a not in y:
                    y[a] = 0
                    self.y_fix_cons.append(self.rmp.add_constraint(self.rmp.y[a] == 0))
        
        elif len(bbnode.y_fix) == len(self.varlinks):
            print("we are getting fixed point because all variables fixed")
            fixed = True
            
            
        
        if fixed:
            status = "fixed"
        
        
        if fixed and self.params.PRINT_BB_INFO:
            print("fixed")     
        
                    
        iteration = 1
        x_l_car = dict()
        x_l_bike = dict()
        
        
        status = ""
        y = None
        x_l_car = None
        x_l_bike = None
        obj_l = self.inf
        
        while iteration <= max_iter:
            
            start_time = time.time()
            
            if not fixed:
                print("are we here")
                status, obj_l, x_l_car, x_l_bike, y = self.CG()

                if status == "infeasible":
                    print("CG infeasible")
                    exit(1)
        
                self.addOAcuts(x_l_car, y)
        
                bbnode.lb = obj_l
            # in khat paeen ezafe kardam
            #if y is not None and x_l_car is not None and x_l_bike is not None:
            print("ofv", self.calcOFV(x_l_car, x_l_bike, y), obj_l)
            
            if fixed or self.isInteger(y):
                #print(fixed, y)
                obj_f = self.getUB(x_l_bike, y)
                #print('we are getting ub from here?1?')
            #else:
            #    rounded = self.roundY(y)
            #    if self.checkBudget(rounded):
            #        obj_f = self.getUB(x_l_bike, rounded)
            #        print('we are getting the ub from here?2?')

                bbnode.ub = min(bbnode.ub, obj_f)
                if obj_f < self.best_ub:
                    self.best_y = y
            
            duration = time.time() - start_time
            if fixed:
                
                break
                
            if self.params.PRINT_BB_BASIC:
                print("BBnode", iteration, obj_l, bbnode.ub, duration)
            
            iteration += 1
        

        
        return status, y
    '''
    def solveNode(self, bbnode, max_iter):
        for cons in self.y_fix_cons:
            self.rmp.remove(cons)
                
        self.y_fix_cons = list()
        used_budget = 0
        open_budget = self.inf
        open_links = 0
        y = dict()
        
        for a in self.varlinks:
            if a in bbnode.y_fix:
                self.y_fix_cons.append(self.rmp.add_constraint(self.rmp.zeta_bike[a] == self.rmp.x_bike[a] * a.getTTBike() * (1 - self.bikelane_bonus * bbnode.y_fix[a])))
                self.y_fix_cons.append(self.rmp.add_constraint(self.rmp.y[a] == bbnode.y_fix[a]))
                used_budget += bbnode.y_fix[a] * a.cost * a.length
                y[a] = bbnode.y_fix[a]
            else:
                open_budget = min(a.cost * a.length, open_budget)
                open_links += 1
        # ... the part where you build y, used_budget, etc. ...

        fixed = False

        if self.budget < used_budget:
            return "infeasible", None

        # budget too tight to buy any new links -> force rest to 0
        if self.budget - used_budget < open_budget or open_links == 0:
            fixed = True
            for a in self.varlinks:
                if a not in y:
                    y[a] = 0
                    # also enforce y[a] == 0 in the RMP so RMP is consistent:
                    self.y_fix_cons.append(self.rmp.add_constraint(self.rmp.y[a] == 0))

        # also catch fully assigned case (BUG FIX: compare lengths, not list to int)
        elif len(bbnode.y_fix) == len(self.varlinks):
            fixed = True
            # y is already fully specified in bbnode.y_fix

        if fixed:
            status = "fixed"
        else:
            status = ""

        iteration = 1
        obj_l = self.inf
        x_l_car = None
        x_l_bike = None
        # IMPORTANT: DO NOT reset y here if fixed
        if not fixed:
            y = None  # we will get y from CG()

        while iteration <= max_iter:

            if not fixed:
                status, obj_l, x_l_car, x_l_bike, y = self.CG()
                if status == "infeasible":
                    return "infeasible", None

                self.addOAcuts(x_l_car, y)
                bbnode.lb = obj_l

            # now y is defined in both cases:
            #   - fixed: constructed above, all 0/1
            #   - not fixed: returned from CG()

            # we ALSO need flows to evaluate UB
            if fixed and x_l_bike is None:
                # solve RMP once to extract flows consistent with fixed y
                _rmp_status, _OFV, *_ = self.solveRMP()
                x_l_car  = {a: self.rmp.x_car[a].solution_value for a in self.network.links}
                x_l_bike = {a: self.rmp.x_bike[a].solution_value for a in self.network.links}

            # Now it's finally safe to print OFV
            print("ofv", self.calcOFV(x_l_car, x_l_bike, y), obj_l)
            bbnode.bikeshareBB = self.bikeshare
            bbnode.carshareBB = self.carshare

            if fixed or self.isInteger(y):
                obj_f = self.getUB(x_l_bike, y)
                bbnode.ub = min(bbnode.ub, obj_f)
                if obj_f < self.best_ub:
                    self.best_y = y

            if fixed:
                break  # no branching down this node any further

            iteration += 1

        return status, y

    def getUB(self, x_l_bike, y):
        obj_f = self.inf
                
        if self.ydict.hasUE(y) == True:
            obj_f = self.ydict.getUE(y)
            self.best_y = y
        else:
            x_f_car, obj_f = self.TAP(x_l_bike, y)
            self.ydict.insertUE(y, obj_f)
                    
                    
            #print("beckmann", self.getBeckmanOFV(x_f_car, y), self.getBeckmanOFV(x_l_car, y))
            self.addVFCuts(x_f_car)
                
            if self.best_ub > obj_f:
                self.best_ub = obj_f
                self.best_x_car = x_f_car
                self.best_x_bike = x_l_bike
                self.best_y = y
                self.best_solution_bikeshare = self.bikeshare
                self.best_solution_carshare = self.carshare * self.w
                
        return obj_f
        

    def isInteger(self, y):
        for a in self.varlinks:
            if abs(y[a] - round(y[a])) > 0.001:
                return False
                
        return True
        
    def pricing(self, link_duals_car, link_duals_bike, dem_duals_car, dem_duals_bike):
        rt_pricing = time.time()
        new = 0
        minrc = self.inf
        
        #print("len check", len(link_duals))
        for a in self.network.links:
            a.dual = link_duals_car[a]
        
        for r in self.network.origins:
            self.network.dijkstras(r,'RC')
            
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
            self.network.dijkstras(r,'RC')
            
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

        self.rt_pricing += time.time() - rt_pricing
        return minrc
        
    def solveRMP(self):
    
        t_solve = time.time()
        self.rmp.solve(log_output=False)
        self.rt_RMP = time.time() - t_solve
        
        

        
        
        if self.rmp.solve_details.status == 'infeasible' or self.rmp.solve_details.status == 'integer infeasible':
            return 'infeasible',self.inf, dict(), dict(), dict(), dict()
        RMP_status = self.rmp.solve_details.status
        
        
        if self.params.PRINT_CG_INFO:
            print("\t\t", RMP_status)
        
        OFV = self.rmp.objective_value
        

        link_duals_car = {a: self.link_cons_car[a].dual_value for a in self.network.links}
        link_duals_bike = {a: self.link_cons_bike[a].dual_value for a in self.network.links}

        dem_duals_car = dict()
        dem_duals_bike = dict()
        for r in self.network.origins:
            for s in self.network.zones:
                if r.getDemand(s) > 0:
                    dem_duals_car[(r,s)] = self.dem_cons_car[(r,s)].dual_value
                if r.getDemandBike(s) > 0:
                    dem_duals_bike[(r,s)] = self.dem_cons_bike[(r,s)].dual_value
        
        
        
        return RMP_status, OFV, link_duals_car, link_duals_bike, dem_duals_car, dem_duals_bike

            
            
    def CG(self):
        conv = False
        nCG = 0
        OFV = self.inf
        
        starttime = time.time()
        
        
        while conv == False:
            RMP_status, OFV, link_duals_car, link_duals_bike, dem_duals_car, dem_duals_bike = self.solveRMP()
            #print("solved?", RMP_status)
            
            
            if RMP_status == 'infeasible':
                CG_status = 'infeasible'
                
                return CG_status, self.inf, dict(), dict(), dict()
            else:
                CG_status = "solved"
                
            minrc = self.pricing(link_duals_car, link_duals_bike, dem_duals_car, dem_duals_bike)

            elapsed = time.time() - starttime
            
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
                elif abs( (xhat_car[a] - x_save[a]) / xhat_car[a] > self.params.VF_TOL_X):
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
            yterm = (self.rmp.u[a] - uhat) * a.intdtdu(xhat, uhat) 

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
    
        #for a in self.varlinks:
        #    a.bikelanes = round(y[a])
        t0_TAP = time.time()  
        self.network.tapas("UE", None)
        self.rt_TAP += time.time() - t0_TAP
        xhat_car = {a:a.x for a in self.network.links}
        obj_f = self.calcOFV(xhat_car, xhat_bike, y)

        return xhat_car, obj_f
    
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
                biketstt += x_bike[a] * (a.getTTBike() - self.bikelane_bonus * y[a])
            else:
                biketstt += x_bike[a] * a.getTTBike()
        
        return biketstt
            
    def calcOFV(self, x_car, x_bike, y):
        TSTT_car = self.calcTSTT_car(x_car, y)
        TSTT_bike = self.calcTSTT_bike(x_bike, y)
        print("This is OFV value separetly car, bike",TSTT_car,TSTT_bike)
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
 