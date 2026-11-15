import time
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

        # ADDED: To store shares of the best solution
        self.best_solution_bikeshare = 0.0  
        self.best_solution_carshare = 0.0   
        
        self.bikeshare = 0.0
        self.carshare = 0.0
        
        self.gap = 100
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

        # ADD THIS LINE to correctly initialize self.varlinks
        self.varlinks = self.network.get_varlinks()

        # CHANGE self.network.varlinks TO self.varlinks
        for a in self.varlinks: 
            self.vf_cuts.append(0)

        
        self.num_bb_nodes = 0
        self.build_RMP()
        
    def solve(self):
        st = time.time()
        
        bbnodes = list()
        
        y_fix_root = YDict.YDict(self.network.links, -1)
        root = BB_node.BB_node(y_fix_root, 0, 0, self.inf, None, 0, False)
        
        bbnodes.append(root)
        
        
        
        lb = 0
        
        # print(self.network.origins)
        # print(self.network.zones)
        # print(self.car_paths)

        
        # print("Num Links", len(self.network.links))
        
        
        self.num_bb_nodes = 0
        
        while len(bbnodes) > 0:
            
            #print("Num BB Nodes", len(bbnodes))
            
            
            # get node with lowest lb
            bbnodes.sort(key=lambda n: n.lb)
            bbnode = bbnodes.pop(0)
            
            #print("LB", bbnode.lb, self.best_ub, self.gap)
            
            if bbnode.lb > self.best_ub - 1e-6:
                continue
                
            if bbnode.solved:
                continue
            
            
            
            self.num_bb_nodes += 1
            
            #print("Node", self.num_bb_nodes, bbnode.y_fix)

            obj_f, x_f_car, x_l_bike, y_f, gap = self.solveNode(bbnode)
            
            
            # Fathom on obj
            if obj_f > self.best_ub - 1e-6:
                #print("Fathom Obj", obj_f, self.best_ub)
                continue
                
            
            
            # Fathom on gap
            if gap < 0.01:
                #print("Fathom Gap", gap)
                bbnode.solved = True
                continue
            
            
            
            # REMOVE THIS BLOCK (it's redundant, self.varlinks is set in __init__)
            # This line must be *after* solveNode is called
            # if not hasattr(self, 'varlinks'):
            #      self.varlinks = self.network.get_varlinks()

            
            max_frac = -1.0
            max_frac_link = None
            
            for a in self.varlinks:
                if bbnode.y_fix[a] == -1:
                    
                    if abs(y_f[a] - 0.5) < abs(max_frac - 0.5):
                        max_frac = y_f[a]
                        max_frac_link = a
            
            
            
            #print("Max Frac", max_frac, max_frac_link)

            # create new nodes
            if max_frac_link is not None:
                
                # left node
                y_fix_left = bbnode.y_fix.copy()
                y_fix_left[max_frac_link] = 0
                
                left = BB_node.BB_node(y_fix_left, bbnode.num_one, bbnode.lb, bbnode.ub, bbnode, self.num_bb_nodes, False)
                
                
                # right node
                y_fix_right = bbnode.y_fix.copy()
                y_fix_right[max_frac_link] = 1
                
                right = BB_node.BB_node(y_fix_right, bbnode.num_one+1, bbnode.lb, bbnode.ub, bbnode, self.num_bb_nodes, False)
                
                bbnodes.append(left)
                bbnodes.append(right)
                
            else:
                bbnode.solved = True
                
            # Update gap
            
            # REMOVED: This was the buggy loop
            # for n in bbnodes:
            #     lb = min(lb, n.lb)
            #     best_node = min(bbnodes, key=lambda n: n.lb)
            #     self.bikeshare = best_node.bikeshareBB
            #     self.carshare = best_node.carshareBB
            
            # ADDED: Correct way to update global lower bound
            if len(bbnodes) > 0:
                lb = min(n.lb for n in bbnodes) # Get the minimum lb from all remaining nodes
            else:
                lb = self.best_ub # If no nodes left, B&B is done, lb = ub
                

            if self.best_ub < self.inf:
                self.gap = 100.0 * (self.best_ub - lb) / self.best_ub
            
            
            if self.num_bb_nodes % 10 == 0:
                print("Iter:", self.num_bb_nodes,"LB:", lb, "UB:", self.best_ub, "Gap:", self.gap, "Num Nodes:", len(bbnodes))

            
            
            
        print("Done B&B")
        
        print("Num BB Nodes", self.num_bb_nodes)
        
        print("Best Obj", self.best_ub)
        
        print("Gap", self.gap)
        
        print("Best Y")
        
        # REMOVE THIS BLOCK (redundant)
        # This check is needed again in case solveNode was never called
        # if not hasattr(self, 'varlinks'):
        #      self.varlinks = self.network.get_varlinks()
             
        for a in self.varlinks:
            if self.best_y[a] > 0.5:
                print(a)
                
        # print("Best X Car")
        
        # for a in self.network.links:
        #     if self.best_x_car[a] > 1e-6:
        #         print(a, self.best_x_car[a])
                
        # print("Best X Bike")
        
        # for a in self.network.links:
        #     if self.best_x_bike[a] > 1e-6:
        #         print(a, self.best_x_bike[a])

        # This is the corrected print statement from the last step
        print("Bike share of objective function", self.best_solution_bikeshare)
        print("Car Sahre of objective function", self.best_solution_carshare)
        
        tot_time = time.time() - st
        
        print('Total Time (s):', tot_time)
        print('TAP Time (s):', self.rt_TAP)
        print('CG Time (s):', self.rt_pricing)
        print('RMP Time (s):', self.rt_RMP)
        
        # ADDED: This is the return statement for run_experiments.py
        return (
            self.best_ub, 
            tot_time, 
            self.rt_TAP, 
            self.rt_pricing, 
            self.rt_RMP, 
            self.num_bb_nodes,
            self.gap, 
            self.best_solution_bikeshare, 
            self.best_solution_carshare
        )

        
    def solveNode(self, bbnode):
        
        # REMOVE THIS BLOCK (redundant)
        # Moved this from __init__ to here, as it's used here
        # and guarantees self.network is fully initialized.
        # if not hasattr(self, 'varlinks'):
        #     self.varlinks = self.network.get_varlinks()

        st_tap = time.time()
        
        x_car, x_bike, duals = self.network.TAP(self.network.TD, bbnode.y_fix)
        
        self.rt_TAP += time.time() - st_tap
        
        obj_f = self.calcOFV(x_car, x_bike, bbnode.y_fix)
        
        # Store shares in the node
        bbnode.bikeshareBB = self.bikeshare
        bbnode.carshareBB = self.carshare
        
        self.getUB(obj_f, x_car, x_bike, bbnode.y_fix)
        
        
        # REMOVE THIS LINE (redundant)
        # This was in your original file at the end of this method.
        # self.varlinks = self.network.get_varlinks()
        
        # Return statement from your original file
        return obj_f, x_f_car, x_l_bike, y_f, gap

        
    def getUB(self, obj_f, x_f_car, x_l_bike, y):
        
        if self.best_ub > obj_f:
            self.best_ub = obj_f
            self.best_x_car = x_f_car
            self.best_x_bike = x_l_bike
            self.best_y = y
            # ADDED: Store the shares when we find a new best solution
            self.best_solution_bikeshare = self.bikeshare
            self.best_solution_carshare = self.carshare

        
        
    def build_RMP(self):
        
        self.rmp = Model('RMP')
        
        # REMOVE THIS BLOCK (redundant)
        # This line must be run before it can be used
        # if not hasattr(self, 'varlinks'):
        #     self.varlinks = self.network.get_varlinks()
        
        self.rmp_y = self.rmp.continuous_var_dict(self.varlinks, name='y', lb=0, ub=1)
        
        self.rmp.add_constraint(self.rmp.sum(self.rmp_y[a] for a in self.varlinks) <= self.network.budget)
        
        self.rmp_theta = self.rmp.continuous_var(name='theta', lb=0)
        
        self.rmp.add_constraint(self.rmp_theta >= 0)
        
        
        self.rmp_zeta = self.rmp.continuous_var_dict(self.network.links, name='zeta', lb=0)
        
        
        self.rmp.minimize(self.rmp_theta + self.rmp.sum(self.rmp_zeta[a] for a in self.network.links))
        
        
        
    def updateCuts(self, x, y):
        
        for a in self.network.links:
            tt = a.getTravelTimeC(x[a], y[a], "UE")
            tt_d = a.getTravelTimeC_d(x[a], y[a], "UE")
            
            self.tstt_alpha[a].append(tt_d)
            self.tstt_beta[a].append(tt - tt_d * x[a])
            
            # This check is vital. rmp_x_car only exists after getPaths is called.
            if hasattr(self, 'rmp_x_car'):
                self.rmp.add_constraint(self.rmp_zeta[a] >= tt_d * self.rmp_x_car[a] + (tt - tt_d * x[a]))
            

        
        
    def getPaths(self, duals):
        
        #print("Get Paths")
        
        
        all_car_paths_found = 0
        all_bike_paths_found = 0
        
        # Find shortest paths car
        for o in self.network.origins:
            #print("Dual", o, duals[o])
            
            #print("Finding shortest paths from", o)
            
            self.network.find_shortest_paths_car(o, duals[o])
            
            for d in self.network.zones:
                
                if self.network.TD[o][d] > 1e-6:
                
                    sp = self.network.get_shortest_path_car(o, d)
                    
                    
                    
                    sp_cost = self.network.get_shortest_path_cost_car(o, d, duals[o])
                    
                    #print("SP Car", o, d, sp_cost)
                    
                    if sp_cost < -self.CG_tol:
                        all_car_paths_found += 1
                        
                        self.car_paths[o][d].append(sp)
                        
                        
                        # Add path to RMP
                        # This needs rmp_h_car to exist
                        if hasattr(self, 'rmp_h_car'):
                            self.rmp.linear_constraint_manager.add(self.rmp_h_car[o][d] >= sp.get_path_cost_RMP(self.rmp_y))
                        
                        
        
        
        # Find shortest paths bike
        for o in self.network.origins:
            #print("Dual", o, duals[o])
            
            #print("Finding shortest paths from", o)
            
            self.network.find_shortest_paths_bike(o)
            
            for d in self.network.zones:
                
                if self.network.TD_bike[o][d] > 1e-6:
                
                    sp = self.network.get_shortest_path_bike(o, d)
                    
                    sp_cost = self.network.get_shortest_path_cost_bike(o, d, duals[o])
                    
                    #print("SP Bike", o, d, sp_cost)
                    
                    if sp_cost < -self.CG_tol:
                        all_bike_paths_found += 1
                        
                        self.bike_paths[o][d].append(sp)
                        
                        
                        # Add path to RMP
                        # This needs rmp_h_bike to exist
                        if hasattr(self, 'rmp_h_bike'):
                            self.rmp.linear_constraint_manager.add(self.rmp_h_bike[o][d] >= sp.get_path_cost_RMP_bike(self.rmp_y, self.bikelane_bonus))

                        
                        
        
        # This flag ensures we only build the RMP components *once*
        if (all_car_paths_found > 0 or all_bike_paths_found > 0) and (not hasattr(self, 'rmp_x_car')):
            
            # Re-build RMP
            
            #print("Re-build RMP")
            
            self.rmp_x_car = self.rmp.continuous_var_dict(self.network.links, name='x_car', lb=0)
            self.rmp_x_bike = self.rmp.continuous_var_dict(self.network.links, name='x_bike', lb=0)
            
            self.rmp_h_car = {r:{s:self.rmp.continuous_var(name='h_car_'+str(r)+'_'+str(s), lb=0) for s in self.network.zones} for r in self.network.origins}
            self.rmp_h_bike = {r:{s:self.rmp.continuous_var(name='h_bike_'+str(r)+'_'+str(s), lb=0) for s in self.network.zones} for r in self.network.origins}
            
            
            self.rmp.add_constraint(self.rmp_theta == self.rmp.sum(self.rmp_h_car[o][d] * self.network.TD[o][d] + self.rmp_h_bike[o][d] * self.network.TD_bike[o][d] for o in self.network.origins for d in self.network.zones))
            
            
            
            for a in self.network.links:
                self.rmp.add_constraint(self.rmp_x_car[a] == self.rmp.sum(self.car_paths[o][d][i].get_nodes_RMP(a) * self.rmp.flow_car(o,d,i) for o in self.network.origins for d in self.network.zones for i in range(len(self.car_paths[o][d]))))
                self.rmp.add_constraint(self.rmp_x_bike[a] == self.rmp.sum(self.bike_paths[o][d][i].get_nodes_RMP(a) * self.rmp.flow_bike(o,d,i) for o in self.network.origins for d in self.network.zones for i in range(len(self.bike_paths[o][d]))))
            
            
            # flow variables
            
            self.rmp.flow_car = {(o,d,i):self.rmp.continuous_var(name='flow_car_'+str(o)+'_'+str(d)+'_'+str(i), lb=0) for o in self.network.origins for d in self.network.zones for i in range(len(self.car_paths[o][d]))}
            self.rmp.flow_bike = {(o,d,i):self.rmp.continuous_var(name='flow_bike_'+str(o)+'_'+str(d)+'_'+str(i), lb=0) for o in self.network.origins for d in self.network.zones for i in range(len(self.bike_paths[o][d]))}

            for o in self.network.origins:
                for d in self.network.zones:
                    if self.network.TD[o][d] > 1e-6:
                        self.rmp.add_constraint(self.rmp.sum(self.rmp.flow_car(o,d,i) for i in range(len(self.car_paths[o][d]))) == self.network.TD[o][d])
                        
                    if self.network.TD_bike[o][d] > 1e-6:
                        self.rmp.add_constraint(self.rmp.sum(self.rmp.flow_bike(o,d,i) for i in range(len(self.bike_paths[o][d]))) == self.network.TD_bike[o][d])


            
            # Add TSTT cuts
            
            for a in self.network.links:
                for i in range(len(self.tstt_alpha[a])):
                    self.rmp.add_constraint(self.rmp_zeta[a] >= self.tstt_alpha[a][i] * self.rmp_x_car[a] + self.tstt_beta[a][i])
                    
            
            # Add VF cuts
            
            # for a in self.varlinks:
            #     for i in range(len(self.vfa[a])):
            #         self.rmp.add_constraint(self.rmp_zeta[a] >= self.vfa[a][i] * self.rmp_y[a] + self.vfb[a][i])
            

            
            
            
            #print("Num vars", self.rmp.number_of_variables)
            #print("Num cons", self.rmp.number_of_constraints)
            
            
            

    def calcTSTT_car(self, x_car, y):
        tstt = 0
        
        # REMOVE THIS BLOCK (redundant)
        # Ensure varlinks is available
        # if not hasattr(self, 'varlinks'):
        #     self.varlinks = self.network.get_varlinks()
            
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
        
        # REMOVE THIS BLOCK (redundant)
        # Ensure varlinks is available
        # if not hasattr(self, 'varlinks'):
        #     self.varlinks = self.network.get_varlinks()

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
        
        return TSTT_car + TSTT_bike
        
        
  
    def getBikePaths(self):
        all_paths = {r:{s:list() for s in self.network.zones} for r in self.network.origins}
        
        
        for o in self.network.origins:
            
            self.network.find_shortest_paths_bike(o)
            
            for d in self.network.zones:
                
                if self.network.TD_bike[o][d] > 1e-6:
                
                    sp = self.network.get_shortest_path_bike(o, d)
                    
                    all_paths[o][d].append(sp)
                    
        return all_paths