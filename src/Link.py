from src import Params

class Link:


    # construct this Link with the given parameters
    def __init__(self, id, start, end, length, t_ff, C, alpha, beta, cost):
        self.id = id
        self.start = start
        self.end = end
        self.t_ff = t_ff
        self.C = C
        self.alpha = alpha
        self.beta = beta
        self.x = 0
        self.y = 1
        self.length = length
        self.bikelanes = 0
        self.cost = cost
        self.OAcuts = []
        self.OABcuts = []
        self.cap_per_bikelane = self.C/4


        self.visit_order = -1
        
        if start is not None:
            start.addOutgoingLink(self)
            
        if end is not None:
            end.addIncomingLink(self)
            
        self.xstar = 0
        self.dual = 0 # for CG

    def setFlow(self, x):
        self.x = x
    
    def __repr__(self):
        return str(self)

    def getTravelTime(self, x, type):
        return self.getTravelTimeC(x, self.bikelanes, type)
        
    def getTTBike(self):
        return self.length / 12.0 * 60
        
    def getTravelTimeC(self, x, bikelanes, type):
        
        if x < 0 and x > -1e-4:
            x = 0.0        
        
        if self.y == 0 and type != 'RC':
            return Params.INFTY
            
        if type == 'UE':
            output = self.t_ff * (1 + self.alpha * pow(x / (self.C - bikelanes * self.cap_per_bikelane), self.beta))
        
        elif type == 'SO' or type == 'SO_OA_cuts':
            output = self.t_ff * (1 + self.alpha * pow(x / (self.C - bikelanes * self.cap_per_bikelane), self.beta))
            
            if self.beta > 1e-4: # for handling the case of beta = 0
                output += x * self.t_ff * self.alpha * self.beta * pow(x / (self.C - bikelanes * self.cap_per_bikelane), self.beta-1) / (self.C - bikelanes * self.cap_per_bikelane)
            
        elif type == 'RC' or type == 'RC2':
            output = self.dual
        elif type == 'dist':
            output = self.length
        elif type == 'bike':
            output = self.getTTBike()
        else:
            raise Exception("wrong type "+str(type))

        return output

    def getDerivativeTravelTime(self, x):
        return self.getDerivativeTravelTimeCx(x, 0)
    
    def getDerivativeTravelTimeCx(self, x, bikelanes):
        
        if x < 0 and x > -1e-4:
            x = 0.0
        
        if self.y == 0:
            return Params.INFTY
            
        if self.beta > 1e-4: # for handling the case of beta = 0
            return self.t_ff * self.alpha * self.beta * pow(x / (self.C - bikelanes * self.cap_per_bikelane), self.beta-1) / (self.C - bikelanes * self.cap_per_bikelane)
        
        else:
            return 0.0
    
    def getDerivativeTravelTimeCy(self, x, bikelanes):
        if x < 0 and x > -1e-4:
            x = 0.0
            
        return -self.t_ff * self.alpha * self.beta * pow(x / (self.C - bikelanes * self.cap_per_bikelane), self.beta) / (self.C - bikelanes * self.cap_per_bikelane)
    '''
    def getDerivativeTravelTimeCy(self, x, bikelanes):
        # ∂t/∂y where y = number of bikelanes
        # = cap_per_bikelane * ∂t/∂u_dec
        u_act = self.C - bikelanes * self.cap_per_bikelane
        if x < 0 and x > -1e-4:
            x = 0.0
        if u_act <= 0:
            return Params.INFTY
        return (self.t_ff * self.alpha * (self.beta)
                * self.cap_per_bikelane
                * (x ** self.beta) / (u_act ** (self.beta + 1)))
    '''
    def getPrimitiveTravelTime(self, x):  
        return self.getPrimitiveTravelTimeC(x, self.bikelanes)
        
        
    def getPrimitiveTravelTimeC(self, x, bikelanes):
        
        if x < 0 and x > -1e-4:
            x = 0.0        
        
        if self.y == 0:
            if x > 0:
                return Params.INFTY
            else:
                return 0
            
        if self.beta > 1e-4: # for handling the case of beta = 0
            return x * self.t_ff + ((self.C - bikelanes * self.cap_per_bikelane) / (self.beta + 1)) * self.t_ff * self.alpha * pow(x / (self.C - bikelanes * self.cap_per_bikelane), self.beta+1)
        
        else:
            return x * self.t_ff       
            
    def intdtdu(self, x, u):
        return  self.t_ff * self.alpha * self.beta * pow(x, self.beta+1) / ( (self.beta+1) * pow(self.C - u, self.beta+1))

    def getCapacity(self):
        return self.C
    
    def getFlow(self):
        return self.x
        
    def __str__(self):
        return "(" + str(self.start.getId()) + ", " + str(self.end.getId()) + ")"
        
    def addXstar(self, flow):
        self.xstar += flow   
    
    def calculateNewX(self, stepsize):        
        self.x = (1 - stepsize) * self.x + stepsize * self.xstar
        self.xstar = 0
        
    def hasHighReducedCost(self, type, percent):
        reducedCost = self.end.cost - self.start.cost
        tt = self.getTravelTime(self.x, type)        
        return tt - reducedCost > tt*percent
 
    def getReducedCost(self, type):
        reducedCost = self.end.cost - self.start.cost
        tt = self.getTravelTime(self.x, type)
        return tt - reducedCost

    def __hash__(self):
        return hash(self.id)