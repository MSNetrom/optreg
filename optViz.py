import sympy
import sympy.vector
import numpy as np
from collections import defaultdict

# Managing our functions and stuff

R = sympy.vector.CoordSys3D('R')

class ColorOrg:
    
    def __init__(self, init_val=0):
        self.id = init_val - 1
        
    def get_color(self):
        self.id += 1
        return 'C'+str(self.id)

class OptViz:
    
    def __init__(self, objective_function, constraints, x_interval, levels_interval):
        # Save data
        self.objective_function = objective_function
        self.constraints = constraints
        self.x_interval = x_interval
        self.levels_interval = levels_interval
        
        self.obj_func_grad = sympy.vector.gradient(self.objective_function, doit=True)
        self.constraint_grads = [sympy.vector.gradient(con, doit=True) for con in self.constraints]
        self.constraints_f = [sympy.lambdify([R.x, R.y], con >= 0) for con in constraints]
        
        # Finding feasable region
        # Set feasable region
        d = np.linspace(x_interval[0],x_interval[1],900)
        x_feas, y_feas = np.meshgrid(d,d)
        self.feas_reg = self.constraints_f[0](x_feas, y_feas)
        for con in self.constraints_f[1:]:
            self.feas_reg = (self.feas_reg & con(x_feas, y_feas))
        self.feas_reg = self.feas_reg.astype(int)
        
        self.extent=(x_feas.min(),x_feas.max(),y_feas.min(),y_feas.max())
        
        # Function for making level curves
        self.c = sympy.Symbol('c')
        self.y_level = sympy.solve(self.objective_function-self.c, R.y)[0]
        self.y_level_f = sympy.lambdify([R.x, self.c], self.y_level)
    
    def point_eval(self, f, x, y):
        return f.evalf(subs={R.x:x, R.y:y, R.z:0})
        
    #def objecive_func_eval()
    
    def objective_grad_eval(self, x_pos, y_pos):
        ''' Evaluate objective function gradients at point
            Return x_length, y_length
        '''
        obj_comps = defaultdict(lambda: 0, self.point_eval(self.obj_func_grad, x_pos, y_pos).components)
        return np.float32(obj_comps[R.i]), np.float32(obj_comps[R.j])
        
    def constraints_grad_eval(self, x_pos, y_pos):
        ''' Evaluate constraint gradients at point
            Return x_length, y_length
        '''
        lengths = [] # [(x_length, y_length)]
        for g in self.constraint_grads:
            g_comps = defaultdict(lambda: 0, self.point_eval(g, x_pos, y_pos).components)
            gx_length, gy_length = np.float32(g_comps[R.i]), np.float32(g_comps[R.j])
            lengths.append((gx_length, gy_length))
            
        return lengths
    
    def level_curve_y(self, x, level):
        return self.y_level_f(x, level)
    
    def get_feasable_region_map(self):
        ''' Returns feasable region, and a rectangle for matplotlib
        '''
        return self.feas_reg, self.extent