from main import *

def func(x):
    rho1, rho2, v1, p1, p2, e1, e2 = x
    eq1 = p1 - constants.gamma_m*rho1*e1
    eq2 = p2 - constants.gamma_m*rho2*e2
    eq3 = p - v1*p1 - (1.0 - v1)*p2
    eq4 = e - y1*e1 - (1.0 - y1)*e2
    eq5 = rho1 - rho*y1/v1
    eq6 = rho2 - rho*(1.0 - y1)/ (1.0 - v1)

def safe_divide(x, y):
    epsilon = 1e-10
    if np.abs(x) > epsilon and np.abs(y) < epsilon:
        return 100.0
    elif np.abs(x) < epsilon and np.abs(y) < epsilon:
        return 0.0
    else:
        return x/y
vsafe_divide = np.vectorize(safe_divide)

class RT(EulerEquation):
    def initialize_sod(self, qleft, qright):
        self.scalar_map = {"k":0, "L": 1, "a_x": 2, "Y_1": 3}
        Q = self.Q.reshape([self.n, self.nvar])

        At = 0.2
        rho_right = 1.0
        rho_left = (1.0 - At)/(1.0 + At)

        T = constants.g/constants.R * self.xc + 293.0
        p_right = rho_right * constants.R * T
        p_left = rho_left * constants.R * T
        
        Q[:self.n/2, 0], Q[:self.n/2, 1], Q[:self.n/2, 2] = self.calc_E(rho_left, 0.0, p_left[:self.n/2]);
        Q[self.n/2:, 0], Q[self.n/2:, 1], Q[self.n/2:, 2] = self.calc_E(rho_right, 0.0, p_right[self.n/2:]);

        idx = self.get_scalar_index("k")
        k_small = 1e-9
        Q[:self.n/2, idx] = k_small
        Q[self.n/2:, idx] = k_small

        idx = self.get_scalar_index("L")
        Q[:, idx] = 0.0
        Q[self.n/2-2:self.n/2+2, idx] = 4.0e-6

        idx = self.get_scalar_index("a_x")
        Q[:, idx] = 0.0

        idx = self.get_scalar_index("Y_1")
        Q[self.n/2:, idx] = 1.0
        Q[:self.n/2, idx] = 0.0

        Q = Q.reshape(self.n*self.nvar)
        # check if any copy happened
        assert(same_address(Q, self.Q))

    
    def temporal_hook(self):
        U = self.U.reshape([self.n+2, self.nvar])
        Q = self.Q.reshape([self.n, self.nvar])
        
        rho = U[:, 0]
        u = U[:, 1]
        p = U[:, 2]
        Y = U[:, 3:]
        idx = self.get_scalar_index("k")
        k = U[:,idx] 

        idx = self.get_scalar_index("a_x")
        a_x = U[:,idx] 

        idx = self.get_scalar_index("L")
        L = U[:,idx] 

        
        idx = self.get_scalar_index("Y_1")
        Y1 = U[:,idx] 
        Y2 = 1.0 - Y1

        # calculate partial density and volume
        # rho1, rho2, v1, v2
        rho1 = rho2 = 10.0
        v1 = rho*Y1/rho1
        v2 = 1.0 - v1
        
        self.mut = constants.c_mu*rho*L*np.sqrt(2.0*k)
        num_b = v1/(rho1 + constants.c*rho) + v2/(rho2 + constants.c*rho)
        den_b = v1*rho1/(rho1 + constants.c*rho) + v2*rho2/(rho2 + constants.c*rho)
        self.b = rho*(num_b/den_b) - 1.0

        ## TEMP FIX
        self.b = self.b*0

        
        self.rhotau = -2.0/3.0*k*rho
        self.ex = self.calc_gradient_face(1.0/constants.gamma_m*p/rho)
        
    def calc_source(self):
        U = self.U.reshape([self.n+2, self.nvar])
        S = self.S.reshape([self.n, self.nvar])
        Ux_center = self.Ux_center.reshape([self.n, self.nvar])
        rho = U[:, 0]
        u = U[:, 1]
        p = U[:, 2]
        Y = U[:, 3:]

        idx = self.get_scalar_index("k")
        k = U[:,idx] 
        idx = self.get_scalar_index("a_x")
        a_x = U[:,idx] 
        idx = self.get_scalar_index("L")
        L = U[:,idx] 
        idx = self.get_scalar_index("Y_1")
        Y1 = U[:,idx] 

        drhodx = Ux_center[:,0]
        dudx = Ux_center[:,1]
        dpdx = Ux_center[:,2]
        S[:,1] = rho[1:-1]*constants.g
        
        S[:,2] = constants.c_d * rho[1:-1] * (2.0*k[1:-1])**(1.5) * vsafe_divide(1.0, L[1:-1]) - a_x[1:-1]*dpdx + rho[1:-1]*u[1:-1]*constants.g
        #print "e ", S[:,idx].max(), S[:,idx].min()
        
        idx = self.get_scalar_index("k")
        S[:,idx] = self.rhotau[1:-1]*dudx + a_x[1:-1]*dpdx - constants.c_d * rho[1:-1] * (2.0*k[1:-1])**(1.5) * vsafe_divide(1.0, L[1:-1])
        #print "k ", S[:,idx].max(), S[:,idx].min()

        idx = self.get_scalar_index("L")
        S[:,idx] = constants.c_l * rho[1:-1] * np.sqrt(k[1:-1])
        #print "L ", S[:,idx].max(), S[:,idx].min()

        idx = self.get_scalar_index("a_x")
        S[:,idx] = constants.c_b**2 * self.b[1:-1] * dpdx - constants.c_a * rho[1:-1] * a_x[1:-1] * np.sqrt(2.0*k[1:-1]) * vsafe_divide(1.0, L[1:-1]) + self.rhotau[1:-1]/rho[1:-1] * drhodx / rho[1:-1]
        #print "a_x ", S[:,idx].max(), S[:,idx].min()

        
    def calc_viscous_flux(self):
        Ux_face = self.Ux_face.reshape([self.n+1, self.nvar])
        Fv = self.Fv.reshape([self.n+1, self.nvar])
        mut_face = 0.5*(self.mut[0:-1] + self.mut[1:])

        idx = self.get_scalar_index("k")
        dkdx_face = Ux_face[:,idx]
        Fv[:,idx] = dkdx_face*mut_face/constants.n_k

        idx = self.get_scalar_index("L")
        dLdx_face = Ux_face[:,idx]
        Fv[:,idx] = dLdx_face*mut_face/constants.n_l

        idx = self.get_scalar_index("a_x")
        da_xdx_face = Ux_face[:,idx]
        Fv[:,idx] = da_xdx_face*mut_face/constants.n_a

        idx = self.get_scalar_index("Y_1")
        dY1dx_face = Ux_face[:,idx]
        Fv[:,idx] = dY1dx_face*mut_face/constants.n_y

        rhotau_face = 0.5*(self.rhotau[0:-1] + self.rhotau[1:])
        Fv[:,1] = rhotau_face

        Fv[:,2] = self.ex*mut_face/constants.n_e
        #print "mut:", mut_face.max(), mut_face.min()
        
        
if __name__ == "__main__":
    qleft = np.array([1.0, 0.0, 1.0])
    qright = np.array([0.125, 0.0, 0.1])
    eqn = RT(n=401, nscalar=4)
    eqn.initialize_sod(qleft, qright)
    eqn.solve(tf=5e-2, dt = 1e-6, animation=True, cfl=0.05, print_step=10)
    rho, rhou, rhoE, rhoY = eqn.get_solution()
    rho, u, p, Y = eqn.get_solution_primvars()

    plt.figure()
    plt.plot(eqn.xc, rho/np.abs(rho).max(), 'r-', lw=1, label="Density")
    plt.plot(eqn.xc, u/np.abs(u).max(), 'gx-', lw=1, label="Velocity")
    plt.plot(eqn.xc, p/np.abs(p).max(), 'b-', lw=1, label="Pressure")
    if eqn.nscalar > 0:
        for i in range(eqn.nscalar):
            plt.plot(eqn.xc, Y[:,i]/np.abs(Y[:,i]).max(), '-', lw=1, label=eqn.scalar_map.keys()[eqn.scalar_map.values().index(i)])
    plt.legend(loc="best")
    plt.show()
    
