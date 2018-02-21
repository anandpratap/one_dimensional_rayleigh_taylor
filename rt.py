from main import *

def safe_divide(x, y):
    epsilon = 1e-16
    return x/(y+epsilon)
# if np.abs(x) > epsilon and np.abs(y) < epsilon:
#         return 1e14
#     elif np.abs(x) < epsilon and np.abs(y) < epsilon:
#         return 0.0
#     else:
#         return x/y

vsafe_divide = np.vectorize(safe_divide)
def calc_volumep(Yh, Yl, rho, At):
    rhoh, rhol = calc_rhop(At)
    vh = rho*Yh/rhoh
    vl = rho*Yl/rhol
    return vh, vl

def calc_rhop(At):
    rho_h = 1.0
    rho_l = (1.0 - At)/(1.0 + At)
    return rho_h, rho_l

class RT(EulerEquation):
    def initialize_sod(self, qleft, qright):
        self.scalar_map = {"k":0, "L": 1, "a_x": 2, "Y_h": 3}
        Q = self.Q.reshape([self.n, self.nvar])

        self.At = 0.05
        rho_h, rho_l = calc_rhop(self.At)


        R = constants.R
        gamma = constants.gamma

        

        
        rho = np.zeros(self.n)
        Yh = np.zeros(self.n)
        L = np.zeros(self.n)


        T = constants.g/constants.R * self.xc + 293.0


        e_h = R/(gamma - 1.0) * T
        e_l = rho_h*e_h/rho_l
        

        Yh[:self.n/2] = 0.0
        Yh[self.n/2:] = 1.0

        Yl = 1.0 - Yh
        e = Yh * e_h + Yl * e_l
        

        p_h = (gamma - 1.0)*rho_h*e_h
        p_l = (gamma - 1.0)*rho_l*e_l

        
        rho[:self.n/2] = rho_l
        rho[self.n/2:] = rho_h


        
       # fac = (1.0 + self.At)/(1.0 - self.At)*vsafe_divide(Y1,Yl)
        
        #v1 = fac/(1.0+fac)
        #v2 = 1.0 - v1

        vh, vl = calc_volumep(Yh, Yl, rho, self.At)
        #rho1 = rho*vsafe_divide(Y1, v1)
        #rho2 = rho*vsafe_divide(Yl, v2)
        #print v1, v2
        p = vh * p_h + vl * p_l
        e = Yh * e_h + Yl * e_l
       
        # plt.figure()
        # plt.ioff()
        # plt.subplot(311)
        # plt.plot(e)
        # plt.subplot(312)
        # plt.plot(rho)
        # plt.plot(rho1)
        # plt.plot(rho2)
        # plt.subplot(313)
        # #plt.plot(ph)
        # #plt.plot(pl)
        # plt.plot(p)

        # plt.plot(p)
        #plt.show()

        
        Q[:self.n/2, 0], Q[:self.n/2, 1], Q[:self.n/2, 2] = self.calc_Ee(rho[:self.n/2], 0.0, e[:self.n/2]);
        Q[self.n/2:, 0], Q[self.n/2:, 1], Q[self.n/2:, 2] = self.calc_Ee(rho[self.n/2:], 0.0, e[self.n/2:]);

        idx = self.get_scalar_index("k")
        k_small = 1e-9
        Q[:self.n/2, idx] = k_small * rho[:self.n/2]
        Q[self.n/2:, idx] = k_small * rho[self.n/2:]
        #k_small =
        idx = self.get_scalar_index("L")
        Q[:, idx] = k_small * rho 
        Q[self.n/2-2:self.n/2+2, idx] = 4.0e-6 * rho[self.n/2-2:self.n/2+2]

        idx = self.get_scalar_index("a_x")
        Q[:, idx] = k_small * rho

        idx = self.get_scalar_index("Y_h")
        Q[:self.n/2, idx] =  Yh[:self.n/2] * rho[:self.n/2]
        Q[self.n/2:, idx] = Yh[self.n/2:] * rho[self.n/2:]
        
        Q = Q.reshape(self.n*self.nvar)
        # check if any copy happened
        assert(same_address(Q, self.Q))

    def calc_E(self, rho, u, p):
        rho = rho
        rhou = rho*u
        E = p/(constants.gamma-1.0) + 0.5*rho*u**2;
        return rho, rhou, E

    def calc_Ee(self, rho, u, e):
        rho = rho
        rhou = rho*u
        E = rho*e + 0.5*rho*u**2;
        return rho, rhou, E

    
    def temporal_hook(self):
        U = self.U.reshape([self.n+2, self.nvar])
        Q = self.Q.reshape([self.n, self.nvar])
        
        rho = U[:, 0]
        u = U[:, 1]
        p = U[:, 2]
        Y = U[:, 3:]
        idx = self.get_scalar_index("k")
        k = U[:,idx] 
        #print rho

        idx = self.get_scalar_index("a_x")
        a_x = U[:,idx] 

        idx = self.get_scalar_index("L")
        L = U[:,idx] 

        
        idx = self.get_scalar_index("Y_h")
        Yh = U[:,idx]
        #print Yh
        #print rho
        Yl = 1.0 - Yh

        
        # calculate partial density and volume
        # rho1, rho2, v1, v2
        #fac = (1.0 + self.At)/(1.0 - self.At)*vsafe_divide(Yh,Yl)
        
        #v1 = fac/(1.0+fac)
        #v2 = 1.0 - v1
        #rho1 = rho*vsafe_divide(Yh, v1)
        #rho2 = rho*vsafe_divide(Yl, v2)
        rhoh, rhol = calc_rhop(self.At)
        vh, vl = calc_volumep(Yh, Yl, rho, self.At)
        
        #print k
        self.mut = constants.c_mu*rho*L*np.sqrt(2.0*k)
        #num_b = vsafe_divide(v1, (rho1 + constants.c*rho)) + vsafe_divide(v2, (rho2 + constants.c*rho))
        #den_b = vsafe_divide(v1*rho1, (rho1 + constants.c*rho)) + vsafe_divide(v2*rho2, (rho2 + constants.c*rho))
        num_b = vh/rhoh + vl/rhol
        den_b = vh + vl
        self.b = rho*(num_b/den_b) - 1.0
        self.b = np.maximum(self.b, 1e-4)
        #self.b = self.b*0
        #print self.b
        ## TEMP FIX
        #self.b = self.b*0

        #print k
        self.rhotau = -2.0/3.0*k*rho

        # not sure if the following expression is correct
        e = np.zeros_like(rho) 
        e[1:-1] = (Q[:,2] - 0.5*rho[1:-1]*u[1:-1]**2)/rho[1:-1]
        e[0] = e[1]
        e[-1] = e[-2]

        #e = 1.0/constants.gamma_m*p/rho
        self.ex = (e[1:] - e[0:-1])/self.dx

    def calc_press(self, Q):
        output = []
        rho = Q[:, 0]
        u = Q[:, 1]/Q[:, 0]

        idx = self.get_scalar_index("Y_h")
        Yh = Q[:,idx]/Q[:,0] 
        Yl = 1.0 - Yh

        rhoh, rhol = calc_rhop(self.At)
        vh, vl = calc_volumep(Yh, Yl, rho, self.At)
        
        #fac = (1.0 + self.At)/(1.0 - self.At)*vsafe_divide(Yh,Yl)
        #v1 = fac/(1.0+fac)
        #v2 = 1.0 - v1
        #rhoh = rho*vsafe_divide(Yh, v1)
        #rhol = rho*vsafe_divide(Yl, v2)

        gamma = constants.gamma

        e = (Q[:,2] - 0.5*rho*u**2)/rho
        
        eh = e / (Yh + rhoh/rhol*Yl)
        el = rhoh*eh/rhol

        ph = (gamma - 1.0)*rhoh*eh
        pl = (gamma - 1.0)*rhol*el

        p = vh * ph + vl * pl
        

        # plt.figure()
        # plt.ioff()
        # plt.subplot(311)
        # plt.plot(e)
        # plt.subplot(312)
        # #plt.plot(rho)
        # plt.plot(rho)
        # plt.plot(rhoh)
        # plt.plot(rhol)

        # plt.subplot(313)
        # plt.plot(p)
        # plt.show()
        
        #fac = Yh + Yl*vsafe_divide(rho1,rho2)
        #e1 = vsafe_divide(e, fac)
        #e2 = e1 * vsafe_divide(rho1, rho2)
        #p1 = (gamma-1.0)*rho1*e1
        #p2 = (gamma-1.0)*rho2*e2
        #p = v1*p1 + v2*p2

        #p = (gamma-1.0) * rho * e
        #plt.ioff()
        #plt.figure()
        #plt.plot(p, label="p1")
        #plt.plot(p1, label="p2")
        #plt.legend()
        #plt.show()
        #print p2
        #p = (constants.gamma-1.0)*(Q[:,2] - 0.5*Q[:,1]**2/Q[:,0])
        output = [rho, u, p]
        Y = Q[:, 3:]/np.tile(Q[:, 0], (self.nscalar,1)).T
        output.append(Y)
        return output
    
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
        idx = self.get_scalar_index("Y_h")
        Yh = U[:,idx] 

        drhodx = Ux_center[:,0]
        dudx = Ux_center[:,1]
        dpdx = Ux_center[:,2]
#        print dpdx
        slice_cells = slice(1,-1)
        S[:,1] = rho[slice_cells]*constants.g

        #k32byL = 
        
        S[:,2] = constants.c_d * rho[slice_cells] * (2.0*k[slice_cells])**(1.5) * vsafe_divide(1.0, L[slice_cells]) - a_x[slice_cells]*dpdx  + rho[slice_cells]*u[slice_cells]*constants.g
        #print "e ", S[:,idx].max(), S[:,idx].min()
        
        idx = self.get_scalar_index("k")
        S[:,idx] = self.rhotau[slice_cells]*dudx + a_x[slice_cells]*dpdx - constants.c_d * rho[slice_cells] * (2.0*k[slice_cells])**(1.5) * vsafe_divide(1.0, L[slice_cells])
        #print "k ", S[:,idx].max(), S[:,idx].min()

        idx = self.get_scalar_index("L")
        S[:,idx] = constants.c_l * rho[slice_cells] * np.sqrt(2.0*k[slice_cells])
        #print "L ", S[:,idx].max(), S[:,idx].min()
        
        idx = self.get_scalar_index("a_x")
        S[:,idx] = constants.c_b**2 * self.b[slice_cells] * dpdx - constants.c_a * rho[slice_cells] * a_x[slice_cells] * np.sqrt(2.0*k[slice_cells]) * vsafe_divide(1.0, L[slice_cells]) + self.rhotau[slice_cells]/rho[slice_cells] * drhodx 
        #print "a_x ", S[:,idx].max(), S[:,idx].min()
        #print "dp", dpdx
        #print "drho ", drhodx
        #print "du ", dudx
        #print "S0", S[0,:]
        #print "S-1", S[-1,:]
        #S[:,:] = 0.0
        
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

        idx = self.get_scalar_index("Y_h")
        dYhdx_face = Ux_face[:,idx]
        Fv[:,idx] = dYhdx_face*mut_face/constants.n_y

        rhotau_face = 0.5*(self.rhotau[0:-1] + self.rhotau[1:])
        Fv[:,1] = rhotau_face

        #Fv[:,:] = 0.0
        Fv[:,2] = self.ex*mut_face/constants.n_e
        #print "mut:", mut_face.max(), mut_face.min()
        
        
if __name__ == "__main__":
    qleft = np.array([1.0, 0.0, 1.0])
    qright = np.array([0.125, 0.0, 0.1])
    eqn = RT(n=201, nscalar=4)
    eqn.initialize_sod(qleft, qright)
    eqn.solve(tf=1.0, cfl = 0.1, animation=True, print_step=10000, integrator="rk2", flux="hllc")
    rho, rhou, rhoE, rhoY = eqn.get_solution()
    rho, u, p, Y = eqn.get_solution_primvars()
    np.savez("data_5.npz", rho=rho, u=u, p=p, Y=Y, x=eqn.xc)
    plt.figure()
    plt.plot(eqn.xc, rho/np.abs(rho).max(), 'r-', lw=1, label="Density")
    plt.plot(eqn.xc, u/np.abs(u).max(), 'gx-', lw=1, label="Velocity")
    plt.plot(eqn.xc, p/np.abs(p).max(), 'b-', lw=1, label="Pressure")
    if eqn.nscalar > 0:
        for i in range(eqn.nscalar):
            plt.plot(eqn.xc, Y[:,i]/np.abs(Y[:,i]).max(), '-', lw=1, label=eqn.scalar_map.keys()[eqn.scalar_map.values().index(i)])
    plt.legend(loc="best")
    plt.show()
    
