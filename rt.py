from main import *
import fortutils as f
def calc_partial_volume(Yh, Yl, rho, At):
    rhoh, rhol = calc_partial_rho(At)
    vh = rho*Yh/rhoh
    vl = rho*Yl/rhol
    return vh, vl

def calc_partial_rho(At):
    rho_h = 1.0
    rho_l = (1.0 - At)/(1.0 + At)
    return rho_h, rho_l

def calc_eosp(rho, e):
    gamma = constants.gamma
    return (gamma - 1.0)*rho*e


class RT(EulerEquation):
    def initialize(self):
        #self.xc = 0.01*self.xc
        self.scalar_map = {"k":0, "L": 1, "a_x": 2, "Y_h": 3}
        Q = self.Q.reshape([self.n, self.nvar])

        self.At = 0.05
        rho_h, rho_l = calc_partial_rho(self.At)
        

        R = constants.R
        gamma = constants.gamma

        rho = np.zeros(self.n)
        Yh = np.zeros(self.n)
        L = np.zeros(self.n)

        first_half = self.xc <= 0.0
        #slice(None, self.n/2)
        second_half = self.xc > 0.0
        #slice(self.n/2, None)

        

        Yh[first_half] = 0.0
        Yh[second_half] = 1.0

        Yl = 1.0 - Yh
               
        
        rho[first_half] = rho_l
        rho[second_half] = rho_h
        
        T_ref = 2293.0
        P_ref = 101325.0
        R = constants.R

        get_mw = lambda rhoi: rhoi*R*T_ref/P_ref
        get_T = lambda mwi: g*(mwi/R)*self.xc + T_ref
        get_cv = lambda mwi: R/(mwi*(gamma-1))
        
        g = constants.g
        mw_h = get_mw(rho_h)
        mw_l = get_mw(rho_l)

        T_l =  get_T(mw_l)
        T_h =  get_T(mw_h)

        cv_l = get_cv(mw_l)
        cv_h = get_cv(mw_h)

        e_l = cv_l*T_l
        e_h = cv_h*T_h

        e = Yh * e_h + Yl * e_l
        
        Q[:, 0], Q[:, 1], Q[:, 2] = self.calc_Ee(rho, 0.0, e);

        idx = self.get_scalar_index("k")
        Q[:, idx] = 0.0
        Q[self.n/2-2:self.n/2+2, idx] = 1e-4 * rho[self.n/2-2:self.n/2+2]
        
        idx = self.get_scalar_index("L")
        Q[:, idx] = 0.0
        Q[self.n/2-2:self.n/2+2, idx] = 4.0e-8 * rho[self.n/2-2:self.n/2+2]

        idx = self.get_scalar_index("a_x")
        Q[:, idx] = 0.0

        idx = self.get_scalar_index("Y_h")
        Q[:, idx] =  Yh * rho
        
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
        
        idx = self.get_scalar_index("a_x")
        a_x = U[:,idx] 

        idx = self.get_scalar_index("L")
        L = U[:,idx] 

        
        idx = self.get_scalar_index("Y_h")
        Yh = U[:,idx]
        Yl = 1.0 - Yh

        
        rhoh, rhol = calc_partial_rho(self.At)
        vh, vl = calc_partial_volume(Yh, Yl, rho, self.At)

        #print "temporal hook", k
        #print k
        #print np.sqrt
        #print k.dtype
        
        self.mut[:] = constants.c_mu*rho*L*np.sqrt(2.0*np.absolute(k))
        # temp fix
        if self.order == 2 or self.order==1:
            self.mut[0] = -(2.0*self.mut[1] - self.mut[2])
            self.mut[-1] = -(2.0*self.mut[-2] - self.mut[-3])
        num_b = vh/rhoh + vl/rhol
        den_b = vh + vl

        self.b[:] = rho*(num_b/den_b) - 1.0
        #print self.b
        #self.b = np.minimum(self.b, 0.0)
        #print self.b
        self.rhotau[:] = -2.0/3.0*k*rho

        # not sure if the following expression is correct
        e = np.zeros_like(rho) 

        e[1:-1] = (Q[:,2] - 0.5*rho[1:-1]*u[1:-1]**2)/rho[1:-1]
        e[0] = e[1]
        e[-1] = e[-2]
        self.ex[:] = (e[1:] - e[0:-1])/self.dx

    def calc_press(self, Q):
        output = []
        rho = Q[:, 0]
        u = Q[:, 1]/Q[:, 0]

        idx = self.get_scalar_index("Y_h")
        Yh = Q[:,idx]/Q[:,0] 
        Yl = 1.0 - Yh

        rhoh, rhol = calc_partial_rho(self.At)
        vh, vl = calc_partial_volume(Yh, Yl, rho, self.At)
        
        e = (Q[:,2] - 0.5*rho*u**2)/rho
        
        eh = e / (Yh + rhoh/rhol*Yl)
        el = rhoh*eh/rhol

        ph = calc_eosp(rhoh, eh)
        pl = calc_eosp(rhol, el)

        p = vh * ph + vl * pl
        

        output = [rho, u, p]
        Y = Q[:, 3:]/np.tile(Q[:, 0], (self.nscalar,1)).T
        idx = self.get_scalar_index("k")
        # clip k to possitive value
        Y[:,idx-3] = np.clip(Y[:,idx-3], 0.0, 1e10)
        #print 'calc press ', Y[:,idx-3]
        idx = self.get_scalar_index("a_x")
        # clip a_x to negative value
        Y[:,idx-3] = np.clip(Y[:,idx-3], -1e10, 0.0)
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

        slice_cells = slice(1,-1)
        S[:,1] = rho[slice_cells]*constants.g

        S[:,2] = constants.c_d * rho[slice_cells] * (2.0*k[slice_cells])**(1.5) * vsafe_divide(1.0, L[slice_cells])
        S[:,2] -= a_x[slice_cells]*dpdx
        S[:,2] += rho[slice_cells]*u[slice_cells]*constants.g
        
        
        idx = self.get_scalar_index("k")
        S[:,idx] = self.rhotau[slice_cells]*dudx
        S[:,idx] += a_x[slice_cells]*dpdx
        S[:,idx] -= constants.c_d * rho[slice_cells] * (2.0*k[slice_cells])**(1.5) * vsafe_divide(1.0, L[slice_cells])
        
        idx = self.get_scalar_index("L")
        S[:,idx] = constants.c_l * rho[slice_cells] * np.sqrt(2.0*k[slice_cells])
        
        idx = self.get_scalar_index("a_x")
        S[:,idx] = constants.c_b**2 * self.b[slice_cells] * dpdx
        S[:,idx] -= constants.c_a * rho[slice_cells] * a_x[slice_cells] * np.sqrt(2.0*k[slice_cells]) * vsafe_divide(1.0, L[slice_cells])
        S[:,idx] += self.rhotau[slice_cells]/rho[slice_cells] * drhodx 
        
    def interpolate_on_face(self, u, ghost=True):
        if ghost:
            u_face = 0.5*(u[0:-1] + u[1:])
        else:
            raise NotImplementedError("")
        return u_face
            
        
    def calc_viscous_flux(self):
        Ux_face = self.Ux_face.reshape([self.n+1, self.nvar])
        Fv = self.Fv.reshape([self.n+1, self.nvar])
        mut_face = self.interpolate_on_face(self.mut)
        
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
        
        rhotau_face = self.interpolate_on_face(self.rhotau)
        Fv[:,1] = rhotau_face
        
        Fv[:,2] = self.ex*mut_face/constants.n_e

    def calc_dt(self):
        rho, u, p, Y = self.get_solution_primvars()
        a = np.sqrt(constants.gamma*p/rho)
        lambda_max = np.max(a + np.abs(u))
        #print lambda_max
        dt_inv = self.dx/lambda_max
        try:
            mut = self.mut
            mut = np.max(np.max(mut), 1e-10)
        except:
            mut = np.array([1e-10])
        dt_viscous = self.dx*self.dx/np.max(mut)/2.0

        return np.minimum(dt_inv, dt_viscous)
        

    def set_bc(self):
        U = self.U.reshape([self.n+2, self.nvar])
        Q = self.Q.reshape([self.n, self.nvar])
        
        rho = U[:, 0]
        u = U[:, 1]
        p = U[:, 2]
        Y = U[:, 3:]

        
        rho[1:-1], u[1:-1], p[1:-1], Y[1:-1,:] = self.calc_press(Q)
        #self.calc_press(Q)
        #print p
        if self.order == 2 or self.order == 1:
            interpolate_bc = lambda q: 2.0*q[1] - q[2]
        elif self.order == 1:
            interpolate_bc = lambda q: q[1]

        u[0] = -interpolate_bc(u)
        rho[0] = interpolate_bc(rho)
        p[0] = interpolate_bc(p)
        Y[0,:] = -interpolate_bc(Y)
        idx = self.get_scalar_index("Y_h")
        Y[0,idx-3] = interpolate_bc(Y[:,idx-3])

        if self.order == 2 or self.order == 1:
            interpolate_bc = lambda q: 2.0*q[-2] - q[-3]
        elif self.order == 1:
            interpolate_bc = lambda q: q[-2]

        u[-1] = -interpolate_bc(u)
        rho[-1] = interpolate_bc(rho)
        p[-1] = interpolate_bc(p)
        Y[-1,:] = -interpolate_bc(Y)
        idx = self.get_scalar_index("Y_h")
        Y[-1,idx-3] = interpolate_bc(Y[:,idx-3])

#        U = U.reshape((self.n+2)*self.nvar)
        #self.calc_gradient_face()
        #print self.x.shape, self.U.shape, self.Ux_face.shape
        Ux_face = self.Ux_face.reshape([self.n+1, self.nvar])
        Ux_center = self.Ux_center.reshape([self.n, self.nvar])
        calc_gradient_face(self.x, self.U, self.Ux_face)
        calc_gradient_center(self.x, self.U, self.Ux_center)
        #Ux_face[:,:] = f.calc_gradient_face(self.dx, U)
        #Ux_center[:,:] = f.calc_gradient_center(self.dx, U)
        #self.calc_gradient_center()
        assert(same_address(U, self.U))

if __name__ == "__main__":
    eqn = RT(n=201, nscalar=4)
    eqn.initialize()
    eqn.solve(tf=0.1, cfl = 0.5, animation=True, print_step=1000, integrator="rk4", flux="hllc", order=1, file_io=True)
    rho, rhou, rhoE, rhoY = eqn.get_solution()
    rho, u, p, Y = eqn.get_solution_primvars()
    #np.savez("data_5.npz", rho=rho, u=u, p=p, Y=Y, x=eqn.xc)
    plt.figure()
    plt.plot(eqn.xc, rho/np.abs(rho).max(), 'r-', lw=1, label="Density")
    plt.plot(eqn.xc, u/np.abs(u).max(), 'gx-', lw=1, label="Velocity")
    plt.plot(eqn.xc, p/np.abs(p).max(), 'b-', lw=1, label="Pressure")
    if eqn.nscalar > 0:
        for i in range(eqn.nscalar):
            plt.plot(eqn.xc, Y[:,i]/np.abs(Y[:,i]).max(), '-', lw=1, label=eqn.scalar_map.keys()[eqn.scalar_map.values().index(i)])
    plt.legend(loc="best")
    plt.show()
    
