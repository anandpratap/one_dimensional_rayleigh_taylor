from main import EulerEquation, same_address, calc_gradient_face, calc_gradient_center, safe_divide, vsafe_divide
import numpy as np
import matplotlib.pyplot as plt
#import adolc as ad

class constants(object):
    gamma = 5.0/3.0
    gamma_m = gamma - 1.0
    c_mu = 0.20364675
    b_t = 29.166667 #8.4
    c_d = 0.35355339
    c_c = 0.0
    c_l = 0.28284271
    n_y = 0.060000
    n_e = 0.060000
    n_k = 0.060000
    n_l = 0.030000
    R = 8.314
    g = 1000 * 9.8 * -1e2
    n_v = 0.06
    c_v1 = 46.67
    c_v2 = 0.849


def get_alpha_t_5(step, alpha):
    n = alpha.size
    assert n == 5
    for i in range(n-1):
        step_start = 10000 + i*5000
        step_end  = 10000 + (i+1)*5000
        if step > step_start and step <= step_end:
            num = alpha[i+1] - alpha[i]
            den = step_end - step_start
            a = alpha[i] + num/den * (step -  step_start)
            return a
    return 1.0

def get_alpha_t(step, alpha, maxstep):
    n = alpha.size
    ends = np.linspace(0, maxstep, alpha.size, dtype=np.int32)
    for i in range(n-1):
        step_start = ends[i]
        step_end  = ends[i+1]
        if step >= step_start and step <= step_end:
            num = alpha[i+1] - alpha[i]
            den = step_end - step_start
            a = alpha[i] + num/den * (step -  step_start)
            return a
    return 1.0


def calc_partial_volume(Yh, Yl, rho, At):
    rhoh, rhol = calc_partial_rho(At)
    vh = rho*Yh/rhoh
    vl = rho*Yl/rhol
    return vh, vl


def calc_partial_rho(At):
    rho_h = 1.0
    rho_l = (1.0 - At)/(1.0 + At) * rho_h
    return rho_h, rho_l


def calc_eosp(rho, e):
    gamma = constants.gamma
    return (gamma - 1.0)*rho*e


def smooth(x, left, right):
    return (np.tanh(80*x) + 1.0)/2.0 * (right-left) + left


class RT(EulerEquation):
    def initialize(self):
        self.At = 0.05
        L_init = 4e-8
        k_init = 1.6e-15
        T_ref = 293.0
        P_ref = 101325.0


        self.scalar_map = {"k":0, "L": 1, "Y_h": 2, "V_h": 3}
        Q = self.Q.reshape([self.n, self.nvar])

        
        rho_h, rho_l = calc_partial_rho(self.At)
        
        
        R = constants.R
        gamma = constants.gamma

        rho = np.zeros(self.n)
        Yh = np.zeros(self.n)
        L = np.zeros(self.n)

        first_half = self.xc <= 0.0
        second_half = self.xc > 0.0

        
        Yh[first_half] = 0.0
        Yh[second_half] = 1.0
        Yh[self.n/2] = 0.5
        Yh[self.n/2-1] = 0.25
        Yh[self.n/2+1] = 0.75

        Yl = 1.0 - Yh
               
        rho = rho_l * Yl + rho_h * Yh

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
        nc = 2
        idx = self.get_scalar_index("k")
        Q[:, idx] = 0.0
        Q[self.n/2-nc:self.n/2+nc, idx] = k_init * rho[self.n/2-nc:self.n/2+nc]
        
        idx = self.get_scalar_index("L")
        Q[:, idx] = 0.0
        Q[self.n/2-nc:self.n/2+nc, idx] = L_init * rho[self.n/2-nc:self.n/2+nc]

        idx = self.get_scalar_index("Y_h")
        Q[:, idx] =  Yh * rho
        
        idx = self.get_scalar_index("V_h")
        Q[:, idx] = 0.0
        dx = self.x[1] - self.x[0]
        h = 2*nc*dx
        yp = 1.0 - ((self.xc)/h)**2
        Vh = 0.022*yp
        Vh[yp<=0.0] = 0.0
        Q[:, idx] = Vh*rho #0.0#-0.0000005 * rho[self.n/2-nc:self.n/2+nc]

        Q = Q.reshape(self.n*self.nvar)
        assert(same_address(Q, self.Q))

    def calc_E(self, rho, u, p):
        rho = rho
        rhou = rho*u
        E = p/(constants.gamma-1.0) + 0.5*rho*u**2;
        return rho, rhou, E

    @staticmethod
    def calc_Ee(rho, u, e):
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
        idx = self.get_scalar_index("L")
        L = U[:,idx]
        #print np.sqrt, k
        self.mut[:] = constants.c_mu * rho * L * np.sqrt(2.0*k)
        self.mut[0] = self.mut[-1] = 0.0
        self.rhotau[:] = -2.0/3.0 * rho * k 
        
        e = np.zeros_like(rho) 

        e[1:-1] = (Q[:,2] - 0.5*rho[1:-1]*u[1:-1]**2)/rho[1:-1]
        e[0] = e[1]
        e[-1] = e[-2]
        self.ex[:] = (e[1:] - e[0:-1])/self.dx

        self.rhotaux = (self.rhotau[2:] - self.rhotau[0:-2])/self.dx/2.0

        
    def calc_primvars(self, Q):
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

        #clip k to be positive
        idx = self.get_scalar_index("k")
        #Y[:,idx-3] = np.clip(Y[:,idx-3], 0.0, 1e10)
        idx_neg = np.where(Y[:,idx-3] < 0.0)
        Y[idx_neg,idx-3] = np.zeros(1, dtype=Y.dtype)[0]*rho[0]
        # clip L to be positive
        idx = self.get_scalar_index("L")
        idx_neg = np.where(Y[:,idx-3] < 0.0)
        Y[idx_neg,idx-3] = np.zeros(1, dtype=Y.dtype)[0]*rho[0]
        #print idx_neg
        #Y[:,idx-3] = np.clip(Y[:,idx-3], 0.0, 1e10)
        
        
        output.append(Y)
        return output
    
    def calc_source(self):
        S = self.S.reshape([self.n, self.nvar])
        U = self.U.reshape([self.n+2, self.nvar])
        Ux_center = self.Ux_center.reshape([self.n, self.nvar])
        drhodx = Ux_center[:,0]
        dudx = Ux_center[:,1]
        dpdx = Ux_center[:,2]
        dYhdx = Ux_center[:,-2]
        rho = U[:, 0]
        u = U[:, 1]
        p = U[:, 2]
        Y = U[:, 3:]
        idx = self.get_scalar_index("k")
        k = U[:,idx]
        idx = self.get_scalar_index("L")
        L = U[:,idx]

        idx = self.get_scalar_index("V_h")
        Vh = U[:,idx]

        idx = self.get_scalar_index("Y_h")
        Yh = U[:,idx]

        S[:,:] = 0.0
        slice_cells = slice(1,-1)
        S[:,1] = rho[slice_cells]*constants.g + self.rhotaux
        S[:,2] = rho[slice_cells]*u[slice_cells]*constants.g

        term = constants.c_d*rho[slice_cells]*(2.0*k[slice_cells])**1.5 * safe_divide(1.0, L[slice_cells]) + constants.b_t*self.mut[slice_cells]/rho[slice_cells]**2 * drhodx*dpdx
        
        S[:,2] = S[:,2] + term
        idx = self.get_scalar_index("k")
        S[:,idx] = self.rhotau[slice_cells]*dudx - term
        idx = self.get_scalar_index("L")
        if self.ml:
            kr = 109.0*100.0
            sc = 10.0
            num = sum(Vh)
            den = sum(Yh*(1.0-Yh))
            theta = (1.0 - num/(den+1e-10))
            t0 = 1.073722e-05
            tb = self.time_step*1e-8/t0
            #print tb, theta
            X = np.array([[kr, sc, tb, theta]])
            Y = self.nn.predict((X-self.xmean)/self.xstd)
            Y = Y[0]
            if tb > 20.0:
                Y = 1.0

        S[:,idx] = constants.c_c*rho[slice_cells]*L[slice_cells]*dudx + get_alpha_t(self.time_step, self.alpha, self.maxstep)*constants.c_l*rho[slice_cells]*np.sqrt(2.0*k[slice_cells])
        if self.ml:
            S[:,idx] = constants.c_c*rho[slice_cells]*L[slice_cells]*dudx + Y*constants.c_l*rho[slice_cells]*np.sqrt(2.0*k[slice_cells])

        idx = self.get_scalar_index("V_h")
        S[:,idx] = constants.c_v1*self.mut[slice_cells]*dYhdx*dYhdx - constants.c_v2*rho[slice_cells]*np.sqrt(2.0*k[slice_cells]) * safe_divide(1.0, L[slice_cells])*Vh[slice_cells]

        
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
        Fv[:,idx] = dkdx_face*mut_face/constants.n_k# * self.alpha[0]
        
        idx = self.get_scalar_index("L")
        dLdx_face = Ux_face[:,idx]
        Fv[:,idx] = dLdx_face*mut_face/constants.n_l# * self.alpha[1]
        
        idx = self.get_scalar_index("Y_h")
        dYhdx_face = Ux_face[:,idx]
        Fv[:,idx] = dYhdx_face*mut_face/constants.n_y # * get_alpha_t(self.time_step, self.alpha) # * self.alpha[2]

        idx = self.get_scalar_index("V_h")
        dVhdx_face = Ux_face[:,idx]
        Fv[:,idx] = dVhdx_face*mut_face/constants.n_v 

        
        rhotau_face = self.interpolate_on_face(self.rhotau)
        Fv[:,2] = self.ex*mut_face/constants.n_e
        
        
    def calc_dt(self):
        rho, u, p, Y = self.get_solution_primvars()
        a = np.sqrt(constants.gamma*p/rho)
        lambda_max = np.max(a + np.abs(u))
        dt_inv = self.dx/lambda_max
        try:
            mut = self.mut
            mut = np.max(np.max(mut), 1e-10)
        except:
            mut = np.array([1e-10])
        dt_viscous = self.dx*self.dx/np.max(mut)/2.0

        return np.minimum(dt_inv, dt_viscous)

    def temporal_hook_post(self):
        Q = self.Q.reshape([self.n, self.nvar])
        Q[0,1] = 0.0
        Q[-1,1] = 0.0

        idx = self.get_scalar_index("k")
        Q[0,idx] = 0.0
        Q[-1,idx] = 0.0

        idx = self.get_scalar_index("L")
        Q[0,idx] = 0.0
        Q[-1,idx] = 0.0

        
    def set_bc(self):
        U = self.U.reshape([self.n+2, self.nvar])
        Q = self.Q.reshape([self.n, self.nvar])
        
        rho = U[:, 0]
        u = U[:, 1]
        p = U[:, 2]
        Y = U[:, 3:]

        
        rho[1:-1], u[1:-1], p[1:-1], Y[1:-1,:] = self.calc_primvars(Q)
        
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

        Ux_face = self.Ux_face.reshape([self.n+1, self.nvar])
        Ux_center = self.Ux_center.reshape([self.n, self.nvar])
        calc_gradient_face(self.x, self.U, self.Ux_face)
        calc_gradient_center(self.x, self.U, self.Ux_center)
        assert(same_address(U, self.U))

if __name__ == "__main__":
    eqn = RT(n=401, nscalar=4, ml=True)
    eqn.initialize()
    #eqn.alpha = np.array([1.00318294, 0.98352435, 0.8])
    #eqn.alpha[0] = 1.0
    eqn.solve(tf=8.4e-4, cfl = 0.5, animation=False, print_step=1, integrator="fe", flux="hllc", order=1, file_io=True, maxstep=40000, jacobian_mode=None,main_run=True)

    #eqn.solve(tf=8.4e-4, cfl = 0.5, animation=False, print_step=1, integrator="fe", flux="hllc", order=1, file_io=True, maxstep=40000, jacobian_mode=None)
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

    from calc_adjoint_mpi import kr_dict, get_configs, get_t0, interpolate, load_data

    filename = "les_data/RT_G0013_S001_A05_NB004-016..ult"
    iteration_buffer, Q_buffer = eqn.buffer.iteration, eqn.buffer.Q_buffer

    Gr, Sc, At = get_configs(filename)
        
    kr_rans =  kr_dict[(Gr, Sc)] * (-1e10) / 100.0 / constants.g
    t0_RANS = get_t0(At, constants.g, kr_rans)
    t0_LES = get_t0(At, -1e10, kr_dict[(Gr, Sc)])
    print t0_RANS, t0_LES
    data = load_data(filename)
    data[6,:,0] = data[6,:,0]/t0_LES*1e-6
    data[6,0,0] = -data[6,0,0]
    t_MAX_LES = data[6,:,0].max() - 1e-10
    print t_MAX_LES
    func_theta = interpolate.interp1d(data[6,:,0], data[6,:,1], bounds_error=True, fill_value=0.0)


    theta_rans = np.zeros(40000)
    theta_les = np.zeros_like(theta_rans)
    for i in range(1, 40001):
        idx = np.where(iteration_buffer == i)[0][0]
        Q_ = Q_buffer[idx,:]
        Q = np.reshape(Q_, [Q_.size/7, 7])
        rho = Q[:,0]
        rhoY = Q[:,3:]
        y = rhoY[:,2]/rho
        v = rhoY[:,3]/rho
        num = sum(v)
        den = sum(y*(1.0-y))
        theta = 1.0 - num/den
        theta_rans[i-1] = theta
        
        time = i*1e-8/t0_RANS
        if time < t_MAX_LES:
            theta_les[i-1] = func_theta(time)




    plt.figure()
    plt.plot(theta_les, label="LES")
    plt.plot(theta_rans, label="RANS")
    plt.legend()

    plt.show()
    
