import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
try:
    import adolc as ad
except:
    logger.warning("ADOLC not found. It is required for adjoint calculations.")
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import sod
def same_address(x, y):
    return x.__array_interface__['data'] == y.__array_interface__['data']
class constants(object):
    gamma = 1.4
    gamma_m = gamma - 1.0
    c_mu = 0.204
    c_a = 0.339
    c_b = 0.857
    c_d = 0.354
    c_l = 0.283
    n_a = 0.060
    n_e = 0.060
    n_k = 0.060
    n_l = 0.030
    n_y = 0.060
    c = 0.0
    R = 8.314
    
class EulerEquation(object):
    def __init__(self, n=11, nscalar=0):
        self.logger = logging.getLogger(__name__)
        self.n = n - 1
        self.nscalar = nscalar
        self.nvar = 3 + self.nscalar
        self.scalar_map = {}
        for i in range(self.nscalar):
            self.scalar_map["Y_%i"%i] = 3 + i
        self.x = np.linspace(-0.5, 0.5, self.n + 1)
        self.dx = self.x[1] - self.x[0]
        self.xc = 0.5*(self.x[0:-1] + self.x[1:])
        self.Q = np.zeros(self.n*self.nvar)

    def calc_gradient_face(self, y):
        dydx = (y[1:] - y[0:-1])/self.dx
        return dydx

    def calc_gradient_center(self, y):
        dydx = (y[2:] - y[0:-2])/(2.0*self.dx)
        return dydx
    
    def calc_dt(self):
        rho, u, p, Y = self.get_solution_primvars()
        a = np.sqrt(constants.gamma*p/rho)
        lambda_max = np.max(a + np.abs(u))
        return self.dx/lambda_max

    def calc_E(self, rho, u, p):
        rho = rho
        rhou = rho*u
        E = p/(constants.gamma-1.0) + 0.5*rho*u**2;
        return rho, rhou, E

    def get_scalar_index(self, scalar):
        if type(scalar) == type(1):
            return 3 + scalar
        elif type(scalar) == type(""):
            return self.scalar_map[scalar]
    def calc_press(self, Q):
        output = []
        rho = Q[:, 0]
        u = Q[:, 1]/Q[:, 0]
        p = (constants.gamma-1.0)*(Q[:,2] - 0.5*Q[:,1]**2/Q[:,0])
        output = [rho, u, p]
        Y = Q[:, 3:]/np.tile(Q[:, 0], (self.nscalar,1)).T
        output.append(Y)
        return output
    
    def initialize_sod(self, qleft, qright):
        Q = self.Q.reshape([self.n, self.nvar])
        Q[:self.n/2, 0], Q[:self.n/2, 1], Q[:self.n/2, 2] = self.calc_E(qleft[0], qleft[1], qleft[2]);
        Q[self.n/2:, 0], Q[self.n/2:, 1], Q[self.n/2:, 2] = self.calc_E(qright[0], qright[1], qright[2]);

        for scalar in range(self.nscalar):
            idx = self.get_scalar_index(scalar)
            Q[:self.n/2, idx] = 1.0 + idx*0.05
            Q[self.n/2:, idx] = 0.0 #+ idx
        
        Q = Q.reshape(self.n*self.nvar)
        # check if any copy happened
        assert(same_address(Q, self.Q))

    def get_solution(self):
        Q = self.Q.reshape([self.n, self.nvar])
        output = [Q[:, 0], Q[:, 1], Q[:, 2]]
        Y = Q[:, 3:]
        output.append(Y)
        return output

    def get_solution_primvars(self):
        Q = self.Q.reshape([self.n, self.nvar])
        rho, u, p, Y = self.calc_press(Q)
        return rho, u, p, Y

    def set_bc(self):
        U = self.U.reshape([self.n+2, self.nvar])
        Q = self.Q.reshape([self.n, self.nvar])
        
        rho = U[:, 0]
        u = U[:, 1]
        p = U[:, 2]
        Y = U[:, 3:]

        
        rho[1:-1], u[1:-1], p[1:-1], Y[1:-1,:] = self.calc_press(Q)
        rho[0] = rho[1]
        u[0] = u[1]
        p[0] = p[1]
        Y[0,:] = Y[1,:]

        
        rho[-1] = rho[-2]
        u[-1] = u[-2]
        p[-1] = p[-2]
        Y[-1,:] = Y[-2,:]
        
        drho = self.calc_gradient_face(rho)
        du = self.calc_gradient_face(u)
        dp = self.calc_gradient_face(p)
        dY = self.calc_gradient_face(Y)

        drhoc = self.calc_gradient_center(rho)
        duc = self.calc_gradient_center(u)
        dpc = self.calc_gradient_center(p)
        dYc = self.calc_gradient_center(Y)

        Ux_face = self.Ux_face.reshape([self.n+1, self.nvar])
        Ux_center = self.Ux_center.reshape([self.n, self.nvar])
        

        Ux_face[:,0] = drho
        Ux_face[:,1] = du
        Ux_face[:,2] = dp
        Ux_face[:,3:] = dY[:,:]

        Ux_center[:,0] = drhoc
        Ux_center[:,1] = duc
        Ux_center[:,2] = dpc
        Ux_center[:,3:] = dYc[:,:]
        #plt.figure()
        #plt.plot(0.5*(dY[1:,0]+dY[0:-1,0]), 'rx-')
        #plt.plot(dYc[:,0], 'gx-')
        #plt.show()
        
        U = U.reshape((self.n+2)*self.nvar)
        assert(same_address(U, self.U))
        Ux_face = Ux_face.reshape((self.n+1)*self.nvar)
        assert(same_address(Ux_face, self.Ux_face))
        Ux_center = Ux_center.reshape(self.n*self.nvar)
        assert(same_address(Ux_center, self.Ux_center))
        
    def reconstruct(self):
        U = self.U.reshape([self.n+2, self.nvar])
        Ul = self.Ul.reshape([self.n+1, self.nvar])
        Ur = self.Ur.reshape([self.n+1, self.nvar])

        Ul[:, :] = U[0:-1,:]
        Ur[:, :] = U[1:,:]

        U = U.reshape((self.n+2)*self.nvar)
        Ul = Ul.reshape((self.n+1)*self.nvar)
        Ur = Ur.reshape((self.n+1)*self.nvar)
        assert(same_address(U, self.U))
        assert(same_address(Ul, self.Ul))
        assert(same_address(Ur, self.Ur))
        
        
    def calc_flux_roe(self):
        F = self.F.reshape([self.n+1, self.nvar])
        Ul = self.Ul.reshape([self.n+1, self.nvar])
        rhol = Ul[:, 0]
        ul = Ul[:, 1]
        pl = Ul[:, 2]
        Yl = Ul[:, 3:]

        

        Ur = self.Ur.reshape([self.n+1, self.nvar])
        rhor = Ur[:, 0]
        ur = Ur[:, 1]
        pr = Ur[:, 2]
        Yr = Ur[:, 3:]
                
        
        el = pl/(constants.gamma_m) + 0.5*rhol*ul*ul;
        er = pr/(constants.gamma_m) + 0.5*rhor*ur*ur;
        hl = (el + pl)/rhol;
        hr = (er + pr)/rhor;

        sqrtrhol = np.sqrt(rhol);
        sqrtrhor = np.sqrt(rhor);
        den_inverse = 1/(sqrtrhol + sqrtrhor);
        uavg = (sqrtrhol*ul + sqrtrhor*ur)*den_inverse;
        havg = (sqrtrhol*hl + sqrtrhor*hr)*den_inverse;
        cavg = np.sqrt(constants.gamma_m*(havg - 0.5*uavg*uavg));
        cavg_inverse = 1.0/cavg;
        
        d1 = rhor - rhol;
        d2 = rhor*ur - rhol*ul;
        d3 = er - el;
        
        alpha_2 = constants.gamma_m*((havg - uavg*uavg)*d1 + uavg*d2 - d3)*cavg_inverse*cavg_inverse;
        alpha_3 = 0.5*(d2 + (cavg - uavg)*d1 - cavg*alpha_2)*cavg_inverse;
        alpha_1 = d1 - alpha_2 - alpha_3;
        
        lambda_1 =  np.abs(uavg - cavg);
        lambda_2 =  np.abs(uavg);
        lambda_3 =  np.abs(uavg + cavg);
        
        f1 = lambda_1*alpha_1 + lambda_2*alpha_2 + lambda_3*alpha_3;
        f2 = lambda_1*alpha_1*(uavg-cavg) + lambda_2*alpha_2*uavg + lambda_3*alpha_3*(uavg+cavg);
        f3 = lambda_1*alpha_1*(havg-cavg*uavg) + 0.5*lambda_2*alpha_2*uavg*uavg + lambda_3*alpha_3*(havg+cavg*uavg);
        
        F[:,0] = 0.5*((rhol*ul + rhor*ur) - f1);
        F[:,1] = 0.5*((rhol*ul*ul + pl + rhor*ur*ur + pr) - f2);
        F[:,2] = 0.5*(ul*hl*rhol + ur*hr*rhor - f3);
        for i in range(self.nscalar):
            F[:,self.get_scalar_index(i)] = 0.5*(rhol*ul*Yl[:,i] + rhor*ur*Yr[:,i]) + 0.5*np.sign(uavg)*(rhol*ul*Yl[:,i] - rhor*ur*Yr[:,i])
        
        F = F.reshape((self.n+1)*self.nvar)
        Ul = Ul.reshape((self.n+1)*self.nvar)
        Ur = Ur.reshape((self.n+1)*self.nvar)
        assert(same_address(Ul, self.Ul))
        assert(same_address(Ur, self.Ur))
        assert(same_address(F, self.F))
        
    def calc_flux_hllc(self):
        F = self.F.reshape([self.n+1, self.nvar])

        Ul = self.Ul.reshape([self.n+1, self.nvar])
        rhol = Ul[:, 0]
        ul = Ul[:, 1]
        pl = Ul[:, 2]
        Yl = Ul[:, 3:]

        

        Ur = self.Ur.reshape([self.n+1, self.nvar])
        rhor = Ur[:, 0]
        ur = Ur[:, 1]
        pr = Ur[:, 2]
        Yr = Ur[:, 3:]
        
        
        el = pl/(constants.gamma_m) + 0.5*rhol*ul*ul;
        er = pr/(constants.gamma_m) + 0.5*rhor*ur*ur;
        hl = (el + pl)/rhol;
        hr = (er + pr)/rhor;
        al = np.sqrt(hl - 0.5*(constants.gamma - 1)*ul*ul)
        ar = np.sqrt(hr - 0.5*(constants.gamma - 1)*ur*ur)

        sqrtrhor = np.sqrt(rhor)
        sqrtrhol = np.sqrt(rhol)

        u_hat = (sqrtrhol*ul + sqrtrhor*ur)/(sqrtrhol + sqrtrhor)
        h_hat = (sqrtrhol*hl + sqrtrhor*hr)/(sqrtrhol + sqrtrhor)
        a_hat= np.sqrt(h_hat - 0.5*(constants.gamma - 1)*u_hat*u_hat)

        Sl = np.maximum(u_hat - a_hat, ul - al)
        Sr = np.maximum(u_hat + a_hat, ur + ar)

        S_star = (pr-pl + rhol*ul*(Sl - ul) - rhor*ur*(Sr - ur))/(rhol*(Sl - ul) - rhor*(Sr - ur))

        region_0 = Sl > 0
        region_1 = np.logical_and(Sl <= 0, S_star >= 0)
        region_2 = np.logical_and(Sr >=0, S_star <= 0)
        region_3 = Sr < 0
    
    
        def calc_ustar(rho, u, p, e, Y, S, Ss):
            fac = rho * (S - u)/(S - Ss)
            U_star_0 = 1.0 * fac
            U_star_1 = Ss * fac
            U_star_2 = e/rho + (Ss-u)*(Ss + p/(rho*(S-u)))
            U_star_2 = U_star_2 * fac
            #print fac.shape
            fac = np.tile(fac, (self.nscalar,1)).T
            #print fac.shape, Y.shape
            U_star_3 = Y * fac
            return U_star_0, U_star_1, U_star_2, U_star_3


        F[region_0,0] = rhol[region_0]*ul[region_0]
        F[region_0,1] = rhol[region_0]*ul[region_0]*ul[region_0] + pl[region_0]
        F[region_0,2] = ul[region_0]*(el[region_0] + pl[region_0])
        for i in range(self.nscalar):
            idx = self.get_scalar_index(i)
            F[region_0,idx] = rhol[region_0]*ul[region_0]*Yl[region_0,i]

        F[region_1,0] = rhol[region_1]*ul[region_1]
        F[region_1,1] = rhol[region_1]*ul[region_1]*ul[region_1] + pl[region_1]
        F[region_1,2] = ul[region_1]*(el[region_1] + pl[region_1])

        for i in range(self.nscalar):
            idx = self.get_scalar_index(i)
            F[region_1,idx] = rhol[region_1]*ul[region_1]*Yl[region_1,i]

        U_star = calc_ustar(rhol[region_1], ul[region_1], pl[region_1], el[region_1], Yl[region_1], Sl[region_1], S_star[region_1])
        F[region_1,0] += Sl[region_1]*(U_star[0] - rhol[region_1])
        F[region_1,1] += Sl[region_1]*(U_star[1] - rhol[region_1]*ul[region_1])
        F[region_1,2] += Sl[region_1]*(U_star[2] - el[region_1])
        for i in range(self.nscalar):
            idx = self.get_scalar_index(i)
            F[region_1,idx] += Sl[region_1]*(U_star[3][:,i] - rhol[region_1]*Yl[region_1,i])

        
        F[region_2,0] = rhor[region_2]*ur[region_2]
        F[region_2,1] = rhor[region_2]*ur[region_2]*ur[region_2] + pr[region_2]
        F[region_2,2] = ur[region_2]*(er[region_2] + pr[region_2])
        for i in range(self.nscalar):
            idx = self.get_scalar_index(i)
            F[region_2,idx] = rhor[region_2]*ur[region_2]*Yr[region_2,i]
            
        U_star = calc_ustar(rhor[region_2], ur[region_2], pr[region_2], er[region_2], Yr[region_2], Sr[region_2], S_star[region_2])
        F[region_2,0] += Sr[region_2]*(U_star[0] - rhor[region_2])
        F[region_2,1] += Sr[region_2]*(U_star[1] - rhor[region_2]*ur[region_2])
        F[region_2,2] += Sr[region_2]*(U_star[2] - er[region_2])
        for i in range(self.nscalar):
            idx = self.get_scalar_index(i)
            F[region_2,idx] += Sr[region_2]*(U_star[3][:,i] - rhor[region_2]*Yr[region_2,i])


        F[region_3,0] = rhor[region_3]*ur[region_3]
        F[region_3,1] = rhor[region_3]*ur[region_3]*ur[region_3] + pr[region_3]
        F[region_3,2] = ur[region_3]*(er[region_3] + pr[region_3])
        for i in range(self.nscalar):
            idx = self.get_scalar_index(i)
            F[region_3,idx] = rhor[region_3]*ur[region_3]*Yr[region_3,i]

        F = F.reshape((self.n+1)*self.nvar)
        Ul = Ul.reshape((self.n+1)*self.nvar)
        Ur = Ur.reshape((self.n+1)*self.nvar)
        assert(same_address(Ul, self.Ul))
        assert(same_address(Ur, self.Ur))
        assert(same_address(F, self.F))


    def calc_viscous_flux(self):
        Ux_face = self.Ux_face.reshape([self.n+1, self.nvar])
        Fv = self.Fv.reshape([self.n+1, self.nvar])
        Fv[:,:] = 0.0
        if self.nscalar > 0:
            Fv[:,3] = Ux_face[:,3] * 1e-3
        #print Fv[:,3].max()

    def calc_source(self):
        S = self.S.reshape([self.n, self.nvar])
        Ux_center = self.Ux_center.reshape([self.n, self.nvar])
        if self.nscalar > 0:
            S[:,self.get_scalar_index("Y_1")] = Ux_center[:,2]*0.1

    def temporal_hook(self):
        pass
            
    def calc_residual(self):
        self.set_bc()
        self.reconstruct()
        self.calc_flux_hllc()
        self.calc_viscous_flux()
        self.calc_source()
        R = self.R.reshape([self.n, self.nvar])
        F = self.F.reshape([self.n+1, self.nvar])
        Fv = self.Fv.reshape([self.n+1, self.nvar])
        S = self.S.reshape([self.n, self.nvar])
        R[:,:] = - (F[1:,:] - F[0:-1,:])/self.dx
        R[:,:] += (Fv[1:,:] - Fv[0:-1,:])/self.dx
        R[:,:] += S[:,:]
        F = F.reshape((self.n+1)*self.nvar)
        R = R.reshape(self.n*self.nvar)
        assert(same_address(F, self.F))
        assert(same_address(R, self.R))
        

    def record_tape(self):
        tag = 0
        Q = self.Q.copy()
        self.Q = ad.adouble(self.Q)

        R = self.R.copy()
        self.R = ad.adouble(self.R)

        U = self.U.copy()
        self.U = ad.adouble(self.U)

        F = self.F.copy()
        self.F = ad.adouble(self.F)

        Ul = self.Ul.copy()
        self.Ul = ad.adouble(self.Ul)

        Ur = self.Ur.copy()
        self.Ur = ad.adouble(self.Ur)

        
        ad.trace_on(tag)
        ad.independent(self.Q)
        self.calc_residual()
        ad.dependent(self.R)
        ad.trace_off()
        print(ad.tapestats(0))
        self.Q = Q
        self.U = U
        self.R = R
        self.Ul = Ul
        self.Ur = Ur
        self.F = F
    def calc_step(self):
        self.calc_residual()

    def solve(self, tf = 0.1, dt = 1e-4):
        self.R = np.zeros(self.n*self.nvar)
        self.S = np.zeros(self.n*self.nvar)
        self.U = np.zeros((self.n + 2)*self.nvar)
        self.Ux_face = np.zeros((self.n + 1)*self.nvar)
        self.Ux_center = np.zeros(self.n*self.nvar)
        self.F = np.zeros((self.n + 1)*self.nvar)
        self.Fv = np.zeros((self.n + 1)*self.nvar)
        self.Ul = np.zeros((self.n+1)*self.nvar)
        self.Ur = np.zeros((self.n+1)*self.nvar)
        #self.record_tape()
        t = 0.0

        while 1:
            #tag = 0
            #options = np.array([0,0,0,0],dtype=int)
            #self.record_tape()
            self.calc_step()
            R = self.R.copy()
            #result = ad.colpack.sparse_jac_no_repeat(tag, self.Q, options)
            #nnz = result[0]
            #ridx = result[1]
            #cidx = result[2]
            #values = result[3]
            #N = self.n*self.nvar

            dt = self.calc_dt()*0.1
            #print dt
            #drdu = -sp.csr_matrix((values, (ridx, cidx)), shape=(N, N)) + sp.eye(N)/dt
            #du = spla.spsolve(drdu, R)
            #print np.linalg.norm(R - self.R)
            #self.Q  = self.Q + du
            self.Q  = self.Q + R*dt
            #print nnz
            #plt.spy(drdu)
            #plt.show()
            t += dt
            if int(t/dt)%100 == 0: 
                self.logger.info("Time = %.2e/%.2e (%.01f%% complete)"%(t, tf, t/tf*100))

            if t > tf:
                self.logger.info("Time = %.2e/%.2e (%.01f%% complete)"%(t, tf, t/tf*100))
                break
if __name__ == "__main__":
    qleft = np.array([1.0, 0.0, 1.0])
    qright = np.array([0.125, 0.0, 0.1])
    eqn = EulerEquation(n=501)
    eqn.initialize_sod(qleft, qright)
    eqn.solve(tf=0.2)
    rho, rhou, rhoE, rhoY = eqn.get_solution()
    rho, u, p, Y = eqn.get_solution_primvars()

    plt.figure()
    plt.plot(eqn.xc, rho, 'r-', lw=1, label="Density")
    plt.plot(eqn.xc, u, 'g-', lw=1, label="Velocity")
    plt.plot(eqn.xc, p, 'b-', lw=1, label="Pressure")
    if eqn.nscalar > 0:
        plt.plot(eqn.xc, rhoY, 'cx-', lw=1, label="Y")

    # #plt.plot(eqn.xc, rhou, 'x-', lw=1)
    # #plt.plot(eqn.xc, rhoE, 'x-', lw=1)

    positions, regions, values = sod.solve(left_state=(1, 1, 0), right_state=(0.1, 0.125, 0.),
                                           geometry=(-0.5, 0.5, 0.0), t=0.2, gamma=1.4, npts=101)

    plt.plot(values['x'], values['rho'], 'r--', lw=1, label="Density")
    plt.plot(values['x'], values['u'], 'g--', lw=1, label="Velocity")
    plt.plot(values['x'], values['p'], 'b--', lw=1, label="Pressure")

    plt.show()
    
