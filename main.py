try:
    profile
except NameError:
    profile = lambda x: x
import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
try:
    import adolc as ad
except:
    logger.warning("ADOLC not found. It is required for adjoint calculations.")
#from mpi4py import MPI as mpi

class Mpi(object):
    def __init__(self):
        #self.comm = mpi.COMM_WORLD
        #self.size = self.comm.Get_size()
        self.rank = 0 #self.comm.Get_rank()

import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import sod
import fortutils as f
from limiters import Limiters
def safe_divide(x, y):
    epsilon = 1e-12
    return x/(y+epsilon)
    # if np.abs(x) > epsilon and np.abs(y) < epsilon:
    #     return x/epsilon
    # elif np.abs(x) < epsilon and np.abs(y) < epsilon:
    #     return 0.0
    # else:
    #     return x/y

vsafe_divide = np.vectorize(safe_divide)
#vsafe_divide = safe_divide

def same_address(x, y):
    return x.__array_interface__['data'] == y.__array_interface__['data']

def slope_limiter(r):
    if r>=0.0:
        return 2.0*r/(1.0 + r)
    else:
        return 0
vslope_limiter = np.vectorize(slope_limiter)

class constants(object):
    gamma = 5.0/3.0
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
    g = 9.8 * -1e2

#@jit(f8(f8[:],f8[:],f8[:]), nopython=True)
def calc_gradient_face(x, U, Ux_face):
    n = x.size - 1
    nvar = U.size/(n+2)
    dx = x[1] - x[0]
    Ux_face = Ux_face.reshape((n+1, nvar))
    U = U.reshape((n+2, nvar))
    Ux_face[:,:] = (U[1:,:] - U[0:-1,:])/dx

#@jit(nopython=True)
def calc_gradient_center(x, U, Ux_center):
    n = x.size - 1
    nvar = U.size/(n+2)
    dx = x[1] - x[0]
    Ux_center = Ux_center.reshape([n, nvar])
    U = U.reshape([n+2, nvar])
    Ux_center[:,:] = (U[2:,:] - U[0:-2,:])/(2.0*dx)
    
class EulerEquation(object):
    """Base class to solve the one dimensional Euler Equation.

    """
    def __init__(self, n=11, nscalar=0):
        self.mpi = Mpi()
        self.logger = logging.getLogger(__name__)
        self.n = n - 1
        self.nscalar = nscalar
        self.nvar = 3 + self.nscalar
        self.scalar_map = {}
        for i in range(self.nscalar):
            self.scalar_map["Y_%i"%i] = i
        self.x = np.linspace(-0.5, 0.5, self.n + 1)
        self.dx = self.x[1] - self.x[0]
        self.xc = 0.5*(self.x[0:-1] + self.x[1:])
        self.Q = np.zeros(self.n*self.nvar)
        self.limiter = Limiters("koren")
        self.alpha = np.ones(1)
    @profile
    def calc_gradient_face(self):
        Ux_face = self.Ux_face.reshape([self.n+1, self.nvar])
        U = self.U.reshape([self.n+2, self.nvar])
        Ux_face[:,:] = (U[1:,:] - U[0:-1,:])/self.dx
        
    @profile
    def calc_gradient_center(self):
        Ux_center = self.Ux_center.reshape([self.n, self.nvar])
        U = self.U.reshape([self.n+2, self.nvar])
        Ux_center[:,:] = (U[2:,:] - U[0:-2,:])/(2.0*self.dx)
        #Ux_center[0,:] = (U[2,:] - U[1,:])/self.dx
        #Ux_center[-1,:] = (U[-2,:] - U[-3,:])/self.dx
    
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
            return 3 + self.scalar_map[scalar]
    @profile
    def calc_press(self, Q):
        output = []
        rho = Q[:, 0]
        inv_rho = 1.0/rho
        u = Q[:, 1]*inv_rho
        p = (constants.gamma-1.0)*(Q[:,2] - 0.5*Q[:,1]*Q[:,1]*inv_rho)
        output = [rho, u, p]
        Y = Q[:, 3:]*inv_rho[:,np.newaxis]
        output.append(Y)
        return output
    
    def initialize(self, qleft, qright):
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
    @profile
    def set_bc(self):
        U = self.U.reshape([self.n+2, self.nvar])
        Q = self.Q.reshape([self.n, self.nvar])
        
        rho = U[:, 0]
        u = U[:, 1]
        p = U[:, 2]
        Y = U[:, 3:]

        
        rho[1:-1], u[1:-1], p[1:-1], Y[1:-1,:] = self.calc_press(Q)
        tmpbc = False

        if tmpbc:
            if self.order == 1:
                u[0] = u[1]
            else:
                u[0] = 2.0*u[1] - u[2]
        else:
            if self.order == 1:
                u[0] = -u[1]
            else:
                u[0] = -(2.0*u[1] - u[2])
            
        if self.order == 1:
            rho[0] = rho[1]
            p[0] = p[1]
            Y[0,:] = Y[1,:]
        else:
            rho[0] = 2.0*rho[1] - rho[2]
            p[0] = 2.0*p[1] - p[2]
            Y[0,:] = 2.0*Y[1,:] - Y[2,:]

        #Y[0,0] = -(2.0*Y[1,0] - Y[2,0])
        #Y[0,3] = -(2.0*Y[1,3] - Y[2,3])

        
        
        if tmpbc:
            if self.order == 1:
                u[-1] = u[-2]
            else:
                u[-1] = 2.0*u[-2] - u[-3]
        else:
            if self.order == 1:
                u[-1] = -u[-2]
            else:
                u[-1] = -(2.0*u[-2] - u[-3])

        if self.order == 1:
            rho[-1] = rho[-2]
            p[-1] = p[-2]
            Y[-1,:] = Y[-2,:]
        else:
            rho[-1] = 2.0*rho[-2] - rho[-3]
            p[-1] = 2.0*p[-2] - p[-3]
            Y[-1,:] = 2.0*Y[-2,:] - Y[-3,:]

        #Y[-1,0] = -(2.0*Y[-2,0] - Y[-3,0])
        #Y[-1,3] = -(2.0*Y[-2,3] - Y[-3,3])

        U = U.reshape((self.n+2)*self.nvar)
        self.calc_gradient_face()
        self.calc_gradient_center()
        assert(same_address(U, self.U))
        
    def reconstruct(self):
        U = self.U.reshape([self.n+2, self.nvar])
        Ul = self.Ul.reshape([self.n+1, self.nvar])
        Ur = self.Ur.reshape([self.n+1, self.nvar])
        if self.order==1:
            Ul[:, :] = U[0:-1,:]
            Ur[:, :] = U[1:,:]
        else:
            rp = vsafe_divide(U[2:-1,:] - U[1:-2,:], U[1:-2,:] - U[0:-3,:])
            rm = vsafe_divide(1.0, rp)
            phip = vslope_limiter(rp)
            phim = vslope_limiter(rm)
            Ul[:, :] = U[0:-1,:]
            Ur[:, :] = U[1:,:]
            #print phip.shape
            #print Ul[1:,:].shape
            #print phip.max(), phip.min()
            #print phim.max(), phim.min()
            Ul[1:-1, :] = U[1:-2, :] + 0.5*phip[:,:] * (U[1:-2,:] - U[0:-3,:])
            Ur[0:-2, :] = U[1:-2, :] - 0.5*phim[:,:] * (U[2:-1,:] - U[1:-2,:])
            # plt.figure()
            # plt.ioff()
            # plt.plot(Ul[:,0])
            # plt.plot(Ur[:,0])
            # plt.plot(U[0:-1,0])
            # plt.plot(U[1:,0])
            # plt.show()
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
        #print lambda_1.min(), lambda_2.max(), lambda_3.max()
        #print (uavg/cavg).max()
        f1 = lambda_1*alpha_1 + lambda_2*alpha_2 + lambda_3*alpha_3;
        f2 = lambda_1*alpha_1*(uavg-cavg) + lambda_2*alpha_2*uavg + lambda_3*alpha_3*(uavg+cavg);
        f3 = lambda_1*alpha_1*(havg-cavg*uavg) + 0.5*lambda_2*alpha_2*uavg*uavg + lambda_3*alpha_3*(havg+cavg*uavg);
        
        F[:,0] = 0.5*((rhol*ul + rhor*ur) - f1);
        F[:,1] = 0.5*((rhol*ul*ul + pl + rhor*ur*ur + pr) - f2);
        F[:,2] = 0.5*(ul*hl*rhol + ur*hr*rhor - f3);
        
        # uavg = 0.5*(ur+ul)
        
        # _Fl = rhol*ul
        # _Fr = rhor*ur 
        # F[:,0] = 0.5*(_Fl + _Fr) + 0.5*np.sign(uavg)*(_Fl - _Fr)

        # _Fl = rhol*ul*ul
        # _Fr = rhor*ur*ur
        # F[:,1] = 0.5*(_Fl + _Fr) + 0.5*np.sign(uavg)*(_Fl - _Fr)
        
        # _Fl = rhol*ul*el/rhol
        # _Fr = rhor*ur*er/rhor
        # F[:,2] = 0.5*(_Fl + _Fr) + 0.5*np.sign(uavg)*(_Fl - _Fr)
        

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
        
        el = pl/(constants.gamma_m) + 0.5*rhol*ul*ul
        er = pr/(constants.gamma_m) + 0.5*rhor*ur*ur
        hl = (el + pl)/rhol
        hr = (er + pr)/rhor
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
    
        @profile
        def calc_ustar(rho, u, p, e, Y, S, Ss):
            fac = rho * (S - u)/(S - Ss)
            U_star_0 = 1.0 * fac
            U_star_1 = Ss * fac
            U_star_2 = e/rho + (Ss-u)*(Ss + p/(rho*(S-u)))
            U_star_2 = U_star_2 * fac
            #print fac.shape
            #print self.nscalar
            #fac = fac
            #fac = np.tile(fac, (self.nscalar,1)).T
            #print fac.shape, Y.shape
            U_star_3 = Y * fac[:,np.newaxis]
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
        F[region_1,0] = F[region_1,0] + Sl[region_1]*(U_star[0] - rhol[region_1])
        F[region_1,1] = F[region_1,1] + Sl[region_1]*(U_star[1] - rhol[region_1]*ul[region_1])
        F[region_1,2] = F[region_1,2] + Sl[region_1]*(U_star[2] - el[region_1])
        for i in range(self.nscalar):
            idx = self.get_scalar_index(i)
            F[region_1,idx] = F[region_1,idx] + Sl[region_1]*(U_star[3][:,i] - rhol[region_1]*Yl[region_1,i])

        
        F[region_2,0] = rhor[region_2]*ur[region_2]
        F[region_2,1] = rhor[region_2]*ur[region_2]*ur[region_2] + pr[region_2]
        F[region_2,2] = ur[region_2]*(er[region_2] + pr[region_2])
        for i in range(self.nscalar):
            idx = self.get_scalar_index(i)
            F[region_2,idx] = rhor[region_2]*ur[region_2]*Yr[region_2,i]
            
        U_star = calc_ustar(rhor[region_2], ur[region_2], pr[region_2], er[region_2], Yr[region_2], Sr[region_2], S_star[region_2])
        F[region_2,0] = F[region_2,0] + Sr[region_2]*(U_star[0] - rhor[region_2])
        F[region_2,1] = F[region_2,1] + Sr[region_2]*(U_star[1] - rhor[region_2]*ur[region_2])
        F[region_2,2] = F[region_2,2] + Sr[region_2]*(U_star[2] - er[region_2])
        for i in range(self.nscalar):
            idx = self.get_scalar_index(i)
            F[region_2,idx] = F[region_2,idx] + Sr[region_2]*(U_star[3][:,i] - rhor[region_2]*Yr[region_2,i])


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

    def calc_source(self):
        S = self.S.reshape([self.n, self.nvar])
        S[:,:] = 0.0
    def temporal_hook(self):
        pass
    @profile
    def calc_residual(self):
        self.set_bc()
        self.temporal_hook()
        self.reconstruct()
        if self.flux == "roe":
            self.calc_flux_roe()
        elif self.flux == "hllc":
            self.calc_flux_hllc()
        elif self.flux == "hllcf":
            F = self.F.reshape([self.n+1, self.nvar])
            Ul = self.Ul.reshape([self.n+1, self.nvar])
            Ur = self.Ur.reshape([self.n+1, self.nvar])
            F[:,:] = f.calc_flux_hllc(Ul, Ur)
        elif self.flux == "none":
            F = self.F.reshape([self.n+1, self.nvar])
            F[:,:] = 0.0
        else:
            raise ValueError("Flux not defined.")

        self.calc_viscous_flux()
        self.calc_source()
        R = self.R.reshape([self.n, self.nvar])
        F = self.F.reshape([self.n+1, self.nvar])
        Fv = self.Fv.reshape([self.n+1, self.nvar])
        S = self.S.reshape([self.n, self.nvar])
        R[:,:] = - (F[1:,:] - F[0:-1,:])/self.dx
        R[:,:] = R[:,:] + (Fv[1:,:] - Fv[0:-1,:])/self.dx
        R[:,:] = R[:,:] + S[:,:]
        #diff = - (F[1:,1] - F[0:-1,1])/self.dx + S[:,1]
        #for i in range(self.n):
        #idx = diff.argmax()
        #print diff.max(), self.xc[idx]
        F = F.reshape((self.n+1)*self.nvar)
        R = R.reshape(self.n*self.nvar)
        assert(same_address(F, self.F))
        assert(same_address(R, self.R))
        

    def record_tape(self, tag=0):
        #import numpy as np
        
        Q = self.Q.copy()
        self.Q = ad.adouble(self.Q)

        alpha = self.alpha.copy()
        self.alpha = ad.adouble(self.alpha)

        R = self.R.copy()
        self.R = ad.adouble(self.R)

        U = self.U.copy()
        self.U = ad.adouble(self.U)

        F = self.F.copy()
        self.F = ad.adouble(self.F)

        Fv = self.Fv.copy()
        self.Fv = ad.adouble(self.Fv)

        S = self.S.copy()
        self.S = ad.adouble(self.S)

        mut = self.mut.copy()
        self.mut = ad.adouble(self.mut)

        b = self.b.copy()
        self.b = ad.adouble(self.b)

        rhotau = self.rhotau.copy()
        self.rhotau = ad.adouble(self.rhotau)

        ex = self.ex.copy()
        self.ex = ad.adouble(self.ex)
        
        Ul = self.Ul.copy()
        self.Ul = ad.adouble(self.Ul)

        Ur = self.Ur.copy()
        self.Ur = ad.adouble(self.Ur)
        
        Ux_face = self.Ux_face.copy()
        self.Ux_face = ad.adouble(self.Ux_face)
        Ux_center = self.Ux_center.copy()
        self.Ux_center = ad.adouble(self.Ux_center)

        
        ad.trace_on(tag)
        if tag == 0:
            ad.independent(self.Q)
        elif tag == 1:
            ad.independent(self.alpha)
        self.calc_residual()
        ad.dependent(self.R)
        ad.trace_off()
        #print(ad.tapestats(0))
        
        self.Q = Q
        self.alpha = alpha
        self.U = U
        self.R = R
        self.Ul = Ul
        self.Ur = Ur
        self.F = F
        self.Fv = Fv
        self.S = S
        self.Ux_face = Ux_face
        self.Ux_center = Ux_center
        self.ex = ex
        self.b = b
        self.rhotau = rhotau
        self.mut = mut


    def complex_step(self, tag=0, stage=0):
        #import numpy as np
        
        Q = self.Q.copy()
        self.Q = (self.Q).astype(complex)

        alpha = self.alpha.copy()
        self.alpha = (self.alpha).astype(complex)

        R = self.R.copy()
        self.R = (self.R).astype(complex)

        U = self.U.copy()
        self.U = (self.U).astype(complex)

        F = self.F.copy()
        self.F = (self.F).astype(complex)

        Fv = self.Fv.copy()
        self.Fv = (self.Fv).astype(complex)

        S = self.S.copy()
        self.S = (self.S).astype(complex)

        mut = self.mut.copy()
        self.mut = (self.mut).astype(complex)

        b = self.b.copy()
        self.b = (self.b).astype(complex)

        rhotau = self.rhotau.copy()
        self.rhotau = (self.rhotau).astype(complex)

        ex = self.ex.copy()
        self.ex = (self.ex).astype(complex)
        
        Ul = self.Ul.copy()
        self.Ul = (self.Ul).astype(complex)

        Ur = self.Ur.copy()
        self.Ur = (self.Ur).astype(complex)
        
        Ux_face = self.Ux_face.copy()
        self.Ux_face = (self.Ux_face).astype(complex)
        Ux_center = self.Ux_center.copy()
        self.Ux_center = (self.Ux_center).astype(complex)

        if tag == 0:
            # calculate drdq
            nsize = self.n*(self.nscalar+3)
            if stage == 0:
                self.dRdQ_complex = np.zeros([nsize, nsize])
            else:
                self.dRdQ_complex_n = np.zeros([nsize, nsize])
                
            dQ = 1e-14
            for i in range(nsize):
                self.Q[i] = self.Q[i] + 1j*dQ
                self.calc_residual()
                self.Q[i] = self.Q[i] - 1j*dQ
                if stage == 0:
                    self.dRdQ_complex[:,i] = np.imag(self.R[:])/dQ
                else:
                    self.dRdQ_complex_n[:,i] = np.imag(self.R[:])/dQ
                               
        elif tag == 1:
            # calculate drdalpha
            nsize = self.n*(self.nscalar+3)
            nalpha = self.alpha.size
            if stage == 0:
                self.dRdalpha_complex = np.zeros([nsize, nalpha])
            else:
                self.dRdalpha_complex_n = np.zeros([nsize, nalpha])
            dalpha = 1e-14
            for i in range(nalpha):
                self.alpha[i] = self.alpha[i] + 1j*dalpha
                self.calc_residual()
                self.alpha[i] = self.alpha[i] - 1j*dalpha
                if stage == 0:
                    self.dRdalpha_complex[:,i] = np.imag(self.R[:])/dalpha
                else:
                    self.dRdalpha_complex_n[:,i] = np.imag(self.R[:])/dalpha

        self.Q = Q
        self.alpha = alpha
        self.U = U
        self.R = R
        self.Ul = Ul
        self.Ur = Ur
        self.F = F
        self.Fv = Fv
        self.S = S
        self.Ux_face = Ux_face
        self.Ux_center = Ux_center
        self.ex = ex
        self.b = b
        self.rhotau = rhotau
        self.mut = mut

        
    def calc_step(self):
        self.calc_residual()

    def temporal_hook_post(self):
        pass
        
    def solve(self, tf = 0.1, dt = 1e-4, animation = False, cfl=1.0, print_step=100, integrator="fe", flux="hllc", order=1, file_io=False, maxstep=1e10):
        if animation and self.mpi.rank == 0:
            plt.ion()
            plt.figure(figsize=(10,10))
        self.R = np.zeros(self.n*self.nvar)
        self.S = np.zeros(self.n*self.nvar)
        self.U = np.zeros((self.n + 2)*self.nvar)

        self.mut = np.zeros((self.n + 2))
        self.rhotau = np.zeros((self.n + 2))
        self.b = np.zeros((self.n + 2))
        self.ex = np.zeros((self.n + 1))
        
        self.Ux_face = np.zeros((self.n + 1)*self.nvar)
        self.Ux_center = np.zeros(self.n*self.nvar)
        self.F = np.zeros((self.n + 1)*self.nvar)
        self.Fv = np.zeros((self.n + 1)*self.nvar)
        self.Ul = np.zeros((self.n+1)*self.nvar)
        self.Ur = np.zeros((self.n+1)*self.nvar)
        self.flux = flux
        self.order = order
        #self.record_tape(/)
        t = 0.0
        step = 0
        
        if integrator == "rk2":
            k1 = np.zeros_like(self.R)
            Qn = np.zeros_like(self.Q)
            k2 = np.zeros_like(self.R)
        elif integrator == "rk4":
            k1 = np.zeros_like(self.R)
            Qn = np.zeros_like(self.Q)
            k2 = np.zeros_like(self.R)
            k3 = np.zeros_like(self.R)
            k4 = np.zeros_like(self.R)
            
        while 1:
            
            if integrator == "be":
                tag = 0
                options = np.array([0,0,0,0],dtype=int)
                self.record_tape(tag=tag)
                result = ad.colpack.sparse_jac_no_repeat(tag, self.Q, options)
                
                nnz = result[0]
                ridx = result[1]
                cidx = result[2]
                values = result[3]
                N = self.n*self.nvar
                self.calc_residual()
                R = self.R.copy()
                #print dt
                #print values
                drdu = -sp.csr_matrix((values, (ridx, cidx)), shape=(N, N)) + sp.eye(N)/dt
                #print drdu
                dQ = spla.spsolve(drdu, R)
                #print np.linalg.norm(R - self.R)
                self.Q  = self.Q + dQ
                
            elif integrator == "fe":
                self.calc_step()
                R = self.R.copy()
                dt = 1e-5 #self.calc_dt()*cfl
                dQ = R*dt
                options = np.array([0,0,0,0],dtype=int)
                tag = 0
                self.record_tape(tag=tag)
                result = ad.colpack.sparse_jac_no_repeat(tag, self.Q, options)
                nnz = result[0]
                ridx = result[1]
                cidx = result[2]
                values = result[3]
                N = self.n*self.nvar
                self.dRdQ = sp.csr_matrix((values, (ridx, cidx)), shape=(N, N))
                self.complex_step(tag=tag)
                print np.linalg.norm(np.nan_to_num(self.dRdQ.toarray()))
                print np.linalg.norm(self.dRdQ_complex)
                tag = 1
                self.record_tape(tag=tag)
                result = ad.colpack.sparse_jac_no_repeat(tag, self.Q, options)
                nnz = result[0]
                ridx = result[1]
                cidx = result[2]
                values = result[3]
                N = self.n*self.nvar
                self.dRdalpha = sp.csr_matrix((values, (ridx, cidx)), shape=(N, 1))
                self.complex_step(tag=tag)
                print np.linalg.norm(np.nan_to_num(self.dRdalpha.toarray()))
                print np.linalg.norm(self.dRdalpha_complex)
                
                self.Q  = self.Q + dQ

                self.complex_step(tag=0, stage=1)
                self.complex_step(tag=1, stage=1)

                
            elif integrator == "rk2":
                dt = self.calc_dt()*cfl

                self.calc_step()
                k1[:] = self.R[:]
                Qn[:] = self.Q[:]
                
                self.Q  = Qn + k1*dt/2.0
                self.calc_step()
                k2[:] = self.R[:]
                dQ = k2*dt
                self.Q = Qn + dQ
                
            elif integrator == "rk4":
                dt = self.calc_dt()*cfl

                self.calc_step()
                k1[:] = self.R[:]
                Qn[:] = self.Q[:]
                
                self.Q  = Qn + k1*dt/2.0
                self.calc_step()
                k2[:] = self.R[:]

                self.Q  = Qn + k2*dt/2.0
                self.calc_step()
                k3[:] = self.R[:]

                self.Q  = Qn + k3*dt
                self.calc_step()
                k4[:] = self.R[:]
                dQ = dt*(k1/6.0 + k2/3.0 + k3/3.0 + k4/6.0)
                self.Q = Qn + dQ
            self.temporal_hook_post()
            #print nnz
            #plt.spy(drdu)
            #plt.show()
            t += dt
            step += 1
            if step%print_step == 0 or 1: 
                self.logger.info("Time = %.2e/%.2e (%.01f%% complete)"%(t, tf, t/tf*100))
                #self.logger.info("Time = %.2e"%(np.linalg.norm(dQ)))
                if animation or file_io:
                    rho, u, p, Y = self.get_solution_primvars()
                if animation and self.mpi.rank == 0:
                    plt.clf()
                    plt.subplot(2,4,1)
                    plt.title("Density")
                    plt.plot(self.xc, rho, 'r-', lw=1, label="Density")
                    plt.xlim(self.xc.min(), self.xc.max())
                    
                    plt.subplot(2,4,2)
                    plt.title("Velocity")
                    plt.plot(self.xc, u, 'r-', lw=1, label="Velocity")
                    plt.xlim(self.xc.min(), self.xc.max())
                    
                    plt.subplot(2,4,3)
                    plt.title("Pressure")
                    plt.plot(self.xc, p, 'r-', lw=1, label="Pressure")
                    plt.xlim(self.xc.min(), self.xc.max())
                    
                    if self.nscalar > 0:
                        for i in range(self.nscalar):
                            plt.subplot(2,4,4+i)
                            label = self.scalar_map.keys()[self.scalar_map.values().index(i)]
                            plt.title(label)
                            plt.plot(self.xc, Y[:,i], '-', lw=1, label=self.scalar_map.keys()[self.scalar_map.values().index(i)])
                            plt.xlim(self.xc.min(), self.xc.max())

                    
                    #plt.legend(loc=1)
                    plt.show()
                    #plt.savefig("figures/%s.png"%name)
                    plt.pause(.00000001)
                if file_io:
                    name = str(step)
                    name = name.zfill(10)
                    np.savez("data_%s.npz"%name, rho=rho, u=u, p=p, Y=Y, x=self.xc, t=t, Q=self.Q)
                    np.savez("data_%s.npz"%name, rho=rho, u=u, p=p, Y=Y, x=self.xc, t=t, Q=self.Q, dRdQ=self.dRdQ, dRdalpha=self.dRdalpha, dRdQ_complex=self.dRdQ_complex, dt=dt, dRdalpha_complex=self.dRdalpha_complex, dRdalpha_complex_n = self.dRdalpha_complex_n, dRdQ_complex_n = self.dRdQ_complex_n)

            
            if t > tf or step > maxstep:
                self.logger.info("Time = %.2e/%.2e (%.01f%% complete)"%(t, tf, t/tf*100))
                if animation:
                    plt.ioff()
                break
            
if __name__ == "__main__":
    qleft = np.array([1.0, 0.0, 1.0])
    qright = np.array([0.125, 0.0, 0.1])
    eqn = EulerEquation(n=401, nscalar=0)
    eqn.initialize(qleft, qright)
    tf = 0.125
    eqn.solve(tf=tf, cfl=0.5, integrator="rk4", flux="hllc", print_step=100, order=2, animation=False)
    rho, rhou, rhoE, rhoY = eqn.get_solution()
    rho, u, p, Y = eqn.get_solution_primvars()

    plt.figure()
    plt.plot(eqn.xc, rho, 'r-', lw=1, label="Density")
    plt.plot(eqn.xc, u, 'g-', lw=1, label="Velocity")
    plt.plot(eqn.xc, p, 'b-', lw=1, label="Pressure")
    if eqn.nscalar > 0:
        plt.plot(eqn.xc, rhoY, 'cx-', lw=1, label="Y")

    eqn = EulerEquation(n=401, nscalar=0)
    eqn.initialize(qleft, qright)
    eqn.solve(tf=tf, cfl=0.5, integrator="rk4", flux="hllc", print_step=100, order=1, animation=False)
    rho, rhou, rhoE, rhoY = eqn.get_solution()
    rho, u, p, Y = eqn.get_solution_primvars()

    plt.plot(eqn.xc, rho, 'r--', lw=1, label="Density")
    plt.plot(eqn.xc, u, 'g--', lw=1, label="Velocity")
    plt.plot(eqn.xc, p, 'b--', lw=1, label="Pressure")
    if eqn.nscalar > 0:
        plt.plot(eqn.xc, rhoY, 'c--', lw=1, label="Y")

        
    # #plt.plot(eqn.xc, rhou, 'x-', lw=1)
    # #plt.plot(eqn.xc, rhoE, 'x-', lw=1)

    positions, regions, values = sod.solve(left_state=(1, 1, 0), right_state=(0.1, 0.125, 0.),
                                           geometry=(-0.5, 0.5, 0.0), t=tf, gamma=constants.gamma, npts=101)

    plt.plot(values['x'], values['rho'], 'r-', lw=2, label="Density")
    plt.plot(values['x'], values['u'], 'g-', lw=2, label="Velocity")
    plt.plot(values['x'], values['p'], 'b-', lw=2, label="Pressure")

    plt.show()
    
