import adolc as ad
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from enum import Enum
import sod
def same_address(x, y):
    return x.__array_interface__['data'] == y.__array_interface__['data']
class Constants(object):
    gamma = 1.4

class EulerEquation(object):
    def __init__(self, n=11):
        self.n = n - 1
        self.nvar = 3
        self.x = np.linspace(-0.5, 0.5, self.n + 1)
        self.dx = self.x[1] - self.x[0]
        self.xc = 0.5*(self.x[0:-1] + self.x[1:])
        self.Q = np.zeros(self.n*self.nvar)

    def calc_dt(self):
        rho, u, p = self.get_solution_primvars()
        a = np.sqrt(Constants.gamma*p/rho)
        lambda_max = np.max(a + np.abs(u))
        return self.dx/lambda_max
        
    def initialize_sod(self, qleft, qright):
        Q = self.Q.reshape([self.n, self.nvar])
        Q[:self.n/2, 0] = qleft[0]
        Q[:self.n/2, 1] = qleft[1]*qleft[0]
        Q[:self.n/2, 2] = qleft[2]/(Constants.gamma-1.0) + 0.5*qleft[0]*qleft[1]**2;

        Q[self.n/2:, 0] = qright[0]
        Q[self.n/2:, 1] = qright[1]*qright[0]
        Q[self.n/2:, 2] = qright[2]/(Constants.gamma-1.0) + 0.5*qright[0]*qright[1]**2;
        Q = Q.reshape(self.n*self.nvar)
        # check if any copy happened
        assert(same_address(Q, self.Q))

    def get_solution(self):
        Q = self.Q.reshape([self.n, self.nvar])
        return Q[:, 0], Q[:, 1], Q[:, 2]

    def get_solution_primvars(self):
        Q = self.Q.reshape([self.n, self.nvar])
        rho = Q[:, 0]
        u = Q[:, 1]/Q[:, 0]
        p = (Constants.gamma-1.0)*(Q[:,2] - 0.5*(rho*u*u))
        return rho, u, p

    def set_bc(self):
        U = self.U.reshape([self.n+2, self.nvar])
        Q = self.Q.reshape([self.n, self.nvar])
        
        rho = U[:, 0]
        u = U[:, 1]
        p = U[:, 2]

        rho[1:-1] = Q[:, 0]
        u[1:-1] = Q[:, 1]/Q[:, 0]
        p[1:-1] = (Constants.gamma-1.0)*(Q[:,2] - 0.5*(rho[1:-1]*u[1:-1]*u[1:-1]))

        rho[0] = rho[1]
        u[0] = u[1]
        p[0] = p[1]

        rho[-1] = rho[-2]
        u[-1] = u[-2]
        p[-1] = p[-2]

        U = U.reshape((self.n+2)*self.nvar)
        assert(same_address(U, self.U))

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
        
        
    def calc_flux(self):
        F = self.F.reshape([self.n+1, self.nvar])

        Ul = self.Ul.reshape([self.n+1, self.nvar])
        rhol = Ul[:, 0]
        ul = Ul[:, 1]
        pl = Ul[:, 2]

        Ur = self.Ur.reshape([self.n+1, self.nvar])
        rhor = Ur[:, 0]
        ur = Ur[:, 1]
        pr = Ur[:, 2]

        GAMMA = Constants.gamma
        GAMMA_M = Constants.gamma - 1.0
        
        el = pl/(GAMMA_M) + 0.5*rhol*ul*ul;
        er = pr/(GAMMA_M) + 0.5*rhor*ur*ur;
        hl = (el + pl)/rhol;
        hr = (er + pr)/rhor;

        sqrtrhol = np.sqrt(rhol);
        sqrtrhor = np.sqrt(rhor);
        den_inverse = 1/(sqrtrhol + sqrtrhor);
        uavg = (sqrtrhol*ul + sqrtrhor*ur)*den_inverse;
        havg = (sqrtrhol*hl + sqrtrhor*hr)*den_inverse;
        cavg = np.sqrt(GAMMA_M*(havg - 0.5*uavg*uavg));
        cavg_inverse = 1.0/cavg;
        
        d1 = rhor - rhol;
        d2 = rhor*ur - rhol*ul;
        d3 = er - el;
        
        alpha_2 = GAMMA_M*((havg - uavg*uavg)*d1 + uavg*d2 - d3)*cavg_inverse*cavg_inverse;
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

        F = F.reshape((self.n+1)*self.nvar)
        Ul = Ul.reshape((self.n+1)*self.nvar)
        Ur = Ur.reshape((self.n+1)*self.nvar)
        assert(same_address(Ul, self.Ul))
        assert(same_address(Ur, self.Ur))
        assert(same_address(F, self.F))

    def calc_residual(self):
        self.set_bc()
        self.reconstruct()
        self.calc_flux()
        R = self.R.reshape([self.n, self.nvar])
        F = self.F.reshape([self.n+1, self.nvar])
        R[:,:] = - (F[1:,:] - F[0:-1,:])/self.dx
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
        self.U = np.zeros((self.n + 2)*self.nvar)
        self.F = np.zeros((self.n + 1)*self.nvar)
        self.Ul = np.zeros((self.n+1)*self.nvar)
        self.Ur = np.zeros((self.n+1)*self.nvar)
        self.record_tape()
        t = 0.0
        while 1:
            tag = 0
            options = np.array([0,0,0,0],dtype=int)
            #self.record_tape()
            self.calc_step()
            R = self.R.copy()
            result = ad.colpack.sparse_jac_no_repeat(tag, self.Q, options)
            nnz = result[0]
            ridx = result[1]
            cidx = result[2]
            values = result[3]
            N = self.n*self.nvar

            dt = self.calc_dt()*1.4
            print dt
            drdu = -sp.csr_matrix((values, (ridx, cidx)), shape=(N, N)) + sp.eye(N)/dt
            du = spla.spsolve(drdu, R)
            #print np.linalg.norm(R - self.R)
            self.Q  = self.Q + du
            #self.Q  = self.Q + R*dt
            print nnz
            #plt.spy(drdu)
            #plt.show()
            t += dt
            if t > tf:
                break
if __name__ == "__main__":
    qleft = np.array([1.0, 0.0, 1.0])
    qright = np.array([0.125, 0.0, 0.1])
    eqn = EulerEquation(n=101)
    eqn.initialize_sod(qleft, qright)
    eqn.solve(t=0.2, integration="explicit_euler")
    rho, rhou, rhoE = eqn.get_solution()
    rho, u, p = eqn.get_solution_primvars()

    plt.figure()
    plt.plot(eqn.xc, rho, 'r-', lw=1, label="Density")
    plt.plot(eqn.xc, u, 'g-', lw=1, label="Velocity")
    plt.plot(eqn.xc, p, 'b-', lw=1, label="Pressure")

    #plt.plot(eqn.xc, rhou, 'x-', lw=1)
    #plt.plot(eqn.xc, rhoE, 'x-', lw=1)

    positions, regions, values = sod.solve(left_state=(1, 1, 0), right_state=(0.1, 0.125, 0.),
                                           geometry=(-0.5, 0.5, 0.0), t=0.2, gamma=1.4, npts=101)

    plt.plot(values['x'], values['rho'], 'r--', lw=1, label="Density")
    plt.plot(values['x'], values['u'], 'g--', lw=1, label="Velocity")
    plt.plot(values['x'], values['p'], 'b--', lw=1, label="Pressure")

    plt.show()
    
