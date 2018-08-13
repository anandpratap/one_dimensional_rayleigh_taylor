import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import os
from pylab import *
from rt_brandon import RT
import time
import adolc as ad
import pickle
import sys
sys.path.append("./les_data")
from load_data import load_data
from scipy import interpolate
def csr_buffer_to_matrix(csrbuffer, full=True):
    indices = csrbuffer.indices[:csrbuffer.n[0]]
    indptr= csrbuffer.indptr[:csrbuffer.n[1]]
    data= csrbuffer.data[:csrbuffer.nnz]
    if full:
        return sp.csr_matrix((data, indices, indptr), shape=(2800, 2800))
    else:
        return sp.csr_matrix((data, indices, indptr), shape=(2800, 4))
    
def load_solution(filename):
    f = open(filename, "rb")
    counter = pickle.load(f)
    t_buffer = pickle.load(f)
    Q_buffer = pickle.load(f)
    iteration_buffer = pickle.load(f)
    return iteration_buffer, Q_buffer

filename = "les_data/RT_G0002_S001_A05_BB04-128..ult"
data = load_data(filename)
#figure()
#plot(data[6,:,0], data[6,:,1])
data[6,0,0] = -data[6,0,0]
func_theta = interpolate.interp1d(data[6,:,0], data[6,:,1], bounds_error=False, fill_value=0.0)

def objective_function(Q, Q_data, step):
    Q = np.reshape(Q, [Q.size/7, 7])
    rho = Q[:,0]
    rhoY = Q[:,3:]
    y = rhoY[:,2]/rho
    v = rhoY[:,3]/rho
    num = sum(v)
    den = sum(y*(1.0-y))
    theta = 1.0 - num/den
    
    Q_data = np.reshape(Q_data, [Q_data.size/7, 7])
    rho = Q_data[:,0]
    rhoY = Q_data[:,3:]
    y = rhoY[:,2]/rho
    v = rhoY[:,3]/rho
    num = sum(v)
    den = sum(y*(1.0-y))
    theta_data = 1.0 - num/den
    fac = 1e1
    #print theta, theta_data
    time = step*1e-8*1e5
    theta_data = func_theta(time)
    #if time > 10.0 and time < 30.0:
    if time < 30.0:
        J = fac*(theta - theta_data)**2
    else:
        J = 0.0
    return J


def calc_objective_total(iteration_buffer, Q_buffer, Q_data_buffer, start=1, end=1):
    J = 0
    for step in range(start, end+1):
        idx = np.where(iteration_buffer == step)[0][0]
        Q = Q_buffer[idx,:]
        Q_data = Q_data_buffer[idx,:]
        J += objective_function(Q, Q_data, step)
    return J

class Adjoint(object):
    def __init__(self, nstart=0, nend=0):
        self.nstart = nstart
        self.nend = nend
        self.J = 0.0
        self.record_tape = True
        self.eqn = RT(n=401, nscalar=4)
        self.eqn.initialize()


    def record_tape_J(self, Q, Q_data):
        tag = 11
        Q = ad.adouble(Q)
        ad.trace_on(tag)
        ad.independent(Q)
        J = self.objective_function(Q, Q_data)
        ad.dependent(J)
        ad.trace_off()

    
    def calc_dJdQ(self, Q, Q_data):
        Qc = Q.astype(complex)
        dJdQ = np.zeros_like(Q)
        dQ = 1e-14
        for i in range(Q.size):
            Qc[i] = Qc[i] + 1j*dQ
            J = self.objective_function(Qc, Q_data)
            dJdQ[i] = np.imag(J)/dQ
            Qc[i] = Qc[i] - 1j*dQ
        return dJdQ

    def load_solution_low(self, step, psi=None):
        idx = np.where(iteration_buffer == step)[0][0]
        Q = Q_buffer[idx,:]
        Q_data = Q_data_buffer[idx,:]
        J = self.objective_function(Q, Q_data)
        if self.record_tape:
            self.record_tape_J(Q, Q_data)
            
        dJdQ = ad.gradient(11, Q)
        #dJdQ = self.calc_dJdQ(Q, Q_data)
        dt = 1e-8
        eqn = self.eqn
        eqn.Q[:] = Q[:]
        #record_tape = True
        start = time.time()
        data_drdq_nm, data_drdalpha_n = eqn.solve(tf=8.4e-4, cfl = 0.5, animation=False, print_step=1, integrator="fe", flux="hllc", order=1, file_io=True, maxstep=50000, jacobian_mode="adolc", do_not_update=True, record_tape=self.record_tape, allocate_vars=True, psi=psi)
        #print data_drdq_nm.shape#, psi
        #if step<49990:
         #   self.record_tape = True
        #print ad.tapestats(0)
        #print data_drdq_nm.nnz, data_drdalpha_n.nnz
        print "Jac calculation time ", time.time()-start
        #drdq =
        dRdQ = data_drdq_nm
        dRdQ.data = np.nan_to_num(dRdQ.data)
        #except:
         #   pass
        dRdalpha = -data_drdalpha_n
        dRdalpha.data = np.nan_to_num(dRdalpha.data)
        #n = dRdQ.shape[0]
        #assert n == rho.size*7
        Ibydt = sp.eye(Q.size)/dt

        #Ibydt = np.diag(np.ones_like(Q)/dt)
        dRdQ = -Ibydt - dRdQ
        #dJdQ = np.zeros_like(Q)
        #dJdQ[3::7] = 2.0*(k[:] - k_inverse)
        dRdQp = Ibydt
        return dJdQ, dRdQ, dRdQp, dRdalpha, J#((k-k_inverse)**2).sum()
    
    def load_solution_buffer(self, step):
        idx = np.where(iteration_buffer == step)[0][0]
        Q = Q_buffer[idx,:]
        Q_data = Q_data_buffer[idx,:]
        J = self.objective_function(Q, Q_data)
        dJdQ = self.calc_dJdQ(Q, Q_data)
        dt = 1e-8
        
        dRdQ = csr_buffer_to_matrix(dRdQ_nm_buffer[idx])#.toarray()
        dRdQ.data = np.nan_to_num(dRdQ.data)
        dRdalpha = csr_buffer_to_matrix(dRdalpha_n_buffer[idx], False)#.toarray()
        dRdalpha.data = -np.nan_to_num(dRdalpha.data)

        Ibydt = sp.eye(Q.size)/dt
        dRdQ = -Ibydt - dRdQ
        dRdQp = Ibydt
        return dJdQ, dRdQ, dRdQp, dRdalpha, J#((k-k_inverse)**2).sum()

    
    def solve(self, run_dir=None):
        if run_dir is not None:
            self.cwd = os.getcwd()
            os.chdir(run_dir)
        #plt.figure()
        J_sum = 0.0
        grad_ = []
        for i in reversed(range(self.nstart, self.nend+1)):
            start = time.time()
            print i
            if i == self.nend:
                dJn, dRdQ_nm, dRdQ_n, dRdalpha, Jn = self.load_solution_low(i)
                load_time = time.time() - start

                #print dRdQ_n.transpose()
                psi = spla.spsolve(-dRdQ_n.transpose(), dJn)
                grad = np.zeros(4)
                #print psi.T.shape
                #print dRdalpha.shape
                dgrad = psi.transpose() *  dRdalpha
                #print dgrad.shape
                #print np.isnan(dRdalpha.data).any()
                #print np.isnan(psi.data).any()
                assert np.isnan(dRdalpha.data).any() == False
                assert np.isnan(psi.data).any() == False
                print grad, dgrad
                grad = grad + dgrad
            else:
                dJn_p1, dRdQ_nm_p1, dRdQ_n_p1, dRdalpha_p1, Jn_p1 = dJn, dRdQ_nm, dRdQ_n, dRdalpha, Jn
                dJn, dRdQ_nm, dRdQ_n, dRdalpha, Jn = self.load_solution_low(i, psi=None)
                #dJn_p1, dRdQ_nm_p1, dRdQ_n_p1, dRdalpha_p1, Jn_p1 = dJn, dRdQ_nm, dRdQ_n, dRdalpha, Jn
                #self.load_solution_buffer(i+1)
                load_time = time.time() - start
                #if i == self.nend - 1:
                rhs = dJn + dRdQ_nm_p1.transpose() * psi
                #else:
                #   rhs = dJn + dRdQ_nm_p1.transpose()
                #  print 'rhs', rhs.shape, dJn.shape
                #psi = spla.spsolve(-dRdQ_n.transpose(), rhs)
                #print dRdQ_n
                dt = 1e-8
                psi = -rhs*dt
                #psi = spla.spsolve(-dRdQ_n.transpose(), rhs)
                #psi_np[:] = psi[:]
                #print dRdalpha.max(), dRdalpha.min()
                #tmp = np.dot(psi.T, dRdalpha)
                dgrad = psi.transpose() *  dRdalpha
                #print rhs.max(), rhs.min()
                #print dRdalpha.max(), dRdalpha.min()

                #print np.isnan(dRdalpha.data).any()
                #print np.isnan(psi.data).any()
                #print grad, dgrad
                assert np.isnan(dRdalpha.data).any() == False
                assert np.isnan(psi.data).any() == False
                assert np.isnan(rhs.data).any() == False
                #assert np.isnan(dgrad).any() == False

                grad = grad +  dgrad
                #r =dRdQ_n.transpose() * psi + rhs
                #assert np.linalg.norm(r) < 1e-10
            J_sum += Jn
            total_time = time.time() - start
            print "Load time % ", load_time/total_time*100.0, "Total time: ", total_time
            
            #plt.plot(psi[3::7], label='%i'%i)
        for i in range(4):
            print "gradient = %.16f"%grad[i]
        #plt.legend()
        print "objective = %.16f"%J_sum
        #plt.show()
        if run_dir is not None:
            os.chdir(self.cwd)
        return J_sum, grad


    
def solve_fd(alpha):
    nend = 60000
    tmp, Q_data_buffer = load_solution("data/final")
    eqn = RT(n=401, nscalar=4)
    eqn.initialize()
    assert eqn.alpha.size == alpha.size
    eqn.alpha[:] = alpha[:]
    eqn.solve(tf=8.4e-4, cfl = 0.5, animation=False, print_step=1, integrator="fe", flux="hllc", order=1, file_io=True, maxstep=nend, jacobian_mode=None)
    iteration_buffer, Q_buffer = eqn.buffer.iteration, eqn.buffer.Q_buffer  #load_solution("final")
    #iteration_buffer, Q_buffer = load_solution("final")
    J = calc_objective_total(iteration_buffer, Q_buffer, Q_data_buffer, start=1, end=nend+1)
    #print J
    #sys.exit(0)
    
    grad = np.zeros(alpha.size)
    for i in range(alpha.size):
        print "fd: var: %i/%i"%(i, alpha.size)
        eqn = RT(n=401, nscalar=4)
        eqn.initialize()
        eqn.alpha[:] = alpha[:]
        dalpha = max(np.abs(eqn.alpha[i]*1e-6), 1e-8)
        eqn.alpha[i] += dalpha
        eqn.solve(tf=8.4e-4, cfl = 0.5, animation=False, print_step=1, integrator="fe", flux="hllc", order=1, file_io=True, maxstep=nend, jacobian_mode=None)
        iteration_buffer, Q_buffer = eqn.buffer.iteration, eqn.buffer.Q_buffer  #load_solution("final")
        Jp = calc_objective_total(iteration_buffer, Q_buffer, Q_data_buffer, start=1, end=nend+1)
        grad[i] = (Jp - J)/dalpha
    return J, grad
if __name__ == "__main__":
    import time
    alpha = np.ones(11)
    #alpha = np.array([ 3.74185502, -1.62855169,  2.41004766,  1.02757932,  0.98879082])
    step_size = 0.1
    for i in range(100):
        start = time.time()
        J, grad = solve_fd(alpha)
        alpha = alpha - grad/np.max(np.abs(grad))*step_size
        print "Time for 1 iteration: ", time.time() - start
        print i, J, grad, alpha
