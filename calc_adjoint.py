import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from pylab import *
class Adjoint(object):
    def __init__(self, nstart=0, nend=0):
        self.nstart = nstart
        self.nend = nend
        self.J = 0.0

    def objective_function(self, Q, Q_data):
        Q = np.reshape(Q, [Q.size/7, 7])
        Q_data = np.reshape(Q_data, [Q_data.size/7, 7])
        kfac = 1.0
        k = Q[:,3]/Q[:,0]
        k_data = Q_data[:,3]/Q_data[:,0]

        Lfac = 1e12
        L = Q[:,4]/Q[:,0]
        L_data = Q_data[:,4]/Q_data[:,0]

        afac = 1e6
        a = Q[:,5]/Q[:,0]
        a_data = Q_data[:,5]/Q_data[:,0]

        Yfac = 1e-2
        Y = Q[:,6]/Q[:,0]
        Y_data = Q_data[:,6]/Q_data[:,0]

        J = sum(kfac*(k - k_data)**2 + Lfac*(L - L_data)**2 + afac*(a - a_data)**2 + Yfac*(Y - Y_data)**2)
        return J

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
        
    def load_solution(self, step):
        filename = str(step).zfill(10)
        filename = "data_%s.npz"%(filename)
        data = np.load(filename)
        data_inverse = np.load('data_for_inverse/'+filename)
        J = self.objective_function(data['Q'], data_inverse['Q'])
        dJdQ = self.calc_dJdQ(data['Q'], data_inverse['Q'])
        rho = data['rho']
        dt = 1e-5
        Q = data['Q']
        k = data['Y'][:,0]
        k_inverse = data_inverse['Y'][:,0]
        dRdQ = data['dRdQ_complex']
        dRdQ = np.nan_to_num(dRdQ)
        dRdalpha = -data['dRdalpha_complex_n']
        n = dRdQ.shape[0]
        assert n == rho.size*7
        Ibydt = np.diag(np.ones_like(Q)/dt)

        dRdQ = -Ibydt - dRdQ

        #dJdQ = np.zeros_like(Q)
        #dJdQ[3::7] = 2.0*(k[:] - k_inverse)
        dRdQp = Ibydt
        return dJdQ, dRdQ, dRdQp, dRdalpha, J#((k-k_inverse)**2).sum()
        
    def solve(self):
        #plt.figure()
        J_sum = 0.0
        grad_ = []
        for i in reversed(range(self.nstart, self.nend+1)):
            print i
            if i == self.nend:
                dJn, dRdQ_nm, dRdQ_n, dRdalpha, Jn = self.load_solution(i)
                psi = np.linalg.solve(-dRdQ_n.transpose(), dJn)
                grad = np.zeros(4)
                grad += np.dot(psi.T, dRdalpha)
            else:
                dJn, dRdQ_nm, dRdQ_n, dRdalpha, Jn = self.load_solution(i)
                dJn_p1, dRdQ_nm_p1, dRdQ_n_p1, dRdalpha_p1, Jn_p1 = self.load_solution(i+1)
                rhs = dJn + np.dot(dRdQ_nm_p1.transpose(), psi)
                psi = np.linalg.solve(-dRdQ_n.transpose(), rhs)
                #psi_np[:] = psi[:]
                #print dRdalpha.max(), dRdalpha.min()
                tmp = np.dot(psi.T, dRdalpha)
                grad += np.dot(psi.T, dRdalpha)
                r = np.dot(dRdQ_n.transpose(), psi) + rhs
                assert np.linalg.norm(r) < 1e-10
            J_sum += Jn
            #plt.plot(psi[3::7], label='%i'%i)
        for i in range(4):
            print "gradient = %.16f"%grad[i]
        #plt.legend()
        print "objective = %.16f"%J_sum
        #plt.show()
        return grad
            
if __name__ == "__main__":
    adj = Adjoint(nstart=1, nend=10)
    grad = adj.solve()
    
