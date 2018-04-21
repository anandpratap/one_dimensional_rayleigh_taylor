from rt import *
from calc_adjoint import Adjoint
from scipy.optimize import minimize
def func_and_grad(alpha):
    eqn = RT(n=101, nscalar=4)
    eqn.alpha = alpha
    eqn.initialize()
    run_dir = eqn.solve(tf=1.5, cfl = 0.5, animation=False, print_step=1000, integrator="fe", flux="hllc", order=1, file_io=True, maxstep=10, jacobian_mode="adolc", tmp_dir=True)
    adj = Adjoint(nstart=1, nend=10)
    J, grad = adj.solve(run_dir=run_dir)
    return J, grad

alpha_init = np.ones(4)
res = minimize(func_and_grad, alpha_init, jac=True, method="BFGS", options={'disp':True, 'maxiter': 100})
print res
