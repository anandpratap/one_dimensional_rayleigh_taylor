from rt import *
from calc_adjoint import Adjoint
from multiprocessing import Pool
factors = np.array([1.0/4.0, 1.0/2.0, 1.0, 2.0, 4.0])

alpha = np.array([1.0, 1.0, 1.0, 1.0])
f = open("inverse.log", "w")
f.write("")
f.close()
step_size = 0.1
for i in range(100):
    eqn = RT(n=401, nscalar=4)
    eqn.alpha = alpha
    eqn.initialize()
    run_dir = eqn.solve(tf=8.4e-4, cfl = 0.5, animation=False, print_step=1, integrator="fe", flux="hllc", order=1, file_io=True, maxstep=10, jacobian_mode="adolc", tmp_dir=True)
    adj = Adjoint(nstart=1, nend=11)
    J, grad = adj.solve(run_dir=run_dir)
    
    # grad_current = grad.copy()
    # alpha_current = alpha.copy()
    # J_search = np.zeros_like(factors)
    
    # for idx, fac in enumerate(factors):
    #     step_size_search = fac*step_size
    #     alpha = alpha_current - (grad_current/abs(grad_current).max())*step_size_search
    #     eqn = RT(n=401, nscalar=4)
    #     eqn.alpha = alpha
    #     eqn.initialize()
    #     run_dir = eqn.solve(tf=8.4e-4, cfl = 0.5, animation=False, print_step=1, integrator="fe", flux="hllc", order=1, file_io=True, maxstep=10, jacobian_mode="adolc", tmp_dir=True)
    #     adj = Adjoint(nstart=1, nend=11)
    #     J_search[idx], grad_search = adj.solve(run_dir=run_dir)

    # idx_min = np.argmin(J_search)
    # step_size = factors[idx_min]*step_size

    # f = open("inverse.log", "a")
    # f.write("%.16f\t"%step_size)
    # f.write("%.16f\t"%J)
    # for idx, a in enumerate(alpha):
    #     if idx == alpha.size - 1:
    #         f.write("%.16f\n"%a)
    #     else:
    #         f.write("%.16f\t"%a)
    # f.close()

    step_size = 0.1
    alpha = alpha - (grad/abs(grad).max())*step_size

    #alpha = alpha_current - (grad/abs(grad).max())*step_size
    print 10*"#"
    print "Alpha = ", alpha
    print 10*"#"
