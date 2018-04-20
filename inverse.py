from rt import *
from calc_adjoint import Adjoint

alpha = np.array([1.0])
for i in range(100):
    eqn = RT(n=101, nscalar=4)
    eqn.alpha = alpha
    eqn.initialize()
    eqn.solve(tf=1.5, cfl = 0.5, animation=False, print_step=1000, integrator="fe", flux="hllc", order=1, file_io=True, maxstep=10)

    adj = Adjoint(nstart=1, nend=10)
    grad = adj.solve()
    alpha = alpha - (grad/abs(grad))*0.05
    print 10*"#"
    print "Alpha = ", alpha[0]
    print 10*"#"
