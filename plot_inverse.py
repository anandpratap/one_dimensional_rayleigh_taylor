from pylab import *

eq = ['k', 'L', 'a_x', 'Y']

d = loadtxt("inverse.log")

figure()
semilogy(d[:,0], 'r.-')
xlabel(r'Iteration No.', fontsize=24)
ylabel(r'Objective function', fontsize=24)
tight_layout()
savefig("inverse_J.pdf")

figure()
semilogy(d[:,1], 'r.-')
xlabel(r'Iteration No.', fontsize=24)
ylabel(r'Step size', fontsize=24)
tight_layout()
savefig("inverse_step_size.pdf")

nc = d.shape[1]
d = np.concatenate((np.ones([1,nc]), d), axis=0)

figure()
for i, k in enumerate(eq):
    plot(d[:,2+i], '.-', label=r'$\alpha_{%s}$'%k)
xlabel(r'Iteration No.', fontsize=24)
ylabel(r'$\alpha$', fontsize=24)
legend(loc="best", fontsize=20)
tight_layout()
savefig('inverse_alpha.pdf')

figure()
for i, k in enumerate(eq):
    semilogy(d[:,2+i], '.-', label=r'$\alpha_{%s}$'%k)
xlabel(r'Iteration No.', fontsize=24)
ylabel(r'$\alpha$', fontsize=24)
legend(loc="best", fontsize=20)
tight_layout()
savefig('inverse_alpha_log.pdf')

show()
