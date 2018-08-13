import glob 
from pylab import *
import numpy as np
from rt_brandon import constants
from analytic import analytic_solution
def get_width(x, y):
    ymin = y[0]
    ymax = y[-1]
    lowfound = False
    highfound = False
    x = data["x"]
    dx = x[1] - x[0]
    for i in range(y.size):
        if y[i] > 0.01 and not lowfound:
            hs = x[i]
            lowfound = True
        if y[i] > 0.99 and not highfound:
            hb = x[i]
            highfound = True
    h = 0.5*(hb - hs)
    if abs(h) < 1e-10:
        h = dx
    xc = 0.5*(hb + hs)
    return h, 0.0, hs, hb


def integrate(x, y):
    n = x.size
    dx = x[1] - x[0]
    iy = 0.0
    for i in range(n):
        iy += y[i]*dx
    return iy

directory = ""
file_list = glob.glob("%sdata_0*[0-9].npz"%(directory))
file_list = sorted(file_list, key=lambda name: int(name[5:-4]))

At = 0.05
L_init = 4e-8

kmax = []
lmax = []
amax = []
t = []

alpha = []
alpha_b = []
alpha_s = []

tau = []
width = []
theta = []

for filename in file_list:
    data = np.load(filename)
    y = data["Y"][:,2]
    h, xc, hs, hb = get_width(data["x"], y)
    
    
    y = data["Y"][:,2]
    v = data["Y"][:,3]

    num = sum(v)
    den = sum(y*(1.0-y))
    theta.append(1.0 - num/den)
    
    width.append(h)
    alpha_ = h/At/abs(constants.g)/data["t"]**2
    alpha_b_ = hb/At/abs(constants.g)/data["t"]**2
    alpha_s_ = hs/At/abs(constants.g)/data["t"]**2
    fac_ = np.sqrt(At*abs(constants.g)/L_init)

    alpha.append(alpha_)
    alpha_b.append(alpha_b_)
    alpha_s.append(alpha_s_)

    tau.append(fac_*data["t"])

    k = data["Y"][:,0]
    kmax.append(k.max())

    l = data["Y"][:,1]
    lmax.append(l.max())
    
    t.append(data["t"]**2*abs(constants.g)*At*100)

figure()
# convert to appropriate units
plot(t, np.array(kmax)*1e-8/1e-9, '.-', label=r"$K_0$")
plot(t, np.array(lmax)*100, '.-', label=r"$L_0$")
plot(t, np.array(width)*100, '.-', label=r"$h_b$")
xlabel(r"$Agt^2$ (cm)", fontsize=24)
legend()
tight_layout()
savefig("amp_vs_time.pdf")

figure()
plot(tau, alpha, "r.-", label="h")
plot(tau, alpha_b, "g--", label="hb")
plot(tau, alpha_s, "b--", label="hs")
ylim(0.00,0.1)
ylabel(r'$\alpha_{b}$', fontsize=24)
xlabel(r'$\tau$', fontsize=24)
tight_layout()
savefig('alpha_vs_tau.pdf')


figure()
plot(tau, theta)
ylabel(r'$\Theta$', fontsize=24)
xlabel(r'$\tau$', fontsize=24)
ylim(0.3, 1.0)
tight_layout()
savefig("theta_tau.pdf")
show()



# plot last solution
filename = file_list[-1]
 
data = np.load(filename)
scalar_map = {"k":0, "L": 1, "Y_h": 2, "V_h":3}


y = data["Y"][:,2]
h, xc, hs, hb = get_width(data["x"], y)
rho = data["rho"]
k = data["Y"][:,0]
l = data["Y"][:,1]
mu = rho*l*np.sqrt(2.0*k)
for i in range(4):
    plt.figure()
    x = data["x"]
    if i != 3:
        y = data["Y"][:,i]/np.linalg.norm(data["Y"][:,i], inf)
    else:
        y = data["Y"][:,i]
    label = scalar_map.keys()[scalar_map.values().index(i)]
    t = data["t"]
    
    yp = 1 - ((x-xc)/h)**2
    yps = np.sqrt(yp)
    yl = 1.0 - 0.5*(1 - ((x-xc)/h))
    if 0:
        plot(data["x"]/h, -y, "g-", label=label)
    else:
        plot(data["x"]/h, y, "g-", label=label)
    if i == 0:
        plot(data["x"]/h, yp, "r--", label='quadratic')
    if i == 1:
        plot(data["x"]/h, yps, "r--", label='quadratic sqrt')
    if i == 2:
        plot(data["x"]/h, yl, "r--", label='linear')
    if i == 3:
        plot(data["x"]/h, 0.05*yp, "r--", label='linear')
    
    ylim(-0.01, 1.01)
    if i == 3:
        ylim(-0.01, 0.06)
    title(label)
    
    ylabel(r'$%s^{\star}$'%label, fontsize=24)
    xlabel(r'$x/h$', fontsize=24)
    tight_layout()
    savefig('%s.pdf'%label)

mu = mu/rho
plt.figure()
plot(data["x"]/h, mu/np.linalg.norm(mu, inf), "g-", label="rho")
plot(data["x"]/h, yp, "r--", label='linear')
ylim(-0.01, 1.01)
title('mu')
ylabel(r'$\mu^{\star}$', fontsize=24)
xlabel(r'$x/h$', fontsize=24)   
tight_layout()
savefig('mu.pdf')

show()
