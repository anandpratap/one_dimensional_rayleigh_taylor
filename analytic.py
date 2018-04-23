from pylab import *

def analytic_solution(x, u, t=0.0, m=10, mu=1e-3, xshift=0.5):
    ua = np.zeros_like(u)
    L = x.max() - x.min()
    dx = x[1] - x[0]
    for i in range(m):
        integral = 0.0
        for j in range(x.size):
            integral += np.sin(i*np.pi*(x[j]+xshift)/L)*u[j]*dx
        ua += 2.0/L*integral*sin(i*np.pi*(x+xshift)/L) * np.exp(-mu*i*i*np.pi*np.pi/L/L*t)
    return ua

if __name__ == "__main__":
    x = np.linspace(-.5, .5, 101)
    u = np.zeros_like(x)
    u[48:53] = 1.0
    ua = analytic_solution(x, u, t=1.0, m=100)
    print ua
    
    
    figure()
    plot(x, u, 'r.-')
    plot(x, ua, 'g.-')
    show()
