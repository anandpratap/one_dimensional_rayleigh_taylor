from main import *

class RT(EulerEquation):
    def initialize_sod(self, qleft, qright):
        Q = self.Q.reshape([self.n, self.nvar])
        Q[:self.n/2, 0], Q[:self.n/2, 1], Q[:self.n/2, 2] = self.calc_E(1.0, 0.0, 1.0);
        Q[self.n/2:, 0], Q[self.n/2:, 1], Q[self.n/2:, 2] = self.calc_E(1.0, 0.0, 1.0);

        for scalar in range(self.nscalar):
            idx = self.get_scalar_index(scalar)
            Q[:self.n/2, idx] = 1.0 + idx*0.05
            Q[self.n/2:, idx] = 0.0 #+ idx
        
        Q = Q.reshape(self.n*self.nvar)
        # check if any copy happened
        assert(same_address(Q, self.Q))

    
    def temporal_hook(self):
        pass
            
    def calc_source(self):
        pass

    def calc_viscous_flux(self):
        pass

if __name__ == "__main__":
    qleft = np.array([1.0, 0.0, 1.0])
    qright = np.array([0.125, 0.0, 0.1])
    eqn = RT(n=501, nscalar=10)
    eqn.initialize_sod(qleft, qright)
    eqn.solve(tf=0.2)
    rho, rhou, rhoE, rhoY = eqn.get_solution()
    rho, u, p, Y = eqn.get_solution_primvars()

    plt.figure()
    plt.plot(eqn.xc, rho, 'r-', lw=1, label="Density")
    plt.plot(eqn.xc, u, 'g-', lw=1, label="Velocity")
    plt.plot(eqn.xc, p, 'b-', lw=1, label="Pressure")
    if eqn.nscalar > 0:
        plt.plot(eqn.xc, rhoY, 'cx-', lw=1, label="Y")

    # #plt.plot(eqn.xc, rhou, 'x-', lw=1)
    # #plt.plot(eqn.xc, rhoE, 'x-', lw=1)

    positions, regions, values = sod.solve(left_state=(1, 1, 0), right_state=(0.1, 0.125, 0.),
                                           geometry=(-0.5, 0.5, 0.0), t=0.2, gamma=1.4, npts=101)

    plt.plot(values['x'], values['rho'], 'r--', lw=1, label="Density")
    plt.plot(values['x'], values['u'], 'g--', lw=1, label="Velocity")
    plt.plot(values['x'], values['p'], 'b--', lw=1, label="Pressure")

    plt.show()
    
