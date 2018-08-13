import sys
import argparse
sys.path.append("/gpfs/gpfs0/groups/duraisamy/anandps/local/scipy-0.19.0/build/lib.linux-ppc64le-2.7")
sys.path.insert(0, "/gpfs/gpfs0/groups/duraisamy/anandps/local/scipy-0.19.0/build/lib.linux-ppc64le-2.7")
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import os
from pylab import *
from rt_brandon import RT, constants
import time
import pickle
sys.path.append("./les_data")
sys.path.append("../../les_data")
from load_data import load_data
from scipy import interpolate
from mpi4py import MPI
import time

fac = 100.0
kr_dict = {}
kr_dict[(2,1)] = 20.0*fac
kr_dict[(2,2)] = 25.0*fac
kr_dict[(2,4)] = 33.0*fac
kr_dict[(2,10)] = 37.0*fac
kr_dict[(2,100)] = 37.0*fac

kr_dict[(13,1)] = 37.0*fac
kr_dict[(13,2)] = 45.0*fac
kr_dict[(13,4)] = 58.0*fac
kr_dict[(13,10)] = 65.0*fac
kr_dict[(13,100)] = 65.0*fac

kr_dict[(81,1)] = 65.0*fac
kr_dict[(81,2)] = 79.0*fac
kr_dict[(81,4)] = 99.0*fac
kr_dict[(81,10)] = 109.0*fac
kr_dict[(81,100)] = 110.0*fac

kr_dict[(512,1)] = 111.0*fac
kr_dict[(512,2)] = 133.0*fac
kr_dict[(512,4)] = 163.0*fac
kr_dict[(512,10)] = 178.0*fac
kr_dict[(512,100)] = 180.0*fac


def load_solution(filename):
    f = open(filename, "rb")
    counter = pickle.load(f)
    t_buffer = pickle.load(f)
    Q_buffer = pickle.load(f)
    iteration_buffer = pickle.load(f)
    return iteration_buffer, Q_buffer


def objective_function(Q, step):
    Q = np.reshape(Q, [Q.size/7, 7])
    rho = Q[:,0]
    rhoY = Q[:,3:]
    y = rhoY[:,2]/rho
    v = rhoY[:,3]/rho
    num = sum(v)
    den = sum(y*(1.0-y))
    theta = 1.0 - num/den
    
    fac = 1e1
    #print theta, theta_data
    time = step*1e-8/t0_RANS
    
    #if time > 10.0 and time < 30.0:
    if time < t_MAX_LES:
        theta_data = func_theta(time)
        J = fac*(theta - theta_data)**2
    else:
        J = 0.0
    return J


def calc_objective_total(iteration_buffer, Q_buffer, alpha, start=1, end=1):
    J = 0
    lam = 1.0
    for step in range(start, end+1):
        idx = np.where(iteration_buffer == step)[0][0]
        Q = Q_buffer[idx,:]
        J += objective_function(Q, step)
    J += sum((alpha - 1.0)**2) * lam
    return J

def run_solve(idx, alpha, dalpha):
    nend = 40000
    eqn = RT(n=401, nscalar=4)
    eqn.initialize()
    assert eqn.alpha.size == alpha.size
    if idx == -1:
        eqn.alpha[:] = alpha[:]
        main_run = True
    else:
        eqn.alpha[:] = alpha[:]
        eqn.alpha[idx] += dalpha
        main_run = False

    eqn.solve(tf=8.4e-4, cfl = 0.5, animation=False, print_step=1, integrator="fe", flux="hllc", order=1, file_io=True, maxstep=nend, jacobian_mode=None,main_run=main_run)
    iteration_buffer, Q_buffer = eqn.buffer.iteration, eqn.buffer.Q_buffer
    J = calc_objective_total(iteration_buffer, Q_buffer, alpha, start=1, end=nend+1)
    return J

    return J, grad

def get_configs(filename):
    filename = filename.split("/")[-1]
    parts = filename.split("_")
    Gr = int(parts[1][1:])
    Sc = int(parts[2][1:])
    At = int(parts[3][1:])/100.0
    return Gr, Sc, At


def get_t0(At, g, kR):
    l_0 = 2.0*np.pi/kR
    t0 = np.sqrt(l_0/(-At*g))
    return t0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('filename', type=str, default="")
    parser.add_argument('--restart', action="store_true")
    args = parser.parse_args()
    filename = args.filename

    Gr, Sc, At = get_configs(filename)

    kr_rans =  kr_dict[(Gr, Sc)] * (-1e10) / 100.0 / constants.g
    t0_RANS = get_t0(At, constants.g, kr_rans)
    t0_LES = get_t0(At, -1e10, kr_dict[(Gr, Sc)])
    print t0_RANS, t0_LES
    data = load_data(filename)
    data[6,:,0] = data[6,:,0]/t0_LES*1e-6
    data[6,0,0] = -data[6,0,0]
    t_MAX_LES = data[6,:,0].max() - 1e-10
    print t_MAX_LES
    func_theta = interpolate.interp1d(data[6,:,0], data[6,:,1], bounds_error=True, fill_value=0.0)

    
    alpha = np.ones(79)
    args.restart = True
    if args.restart:
        alpha = np.loadtxt("alpha.dat")
    dalpha = 1e-6
    step_size = 0.05

    
    
    nrun = alpha.size + 1
    mpi_comm = MPI.COMM_WORLD
    mpi_rank = mpi_comm.Get_rank()
    mpi_size = mpi_comm.Get_size()
    nrun_local = int(np.ceil(nrun/float(mpi_size)))
    global_run_idx = [-1]
    global_run_idx.extend(range(0, alpha.size))

    run_idx = global_run_idx[mpi_rank::mpi_size]

    if mpi_rank == 0:
        J_global = np.zeros(len(global_run_idx))
        idx_map = {}
        idx_map[0] = run_idx
    if mpi_rank != 0:
        mpi_comm.send(run_idx, dest=0, tag=11)
    else:
        for i in range(1, mpi_size):
            idx_map[i] = mpi_comm.recv(source=i, tag=11)

    J = [None]*len(run_idx)
    for iteration_ in range(80):
        start = time.time()
        for jidx, j in enumerate(run_idx):
            J[jidx] = run_solve(j, alpha, dalpha)
            
        if mpi_rank == 0:
            J_global[run_idx] = J[:]

        if mpi_rank != 0:
            mpi_comm.send(J, dest=0, tag=12)
        else:
            for i in range(1, mpi_size):
                J_i = mpi_comm.recv(source=i, tag=12)
                J_global[idx_map[i]] = J_i[:]
        
        if mpi_rank == 0:
            grad = np.zeros(alpha.size)
            Jb = J_global[0]
            for i in range(grad.size):
                grad[i] = (J_global[i+1] - Jb)/dalpha
            alpha = alpha - grad/np.max(np.abs(grad))*step_size
            np.savetxt("alpha.dat", alpha)

        mpi_comm.Barrier()
        mpi_comm.Bcast(alpha, root=0)

        if 0 == mpi_rank:
            print "Time for 1 iteration: ", time.time() - start
            print iteration_, J_global[0], alpha
        mpi_comm.Barrier()

        
    #     start = time.time()
    #     J, grad = solve_fd(alpha)
    #     alpha = alpha - grad/np.max(np.abs(grad))*step_size
    #     print "Time for 1 iteration: ", time.time() - start
    #     print i, J, grad, alpha

