from pathlib import Path

from functions import *
from quadratic import *
from optimizers import *

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import csv
import os, glob

from collections import defaultdict

opts = [
    LinearConjugateGradient,
    FletcherReeves,
    PolakRibiere,
    NewtonCholesky,
    NewtonDirection,
    PowellsDogLeg,
    DFP,
    BFGS,
    LBFGS,
    NewtonCG
]

EPSILON = 1e-8

def prepare(output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # [f.unlink() for f in Path(output_dir).iterdir() if f.is_file()]


def prepare_path(output_path):
    output_path.mkdir(parents=True, exist_ok=True)
    # for f in output_path.glob("*"):
    #     if f.is_file():
    #         f.unlink()


def task1table(output_dir, filename_base, data):
    output_csv = filename_base + "_table.csv"
    with open(os.path.join(output_dir, output_csv), 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['n', 'k', 'Nit', 'Nfev', 'Ngev', 'Nhev', 'StopReason'])
        for n, data_n in data.items():
            for k, res in data_n.items():
                nit, nfev, ngev, nhev, reason = res
                writer.writerow([n, k, nit, nfev, ngev, nhev, reason])


RED = "#d7191c"
ORANGE = "#fdae61"
GREEN = "#006400"
BLUE = "#2b83ba"


def task1plt_plot(filepath, label, title, x, nits, nfevs, ngevs, nhevs):
    plt.plot(x, nits, label='Iteration count', linewidth=2, linestyle='--', color=RED, marker='o')
    plt.plot(x, nfevs, label='Function call count', linewidth=2, linestyle='--', color=ORANGE, marker='^')
    plt.plot(x, ngevs, label='Gradient call count', linewidth=2, linestyle='--', color=GREEN, marker='s')
    plt.plot(x, nhevs, label='Hessian call count', linewidth=2, linestyle='--', color=BLUE, marker='x')
    plt.legend()
    plt.xlabel(label)
    plt.xscale('log')
    plt.title(title)
    plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
    plt.grid(True, which='both')
    plt.savefig(filepath, dpi=300)
    plt.close()


def task1plt(output_dir, filename_base, data, ns, ks):
    fixed_n_dir = "fixed_n"
    fixed_k_dir = "fixed_k"
    output_n = Path(output_dir) / fixed_n_dir
    output_k = Path(output_dir) / fixed_k_dir

    for n in ns:
        filename_plt = filename_base + f"_fix_n_{n}.png"
        file_path = os.path.join(output_n, filename_plt)
        ks_ = list(data[n].keys())
        vals = data[n].values()
        nits, nfevs, ngevs, nhevs, _ = map(list, zip(*vals))
        label = "k (condition number)"
        title = f'Task 1 (fixed n={n}).'
        task1plt_plot(file_path, label, title, ks_, nits, nfevs, ngevs, nhevs)

    for k in ks:
        filename_plt = filename_base + f"_fix_k_{k}.png"
        file_path = os.path.join(output_k, filename_plt)
        ns_ = list(data.keys())
        nits, nfevs, ngevs, nhevs = [], [], [], []
        for n in ns_:
            nit, nfev, ngev, nhev, _ = data[n][k]
            nits.append(nit)
            nfevs.append(nfev)
            ngevs.append(ngev)
            nhevs.append(nhev)
        label = "n (dimension number)"
        title = f'Task 1 (fixed k={k}).'
        task1plt_plot(file_path, label, title, ns_, nits, nfevs, ngevs, nhevs)


def task1():
    output_dir = "task1"
    prepare(output_dir)
    fixed_n_dir = "fixed_n"
    fixed_k_dir = "fixed_k"
    output_n = Path(output_dir) / fixed_n_dir
    output_k = Path(output_dir) / fixed_k_dir
    prepare_path(output_n)
    prepare_path(output_k)

    ns = [int(2 ** i) for i in range(1, 10)]
    ks = [int(1.5 ** i) for i in range(35)]

    for opt in opts[9:]:
        filename_base = f"task1_{opt.__name__}"
        data = defaultdict(dict)
        for n in ns:
            for k in ks:
                quad = Quadratic(n, k)
                starting_pos = 5.0 * np.ones(n)
                optimizer = opt(quad.eval_func, quad.eval_grad, quad.eval_hess, EPSILON, starting_pos)
                res = optimizer.find_minimum()
                if res is not None:
                    x, _ = res
                    print(f'({opt.__name__},n={n},k={k}):expected={quad._xmin}, actual:{x}')
                else:
                    print(f'({opt.__name__},n={n},k={k}):expected={quad._xmin}, actual: not found')
                nit = optimizer.iteration_count
                nfev, ngev, nhev = optimizer.func_call_count, optimizer.grad_call_count, optimizer.hess_call_count
                stop_reason = optimizer.stop_reason

                data[n][k] = (nit, nfev, ngev, nhev, stop_reason)
        task1table(output_dir, filename_base, data)
        task1plt(output_dir, filename_base, data, ns, ks)


def draw_trajectory(trajectory, color):
    trajectory = np.array(trajectory)
    x_coords = trajectory[:, 0]
    y_coords = trajectory[:, 1]
    x_coords = np.nan_to_num(x_coords, nan=np.nan, posinf=np.nan, neginf=np.nan)
    y_coords = np.nan_to_num(y_coords, nan=np.nan, posinf=np.nan, neginf=np.nan)
    plt.plot(x_coords, y_coords, '-o', color=color, markersize=6)
    plt.plot(x_coords[0], y_coords[0], 's', color=color, markersize=10)


def draw_plot(f, xmins):
    avg = np.mean(xmins, axis=0)
    x, y = avg[0], avg[1]
    x_range = (x - 6, x + 6)
    y_range = (y - 6, y + 6)

    x_grid = np.linspace(x_range[0], x_range[1], 200)
    y_grid = np.linspace(y_range[0], y_range[1], 200)
    X, Y = np.meshgrid(x_grid, y_grid)

    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = f(np.array([X[i, j], Y[i, j]]))
    plt.figure(figsize=(8, 8))
    plt.xlim(x - 6, x + 6)
    plt.ylim(y - 6, y + 6)
    contours = plt.contour(X, Y, Z, levels=50, cmap='viridis', linewidths=0.8)
    plt.clabel(contours, inline=True, fontsize=8, fmt='%1.1f')
    plt.xlabel('X')
    plt.ylabel('Y')
    xmins = np.asarray(xmins)

    plt.scatter(xmins[:, 0], xmins[:, 1],
                marker='*', s=600, c='black',
                label='Точка минимума')


def task23table(output_dir, output_csv, data):
    with open(os.path.join(output_dir, output_csv), 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Xstart', 'Ystart', 'Nit', 'Nfev', 'Ngev', 'Nhev', 'StopReason'])
        for res in data:
            xstart, ystart, nit, nfev, ngev, nhev, reason = res
            writer.writerow([xstart, ystart, nit, nfev, ngev, nhev, reason])


colors = [
    "#e50000",
    "#d4d100",
    "#18e100",
    "#007ba9",
    "#f508cb"
]

def task2():
    output_dir = "task2"
    prepare(output_dir)
    quad = Quadratic(n=2, k=10)
    xmin = quad._xmin
    Npoints = 5
    angles = 2 * np.pi * np.arange(Npoints) / Npoints
    deltas = np.column_stack((np.cos(angles), np.sin(angles)))
    deltas = [(1.0 + i) * deltas[i] for i in range(Npoints)]
    starting_points = [xmin + delta for delta in deltas]

    for opt in opts[:-1]:
        filename_base = f"task2_{opt.__name__}"
        filename_plt = filename_base + "_plot.png"
        file_path = os.path.join(output_dir, filename_plt)
        draw_plot(quad.eval_func, [xmin])
        data = []
        for i in range(Npoints):
            starting_point = starting_points[i]
            optimizer = opt(quad.eval_func, quad.eval_grad, quad.eval_hess, EPSILON, starting_point,
                            do_history=True)
            optimizer.find_minimum()
            draw_trajectory(optimizer.get_history(), colors[i])
            nit = optimizer.iteration_count
            stop_reason = optimizer.stop_reason
            nfev, ngev, nhev = optimizer.func_call_count, optimizer.grad_call_count, optimizer.hess_call_count
            data.append((starting_point[0], starting_point[1], nit, nfev, ngev, nhev, stop_reason))
        plt.legend()
        plt.savefig(file_path, dpi=300)
        plt.close()

        filename_csv = filename_base + "_table.csv"
        task23table(output_dir, filename_csv, data)

tricky_functions = [
    rosenbrock,
    himmelblau
]

def task3():
    output_dir = "task3"
    prepare(output_dir)

    Npoints = 5
    angles = 2 * np.pi * np.arange(Npoints) / Npoints
    deltas = np.column_stack((np.cos(angles), np.sin(angles)))
    starting_points = [(1.0 + i) * deltas[i] for i in range(Npoints)]

    for opt in opts:
        for f in tricky_functions:
            filename_base = f"task3_{opt.__name__}_{f.name}"
            filename_plt = filename_base + "_plot.png"
            file_path = os.path.join(output_dir, filename_plt)
            draw_plot(f.f, f.xmins)
            data = []

            for i in range(Npoints):
                starting_point = starting_points[i]
                optimizer = opt(f.f, f.grad_f, f.hess_f, EPSILON, starting_point,
                                do_history=True)
                optimizer.find_minimum()
                draw_trajectory(optimizer.get_history(), colors[i])
                nit = optimizer.iteration_count
                stop_reason = optimizer.stop_reason
                nfev, ngev, nhev = optimizer.func_call_count, optimizer.grad_call_count, optimizer.hess_call_count
                data.append((starting_point[0], starting_point[1], nit, nfev, ngev, nhev, stop_reason))

            plt.legend()
            plt.savefig(file_path, dpi=300)
            plt.close()

            filename_csv = filename_base + "_table.csv"
            task23table(output_dir, filename_csv, data)

def task4table(output_dir, output_csv, data):
    with open(os.path.join(output_dir, output_csv), 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['m', 'Nit', 'Nfev', 'Ngev', 'Nhev', 'Delta', 'StopReason'])
        for res in data:
            m, nit, nfev, ngev, nhev, delta, reason = res
            writer.writerow([m, nit, nfev, ngev, nhev, delta, reason])

def task4():
    output_dir = "task4"
    prepare(output_dir)

    MAX_M = 100
    DIM = 30

    quad = Quadratic(n=DIM, k=10000)
    starting_point = quad._xmin + np.ones(DIM)
    end_point = quad._xmin
    ms = [i + 1 for i in range(MAX_M)]
    data = []

    filename_base = f"task4_"

    for m in ms:
        optimizer = LBFGS(quad.eval_func, quad.eval_grad, quad.eval_hess, EPSILON,
                          starting_point, m=m)
        optimizer.find_minimum()
        delta = np.linalg.norm(end_point - optimizer._cur_pos)
        nit = optimizer.iteration_count
        stop_reason = optimizer.stop_reason
        nfev, ngev, nhev = optimizer.func_call_count, optimizer.grad_call_count, optimizer.hess_call_count
        data.append((m, nit, nfev, ngev, nhev, delta, stop_reason))

    filename_csv = filename_base + "table.csv"
    task4table(output_dir, filename_csv, data)
    nits = [item[1] for item in data]
    nfevs = [item[2] for item in data]
    ngevs = [item[3] for item in data]

    filename_plt = filename_base + "plot.png"
    file_path = os.path.join(output_dir, filename_plt)
    plt.plot(ms, nits, label='Iteration count', linewidth=2, linestyle='--', color=RED)
    plt.plot(ms, nfevs, label='Function call count', linewidth=2, linestyle='--', color=ORANGE)
    plt.plot(ms, ngevs, label='Gradient call count', linewidth=2, linestyle='--', color=GREEN)
    plt.legend()
    plt.xlabel("Amount of stored last pairs m")
    plt.title("LBFGS efficiency depending on number m")
    plt.grid(True, which='both')
    plt.savefig(file_path, dpi=300)
    plt.close()


def main():
    # task1()
    # task2()
    # task3()
    task4()
    return

if __name__ == "__main__":
    main()
