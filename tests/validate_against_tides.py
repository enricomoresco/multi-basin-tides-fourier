# -*- coding: utf-8 -*-
"""
Validation of multi_basin_fourier against the original time-stepping models.

1. tides4bo (one basin, 4 inlets) and tides2ba (two basins, 7 channels):
   the main loop of each program is replicated here (explicit Euler, same
   non-dimensional equations) and its last-cycle tidal range is compared with
   the lumped harmonic-balance solution.
2. Small-bay limit: the demo network with propagating basins is shrunk until
   w L_bay / sqrt(g d_bay) << 1 and compared with the lumped solution.

Run from the repository root:
    python tests/validate_against_tides.py
"""
import copy
import math
import os
import sys
from configparser import ConfigParser

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import multi_basin_fourier as mbf  # noqa: E402


def fourier_ranges(const, extra, shrink=None):
    cfg = mbf.read_config(const, extra)
    if shrink is not None:
        for b in cfg['basins'].values():
            if b['type'] == 'propagating':
                b['Lb'] *= shrink
                b['x_river'] *= shrink
                b['x_obs'] = []
        for ch in cfg['channels']:
            for key in ('a', 'b'):
                e = ch[key]
                if e[0] == 'basin':
                    ch[key] = (e[0], e[1], e[2] * shrink)
    s = mbf.MultiBasinSystem(cfg)
    s.solve()
    r = s.results()
    out = {}
    for b, bs in enumerate(s.basins):
        rng = [e.max() - e.min() for e in r['node_eta'][b]]
        out[bs.name] = (min(rng), max(rng))
    return out


def euler_tides4bo(path, nt, n):
    p = ConfigParser(inline_comment_prefixes=('#',))
    p.read(path)
    ph = p['physical_variables']
    f = ph.getfloat
    L = [f(f'L{i}') for i in range(1, 5)]
    D = [f(f'D{i}') for i in range(1, 5)]
    B = [f(f'B{i}') for i in range(1, 5)]
    S, a0, nu, g, kse = f('S'), f('a0'), f('nu'), f('g', fallback=9.81), f('kse', fallback=30)
    w = np.sqrt(g / S * sum(B[i] * D[i] / L[i] for i in range(4)))
    dim = int(nt * n * w / nu) + 1
    dt = nu * 2 * np.pi / (nt * w)
    C = [kse * D[i]**(1 / 6) / np.sqrt(g) for i in range(4)]
    U0 = S * nu * a0 / sum(B[i] * D[i] for i in range(4))
    d1 = [nu * U0 * L[i] / (g * a0) for i in range(4)]
    d2 = [U0 * U0 * L[i] / (g * C[i]**2 * a0 * D[i]) for i in range(4)]
    U, h, last = [0.0] * 4, 0.0, []
    for i in range(dim):
        s, Uo = math.sin(i * dt), U[:]
        for j in range(4):
            U[j] += dt * ((-h + s) / d1[j] - (d2[j] / d1[j]) * U[j] * abs(U[j]))
        h += dt * sum(Uo[j] * D[j] * B[j] for j in range(4)) * U0 / (S * a0 * nu)
        if i * dt > 2 * np.pi * (n - 1):
            last.append(h)
    return {'bay': a0 * (max(last) - min(last))}


def euler_tides2ba(path, nt, n):
    p = ConfigParser(inline_comment_prefixes=('#',))
    p.read(path)
    ph = p['physical_variables']
    f = ph.getfloat
    L = [f(f'L{i}') for i in range(1, 8)]
    D = [f(f'D{i}') for i in range(1, 8)]
    B = [f(f'B{i}') for i in range(1, 8)]
    ks = [f(f'kse{i}', fallback=30) for i in range(1, 8)]
    S, Sin, a0, nu, g = f('S'), f('Sin'), f('a0'), f('nu'), f('g', fallback=9.81)
    w = np.sqrt(g / ((Sin) * (B[4] * D[4] / L[4]) + (B[5] * D[5] / L[5])))   # as in tides2ba
    dim = int(nt * n * w / nu) + 1
    dt = nu * 2 * np.pi / (nt * w)
    C = [ks[i] * D[i]**(1 / 6) / np.sqrt(g) for i in range(7)]
    U0 = (S + Sin) * nu * a0 / sum(B[i] * D[i] for i in range(6))
    d1 = [nu * U0 * L[i] / (g * a0) for i in range(7)]
    d2 = [U0 * U0 * L[i] / (g * C[i]**2 * a0 * D[i]) for i in range(7)]
    U, h2, hin, ls, ln = [0.0] * 7, 0.0, 0.0, [], []
    for i in range(dim):
        s, Uo = math.sin(i * dt), U[:]
        for j in range(4):
            U[j] += dt * ((-h2 + s) / d1[j] - (d2[j] / d1[j]) * U[j] * abs(U[j]))
        for j in (4, 5):
            U[j] += dt * ((-hin + s) / d1[j] - (d2[j] / d1[j]) * U[j] * abs(U[j]))
        U[6] += dt * ((hin - h2) / d1[6] - (d2[6] / d1[6]) * U[6] * abs(U[6]))
        h2 += dt * (sum(Uo[j] * D[j] * B[j] for j in range(4)) + Uo[6] * D[6] * B[6]) * U0 / (S * a0 * nu)
        hin += dt * (Uo[4] * D[4] * B[4] + Uo[5] * D[5] * B[5] - Uo[6] * D[6] * B[6]) * U0 / (Sin * a0 * nu)
        if i * dt > 2 * np.pi * (n - 1):
            ls.append(h2)
            ln.append(hin)
    return {'south': a0 * (max(ls) - min(ls)), 'north': a0 * (max(ln) - min(ln))}


def main():
    ok = True
    ex = os.path.join(ROOT, 'examples')

    print("1) lumped harmonic balance vs explicit-Euler replicas (last-cycle tidal range, m)")
    cases = [('one_basin_4inlets', 'const4bo.txt', euler_tides4bo),
             ('two_basins_7channels', 'const2ba.txt', euler_tides2ba)]
    for folder, const, euler in cases:
        cpath = os.path.join(ex, folder, const)
        fr = fourier_ranges(cpath, cpath[:-4] + '_extra.txt')
        e500 = euler(cpath, 500, 8)
        e5000 = euler(cpath, 5000, 8)
        for name in e5000:
            ff = fr[name][0]
            err = abs(ff - e5000[name]) / e5000[name]
            ok &= err < 5e-4
            print(f"   {folder:>22} / {name:>5}: Fourier {ff:.4f} | Euler nt=500 {e500[name]:.4f} "
                  f"| Euler nt=5000 {e5000[name]:.4f} | rel. diff {err:.1e}")

    print("\n2) small-bay limit: propagating demo network shrunk 1000x vs lumped network")
    demo = os.path.join(ex, 'two_basins_propagating_demo', 'const2ba.txt')
    lump = fourier_ranges(os.path.join(ex, 'two_basins_7channels', 'const2ba.txt'),
                          os.path.join(ex, 'two_basins_7channels', 'const2ba_extra.txt'))
    small = fourier_ranges(demo, demo[:-4] + '_extra.txt', shrink=1e-3)
    for name in lump:
        err = max(abs(small[name][0] - lump[name][0]), abs(small[name][1] - lump[name][0])) / lump[name][0]
        ok &= err < 1e-3
        print(f"   {name:>5}: lumped {lump[name][0]:.5f} | propagating (min..max along bay) "
              f"{small[name][0]:.5f}..{small[name][1]:.5f} | rel. diff {err:.1e}")

    print("\nALL CHECKS PASSED" if ok else "\nSOME CHECKS FAILED")
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
