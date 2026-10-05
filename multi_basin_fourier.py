# -*- coding: utf-8 -*-
"""
Multi-basin, multi-channel tidal model by harmonic balance (Fourier).

Generalisation of tides4bo / tides2ba: M basins connected to each other and to
one or more open seas by N channels, with arbitrary topology.

Channels (nonlinear, as in tides4bo / tides2ba). Channel j goes from end a to
end b (each end is a sea or a point x of a basin); u_j > 0 flows from a to b:

    (L_j/g) du_j/dt + R_j u_j|u_j| = eta_a(t) - eta_b(t)
    R_j = L_j / (kse_j^2 D_j^(4/3)) + m_j / (2 g)

Basins, harmonic by harmonic (x(t) = Re sum_k X_k exp(i k w t)):

  lumped       one level:  i k w S eta_k = sum of channel inflows    (tides*)
  propagating  1D prismatic bay (surface S, length L_bay, depth d_bay), closed
               ends, linearised friction. Nodes at 0, L_bay and at the channel
               openings; each reach between nodes is an exact two-port of the
               damped long-wave equation:
                   flux a->reach = Y0 (eta_b csc(kl) - eta_a cot(kl))
                   Y0 = i k w B / kappa,  kappa^2 = (k w/c)^2 (1 - i r_k/(k w))
               k = 0: resistive reach, flux = g d B (eta_a - eta_b)/(r_0 l).

For w L_bay / sqrt(g d_bay) -> 0 every propagating basin collapses onto a single
level and the model coincides with the lumped one, i.e. with tides4bo/tides2ba.

Unknowns: Fourier coefficients of every channel velocity and the mean level of
every basin. Friction is evaluated in time and projected on the harmonics
(pseudo-spectral); the nonlinear system is solved with a Newton-type method,
starting from a linearised (Lorentz) network solution. Bay friction of the
propagating basins is updated iteratively (energy-consistent Lorentz).

Input: the parameter file of the tides program (const4bo.txt, const2ba.txt, ...:
L1.., D1.., B1.., kse or kse1.., a0, nu, g, surfaces) plus an extra file with the
topology and the additional parameters.

Usage:
    python multi_basin_fourier.py const2ba.txt const2ba_extra.txt [--check] [--no-plot]
"""

import argparse
import copy
import os
import sys
from configparser import ConfigParser

import numpy as np
from scipy.optimize import root

_trapz = getattr(np, 'trapezoid', None) or np.trapz   # numpy < 2.0 compatibility


# ============================================================================
# Input
# ============================================================================
def _list(sec, key, n, cast=float, fallback=None):
    if sec is None or not sec.get(key, '').strip():
        return [fallback] * n
    vals = [v.strip() for v in sec.get(key).split(',') if v.strip()]
    if len(vals) == 1:
        vals *= n
    if len(vals) != n:
        sys.exit(f"'{key}' needs 1 or {n} comma-separated values, got {len(vals)}")
    return [cast(v) for v in vals]


def _harmonics(sec, K, amp, phase_deg, mean=0.0):
    """Complex harmonics of a sea level: a*cos(wt - phase) + optional overtides."""
    get = lambda k, fb: sec.getfloat(k, fallback=fb) if sec is not None else fb
    Z = np.zeros(K + 1, dtype=complex)
    Z[0] = get('mean', mean)
    Z[1] = get('amplitude', amp) * np.exp(-1j * np.deg2rad(get('phase_deg', phase_deg)))
    for k in range(2, K + 1):
        ak = get(f'amp_{k}', None)
        if ak is not None:
            Z[k] = ak * np.exp(-1j * np.deg2rad(get(f'phase_{k}_deg', 0.0)))
    return Z


def read_config(const_path, extra_path):
    pc = ConfigParser(inline_comment_prefixes=('#', ';'))
    if not pc.read(const_path):
        sys.exit(f"Cannot read {const_path}")
    pe = ConfigParser(inline_comment_prefixes=('#', ';'))
    if not pe.read(extra_path):
        sys.exit(f"Cannot read {extra_path}")
    ph = pc['physical_variables']
    sec = lambda name: pe[name] if pe.has_section(name) else None
    getf = lambda s, k, fb: s.getfloat(k, fallback=fb) if s is not None else fb

    g = ph.getfloat('g', fallback=9.81)
    kse_glob = ph.getfloat('kse', fallback=30.0)
    en = sec('numerics')
    K, M = int(getf(en, 'K', 30)), int(getf(en, 'M', 2048))
    ef = sec('forcing')
    w = getf(ef, 'omega', ph.getfloat('nu'))

    def const_or_number(v):
        try:
            return float(v)
        except ValueError:
            if ph.get(v) is None:
                sys.exit(f"'{v}' is neither a number nor a key of the const file")
            return ph.getfloat(v)

    # ---- seas: default 'sea' from a0 (tides: a0 sin(nu t) -> phase 90 deg)
    seas = {'sea': _harmonics(ef, K, ph.getfloat('a0'), 90.0)}
    for s in pe.sections():
        if s.lower().startswith('sea:'):
            seas[s.split(':', 1)[1].strip().lower()] = _harmonics(pe[s], K, ph.getfloat('a0'), 90.0)

    # ---- basins
    basins = {}
    for s in pe.sections():
        if not s.lower().startswith('basin:'):
            continue
        name = s.split(':', 1)[1].strip().lower()
        b = pe[s]
        spec = dict(name=name, type=b.get('type', 'lumped').strip().lower(),
                    S=const_or_number(b.get('surface', 'S')),
                    Q_river=getf(b, 'Q_river', 0.0))
        if spec['type'] == 'propagating':
            for key in ('L_bay', 'd_bay'):
                if b.get(key) is None:
                    sys.exit(f"[{s}] type = propagating needs {key}")
            spec.update(Lb=b.getfloat('L_bay'), d=b.getfloat('d_bay'),
                        ks=getf(b, 'kse_bay', kse_glob),
                        minor_factor=getf(b, 'minor_factor', 1.5),
                        r_fixed=getf(b, 'r_bay', np.nan),
                        x_river=getf(b, 'x_river', b.getfloat('L_bay')))
        elif spec['type'] == 'lumped':
            spec.update(x_river=0.0)
        else:
            sys.exit(f"[{s}]: unknown type {spec['type']}")
        xo = b.get('x_obs', fallback='').strip()
        spec['x_obs'] = [float(v) for v in xo.split(',')] if xo else []
        basins[name] = spec
    if not basins:
        sys.exit("The extra file needs at least one [basin:<name>] section")

    # ---- channels: L1, D1, B1, ... from the const file, topology from [channels]
    n = 0
    while all(ph.get(f'{v}{n + 1}') is not None for v in ('L', 'D', 'B')):
        n += 1
    if n == 0:
        sys.exit("No channels (L1, D1, B1, ...) in the const file")
    ec = sec('channels')
    if ec is None:
        sys.exit("The extra file needs a [channels] section")
    names = _list(ec, 'names', n, str, None)
    ms = _list(ec, 'm_loss', n, float, 0.0)

    def parse_end(txt, j):
        txt = txt.strip().lower()
        x = 0.0
        if '@' in txt:
            txt, xs = txt.split('@')
            x = float(xs)
        txt = txt.strip()
        if txt in seas:
            return ('sea', txt, 0.0)
        if txt in basins:
            return ('basin', txt, x)
        sys.exit(f"channel {j}: unknown end '{txt}' (seas: {list(seas)}, basins: {list(basins)})")

    channels = []
    for j in range(1, n + 1):
        if ec.get(f'c{j}') is None:
            sys.exit(f"[channels] needs c{j} = <from> -> <to>")
        a, b = ec.get(f'c{j}').split('->')
        L, D, B = (ph.getfloat(f'{v}{j}') for v in ('L', 'D', 'B'))
        ks = ph.getfloat(f'kse{j}', fallback=kse_glob)
        ks = ec.getfloat(f'kse{j}', fallback=ks)
        channels.append(dict(name=names[j - 1] or f'ch{j}', L=L, B=B, D=D,
                             a=parse_end(a, j), b=parse_end(b, j),
                             R=L / (ks**2 * D**(4 / 3)) + ms[j - 1] / (2 * g)))
    return dict(g=g, w=w, K=K, M=M, seas=seas, basins=basins, channels=channels)


# ============================================================================
# Spectral helpers
# ============================================================================
def synth(X, M):
    F = np.zeros(X.shape[:-1] + (M // 2 + 1,), dtype=complex)
    K = X.shape[-1] - 1
    F[..., 0] = X[..., 0] * M
    F[..., 1:K + 1] = X[..., 1:] * M / 2
    return np.fft.irfft(F, n=M, axis=-1).real


def analyse(x, K):
    M = x.shape[-1]
    F = np.fft.rfft(x, axis=-1)
    X = np.empty(x.shape[:-1] + (K + 1,), dtype=complex)
    X[..., 0] = F[..., 0] / M
    X[..., 1:] = 2 * F[..., 1:K + 1] / M
    return X


def _kappa(omega_k, g, d, r):
    kap = omega_k / np.sqrt(g * d) * np.sqrt(1 - 1j * r / omega_k)
    return np.where(kap.imag > 0, -kap, kap)


# ============================================================================
# Basin
# ============================================================================
class Basin:
    def __init__(self, spec, positions, g, w, K):
        self.spec, self.name, self.g, self.w, self.K = spec, spec['name'], g, w, K
        self.S, self.Qr = spec['S'], spec['Q_river']
        self.prop = spec['type'] == 'propagating'
        self.k = np.arange(1, K + 1)
        self.iwk = 1j * self.k * w
        if self.prop:
            Lb = spec['Lb']
            pos = np.array(list(positions) + [0.0, Lb, spec['x_river']])
            if np.any(pos < 0) or np.any(pos > Lb):
                sys.exit(f"basin {self.name}: positions must lie in [0, L_bay]")
            pos = np.unique(pos)
            self.nodes = pos[np.concatenate([[True], np.diff(pos) > 1e-6 * Lb])]
            self.B = self.S / Lb
        else:
            self.nodes = np.array([0.0])
        self.Nn = self.nodes.size
        self.river_node = self.node(spec['x_river'])
        self.r1 = 0.0

    def node(self, x):
        return int(np.argmin(np.abs(self.nodes - x))) if self.prop else 0

    def r_k(self, k):
        return self.r1 if k == 1 else self.spec['minor_factor'] * self.r1

    def set_admittance(self):
        K, Nn = self.K, self.Nn
        if not self.prop:
            self.Y = (self.iwk * self.S)[:, None, None]
            self.Yinv = 1 / self.Y
            self.G0 = None
            return
        g, d = self.g, self.spec['d']
        self.kap = np.array([_kappa(k * self.w, g, d, self.r_k(k)) for k in self.k])
        Y = np.zeros((K, Nn, Nn), dtype=complex)
        G = np.zeros((Nn, Nn))
        r0 = max(self.r_k(0), 1e-9)
        Y0 = self.iwk * self.B / self.kap
        for i, l in enumerate(np.diff(self.nodes)):
            z = self.kap * l
            cot, csc = np.cos(z) / np.sin(z), 1 / np.sin(z)
            Y[:, i, i] -= Y0 * cot
            Y[:, i + 1, i + 1] -= Y0 * cot
            Y[:, i, i + 1] += Y0 * csc
            Y[:, i + 1, i] += Y0 * csc
            c = g * d * self.B / (r0 * l)
            G[i, i] += c; G[i + 1, i + 1] += c
            G[i, i + 1] -= c; G[i + 1, i] -= c
        self.Y, self.Yinv, self.G0 = Y, np.linalg.inv(Y), G

    def levels(self, Qinj, eta_ref):
        """Qinj: (Nn, K+1) inflow at nodes -> node levels (Nn, K+1), total mean inflow."""
        E = np.empty((self.Nn, self.K + 1), dtype=complex)
        E[:, 1:] = np.einsum('knm,mk->nk', self.Yinv, Qinj[:, 1:])
        q0 = Qinj[:, 0].real.copy()
        q0[self.river_node] += self.Qr
        E[0, 0] = eta_ref
        if self.Nn > 1:
            G = self.G0
            E[1:, 0] = np.linalg.solve(G[1:, 1:], q0[1:] - G[1:, 0] * eta_ref)
        return E, q0.sum()

    def friction_update(self, E):
        sp = self.spec
        kap, iw = self.kap[0], self.iwk[0]
        I3 = I2 = 0.0
        for i, l in enumerate(np.diff(self.nodes)):
            s = np.linspace(0, l, max(20, int(400 * l / sp['Lb'])))
            ea, eb = E[i, 1], E[i + 1, 1]
            deta = kap * (-ea * np.cos(kap * (l - s)) + eb * np.cos(kap * s)) / np.sin(kap * l)
            u = np.abs(iw * deta / kap**2) / sp['d']
            I3 += _trapz(u**3, s); I2 += _trapz(u**2, s)
        if I2 <= 0:
            return 0.0
        return 8 / (3 * np.pi) * self.g / (sp['ks']**2 * sp['d']**(4 / 3)) * I3 / I2

    def profile(self, E, x):
        x = np.atleast_1d(np.asarray(x, float))
        H = np.empty((x.size, self.K + 1), dtype=complex)
        if not self.prop:
            H[:] = E[0]
            return H
        for n, xx in enumerate(x):
            i = int(np.clip(np.searchsorted(self.nodes, xx, side='right') - 1, 0, self.Nn - 2))
            l, s = self.nodes[i + 1] - self.nodes[i], xx - self.nodes[i]
            ea, eb = E[i], E[i + 1]
            H[n, 1:] = (ea[1:] * np.sin(self.kap * (l - s)) + eb[1:] * np.sin(self.kap * s)) \
                / np.sin(self.kap * l)
            H[n, 0] = ea[0] + (eb[0] - ea[0]) * s / l
        return H


# ============================================================================
# System
# ============================================================================
class MultiBasinSystem:
    def __init__(self, cfg):
        self.cfg = cfg
        self.g, self.w, self.K, self.M = cfg['g'], cfg['w'], cfg['K'], cfg['M']
        if self.M <= 4 * self.K:
            raise ValueError("Use M > 4K to limit aliasing of u|u|")
        ch = cfg['channels']
        self.N = len(ch)
        self.L = np.array([c['L'] for c in ch])
        self.A = np.array([c['B'] * c['D'] for c in ch])
        self.R = np.array([c['R'] for c in ch])
        self.sea_names = list(cfg['seas'])
        self.Zs = np.array([cfg['seas'][s] for s in self.sea_names])
        self.bnames = list(cfg['basins'])
        pos = {b: [] for b in self.bnames}
        for c in ch:
            for e in (c['a'], c['b']):
                if e[0] == 'basin':
                    pos[e[1]].append(e[2])
        self.basins = [Basin(cfg['basins'][b], pos[b], self.g, self.w, self.K) for b in self.bnames]
        self.Mb = len(self.basins)
        # channel ends -> ('sea', s) or ('basin', b, node); incidence matrices
        self.ends = []
        self.P = [np.zeros((bs.Nn, self.N)) for bs in self.basins]
        for j, c in enumerate(ch):
            pair = []
            for e, sign in ((c['a'], -1.0), (c['b'], +1.0)):
                if e[0] == 'sea':
                    pair.append(('sea', self.sea_names.index(e[1])))
                else:
                    b = self.bnames.index(e[1])
                    nd = self.basins[b].node(e[2])
                    pair.append(('basin', b, nd))
                    self.P[b][nd, j] += sign
            self.ends.append(pair)
        for b, bs in enumerate(self.basins):
            if not np.any(self.P[b]):
                sys.exit(f"basin {bs.name} is not connected to any channel")
        self.offs = np.cumsum([0] + [bs.Nn for bs in self.basins])
        self.k = np.arange(1, self.K + 1)
        self.iwk = 1j * self.k * self.w
        for bs in self.basins:
            bs.set_admittance()

    # --------------------------------------------------------------- unknowns
    def unpack(self, x):
        n = self.N * (2 * self.K + 1)
        blk = x[:n].reshape(self.N, 2 * self.K + 1)
        U = np.empty((self.N, self.K + 1), dtype=complex)
        U[:, 0] = blk[:, 0]
        U[:, 1:] = blk[:, 1:self.K + 1] + 1j * blk[:, self.K + 1:]
        return U, x[n:]

    def pack(self, U, eta_ref):
        blk = np.hstack([U[:, :1].real, U[:, 1:].real, U[:, 1:].imag])
        return np.concatenate([blk.ravel(), np.real(eta_ref)])

    def all_levels(self, U, eta_ref):
        AU = self.A[:, None] * U
        E, mass = [], np.empty(self.Mb)
        for b, bs in enumerate(self.basins):
            Eb, mass[b] = bs.levels(self.P[b] @ AU, eta_ref[b])
            E.append(Eb)
        return E, mass

    def end_levels(self, E):
        La = np.empty((self.N, self.K + 1), dtype=complex)
        Lb = np.empty_like(La)
        for j, (ea, eb) in enumerate(self.ends):
            for arr, e in ((La, ea), (Lb, eb)):
                arr[j] = self.Zs[e[1]] if e[0] == 'sea' else E[e[1]][e[2]]
        return La, Lb

    def residual(self, x):
        U, eta_ref = self.unpack(x)
        E, mass = self.all_levels(U, eta_ref)
        La, Lb = self.end_levels(E)
        u = synth(U, self.M)
        F = analyse(self.R[:, None] * u * np.abs(u), self.K)
        res = np.empty_like(U)
        res[:, 0] = F[:, 0] - (La[:, 0] - Lb[:, 0])
        res[:, 1:] = (self.L[:, None] / self.g) * self.iwk * U[:, 1:] + F[:, 1:] - (La[:, 1:] - Lb[:, 1:])
        return np.concatenate([np.hstack([res[:, :1].real, res[:, 1:].real, res[:, 1:].imag]).ravel(),
                               mass / self.A.sum()])

    # ------------------------------------------------------- initial guess
    def lorentz_guess(self, n_iter=60):
        """Linear network (channel admittances a_j) solved for all node levels."""
        Nt = self.offs[-1]
        amp = np.full(self.N, 0.5)
        for _ in range(n_iter):
            r = 8 / (3 * np.pi) * self.R * amp
            U = np.zeros((self.N, self.K + 1), dtype=complex)
            for ik in range(self.K):
                a = 1 / (self.iwk[ik] * self.L / self.g + r)
                Ymat = np.zeros((Nt, Nt), dtype=complex)
                rhs = np.zeros(Nt, dtype=complex)
                for b, bs in enumerate(self.basins):
                    o = self.offs[b]
                    Ymat[o:o + bs.Nn, o:o + bs.Nn] = bs.Y[ik]
                gidx = []
                for j, pair in enumerate(self.ends):
                    idx = [self.offs[e[1]] + e[2] if e[0] == 'basin' else None for e in pair]
                    gidx.append(idx)
                    for m, e in enumerate(pair):
                        other = pair[1 - m]
                        if idx[m] is None:
                            continue
                        Ymat[idx[m], idx[m]] += a[j]
                        if other[0] == 'basin':
                            Ymat[idx[m], idx[1 - m]] -= a[j]
                        else:
                            rhs[idx[m]] += a[j] * self.Zs[other[1], ik + 1]
                eta = np.linalg.solve(Ymat, rhs)
                for j, pair in enumerate(self.ends):
                    lv = [eta[gidx[j][m]] if pair[m][0] == 'basin' else self.Zs[pair[m][1], ik + 1]
                          for m in (0, 1)]
                    U[j, ik + 1] = a[j] * (lv[0] - lv[1])
            new = np.abs(synth(U, self.M)).max(1)
            amp = 0.5 * amp + 0.5 * np.maximum(new, 1e-6)
        return U

    # ---------------------------------------------------------------- solve
    def _root(self, x0, tol):
        sol = root(self.residual, x0, method='hybr', tol=tol)
        if not sol.success:
            sol = root(self.residual, sol.x, method='lm', tol=tol)
        return sol

    def solve(self, tol=1e-12, max_outer=80, rtol_r=1e-5, verbose=False):
        props = [bs for bs in self.basins if bs.prop]
        for bs in props:
            rf = bs.spec['r_fixed']
            bs.r1 = rf if np.isfinite(rf) else (bs.r1 or 0.1 * self.w)
            bs.set_admittance()
        free = [bs for bs in props if not np.isfinite(bs.spec['r_fixed'])]
        U0 = self.lorentz_guess()
        x = self.pack(U0, np.full(self.Mb, self.Zs[0, 0].real))
        self.outer_iterations = 0
        for it in range(max_outer):
            sol = self._root(x, tol)
            x = sol.x
            self.outer_iterations = it + 1
            if not free:
                break
            U, eta_ref = self.unpack(x)
            E, _ = self.all_levels(U, eta_ref)
            change = 0.0
            for bs in free:
                b = self.basins.index(bs)
                rn = bs.friction_update(E[b])
                change = max(change, abs(rn - bs.r1) / max(rn, 1e-12))
                bs.r1 = 0.5 * (bs.r1 + rn)
                bs.set_admittance()
            if verbose:
                print(f"  outer {it + 1}: r_bay = "
                      + ', '.join(f'{bs.name} {bs.r1:.3e}' for bs in free) + f"  (rel. change {change:.1e})")
            if change < rtol_r:
                sol = self._root(x, tol)
                x = sol.x
                break
        self.U, self.eta_ref = self.unpack(x)
        self.E, _ = self.all_levels(self.U, self.eta_ref)
        self.La, self.Lb = self.end_levels(self.E)
        self.sol = sol
        return sol

    # --------------------------------------------------------------- output
    def results(self):
        t = np.arange(self.M) / self.M * 2 * np.pi / self.w
        u = synth(self.U, self.M)
        Q = self.A[:, None] * u
        out = dict(t=t, u=u, Q=Q, zeta=synth(self.Zs, self.M), Q_mean=Q.mean(1),
                   Q1=self.A * self.U[:, 1], dt=t[1] - t[0])
        out['node_eta'] = [synth(Eb, self.M) for Eb in self.E]
        out['stations'] = []
        for b, bs in enumerate(self.basins):
            xo = bs.spec['x_obs']
            H = bs.profile(self.E[b], xo) if xo else None
            out['stations'].append((xo, H, synth(H, self.M) if xo else None))
        return out

    def _end_label(self, e):
        if e[0] == 'sea':
            return self.sea_names[e[1]]
        bs = self.basins[e[1]]
        return bs.name + (f"@{bs.nodes[e[2]]:.0f}" if bs.prop else '')

    def report(self):
        r = self.results()
        Zref = self.Zs[0, 1]
        names = [c['name'] for c in self.cfg['channels']]
        print(f"\nsolver: {self.sol.message.strip()}  |residual|max = {np.abs(self.sol.fun).max():.2e}"
              f"  (outer iterations: {self.outer_iterations})")
        for s, Z in zip(self.sea_names, self.Zs):
            print(f"sea '{s}': amplitude {abs(Z[1]):.4f} m, phase {np.rad2deg(-np.angle(Z[1])):.1f} deg")
        print("(lags below are relative to the first sea)")

        for b, bs in enumerate(self.basins):
            print(f"\n=== basin '{bs.name}' ({bs.spec['type']}, S = {bs.S:.3e} m2)")
            if bs.prop:
                c = np.sqrt(self.g * bs.spec['d'])
                print(f"    w L_bay / c = {self.w * bs.spec['Lb'] / c:.3f}  (<< 1: lumped limit), "
                      f"r_bay/w = {bs.r1 / self.w:.3f}")
            print(f"    {'x m':>8} {'range m':>8} {'max m':>8} {'min m':>8} {'mean m':>8} {'|eta1| m':>9} {'lag deg':>8}")
            for n in range(bs.Nn):
                e = r['node_eta'][b][n]
                E1 = self.E[b][n, 1]
                print(f"    {bs.nodes[n]:8.0f} {e.max() - e.min():8.4f} {e.max():8.4f} {e.min():8.4f} "
                      f"{self.E[b][n, 0].real:+8.4f} {abs(E1):9.4f} {np.rad2deg(np.angle(Zref / E1)):8.1f}")
            xo, H, eo = r['stations'][b]
            if xo:
                print("    stations:")
                for xx, Hh, ee in zip(xo, H, eo):
                    print(f"    {xx:8.0f} {ee.max() - ee.min():8.4f} {ee.max():8.4f} {ee.min():8.4f} "
                          f"{Hh[0].real:+8.4f} {abs(Hh[1]):9.4f} {np.rad2deg(np.angle(Zref / Hh[1])):8.1f}")
            # exchange through each connected channel (flow into the basin)
            js = [j for j in range(self.N) if np.any(self.P[b][:, j])]
            sgn = np.array([self.P[b][:, j].sum() for j in js])
            Qin = sgn[:, None] * r['Q'][js]
            prism = np.clip(Qin, 0, None).sum(1) * r['dt']
            tot = prism.sum()
            print(f"    {'channel':>14} {'flood prism m3':>15} {'share %':>8} {'mean inflow m3/s':>17}")
            for jj, j in enumerate(js):
                print(f"    {names[j]:>14} {prism[jj]:15.3e} {100 * prism[jj] / tot:8.2f} {Qin[jj].mean():+17.2f}")

        hdr = (f"{'channel':>14} {'from -> to':>26} {'|Q1| m3/s':>10} {'ph deg':>7} {'u_max':>6} "
               f"{'prism a->b':>11} {'prism b->a':>11} {'Q_mean':>9}")
        print("\n" + hdr + "\n" + "-" * len(hdr))
        for j, pair in enumerate(self.ends):
            Q = r['Q'][j]
            lab = f"{self._end_label(pair[0])} -> {self._end_label(pair[1])}"
            print(f"{names[j]:>14} {lab:>26} {abs(r['Q1'][j]):10.1f} {np.rad2deg(np.angle(r['Q1'][j])):7.1f} "
                  f"{np.abs(r['u'][j]).max():6.3f} {np.clip(Q, 0, None).sum() * r['dt']:11.3e} "
                  f"{-np.clip(Q, None, 0).sum() * r['dt']:11.3e} {r['Q_mean'][j]:+9.2f}")
        return r


# ============================================================================
# Independent check: time integration of the same equations (method of lines)
# ============================================================================
def time_domain_check(sysm, cycles=40, nx=200):
    from scipy.integrate import solve_ivp
    if any(bs.prop and bs.spec['minor_factor'] != 1 for bs in sysm.basins):
        print("\n(check uses one bay friction coefficient for all harmonics: re-solving with minor_factor = 1)")
        sysm = copy.deepcopy(sysm)
        for bs in sysm.basins:
            if bs.prop:
                bs.spec['minor_factor'] = 1.0
        sysm.solve()
    g, L, R, A, N = sysm.g, sysm.L, sysm.R, sysm.A, sysm.N
    T = 2 * np.pi / sysm.w
    kk = np.arange(sysm.K + 1)

    layout, off = [], N
    for bs in sysm.basins:
        if bs.prop:
            dx = bs.spec['Lb'] / nx
            xc = (np.arange(nx) + 0.5) * dx
            layout.append(dict(off=off, n=2 * nx - 1, dx=dx, xc=xc))
            off += 2 * nx - 1
        else:
            layout.append(dict(off=off, n=1))
            off += 1

    def cell_and_weights(b, x):
        lay = layout[b]
        cell = min(int(x / lay['dx']), nx - 1)
        il = int(np.clip(np.searchsorted(lay['xc'], x) - 1, 0, nx - 2))
        return cell, il, (x - lay['xc'][il]) / lay['dx']

    ends = []
    for j, pair in enumerate(sysm.ends):
        ee = []
        for e in pair:
            if e[0] == 'sea':
                ee.append(('sea', e[1]))
            else:
                bs = sysm.basins[e[1]]
                x = bs.nodes[e[2]]
                ee.append(('basin', e[1], *cell_and_weights(e[1], x)) if bs.prop else ('basin', e[1]))
        ends.append(ee)

    def level(y, t, e):
        if e[0] == 'sea':
            return (sysm.Zs[e[1]] * np.exp(1j * kk * sysm.w * t)).real.sum()
        lay = layout[e[1]]
        if len(e) == 2:
            return y[lay['off']]
        eta = y[lay['off']:lay['off'] + nx]
        return (1 - e[4]) * eta[e[3]] + e[4] * eta[e[3] + 1]

    def rhs(t, y):
        u = y[:N]
        dy = np.zeros_like(y)
        Qc = A * u
        lv = np.array([[level(y, t, e) for e in ee] for ee in ends])
        dy[:N] = g / L * (lv[:, 0] - lv[:, 1] - R * u * np.abs(u))
        for b, bs in enumerate(sysm.basins):
            lay = layout[b]
            o = lay['off']
            if not bs.prop:
                dy[o] = (sysm.P[b].sum(0) @ Qc + bs.Qr) / bs.S
                continue
            dx, d, B, rb = lay['dx'], bs.spec['d'], bs.B, bs.r1
            eta, q = y[o:o + nx], y[o + nx:o + 2 * nx - 1]
            qf = np.concatenate([[0.0], q, [0.0]])
            deta = -(qf[1:] - qf[:-1]) / dx
            for j, ee in enumerate(ends):
                for m, e in enumerate(ee):
                    if e[0] == 'basin' and e[1] == b:
                        deta[e[2]] += (1 if m == 1 else -1) * Qc[j] / (B * dx)
            deta[min(int(bs.spec['x_river'] / dx), nx - 1)] += bs.Qr / (B * dx)
            dy[o:o + nx] = deta
            dy[o + nx:o + 2 * nx - 1] = -g * d * (eta[1:] - eta[:-1]) / dx - rb * q
        return dy

    tt = np.linspace((cycles - 1) * T, cycles * T, sysm.M, endpoint=False)
    s = solve_ivp(rhs, (0, cycles * T), np.zeros(off), t_eval=tt, method='RK45',
                  rtol=1e-8, atol=1e-10, max_step=T / 400)
    r = sysm.results()
    print(f"\ntime-domain check ({cycles} cycles, RK45, nx = {nx} per propagating basin):")
    names = [c['name'] for c in sysm.cfg['channels']]
    for j in range(N):
        Qtd = A[j] * s.y[j]
        errQ = np.abs(r['Q'][j] - Qtd).max() / np.abs(Qtd).max()
        lf = np.array([level(s.y[:, i], tt[i], ends[j][1]) for i in range(tt.size)])
        errL = np.abs(sysm_level_series(sysm, r, sysm.ends[j][1]) - lf).max()
        print(f"  {names[j]:>14}: max rel. error on Q = {errQ:.2e}, "
              f"max error on level at the 'to' end = {errL:.2e} m")


def sysm_level_series(sysm, r, e):
    return r['zeta'][e[1]] if e[0] == 'sea' else r['node_eta'][e[1]][e[2]]


# ============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('const', help='parameter file of the tides program, or a folder')
    ap.add_argument('extra', nargs='?', default=None, help='topology + additional parameters')
    ap.add_argument('--check', action='store_true', help='compare with time integration')
    ap.add_argument('--no-plot', action='store_true')
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    const = a.const
    if os.path.isdir(const):
        cands = [f for f in os.listdir(const) if f.startswith('const') and f.endswith('.txt')
                 and not f.endswith('_extra.txt')]
        if len(cands) != 1:
            sys.exit(f"folder {const}: expected one const*.txt file, found {cands}")
        const = os.path.join(const, cands[0])
    extra = a.extra or const[:-4] + '_extra.txt'
    out = a.out or os.path.splitext(const)[0] + '_fourier'

    cfg = read_config(const, extra)
    sysm = MultiBasinSystem(cfg)
    sysm.solve(verbose=any(bs.prop for bs in sysm.basins))
    r = sysm.report()
    if a.check:
        time_domain_check(sysm)

    names = [c['name'] for c in cfg['channels']]
    cols, head = [r['t']], ['t_s']
    for s, z in zip(sysm.sea_names, r['zeta']):
        cols.append(z); head.append(f'zeta_{s}_m')
    for b, bs in enumerate(sysm.basins):
        for n in range(bs.Nn):
            cols.append(r['node_eta'][b][n])
            head.append(f'eta_{bs.name}' + (f'_x{bs.nodes[n]:.0f}' if bs.prop else '') + '_m')
    for j, nm in enumerate(names):
        cols.append(r['Q'][j]); head.append(f'Q_{nm}_m3s')
    np.savetxt(out + '_timeseries.csv', np.column_stack(cols), delimiter=',',
               header=','.join(head), comments='')
    np.savez(out + '_harmonics.npz', U=sysm.U, omega=sysm.w, A=sysm.A, channel_names=np.array(names),
             zeta=sysm.Zs, **{f'eta_{bs.name}': sysm.E[b] for b, bs in enumerate(sysm.basins)},
             **{f'nodes_{bs.name}': bs.nodes for bs in sysm.basins})

    if not a.no_plot:
        import matplotlib.pyplot as plt
        th = r['t'] / 3600
        fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
        for s, z in zip(sysm.sea_names, r['zeta']):
            ax[0].plot(th, z, 'k', lw=1.5, label=s)
        for b, bs in enumerate(sysm.basins):
            for n in range(bs.Nn):
                lab = bs.name + (f' x={bs.nodes[n] / 1e3:.1f} km' if bs.prop else '')
                ax[0].plot(th, r['node_eta'][b][n], label=lab)
        ax[0].set(xlabel='t [h]', ylabel='level [m]'); ax[0].legend(fontsize=8)
        for j, nm in enumerate(names):
            ax[1].plot(th, r['Q'][j], label=nm)
        ax[1].set(xlabel='t [h]', ylabel='Q [m$^3$/s] (positive from -> to)'); ax[1].legend(fontsize=8)
        fig.tight_layout(); fig.savefig(out + '.png', dpi=200); plt.show()


if __name__ == '__main__':
    main()
