from numpy import exp, log, sqrt, empty, sum
import numpy as np
from scipy import integrate as ode, optimize as root

from PyCloudParcel.phys import phys
from PyCloudParcel.coord import x_ln

# state vector indices and units     
class indices:
    def __init__(self, si, n, fn, thermo, micro):
        self.p_unit = si.pascal
        self.T_unit = si.kelvin
        self.q_unit = si.dimensionless
        self.t_unit = si.second
        self.S_unit = si.dimensionless
        self.r_unit = si.metre
        self.n_unit = 1/si.kilogram
        
        assert 'p' in thermo
        self.p = 0 

        if 'T' in thermo: self.T = 1
        elif 'S' in thermo: self.S = 1
        else: assert False

        assert 'r' in micro
        assert(n > 0)
        self.x = slice(2, 2 + n)
        self.n = self.x.stop
        self.x_unit = fn.x(1 * self.r_unit).units

        if 'T' in micro:
            self.Td = slice(self.x.stop, self.x.stop + n)
            self.n = self.Td.stop
        else:
            if len(micro) > 1: assert False

# parcel model with monodisperse aerosol/droplet population
class eqsys:
    def __init__(self, ph, ix, fn):
        self.ix = ix
        self.fn = fn
        self.ph = ph
        
    def __call__(self, t, y):
        # extracting physical vars from y
        ix = self.ix
        fn = self.fn
        ph = self.ph

        t = t * ix.t_unit
        p = y[ix.p] * ix.p_unit
        x = y[ix.x] * ix.x_unit

        r = fn.r(x)
        q = ph.q(self.q1, self.nd, r)

        if hasattr(ix, 'T'):
            T = y[ix.T] * ix.T_unit
            S = ph.RH(T,p,q) - 1
        elif hasattr(ix, 'S'):
            S = y[ix.S] * ix.S_unit
            T = ph.T(S, p, q)
        else:
            assert False

        if hasattr(ix, 'Td'):
            Td = y[ix.Td] * ix.T_unit

        # physics
        dp_dt = ph.dp_dt(p, T, self.w(t), q)

        if hasattr(ix, 'Td'):
            dr_dt = ph.dr_dt_Fick(r, T, S, self.kp, self.rd, Td)
        else:
            dr_dt = ph.dr_dt_MM(r, T, p, S, self.kp, self.rd)

        if hasattr(ix, 'Td'):
            dTd_dt = ph.dTd_dt(r, T, p, Td, dr_dt)

        dq_dt = ph.dq_dt(self.nd, r, dr_dt)

        if hasattr(ix, 'T'):
            dT_dt = ph.dT_dt(T, p, dp_dt, q, dq_dt)
        elif hasattr(ix, 'S'):
            dS_dt = ph.dS_dt(T, p, dp_dt, q, dq_dt, S)
        else:
            assert False

        # preparing dy_dt to be returned
        dy_dt = empty(ix.n)
        dy_dt[ix.p] = ph.mgn(dp_dt, ix.p_unit / ix.t_unit)
        dy_dt[ix.x] = ph.mgn(fn.dx_dr(r) * dr_dt, ix.x_unit / ix.t_unit)

        if hasattr(ix, 'T'):
            dy_dt[ix.T] = ph.mgn(dT_dt, ix.T_unit / ix.t_unit)
        elif hasattr(ix, 'S'):
            dy_dt[ix.S] = ph.mgn(dS_dt, ix.S_unit / ix.t_unit)
        else:
            assert False

        if hasattr(ix, 'Td'):
            dy_dt[ix.Td] = ph.mgn(dTd_dt, ix.T_unit / ix.t_unit)

        return dy_dt

class minfun:
    def __init__(self, ph, ix, T0, S0, p0, kp, rd):
        self.ph = ph
        self.ix = ix
        self.T0 = T0
        self.S0 = S0
        self.p0 = p0
        self.kp = kp
        self.rd = rd

    def __call__(self, r):
        if hasattr(self.ix, 'Td'):
            drdt = self.ph.dr_dt_Fick(
                r * self.ix.r_unit, self.T0, self.S0, self.kp, self.rd, self.T0
            )
        else:
            drdt = self.ph.dr_dt_MM(
                r * self.ix.r_unit, self.T0, self.p0, self.S0, self.kp, self.rd
            )
        return self.ph.mgn(drdt, self.ix.r_unit / self.ix.t_unit)


def parcel(*, si, t, T0, p0, w, q0, kp, rd, nd, rtol=1e-3, thermo=('T','p'), micro=('r',)):
    assert len(rd) == len(nd)
    nr = len(rd)

    fn = x_ln(si)
    ix = indices(si, nr, fn, thermo, micro)
    ph = phys(si)
    
    sys = eqsys(ph, ix, fn)
    sys.w = w 
    sys.kp = kp # TODO: multiple kappas
    sys.rd = rd.to(ix.r_unit)
    sys.nd = nd.to(ix.n_unit)
        
    r0 = empty(nr) * ix.r_unit
    S0 = ph.RH(T0, p0, q0) - 1

    for i in range(nr):
        a = ph.mgn(sys.rd[i], ix.r_unit)
        b = ph.mgn(ph.r_cr(kp, sys.rd[i], T0), ix.r_unit)
        f = minfun(ph, ix, T0, S0, p0, kp, sys.rd[i])
        r0.magnitude[i] = root.brentq(f, a, b)

    # introducing q1 so that q @ t=0 equals q0
    sys.q1 = q0 - ph.q(0, sys.nd, r0)
    
    # y0
    y0 = empty(ix.n)
    y0[ix.p] = ph.mgn(p0, ix.p_unit)
    y0[ix.x] = ph.mgn(fn.x(r0), ix.x_unit)

    if hasattr(ix, 'T'):
        y0[ix.T] = ph.mgn(T0, ix.T_unit)
    elif hasattr(ix, 'S'):
        y0[ix.S] = ph.mgn(S0, ix.S_unit)
    else:
        assert False

    if hasattr(ix, 'Td'):
        y0[ix.Td] = ph.mgn(T0, ix.T_unit)

    integ = ode.solve_ivp(
        sys,
        [0, ph.mgn(t, ix.t_unit)],
        y0,
        method='BDF',
        rtol=rtol,
        atol=1e-6
    )
    assert integ.success, integ.message
  
    return integ, sys

