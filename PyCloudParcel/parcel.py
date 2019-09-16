from numpy import exp, log, sqrt, empty, sum
import numpy as np
from scipy import integrate as ode, optimize as root

from PyCloudParcel.phys import phys
from PyCloudParcel.coord import x_ln

# state vector indices and units     
class indices:
    def __init__(self, si, n, fn, thermo):
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

        assert(n > 0)
        self.x = slice(2, 2 + n)
        self.n = self.x.stop
        self.x_unit = fn.x(1 * self.r_unit).units

# parcel model with monodisperse aerosol/droplet population
class eqsys:
    def __init__(self, ph, ix, fn):
        self.ix = ix
        self.fn = fn
        self.ph = ph
        
    def __call__(self, t, y):
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
            
        dp_dt = ph.dp_dt(p, T, self.w(t), q)
        dr_dt = ph.dr_dt(r, T, p, S, self.kp, self.rd) 
        dq_dt = ph.dq_dt(self.nd, r, dr_dt)

        dy_dt = empty(ix.n)
        dy_dt[ix.p] = ph.mgn(dp_dt, ix.p_unit / ix.t_unit)
        dy_dt[ix.x] = ph.mgn(fn.dx_dr(r) * dr_dt, ix.x_unit / ix.t_unit)
        
        if hasattr(ix, 'T'):
            dT_dt = ph.dT_dt(T, p, dp_dt, q, dq_dt)
            dy_dt[ix.T] = ph.mgn(dT_dt, ix.T_unit / ix.t_unit)
        elif hasattr(ix, 'S'):
            dS_dt = ph.dS_dt(T, p, dp_dt, q, dq_dt, S)
            dy_dt[ix.S] = ph.mgn(dS_dt, ix.S_unit / ix.t_unit)
        else: 
            assert False
         
        return dy_dt


#wszystkie rd?
def parcel(*, si, t, T0, p0, w, q0, kp, rd, nd, dt_max, thermo=('T','p')):
    assert len(rd) == len(nd)
    nr = len(rd)

    fn = x_ln(si)
    ix = indices(si, nr, fn, thermo)
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
        f = lambda r: ph.mgn(ph.dr_dt(r * ix.r_unit, T0, p0, S0, kp, sys.rd[i]), ix.r_unit/ix.t_unit)
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

    integ = ode.solve_ivp(sys, [0, ph.mgn(t, ix.t_unit)], y0, method='BDF', max_step=ph.mgn(dt_max,ix.t_unit))
    assert integ.success, integ.message
  
    return integ, sys

