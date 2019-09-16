import numpy as np

class phys:
    def __init__(self, si):
        from scipy import constants as sci
        import mendeleev as pt
        import numpy as np
        
        self.si = si
        
        self.g     = sci.g * si.metre / si.second**2
        self.pi    = sci.pi
        self.R_str = sci.R * si.joule / si.kelvin / si.mole       
        self.T0    = sci.zero_Celsius * si.kelvin
        
        self.c_pd  = 1005 * si.joule / si.kilogram / si.kelvin
        self.c_pv  = 1850 * si.joule / si.kilogram / si.kelvin
        self.c_pw  = 4218 * si.joule / si.kilogram / si.kelvin
    
        self.p_tri = 611.73  * si.pascal
        self.T_tri = 273.16  * si.kelvin
        self.l_tri = 2.5e6   * si.joule / si.kilogram
        
        self.ARM_C1 = 6.1094 * 100 * si.pascal
        self.ARM_C2 = 17.625 
        self.ARM_C3 = 243.04 * si.kelvin
    
        self.sgm   = 0.072   * si.joule / si.metre**2
        self.rho_w = 1 * si.kilograms / si.litres
    
        awgh = lambda x: x.atomic_weight * si.gram / si.mole
        self.Md    = 0.78 * awgh(pt.N) * 2 + 0.21 * awgh(pt.O) * 2 + 0.01 * awgh(pt.Ar)
        self.Mv    = awgh(pt.O) + awgh(pt.H) * 2

        self.eps   = self.Mv / self.Md
        self.Rd    = self.R_str / self.Md
        self.Rv    = self.R_str / self.Mv
        self.D0    = 2.26e-5 * si.metre**2 / si.second
        self.K0    = 2.4e-2  * si.joules / si.metres / si.seconds / si.kelvins
        

    def mgn(self, q, u = None):
        if u is None: return q.to_base_units().magnitude
        return (q/u).to(self.si.dimensionless).magnitude
                
    def dot(self, a, b):
        return np.dot(a.magnitude, b.magnitude) * a.units * b.units
        
    def dp_dt(self, p, T, w, q): 
        # pressure deriv. (hydrostatic) 
        return -(p / self.R(q) / T) * self.g * w

    def dT_dt(self, T,  p, dp_dt, q, dq_dt):
        # temperature deriv. (adiabatic)
        return (T * self.R(q) / p * dp_dt - self.lv(T)*dq_dt ) / self.c_p(q)

    def dS_dt(self, T, p, dp_dt, q, dq_dt, S):
        lv = self.lv(T)
        cp = self.c_p(q)
        return (S+1) * (
            dp_dt / p * (1 - lv * self.R(q) / cp / self.Rv / T) + 
            dq_dt * (lv**2 / cp / self.Rv / T**2 + 1/(q + q**2 / self.eps))
        )

    def q(self, q0, n, r):
        return q0 - 4/3 * self.pi * self.dot(n, r**3) * self.rho_w

    def dq_dt(self, n, r, dr_dt):
        # specific humidity deriv.
        return -4 * self.pi * np.sum(n * r**2 * dr_dt) * self.rho_w

    def dr_dt_Fick(self, r, T, S, kp, rd, Td):
        rho_v = (S+1) * self.pvs(T) / self.Rv / T
        rho_eq = self.pvs(Td) * (1 + self.A(T)/r - self.B(kp,rd)/r**3) / self.Rv / Td
        D = self.D(r, Td) # TODO: K(T) vs. K(Td) ???
        return D / r / self.rho_w * (rho_v - rho_eq)

    def dr_dt_MM(self, r, T, p, S, kp, rd):
        return 1 / r * (
            S - self.A(T)/r + self.B(kp,rd)/r**3
        ) / (
            self.Fd(T, self.D(r, T)) +
            self.Fk(T, self.K(r, T, p), self.lv(T))
        )

    def dTd_dt(self, r, T, p, Td, dr_dt):
        return 3/r/self.c_pw * (
            self.K(r, T, p) / self.rho_w / r * (T - Td) + # TODO: K(T) vs. K(Td) ???
            self.lv(Td) * dr_dt
        )

    def RH(self, T, p, q): 
        # RH from mixing ratio, temperature and pressure
        return p / (1 + self.eps/q) / self.pvs(T)

    def mix(self, q, dry, wet): 
        return wet/(1/q + 1) + dry/(1 + q)
    
    def c_p(self, q): 
        return self.mix(q, self.c_pd, self.c_pv)

    def R(self, q): 
        return self.mix(q, self.Rd, self.Rv)

    def lv(self, T):
        # latent heat of evaporation 
        return self.l_tri + (self.c_pv - self.c_pw) * (T - self.T_tri)
    
    def lambdaD(self,T):
        return self.D0 / np.sqrt(2 * self.Rv * T)
              
    def lambdaK(self,T,p):
        return (4/5) * self.K0 * T / p / np.sqrt(2 * self.Rd * T)

    def beta(self, Kn):
        return (1 + Kn) / (1 + 1.71 * Kn + 1.33 * Kn*Kn) 
        
    def D(self, r, T): 
        Kn = self.lambdaD(T) / r
        return self.D0 * self.beta(Kn)

    def K(self, r, T, p):
        Kn = self.lambdaK(T, p) / r
        return self.K0 * self.beta(Kn)

# Maxwel-Mason coefficients:
    def Fd(self, T, D):
        return self.rho_w * self.Rv * T / D / self.pvs(T)
    
    def Fk(self, T, K, lv):
        return self.rho_w * lv / K / T * (lv / self.Rv / T - 1)

# Koehler curve (expressed in partial pressure):
    def A(self, T):
        return 2 * self.sgm / self.Rv / T / self.rho_w

    def B(self, kp, rd):
        return kp * rd**3

    def r_cr(self, kp, rd, T):
        # critical radius
        return np.sqrt(3 * kp * rd**3 / self.A(T))

    def pvs(self, T):
        # August-Roche-Magnus formula 
        return self.ARM_C1 * np.exp((self.ARM_C2 * (T-self.T0)) / (T-self.T0 + self.ARM_C3))

    def bpt(self, p):
        # inverse of the above
        bpt_log = np.log(p/self.ARM_C1)/self.ARM_C2
        return self.ARM_C3 * bpt_log / (1 - bpt_log) + self.T0
    
    def T(self, S, p, q):
        return self.bpt(p / (S+1) / (1 + self.eps/q))
