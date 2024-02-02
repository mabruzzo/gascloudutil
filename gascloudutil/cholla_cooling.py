import numpy as np
from scipy.interpolate import RegularGridInterpolator

class _ChollaConstants:
    kboltz_cgs: float = 1.380658e-16
    mh_cgs: float = 1.672622e-24

def _cholla_CIE_cooling_func(n, T):
    # n is number denstiy (in units of cm**-3) and T is temperature (in Kelvin)
    #
    # returns volumetric rate of cooling (positive values correspond to coolin)
    n, T = np.asanyarray(n), np.asanyarray(T)
    assert np.ndim(n) <= 1
    assert np.ndim(T) <= 1

    cooling_rate = np.empty(dtype = 'f8', shape = T.shape)

    logT = np.log10(T)
    cooling_rate[logT < 4.0] = 0.0
    w = np.logical_and(logT >= 4.0, logT < 5.9)
    cooling_rate[w] = 10.0**(-1.3 * (logT[w] - 5.25) * (logT[w] - 5.25) - 21.25)
    w = np.logical_and(logT >= 5.9, logT < 7.4)
    cooling_rate[w] = 10.0**(0.7 * (logT[w] - 7.1) * (logT[w] - 7.1) - 22.8)
    w = logT >= 7.4
    cooling_rate[w] = 10.0**(0.45 * logT[w] - 26.065)

    return (n*n)*cooling_rate

class _ChollaCloudyCoolingFunc:
    def __init__(self, path):
        log_n, log_T, log_cool_div_n2, log_heat_div_n2 = np.loadtxt(
            path, dtype = 'f4', unpack = True)

        shape = (np.unique(log_n).size, np.unique(log_T).size)
        log_n.shape = shape
        log_T.shape = shape
        log_cool_div_n2.shape = shape
        log_heat_div_n2.shape = shape

        assert np.all(log_n[:,0:1] == log_n) and np.all(log_T[0:1,:] == log_T)
        kw = {'points' : (log_n[:,0], log_T[0,:]), 'method' : 'linear',
              'bounds_error' : True}

        self._log_interp_div_n2 = {
            'cool' : RegularGridInterpolator(values = log_cool_div_n2, **kw),
            'heat' : RegularGridInterpolator(values = log_heat_div_n2, **kw)}
        self.log_n_bounds = (log_n.min(), log_n.max())
        self.log_T_bounds = (log_T.min(), log_T.max())

    def __call__(self, n, T):
        # expects arguments to be in units of cgs

        assert np.shape(n) == np.shape(T)
        if np.ndim(n) == 0:
            return_scalar = True
            pts = np.log10(np.array([[n,T]]))
        else:
            assert np.ndim(n) == 1
            return_scalar = False
            pts = np.log10(np.column_stack([n,T]))

        # we aren't currently handling bounds_errors consistently. Currently,
        # we raise an error while I suspect Cholla may just pick the nearest
        # value (although, they set cooling to 0 below 10 K)
        cooling_rate = self._log_interp_div_n2['cool'](pts)
        heating_rate = self._log_interp_div_n2['heat'](pts)

        # cool has units of erg/s/cm**3 (it's volumetric cooling rate)
        cool = n * n * (10.0 ** cooling_rate - 10.0** heating_rate)

        if return_scalar:
            return cool[0]
        return cool


class ChollaEOS:

    def __init__(self, cloudy_data_path = None, mmw = 0.6, gamma = 5.0/3.0,
                 constants = _ChollaConstants()):

        self._mmw = mmw # mean molecular weight
        self._gamma = gamma # adiabatic index

        if cloudy_data_path is not None:
            self._cooling_func = _ChollaCloudyCoolingFunc(cloudy_data_path)
        else:
            self._cooling_func = _cholla_CIE_cooling_func
        self._constants = constants

    def using_CIE_cooling(self):
        return not isinstance(self._cooling_func, _ChollaCloudyCoolingFunc)

    def _calculate_nT(self, rho, eint):
        # rho is mass density and eint is specific internal energy (aka
        # specific thermal energy).
        # -> They both have cgs units (g/cm**3 & (cm/s)**2, respectively)
        n = rho / (self._mmw * self._constants.mh_cgs)
        T = (eint * (self._mmw * self._constants.mh_cgs * (self._gamma - 1.0))
             / (self._constants.kboltz_cgs))
        return n,T

    def calculate_T(self, rho, eint):
        # rho is mass density and eint is specific internal energy (aka
        # specific thermal energy).
        # -> They both have cgs units (g/cm**3 & (cm/s)**2, respectively)
        return self._calculate_nT(rho, eint)[1]

    def rho_eint_from_nT(self, number_density, T):
        rho = number_density * (self._mmw * self._constants.mh_cgs)
        eint = ( (self._constants.kboltz_cgs * T) /
                 (self._mmw * self._constants.mh_cgs * (self._gamma - 1.0)) )
        return rho, eint

    def _calculate_tcool(self, n, T, eint_dens):
        fn = self._cooling_func
        cool = -1.0 * fn(n = n, T=T)
        #print(eint_dens)
        return eint_dens / cool

    def calculate_tcool(self, rho, eint):
        # rho is mass density and eint is specific internal energy (aka
        # specific thermal energy).
        # -> They both have cgs units (g/cm**3 & (cm/s)**2, respectively)
        #
        # The result has units of seconds. A negative value corresponds to
        # cooling. A positive value corresponds to heating
        n, T = self._calculate_nT(rho, eint)
        return self._calculate_tcool(n, T, eint_dens = rho * eint)

    def calculate_tcool_from_nT(self, number_density, T):
        # The arguments should have cgs units!
        #
        # The result has units of seconds. A negative value corresponds to
        # cooling. A positive value corresponds to heating
        eint_dens = (number_density * T *self._constants.kboltz_cgs /
                     (self._gamma - 1.0))
        return self._calculate_tcool(number_density, T, eint_dens)

    def get_gamma(self):
        return self._gamma

if __name__ == '__main__':
    my_eos = ChollaEOS(
        cloudy_data_path = ('/home/mabruzzo/hydro/cholla/src/cooling/'
                            'cloudy_coolingcurve.txt')
    )
    print(my_eos.calculate_tcool_from_nT(0.1, 20000.0))
    my_eos2 = ChollaEOS()
    print(my_eos2.calculate_tcool_from_nT(0.1, 20000.0))

    
