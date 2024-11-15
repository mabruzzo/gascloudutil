import numpy as np
from scipy.interpolate import RegularGridInterpolator
import unyt
import unyt.dimensions as udims
from enum import auto, Enum

class _ChollaConstants:
    kboltz_cgs: float = 1.380658e-16
    mh_cgs: float = 1.672622e-24

    def kboltz_quan(self):
        return unyt.unyt_quantity(self.kboltz_cgs, "cm**2*g/(K*s**2)")
    def mh_quan(self):
        return unyt.unyt_quantity(self.mh_cgs, "g")

def _CIE_cooling_update(cooling_rate, logT):
    cooling_rate[logT < 4.0] = 0.0
    w = np.logical_and(logT >= 4.0, logT < 5.9)
    cooling_rate[w] = 10.0**(-1.3 * (logT[w] - 5.25) * (logT[w] - 5.25) - 21.25)
    w = np.logical_and(logT >= 5.9, logT < 7.4)
    cooling_rate[w] = 10.0**(0.7 * (logT[w] - 7.1) * (logT[w] - 7.1) - 22.8)
    w = logT >= 7.4
    cooling_rate[w] = 10.0**(0.45 * logT[w] - 26.065)

def _cholla_CIE_cooling_func(n, T):
    # n is number denstiy (in units of cm**-3) and T is temperature (in Kelvin)
    #
    # returns volumetric rate of cooling (positive values correspond to coolin)
    n, T = np.asanyarray(n), np.asanyarray(T)
    assert np.ndim(n) <= 1
    assert np.ndim(T) <= 1

    cooling_rate = np.empty(dtype = 'f8', shape = T.shape)
    logT = np.log10(T)
    _CIE_cooling_update(cooling_rate, logT)
    return (n*n)*cooling_rate

def _cholla_TI_cooling_func_NOHEAT(n, T):
    # n is number denstiy (in units of cm**-3) and T is temperature (in Kelvin)
    #
    # returns volumetric rate of cooling (positive values correspond to coolin)
    n, T = np.asanyarray(n), np.asanyarray(T)
    assert np.ndim(n) <= 1
    assert np.ndim(T) <= 1

    cooling_rate = np.empty(dtype = 'f8', shape = T.shape)
    logT = np.log10(T)
    _CIE_cooling_update(cooling_rate, logT)
    w = np.logical_and(logT >= 1.0, logT < 4.0)
    cooling_rate[w] = 2e-26 * (1e7 * np.exp(-1.148e5 / (T[w] + 1000.0)) + 1.4e-2 * np.sqrt(T[w]) * np.exp(-92.0 / T[w]))
    return (n*n)*cooling_rate
    

class CoolingKind(Enum):
    CIE=auto()
    TI_COOL_NOHEAT=auto()
    TI_COOL_AND_PHOTOELECTRIC=auto()
    PHOTOELECTRIC_HEATING=auto()

class CloudyCoolKind(Enum):
    HEAT_AND_COOL = auto()
    HEAT_ONLY = auto()
    COOL_ONLY = auto()

class _ChollaCloudyCoolingFunc:
    def __init__(self, path, cool_kind = None):
        if cool_kind is None:
            cool_kind=CloudyCoolKind.HEAT_AND_COOL
        self.cool_kind = cool_kind
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
        if self.cool_kind is CloudyCoolKind.HEAT_ONLY:
            cooling_rate = 0.0
        else:
            cooling_rate = 10.0**self._log_interp_div_n2['cool'](pts)
        if self.cool_kind is CloudyCoolKind.COOL_ONLY:
            heating_rate = 0.0
        else:
            heating_rate = 10.0**self._log_interp_div_n2['heat'](pts)

        # cool has units of erg/s/cm**3 (it's volumetric cooling rate)
        cool = n * n * (cooling_rate - heating_rate)

        if return_scalar:
            return cool[0]
        return cool

class PhotoelectricHeating:
    def __init__(self, n_av, T_cut=1e4):
        self.n_av = n_av
        self.T_cut = T_cut

    def __call__(self, n, T):
        # expects arguments to be in units of cgs
        assert np.shape(n) == np.shape(T)
        cooling_rate = np.empty(dtype = 'f8', shape = T.shape)
        w = T < self.T_cut
        cooling_rate[w] = n[w]*self.n_av*-1e-26
        cooling_rate[~w] = 0.0
        return cooling_rate

class HybridCoolHeating:
    def __init__(self, fn_l):
        self.fn_l = fn_l

    def __call__(self, n, T):
        # expects arguments to be in units of cgs
        assert np.shape(n) == np.shape(T)
        cooling_rate = np.zeros(dtype = 'f8', shape = T.shape)
        for fn in self.fn_l:
            cooling_rate += fn(n=n,T=T)
        return cooling_rate

class ChollaEOS:

    def __init__(self, cooling_arg = None, mmw = 0.6, gamma = 5.0/3.0,
                 cloudy_cool_kind = None, photoelectric_heating_n_av = None,
                 constants = _ChollaConstants()):

        self._mmw = mmw # mean molecular weight
        self._gamma = gamma # adiabatic index

        photoheating = None
        if photoelectric_heating_n_av is not None:
            photoheating = PhotoelectricHeating(
                n_av = photoelectric_heating_n_av
            )

        if cooling_arg is None or isinstance(cooling_arg, CoolingKind):
            assert cloudy_cool_kind is None
            if (cooling_arg is None) or (cooling_arg is CoolingKind.CIE):
                assert photoheating is None
                self._cooling_func = _cholla_CIE_cooling_func
            elif (cooling_arg is CoolingKind.PHOTOELECTRIC_HEATING):
                assert photoheating is not None
                self._cooling_func = photoheating
            elif (cooling_arg is CoolingKind.TI_COOL_NOHEAT):
                assert photoheating is None
                self._cooling_func = _cholla_TI_cooling_func_NOHEAT
            elif (cooling_arg is CoolingKind.TI_COOL_AND_PHOTOELECTRIC):
                assert photoheating is not None
                self._cooling_func = HybridCoolHeating(
                    [_cholla_TI_cooling_func_NOHEAT, photoheating]
                )
            else:
                raise ValueError(f"we haven't added support for {cooling_arg}")
        else:
            fn = _ChollaCloudyCoolingFunc(cooling_arg, cloudy_cool_kind)
            if photoheating is None:
                self._cooling_func = fn
            else:
                self._cooling_func = HybridCoolHeating([fn, photoheating])

        self._constants = constants

    def using_CIE_cooling(self):
        return not isinstance(self._cooling_func, _ChollaCloudyCoolingFunc)

    def _calculate_nT_CGS(self, rho_cgs, eint_cgs):
        # rho_cgs is mass density and eint_cgs is specific internal energy (aka
        # specific thermal energy).
        # -> neither argument should be a unyt.unyt_array instance. Instead
        #    they ordinary `float` instances or ordinary numpy arrays (with
        #    broadcastable shapes)
        # -> This function assumes that they have cgs units (g/cm**3 &
        #    (cm/s)**2, respectively). The returned values are in cgs units
        n = rho_cgs / (self._mmw * self._constants.mh_cgs)
        gm1 = self._gamma - 1.0
        T = ( eint_cgs * (self._mmw * self._constants.mh_cgs * gm1) /
              self._constants.kboltz_cgs )
        return n,T

    @udims.returns(udims.temperature)
    @udims.accepts(rho = udims.density, eint = udims.specific_energy)
    def calculate_T(self, rho, eint):
        # rho is mass density and eint is specific internal energy (aka
        # specific thermal energy).
        return self._calculate_nT_CGS(rho.in_cgs().v,
                                      eint.in_cgs().v)[1] * unyt.K

    def rho_eint_from_nT_CGS(self, number_density, T):
        rho = number_density * (self._mmw * self._constants.mh_cgs)
        eint = ( (self._constants.kboltz_cgs * T) /
                 (self._mmw * self._constants.mh_cgs * (self._gamma - 1.0)) )
        return rho, eint

    @udims.accepts(number_density = udims.number_density, T = udims.temperature)
    def rho_eint_from_nT(self, number_density, T):
        """
        Computes the mass density and specific internal energy

        Parameters
        ----------
        number_density : unyt.unyt_array
            number density of gas
        T : unyt.unyt_array
            gas temperature

        Returns
        -------
        rho, eint : unyt.unyt_array
            The equivalent mass density and specific thermal energy values
        """
        rho_cgs, eint_cgs = self.rho_eint_from_nT_CGS(
            number_density = number_density.in_cgs().ndview,
            T = T.to('K').in_cgs().ndview)
        rho = unyt.unyt_array(rho_cgs, 'g/cm**3')
        eint = unyt.unyt_array(eint_cgs, 'cm**2/s**2')
        return rho, eint

    def _calculate_tcool(self, n, T, eint_dens):
        fn = self._cooling_func
        cool = -1.0 * fn(n = n, T=T)
        with np.errstate(divide = 'ignore'):
            return eint_dens / cool

    @udims.returns(udims.energy/udims.time/ udims.volume)
    @udims.accepts(number_density = udims.number_density, T = udims.temperature)
    def cooling_rate_from_nT(self, n, T):
        fn = self._cooling_func
        cool = -1.0 * fn(n = n.in_cgs().ndview, T=T.in_cgs().ndview)
        return unyt.unyt_array(cool, 'erg/s/cm**3')
        

    def calculate_tcool_CGS(self, rho, eint):
        # just like calculate_tcool, but CGS units are implied!
        n, T = self._calculate_nT_CGS(rho_cgs = rho, eint_cgs = eint)
        return self._calculate_tcool(n, T, eint_dens = rho * eint)

    @udims.returns(udims.time)
    @udims.accepts(rho = udims.density, eint = udims.specific_energy)
    def calculate_tcool(self, rho, eint):
        """
        Computes a cooling timescale

        Parameters
        ----------
        rho : unyt.unyt_array
            Mass density
        eint : unyt.unyt_array
            specific internal energy (aka specific thermal energy)

        Returns
        -------
        tcool : unyt.unyt_array
            The cooling timescale. A positive value corresponds to heating. A
            negative value corresponds to cooling.
        """
        tcool_seconds = self.calculate_tcool_CGS(rho = rho.in_cgs().ndview,
                                                 eint = eint.in_cgs().ndview)
        return unyt.unyt_array(tcool_seconds,'s')

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

    
