from warnings import warn

import numpy as np
import unyt
import unyt.dimensions as udim

from cool_func_stats import find_special_locations_isobar, EnergyFlow
from cholla_cooling import ChollaEOS


@udim.accepts(Tmin = udim.temperature, Tmax = udim.temperature,
              pressure = udim.pressure)
def _find_loc_mintcool(eos, Tmin, Tmax, 
                       pressure = unyt.unyt_quantity(1e3,'K/cm**3')* unyt.kboltz_cgs):
    # find the location in phase-space where tcool is minimized at a given
    # pressure (between Tmin and Tmax)
    #
    # At the moment, this will only work with Cholla's CIE cooling curve!

    # strip off units (if applicable)

    _, specific_eint_min = eos.rho_eint_from_nT(
        pressure/(Tmin*unyt.kboltz_cgs), Tmin)
    _, specific_eint_max = eos.rho_eint_from_nT(
        pressure/(Tmax*unyt.kboltz_cgs), Tmax)

    log_specific_eint_step = 0.1

    gm1 = (eos.get_gamma() - 1)
    heat_cool_intervals, eint_cgs_extrema = find_special_locations_isobar(
        pthermal_div_gm1 = (pressure / gm1).in_cgs().ndview,
        specific_eint_bounds = [specific_eint_min, specific_eint_max],
        brute_step = log_specific_eint_step,
        is_log10_brute_step = True,
        cooling_eos = eos,
        maxiter_each = 500, s_rel_tol = 1e-8)

    # sanity checks
    assert np.all(eint_cgs_extrema.min() >= heat_cool_intervals.intervals[0])
    assert np.all(eint_cgs_extrema.max() <= heat_cool_intervals.intervals[-1])

    eint_extrema = unyt.unyt_array(eint_cgs_extrema, 'cm**2/s**2')
    rho_extrema = pressure / gm1 / eint_extrema
    tcool_at_extrema = eos.calculate_tcool(rho = rho_extrema,
                                           eint = eint_extrema)

    # here we enforce some sanity checks!
    num_intervals = heat_cool_intervals.num_intervals
    if num_intervals == 0:
        raise RuntimeError("Something went horribly wrong")
    elif ((num_intervals == 1) and
          (heat_cool_intervals.get_energy_flow(0) != EnergyFlow.COOL)):
        warn("The specified cooling curve does not have any cooling")
        return np.nan, np.nan, np.nan
    elif (num_intervals == 1):
        # the entire cooling function range just has cooling
        eint_mincool = eint_extrema[np.argmin(np.abs(tcool_at_extrema))]
    elif (num_intervals == 2):
        # todo: add support for the case where cooling is turned off at higher
        #       temperatures (to prevent cooling of the wind)
        if ((heat_cool_intervals.get_energy_flow(0) == EnergyFlow.COOL) or
            (heat_cool_intervals.get_energy_flow(1) != EnergyFlow.COOL)):
            # listing both of these conditions, rather than just 1 includes the
            # pathological case where there is an extrema that corresponds to
            # a cooling rate of 0.0 at a single Temperature
            raise RuntimeError(
                "For a cooling curve with 2 heating/cooling/neutral regions, "
                "we can't currently handle the case where the lower interval "
                "corresponds to cooling and/or the upper interval corresponds "
                "to heating/neutral energy flow")

        # identify the extrema candidates in the higher temperature interval
        # (that interval corresponds to cooling)
        w_cooling = (tcool_at_extrema < 0)
        mincool_index = np.argmin(np.abs(tcool_at_extrema * w_cooling))
        eint_mincool = eint_extrema[w_cooling]
    else:
        raise RuntimeError(
            "can't handle cases with more than 2 cooling intervals yet!")

    rho_mincool = pressure / gm1 / eint_mincool
    T_mincool = eos.calculate_T(rho = rho_mincool, eint = eint_mincool)
    mintcool = eos.calculate_tcool(rho = rho_mincool, eint = eint_mincool)

    if False:
        phat = float((pressure / unyt.kboltz_cgs).to('K/cm**3').v)
        print(f"min tcool for {float(Tmin.to('K').v):.3e} <= (T/K) <= "
              f"{float(Tmax.to('K').v):.3e}, at pressure/kB = "
              f"{float(phat):.3e} K/cm**3 is:\n"
              f"     |tcool| = {np.abs(mintcool)}\n"
              f"     it occurs at T = {T_mincool}"#or equivalently\n"
              #f"     at eint = {special_eint_vals[min_index]}"
        )
    return eint_mincool, T_mincool, mintcool

def find_minmix(Tcl, Tw, assumed_p, eos):
    """
    Calculate tcool_minmix at the given pressure.
    
    This assumes that cloud will not cool below Tcl
    
    Parameters
    ----------
    Tcl : unyt.unyt_quantity
        The cloud temperature
    Tw : unyt.unyt_quantity
        The wind temperature
    assumed_p : unyt.unyt_quantity
        The assumed pressure. This assumes things are at
        constant pressure.
    valid_coolingcurve_Trange : seq of 2 unyt.unyt_quantities
        The range temperatures where cooling is valid.
    eos : equation of state
    
    TODO:
    -----
    In the future, rewrite this function so that it takes eint as arguments
    rather than temperature!
    """

    if isinstance(eos, ChollaEOS) and eos.using_CIE_cooling():
        # this is a crude hack
        # ->we know heating and cooling is 0 below 1e4 Kelvin (in other words,
        #   tcool is infinite), so no need to search for the location where 
        #   tcool is minimized below 1e4 K
        min_T_search = np.maximum(Tcl.to('K'), 1e4 * unyt.K)
        max_T_search = 1e12 * unyt.K # arbitrarily large value
        assert (1e4 * unyt.K) < Tw <= (1e12 * unyt.K)
    else:
        raise RuntimeError("At the moment, this function is not general "
                           "purpose enough to work with anything other "
                           "than Cholla's CIE cooling curve")

    # find the specific internal energy & temperature where cooling is minimized
    # along the isobar
    e_mincool, T_mincool, _ = _find_loc_mintcool(
        eos, Tmin = min_T_search, Tmax = max_T_search,
        pressure = assumed_p.to('dyne/cm**2'))

    _, e_wind = eos.rho_eint_from_nT(
        number_density = (assumed_p.to('dyne/cm**2')/(Tw*unyt.kboltz_cgs)),
        T = Tw)

    e_minmix = np.sqrt(e_mincool * e_wind)
    rho_minmix = (assumed_p / ((eos.get_gamma() - 1) * e_minmix)).to('g/cm**3')

    return {
        'rho_minmix' : rho_minmix,
        'e_minmix' : e_minmix,
        'T_minmix' : eos.calculate_T(rho = rho_minmix, eint = e_minmix),
        'tcool_minmix' : eos.calculate_tcool(rho = rho_minmix, eint = e_minmix)
    }

@udim.accepts(vwind = udim.velocity, pressure = udim.pressure)
def estimate_survival_radius(eos, vwind, density_contrast, pressure, alpha = 7,
                             *, temperature_cl = None, temperature_w = None):
    """
    Estimate the minimum radius for cloud survival,
        `tcoolminmix < alpha * t_shear`

    The user needs to specify either `temperature_cl` or `temperature_w`.

    Parameters
    ----------
    vwind : unyt.unyt_quantity
        Speed of the wind (in the cloud's reference frame)
    density_contrast : float
        ratio between cloud mass density and wind mass density
    pressure : unyt.unyt_quantity
        The thermal_pressure that assumed to be constant for the
        cloud and wind
    alpha : float
        A common choice is 7
    """

    # given a pressure and temperature, this computes the temperature
    # when the mass_density has been multiplied by mass_density_factor
    # (and pressure is held constant)
    def _get_matching_T(cur_T, mass_density_factor, pressure, eos):
        # at a given pressure and cur_T, we compute number density
        cur_number_density = pressure / (cur_T * unyt.kboltz_cgs)

        # get the mass-density, specific internal energy equivalent to
        # (cur_number_density, cur_T)
        cur_rho, cur_eint = eos.rho_eint_from_nT(
            number_density = cur_number_density, T = cur_T)

        new_rho = cur_rho * mass_density_factor
        new_eint = cur_rho * cur_eint / new_rho
        return eos.calculate_T(new_rho, new_eint)

    if np.ndim(pressure) > 0:
        raise ValueError("The pressure argument must be a scalar")
    elif (temperature_cl is None) == (temperature_w is None):
        raise ValueError(
            "the user must specify one either temperature_cl OR "
            "temperature_w (BUT NOT BOTH)")
    elif (temperature_cl is not None):
        T_cl = temperature_cl.to('K')
        T_w = _get_matching_T(cur_T = T_cl,
                              mass_density_factor = 1.0/density_contrast,
                              pressure = pressure, eos = eos)
    else:
        T_w = temperature_w.to('K')
        T_cl = _get_matching_T(cur_T = T_w, 
                               mass_density_factor = density_contrast,
                               pressure = pressure, eos = eos)
    rslt = find_minmix(Tcl = T_cl, Tw = T_w, assumed_p = pressure,
                       eos = eos)
    print(rslt)
    tcool_minmix = rslt['tcool_minmix']
    if tcool_minmix >= 0:
        raise RuntimeError(
            "Something is wrong - tcool_minmix should be negative")

    # solve: t_shear * alpha == tcool_minmix
    #     -> (R_cl / v_wind) * alpha == tcool_minmix
    #     -> R_cl == tcool_minmix * v_wind / alpha
    return (np.abs(tcool_minmix) * vwind / alpha).to('pc')
