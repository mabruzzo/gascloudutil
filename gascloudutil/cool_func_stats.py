from warnings import warn

import numpy as np
from scipy.optimize import (
    minimize_scalar, root_scalar
)

def _get_search_vals(step, lo, hi, start = None, log10step = False):
    # build a sorted array containing integer numbers of steps out from start
    # toward lo and hi.
    # -> We ensure that out[0] == lo and out[-1] == hi
    # -> consequently, the leftmost and rightmost steps may be smaller than step

    _lo, _hi, _start = lo, hi, start
    if log10step:
        lo, hi = np.log10(lo), np.log10(hi)
        if start is not None: start = np.log10(start)

    assert step > 0
    if start is None: # treat hi as start
        nsteps_L = int(np.ceil((hi - lo)/step))
        nsteps_R = 0
        tmp = hi + np.arange(-nsteps_L, nsteps_R+1) * step
    else:
        nsteps_L = int(np.ceil((start - lo)/step))
        nsteps_R = int(np.ceil((hi - start)/step))
        tmp = start + np.arange(-nsteps_L, nsteps_R+1) * step

    out = 10.0**tmp if log10step else tmp

    out[0] = _lo
    out[-1] = _hi
    if _start is not None: out[nsteps_L] = _start
    search_start_index = nsteps_L
    return search_start_index, out

def _report_zeroslope(s0, s1, inv_tcool_val):
    _msg_template = (
        f"Encountered an interval of {{direction}} between parameter "
        f"values of {s0!r} and {s1!r} where the reciprocal of the "
        f"{{direction}} timescale appears to have a constant value of "
        f"{inv_tcool_val}.\n\n{{cust}}\n\n"
        f"If this is unexpected, consider reducing the brute_step arg"
    )
    if inv_tcool_vals[i0] > 0:
        custom_part = (
            "Since this only affects heating, the function will carry on. But, "
            "of the extrema when heating dominates may not be well behaved")
        warn(_msg_template.format(direction = "heating", cust = custom_part))
    else:
        raise RuntimeError(
            _msg_template.format(direction = "cooling", cust = "ABORTING NOW"))

def _find_important_locations(inv_tcool, s_bounds, brute_step,
                              is_log10_brute_step, s_brute_start = None,
                              maxiter_each = 500, s_rel_tol = 1e-8,
                              skip_unstable_equilibrium = True):
    # I thought I could be a lot more clever here... Unfortunately you can't
    # if you want to get a general solution....

    # prologue:
    s_min, s_max = s_bounds
    assert s_min < s_max
    if s_brute_start is not None:
        assert s_min < s_brute_start < s_max

    # in principle, we can get away with doing less work!
    # -> let `my_s = s_max if s_brute_start is None else s_brute_start`
    # -> if tcool(my_s) >= 0, then we can't save time
    # -> otherwise, the only thermal-equilibrium point at s < my_s is the
    #    one with the max value. Let's call that s0 (if it exists). We also
    #    care about all thermal-equilibrium points above my_s.
    # -> We then only care about local extrema between s0 and s_max. If s0
    #    doesn't exist, then we can't save any time

    # STEP 1: construct the grid of s values to use for brute-force search
    search_start_index, brute_search_s_vals = _get_search_vals(
        step = brute_step, lo = s_min, hi = s_max, start = s_brute_start,
        log10step = is_log10_brute_step)
    assert np.all(brute_search_s_vals[:-1] < brute_search_s_vals[1:])
    if brute_search_s_vals.size <= 2:
        raise ValueError("The stepsize is too small! The only brute-force "
                         "grid-points we will search have s values of: "
                         f"{brute_search_s_vals!r}. We need to search at "
                         "least 3 s values to find local minima/maxima")

    # STEP 1a: evaluate inverse cooling time at all brute-force grid points
    inv_tcool_vals = inv_tcool(brute_search_s_vals)

    # STEP 2: find the roots that correspond to temperature equilibrium)
    equilibrium_s_vals = []
    sgn = np.sign(inv_tcool_vals)
    for i in np.where(np.abs(sgn[:-1] + sgn[1:]) <= 1)[0]:
        bracket = [brute_search_s_vals[i], brute_search_s_vals[i+1]]

        if (inv_tcool_vals[i] != 0.0) and (inv_tcool_vals[i+1] != 0.0):
            if ((inv_tcool_vals[i] < 0.0 < inv_tcool_vals[i+1]) and
                skip_unstable_equilibrium):
                # Since cooling occurs at lower thermal energy and heating
                # occurs at higher thermal energy, this is unstable
                continue
            root_rslt = root_scalar(
                f = inv_tcool, args = (), method='bisect', bracket = bracket,
                rtol = s_rel_tol, maxiter = maxiter_each
            )
            if not root_rslt.converged:
                raise RuntimeError("Not converged. Termination flag:\n " +
                                   "   " + root_rslt.flag)
            equilibrium_s_vals.append(root_rslt.root)
        else:
        
            is_first_pair,is_final_pair = (i==0), ((i+2) == inv_tcool_vals.size)
            if (inv_tcool_vals[i] == 0) and (inv_tcool_vals[i+1] == 0):
                warn("encountered 2 contiguous brute-force grid points, with "
                     f"parameter vals of {bracket}, where heating/cooling "
                     "cancel. Doing our best to handle this case (but we may "
                     "not handle it properly). If this is unintentional, "
                     "consider changing brute_step")
                equilibrium_s_vals.append(brute_search_s_vals[i])
            elif inv_tcool_vals[i] == 0:
                if ( is_first_pair and inv_tcool_vals[i] == 0):
                    # ignore any stability concerns
                    equilibrium_s_vals.append(brute_search_s_vals[i])
                elif ( (inv_tcool_vals[i-1] > 0.0 > inv_tcool_vals[i+1]) or
                       (not skip_unstable_equilibrium) ):
                    equilibrium_s_vals.append(brute_search_s_vals[i])
            else: # inv_tcool_vals[i+1] == 0
                if is_final_pair:
                    # skip stability concerns
                    equilibrium_s_vals.append(brute_search_s_vals[i+1])
                else:
                    pass # we consider appending brute_search_s_vals[i+1] to
                         # root_s_vals in a future iteration

    # Step 3: find the local extrema
    # -> in principle, we could use the roots to find better starting
    #    guesses, but that's more trouble than its worth at the moment
    extrema_s_vals = []

    neg_inv_tcool = lambda s: -1 * inv_tcool(s) # for finding maxima

    # A bracket-ed region of 3 points is needed to locate an extrema
    sgn_diff = np.sign(np.diff(inv_tcool_vals))
    idx_potential_bracket_center = np.where(
        np.abs(sgn_diff[:-1] + sgn_diff[1:]) <= 1 )[0] + 1

    for i in idx_potential_bracket_center:
        i_left, i_center, i_right = (i-1), i, (i+1)
        s_left, s_center, s_right = brute_search_s_vals[i-1:i+2]

        if ( (inv_tcool_vals[i_left] != inv_tcool_vals[i_center]) and
             (inv_tcool_vals[i_center] != inv_tcool_vals[i_right]) ):
            if inv_tcool_vals[i_center] > inv_tcool_vals[i_left]:
                f = neg_inv_tcool # we need to search for a maximum
            else:
                f = inv_tcool # we need to search for a minima
            minimize_rslt = minimize_scalar(
                fun = f, bracket = (s_left, s_center, s_right),
                method = 'golden', tol = s_rel_tol,
                options = {'maxiter' : maxiter_each}
            )
            if not minimize_rslt.success:
                raise RuntimeError(
                    "Encountered a problem during call to minimize_scalar:\n"
                    + "  " + minimize_rslt.message)
            extrema_s_vals.append(minimize_rslt.x)
        else:
            # we make a point of reporting the zero-slope. We may abort
            # with an error during reporting. If we don't abort just
            # continue onwards
            if (inv_tcool_vals[i_left] == inv_tcool_vals[i_center]):
                _report_zeroslope(s0 = s_left, s1 = s_center,
                                  inv_tcool_val = inv_tcool_vals[i_center])
            elif (i_right+1 == inv_tcool_vals.size):
                _report_zeroslope(s0 = s_left, s1 = s_center,
                                  inv_tcool_val = inv_tcool_vals[i_center])
            else:
                pass # we will report this interval in the next loop

    # some extra bookkeeping
    if (s_min not in extrema_s_vals) and (s_min not in equilibrium_s_vals):
        extrema_s_vals.append(s_min)
    if (s_max not in extrema_s_vals) and (s_max not in equilibrium_s_vals):
        extrema_s_vals.append(s_max)

    # now let's format outputs
    _s_vals, _is_thermal_eq = list(zip(*sorted(
        [(s, True) for s in equilibrium_s_vals] +
        [(s, False) for s in extrema_s_vals]
    )))
    return np.array(_s_vals), np.array(_is_thermal_eq)
        

def find_special_locations(parametric_curve, *, s_bounds, brute_step,
                           is_log10_brute_step, cooling_curve,
                           maxiter_each = 500, s_rel_tol = 1e-8,
                           skip_unstable_equilibrium = False):
    """
    Find the "special points" for some cooling curve.

    In more detail, these points correspond to the location where:
      - gas is in thermal equilibrium. In other words, heating/cooling has no
        effect (heating cancels out cooling).
      - local and global minima and maxima of the cooling time, `tcool`.
        Note that `tcool = eint_dens / ((d/dt) eint_dens)`, where `eint_dens`
        is the internal energy density (aka thermal energy density).

    The function searches for these "special" points along a 1D "curve" through
    phase space. For now, phase space is simply parameterized by mass-density
    and specific thermal energy. In the future, one could imagine parameterizing
    phase-space in terms of additional quantities such as metallicity or
    some description of background heating (e.g. how self-sheilded the gas is).

    This "curve" is specified by the `parametric_curve` argument, which maps
    the parameter ``s`` to coordinates in this phase-space along this "curve".
    More details are provided below. This "curve" is usually an isobar.

    Parameters
    ----------
    parametric_curve: callable
        This is a callable accepting a single argument `s`. It represents a
        1D curve through phase space. In more detail:
          - this callable should return a tuple of 2 floats. The first entry
            is treated as the mass-density. The second entry is the specific
            internal energy. Both are implicitly assumed to have cgs units
            (so "g/cm**3" and "cm**2/s**2", respectively)
          - Both mass-density and specific-internal-energy should be continuous
            functions of `s`. In particular, specific-internal-energy should be
            monotonically non-decreasing function of `s`.
          - This "curve" is usually an isobar and `s` is commonly just the
            specific internal energy.
    s_bounds : tuple of 2 floats
        The minimum and maximum values `s` that can be passed to 
        `parametric_curve`.
    brute_step : float
        The step-size of s (to use for the initial brute-force search)
    is_log10_brute_step: bool
        Whether `brute_step` is specified in linear or logarithmic space
    cooling_curve
        This represents the cooling function. It must have a function called
        calculated_tcool, that expects each function returned by a call to
        `parametric_curve` to be forwarded on, in the same order.
    maxiter_each: int
        The maximum number of iterations used in each rootfinding call and each 
        call to identify extrema locations
    s_rel_tol : float
        The relative tolerance used to dictate when each rootfinding/extrema 
        finding call terminates.


    Returns
    -------
    special_s_vals: np.ndarray
        sorted 1D array specifying locations along the curve where "special"
        points occur
    is_thermal_eq: np.ndarray
        1D array of booleans of the same shape `special_s_vals`. Values that
        are `True`, indicate that the corresponding entry in `special_s_vals`
        specifies thermodynamic equilibrium.

    Notes
    -----
    Currently we use a fairly exhaustive approach. In principle we don't need
    to be so exhaustive.
    """

    def inv_tcool(s):
        vals = parametric_curve(s)
        return 1.0/cooling_curve.calculate_tcool(*vals)

    return _find_important_locations(
        inv_tcool = inv_tcool, s_bounds = s_bounds, brute_step = brute_step,
        is_log10_brute_step = is_log10_brute_step, s_brute_start = None,
        maxiter_each = maxiter_each, s_rel_tol = s_rel_tol,
        skip_unstable_equilibrium = skip_unstable_equilibrium)

def find_special_locations_isobar(pthermal_div_gm1, *, specific_eint_bounds,
                                  brute_step, is_log10_brute_step, cooling_eos,
                                  maxiter_each = 500, s_rel_tol = 1e-8,
                                  skip_unstable_equilibrium = False):
    """
    A convenience function that wraps find_special_locations for the simple
    case where gas is assumed to be distributed along an isobar

    TODO: make it possible to specify temperature, rather than thermal-energy

    Parameters
    ----------
    pthermal_div_gm1 : float
        The thermal pressure divided by the `(gamma - 1)`. As the specific
        thermal energy is varied, this is held constant
    eint_bounds : float
        The bounds of the specific internal energy (aka thermal energy). This 
        should be specified in cgs units (i.e. cm**2/s**2)
    brute_step: float
        Specifies the step-size to use to vary the specific internal energy
        while doing a brute force search to try to bracket the roots and the
        extrema
    is_log10_brute_step : bool
        Specifies whether brute_step is in log-space or linear space. In the 
        latter case, then `brute_step` should have cgs units
    cooling_eos
        An eos object is used to compute the cooling time.

    Returns
    -------
    special_eint_vals: np.ndarray
        sorted 1D array specifying locations along the curve where "special"
        points occur. These are values of the specific internal energy.
    is_thermal_eq: np.ndarray
        1D array of booleans of the same shape `special_e_vals`. Values that
        are `True`, indicate that the corresponding entry in `special_e_vals`
        specifies thermodynamic equilibrium.
    """

    def isobar_parametric_curve(specific_eint):
        # the parametric curve is parameterized in terms of the specific
        # internal energy
        rho = pthermal_div_gm1 / specific_eint
        return rho, specific_eint

    return find_special_locations(
        parametric_curve = isobar_parametric_curve,
        s_bounds = specific_eint_bounds, cooling_curve = cooling_eos,
        brute_step = brute_step,
        is_log10_brute_step = is_log10_brute_step,
        maxiter_each = maxiter_each, s_rel_tol = 1e-8,
        skip_unstable_equilibrium = skip_unstable_equilibrium)

if __name__ == '__main__':
    def inv_tcool(x):
        return x**3 - 3 * x**2 - 144 * x + 432

    rslt = _find_important_locations(
        inv_tcool, s_bounds = (-15,12), brute_step = 0.2,
        is_log10_brute_step = False, s_brute_start = None,
        maxiter_each = 500, s_rel_tol = 1e-8,
        skip_unstable_equilibrium = False)
    print(rslt)
