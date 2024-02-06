from warnings import warn

import numpy as np
from scipy.optimize import (
    minimize_scalar, root_scalar
)

from enum import Enum
class EnergyFlow(Enum):
    COOL = -1
    NEUTRAL = 0
    HEAT = 1

class HeatCoolIntervals:
    def __init__(self, intervals, energy_flow):
        self.intervals = np.array(intervals)
        self.energy_flow = energy_flow
        assert len(intervals) > 1
        assert len(energy_flow) == (self.intervals.size - 1)

    def __repr__(self):
        return (f"HeatCoolIntervals(intervals = {self.intervals!r}, " +
                f"energy_flow = {self.energy_flow!r}")

    @property
    def num_intervals(self): return self.intervals.size - 1

    def identify_interval(self, x):
        interval_index = np.array(np.digitize(x, self.intervals))
        if 0 in interval_index:
            raise ValueError('1+ specified val lies outside of the intervals')
        elif interval_index.max() > self.num_intervals:
            # it's ok to be equal to num_intervals right now, since we
            # will subtract 1 in a moment
            raise ValueError('1+ specified val lies outside of the intervals')
        return interval_index - 1

    def get_energy_flow(self, index):
        index = np.asanyarray(index)
        assert np.all(np.logical_and(index>=0, index<self.num_intervals))
        if np.ndim(index) == 0:
            return self.energy_flow[index]
        elif np.ndim(index):
            return [self.energy_flow[int(i)] for i in index]
        else:
            raise ValueError("invalid choice...")

def _max_ind_in_constval_segment(arr, start_ind):
    # Imagine the elements of arr are grouped into a series of segments. In a
    # given segment, all contiguous elements have the same element.
    # -> this function returns the maximum index in the segment containing
    #    start_ind
    segment_val = arr[start_ind]
    for i in range(start_ind + 1, len(arr)):
        if arr[i] != segment_val:
            return i - 1
    return len(arr) - 1


def zerointerval_transition(f, bracket, rtol, maxiter):
    """
    Local search for the transition-location for a scalar function between
    a region where it evaluates to all positive (or all negative) values
    and a region where it always evaluates to 0
    """

    # we might be able to be more clever using first derivatives. The first
    # derivative will have a discontinuity at the exact transition point, but
    # we may be able to use it to help us find the point faster...

    if len(bracket) != 2:
        raise ValueError(f"{bracket!r} isn't a valid bracket: it must be a 2 "
                         "element sequence")
    elif (bracket[0] >= bracket[1]):
        raise ValueError(f"{bracket!r} isn't a valid bracket: bracket[1] must "
                         "exceed bracket[0]")
    elif any(e <= 0.0 for e in bracket):
        # this is a requirement as long as we just make use of rtol (and don't
        # use atol)
        raise ValueError(f"{bracket!r} isn't a valid bracket: both elements "
                         "must currently be positive")
    elif not (0.0 < rtol < 1.0):
        raise ValueError("rtol must lie between 0.0 and 1.0")
    elif (maxiter <= 0) or (int(maxiter) != maxiter):
        raise ValueError("maxiter must be a positive integer")

    l,r = bracket
    f_l, f_r = f(l), f(r)
    if (f_l == 0.0 and f_r != 0.0):
        # return largest value that corresponds to a value of 0 (to do this,
        # always keep the zero-region on the left)

        # ATTEMPT AT CLEVERNESS (try to avoid returning bracket[0])
        small_l_offset = min(r, l * (1+rtol))
        if f(small_l_offset) == 0.0: 
            l = small_l_offset
        else:
            r = small_l_offset

        for count in range(maxiter):
            midpoint = 0.5 * (l + r)
            if (l * (1+rtol)) > midpoint:
                break
            elif f(midpoint) == 0.0:
                l = midpoint
            else:
                r = midpoint
        return l
    elif (f_l != 0.0 and f_r == 0.0):
        # return smallest value that corresponds to a value of 0 (to do this,
        # always keep the zero-region on the right)

        # ATTEMPT AT CLEVERNESS (try to avoid returning bracket[1])
        small_r_offset = max(l, r * (1-rtol))
        if f(small_r_offset) == 0.0: 
            r = small_r_offset
        else:
            l = small_r_offset

        for count in range(maxiter):
            midpoint = 0.5 * (l + r)
            if (r * (1-rtol)) < midpoint:
                break
            elif f(midpoint) == 0.0:
                r = midpoint
            else:
                l = midpoint
        return r
    else:
        raise ValueError("Either f(bracket[0]) or f(bracket[1]) MUST evaluate "
                         "to 0.0 (BUT NOT BOTH)")

def get_heat_cool_intervals(invtcool_fn, s_grid, invtcool_grid = None,
                            maxiter_each = 500, s_rel_tol = 1e-8):
    """
    This function identifies the intervals on which a cooling function has net
    heating, net cooling, or produces no net energy change.

    In more detail, we consider the cooling function when it is parameterized,
    by a single scalar parameter s. This uses a brute-force technique: it
    performs the search for these intervals based on a grid of s values

    Parameters
    ----------
    invtcool_fn : callable
        A function that returns the reciprical of the cooling times scale at a
        given value of s
    s_grid : np.ndarray
        A 1D array of s values that is used to control the search for these
        intervals. This must contain at least 3 elements and the elements must
        monotonically increase.
    invtcool_grid : np.ndarray, Optional
        Optional parameter used to specify the precomputed values of invtcool_fn
        at each s value in s_grid
    maxiter_each: int
        The maximum number of iterations used in each rootfinding call and each 
        call to identify extrema locations
    s_rel_tol : float
        The relative tolerance used to dictate when each rootfinding/extrema 
        finding call terminates.
    """
    assert 0 < s_rel_tol < 1
    assert isinstance(maxiter_each, int) and (maxiter_each > 0)
    assert (s_grid.ndim == 1) and (s_grid.size >= 3)
    if invtcool_grid is None:
        invtcool_grid = invtcool_fn(s_grid)
    assert (s_grid.shape == invtcool_grid.shape)
    sign_grid = np.sign(invtcool_grid)

    def left_s_val(s): return s * (1.0 - s_rel_tol)
    def right_s_val(s): return s * (1.0 + s_rel_tol)

    commonkw = {'f' : invtcool_fn, 'rtol' : s_rel_tol, 'maxiter' : maxiter_each}

    # we are going to fill up the following 2 lists with interval data.
    edges = [s_grid[0]]  # <-- will have N+1 items for N intervals
    interval_signs = []  # <-- will have N items for N intervals

    grid_length = s_grid.size
    # Each time we enter the while loop, we are considering a new interval.
    # - in the body of the while loop, we determine the extent of the interval
    #   and record its properties
    max_grid_ind_curinterval = -1
    while (max_grid_ind_curinterval + 1) != grid_length:
        # the current interval extends from s = edges[-1]

        # First, find the min element from s_grid contained in cur interval
        min_grid_ind_curinterval = max_grid_ind_curinterval + 1
        # invariant: s_grid[min_grid_ind_curinterval] >= edges[-1]

        # Next, find the max element from s_grid contained in cur interval
        max_grid_ind_curinterval = _max_ind_in_constval_segment(
            sign_grid, start_ind = min_grid_ind_curinterval)

        # Now, identify the sign of the current interval:
        cur_interval_sign = sign_grid[min_grid_ind_curinterval]

        # Finally, we do the work of finding the precise end point and we
        # record things
        transition_bracket = (
            s_grid[max_grid_ind_curinterval:(max_grid_ind_curinterval+2)])
        if cur_interval_sign != 0.0:
            interval_signs.append(cur_interval_sign)
            if (max_grid_ind_curinterval + 1) == grid_length:
                edges.append(s_grid[max_grid_ind_curinterval])
            elif sign_grid[max_grid_ind_curinterval + 1] != 0.0:
                root_rslt = root_scalar(bracket = transition_bracket,
                                        method='bisect', **commonkw)
                if not root_rslt.converged:
                    raise RuntimeError(
                        f"Not converged. Termination flag: {root_rslt.flag!r}")
                edges.append(root_rslt.root)
            else: # sign_grid[max_grid_ind_curinterval + 1] == 0.0
                # POTENTIAL CHALLENGE: s_grid[max_grid_ind_curinterval + 1] may
                # exactly coincide with an isolated root of invtcool_fn OR it
                # may coincide with an extended interval of zeros.
                # -> right now, we only care how this affects the right edge of
                #    current interval
                edges.append(zerointerval_transition(
                    bracket = transition_bracket, **commonkw))
        else: # cur_interval_sign == 0.0
            # Check whether the current "interval" is just an isolated point
            # or is part of an extended region...
            min_s_grid_val = s_grid[min_grid_ind_curinterval]
            sLeft = max(left_s_val(min_s_grid_val), edges[-1])
            max_s_grid_val = s_grid[max_grid_ind_curinterval]
            sRight = min(right_s_val(max_s_grid_val), s_grid[-1])
            is_isolated_point = (
                # to be isolated, require:
                # -> that there's only 1 point in s_grid in cur "interval" AND
                (min_grid_ind_curinterval == max_grid_ind_curinterval) and
                # -> that the interval does NOT extend to the left AND
                ((sLeft == min_s_grid_val) or (invtcool_fn(sLeft) != 0.0)) and
                # -> that the interval does NOT extend to the right
                ((sRight == min_s_grid_val) or (invtcool_fn(sRight) != 0.0))
            )
                
            if is_isolated_point:
                # The current "interval" corresponds to a value in "s_grid"
                # that just happens to __EXACTLY__ correspond to a isolated
                # number where invtcool_fn evaluates to 0. This corresponds to
                # the point where invtcool_fn either:
                # - transitions between a positive & negative interval
                # - has a local extrema that exactly evaluates to 0
                #
                # While one could think of this as infinitely thin interval,
                # then you would have to treat all root-locations that way.
                # so we effectively skip this interval
                pass # by doing nothing, we skip this interval!
            else:
                # record the interval's sign and right edge
                interval_signs.append(0.0)
                if (max_grid_ind_curinterval + 1) == grid_length:
                    assert sRight == max_s_grid_val # sanity check!
                    edges.append(max_s_grid_val)
                else:
                    edges.append(zerointerval_transition(
                        bracket = transition_bracket, **commonkw))

    interval_vals = [EnergyFlow(e) for e in interval_signs]
    return HeatCoolIntervals(intervals = edges, energy_flow = interval_vals)


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

def _report_zeroslope(s0, s1, inv_tcool_val, heat_cool_intervals):
    if inv_tcool_val == 0.0:
        intervals = heat_cool_intervals.identify_interval(np.array([s0,s1]))
        eflow = heat_cool_intervals.get_energy_flow(intervals[0])
        if ((intervals[0] == intervals[1]) and (eflow == EnergyFlow.NEUTRAL)):
            # if the zero-slope is in a known region without heat-flow, no need
            # to report it...
            return

    _msg_template = (
        f"Encountered an interval of {{direction}} between parameter "
        f"values of {s0!r} and {s1!r} where the reciprocal of the "
        f"{{direction}} timescale appears to have a constant value of "
        f"{inv_tcool_val}.\n\n{{cust}}\n\n"
        f"If this is unexpected, consider reducing the brute_step arg"
    )
    if inv_tcool_val >= 0:
        custom_part = (
            "Since this zero-slope region doesn't affect cooling, the function "
            "will carry on. But, the locations of the extrema when heating "
            "dominates may not be well behaved")
        warn(_msg_template.format(direction = "heating", cust = custom_part))
    else:
        raise RuntimeError(
            _msg_template.format(direction = "cooling", cust = "ABORTING NOW"))

def _find_important_locations(inv_tcool, s_bounds, brute_step,
                              is_log10_brute_step, s_brute_start = None,
                              maxiter_each = 500, s_rel_tol = 1e-8):
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

    # Step 2: find the intervals of heating/cooling
    heat_cool_intervals = get_heat_cool_intervals(
        invtcool_fn = inv_tcool, s_grid = brute_search_s_vals,
        invtcool_grid = inv_tcool_vals,
        maxiter_each = maxiter_each, s_rel_tol = s_rel_tol)

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

        # Note: In theory, it would be nice if we updated the h_c_intervals on
        #       the offchance that we encountered an extrema that shifts between
        #       heating and cooling (or just a single point where
        #       heating/cooling cancel)

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
                                  inv_tcool_val = inv_tcool_vals[i_center],
                                  heat_cool_intervals = heat_cool_intervals)
            elif (i_right+1 == inv_tcool_vals.size):
                _report_zeroslope(s0 = s_center, s1 = s_right,
                                  inv_tcool_val = inv_tcool_vals[i_center],
                                  heat_cool_intervals = heat_cool_intervals)
            else:
                pass # we will report this interval in the next loop

    # some extra bookkeeping
    if (s_min not in extrema_s_vals): extrema_s_vals.append(s_min)
    if (s_max not in extrema_s_vals): extrema_s_vals.append(s_max)

    return heat_cool_intervals, np.array(sorted(extrema_s_vals))
        

def find_special_locations(parametric_curve, *, s_bounds, brute_step,
                           is_log10_brute_step, cooling_curve,
                           maxiter_each = 500, s_rel_tol = 1e-8):
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
    heat_cool_intervals : HeatCoolIntervals
        object representing the heating and cooling intervals
    critical_s_vals: np.ndarray
        sorted 1D array specifying locations along the curve where extrema
        may occur

    Notes
    -----
    Currently we use a fairly exhaustive approach. In principle we don't need
    to be so exhaustive.
    """
    if hasattr(cooling_curve, 'calculate_tcool_CGS'):
        def inv_tcool(s):
            vals = parametric_curve(s)
            return 1.0/cooling_curve.calculate_tcool_CGS(*vals)
    elif hasattr(cooling_curve, 'calculate_tcool'):
        warn("assuming that parametric_curve returns mass_density followed by "
             "specific internal energy")
        import unyt
        def inv_tcool(s):
            rho_cgs, eint_cgs = parametric_curve(s)
            tcool = cooling_curve.calculate_tcool(
                unyt.unyt_array(rho_cgs,'g/cm**3'),
                unyt.unyt_array(eint_cgs,'cm**2/s**2'))
            return 1.0/(tcool.to('s').ndview)
    else:
        raise TypeError("invalid cooling_curve argument")

    return _find_important_locations(
        inv_tcool = inv_tcool, s_bounds = s_bounds, brute_step = brute_step,
        is_log10_brute_step = is_log10_brute_step, s_brute_start = None,
        maxiter_each = maxiter_each, s_rel_tol = s_rel_tol)

def find_special_locations_isobar(pthermal_div_gm1, *, specific_eint_bounds,
                                  brute_step, is_log10_brute_step, cooling_eos,
                                  maxiter_each = 500, s_rel_tol = 1e-8):
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
    heat_cool_intervals : HeatCoolIntervals
        object representing the heating and cooling intervals. The intervals
        are indexed in terms of the specific internal energy
    critical_eint_vals: np.ndarray
        sorted 1D array specifying eint along the isobar where extrema
        may occur
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
        maxiter_each = maxiter_each, s_rel_tol = 1e-8)

if __name__ == '__main__':
    def inv_tcool(x):
        return x**3 - 3 * x**2 - 144 * x + 432

    rslt = _find_important_locations(
        inv_tcool, s_bounds = (-15,12), brute_step = 0.2,
        is_log10_brute_step = False, s_brute_start = None,
        maxiter_each = 500, s_rel_tol = 1e-8)
    print(rslt)
