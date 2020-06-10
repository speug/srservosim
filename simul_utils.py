import numpy as np
from scipy.signal import peak_widths
from scipy.stats import binom


def lineshape(delta, t, state_prep=False):
    """Calculate theoretical lineshape from ideal Rabi flopping.

    Parameters
    ----------
    delta : scalar or np.array
        Detunings for which calculate the lineshape value
    t : scalar or np.array
        Pulse length(s)
    state_prep : bool, optional
        Flag for whether the ions have been state-prepped, yielding
        a different resulting max probability.
    Returns
    -------
    Lineshape array as probabilities for state change as a function
    of detuning"""
    sinc_arg = (0.5 * np.sqrt(np.pi ** 2 + np.square(delta * (2 * np.pi))
                * np.square(t)))
    out = (np.pi/2.)**2 * np.square(np.divide(np.sin(sinc_arg), sinc_arg))
    if not state_prep:
        out = out / 2.
    return out


def FWHM(lineshape, delta):
    """Calculate FWMH from data."""
    width = peak_widths(lineshape, [np.argmax(lineshape)], rel_height=0.5)
    out = delta[int(np.rint(width[3][0]))] - delta[int(np.rint(width[2][0]))]
    return out


def k_p(lineshape_func, delta, t, n=1000, state_prep=False):
    """"Calculate k_p for optimal gain."""
    fwhm = FWHM(lineshape_func(delta, t, state_prep), delta)
    d_R = np.linspace(start=delta[0], stop=delta[-1] - fwhm, num=n)
    d_B = d_R + fwhm
    d_C = d_R + fwhm / 2.
    p_R = lineshape_func(d_R, t, state_prep)
    p_B = lineshape_func(d_B, t, state_prep)
    k = p_B - p_R
    d_k = np.diff(k) / (d_C[1] - d_C[0])
    origin = d_k[np.argmin(np.abs(d_C))]
    return origin, k, d_C


def allan_deviation(p_X, kp, eta0, T_c, tau):
    """Calculate theoretical Allan deviation."""
    sigma = (-2. * np.sqrt(p_X * (1 - p_X)) * (1 / (kp * eta0))
             * np.sqrt(T_c / tau))
    return sigma


def sampling_cycle(f0,
                   T_s,
                   n_m,
                   theoretical_delta,
                   tau_pi=6e-3,
                   laser_drift=20e-6,
                   state_prep=False):
    """Perform one cycle of sampling from data, including laser cavity
    drift.
    Parameters
    ----------
    f0 : scalar
        Initial detuning of center
    T_s : scalar
        Total sampling time (single sample time * samples)
    n_m : scalar
        Number of samples to take
    theoretical_delta: scalar
        Theoretical detuning of HWHM
    tau_pi : optional
        Pulse length in seconds
    laser_drift : optional
        Laser cavity drift in Hz/s
    state_prep : bool
        State preparation flag
    """
    time_step = T_s / n_m
    if laser_drift == 0.:
        detunings = np.ones(n_m) * f0 + theoretical_delta
    else:
        detunings = np.arange(start=f0,
                              stop=f0 + n_m * laser_drift * time_step,
                              step=laser_drift * time_step)
        detunings += theoretical_delta
    # quantum jump p from theory
    jump_probabilities = lineshape(detunings, tau_pi, state_prep)
    # draws from binomial
    measured_results = binom.rvs(n=1, p=jump_probabilities)
    p_X = np.sum(measured_results) / n_m
    center_f = detunings[-1] - theoretical_delta
    total_drift = n_m * laser_drift * time_step
    return p_X, center_f, total_drift
