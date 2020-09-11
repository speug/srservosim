import numpy as np
from scipy.signal import peak_widths
from scipy.stats import binom
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def linear_zeeman(B, zeeman_coeff=5.6):
    return B * zeeman_coeff


def lineshape(delta, t, state_prep=False, center=0.):
    """Calculate theoretical lineshape from ideal Rabi flopping.

    Parameters
    ----------
    delta : scalar or np.array
        Detunings for which calculate the lineshape value
    t : scalar or np.array
        Pulse length(s)
    state_prep : bool, optional
        Flag for whether the ions have been state-prepped, yielding
        a different resulting max probability. Default: False
    center : scalar, optional
        Offset of the central wavelength. Default: 0
    Returns
    -------
    Lineshape array as probabilities for state change as a function
    of detuning"""
    sinc_arg = (0.5 * np.sqrt(np.pi ** 2
                              + np.square((delta-center) * (2 * np.pi))
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


def k_p(delta, t, n=1000, state_prep=False, center=0.):
    """"Calculate k_p for optimal gain."""
    fwhm = FWHM(lineshape(delta, t, state_prep, center), delta)
    d_R = np.linspace(start=delta[0], stop=delta[-1] - fwhm, num=n)
    d_B = d_R + fwhm
    d_C = d_R + fwhm / 2.
    p_R = lineshape(d_R, t, state_prep, center)
    p_B = lineshape(d_B, t, state_prep, center)
    k = p_B - p_R
    d_k = np.diff(k) / (d_C[1] - d_C[0])
    origin = d_k[np.argmin(np.abs(d_C-center))]
    return origin, k, d_C


def allan_deviation(p_X, kp, eta0, T_c, tau):
    """Calculate theoretical Allan deviation."""
    sigma = (-2. * np.sqrt(p_X * (1 - p_X)) * (1 / (kp * eta0))
             * np.sqrt(T_c / tau))
    return sigma


def sampled_lineshape(lineshape_func,
                      delta,
                      tau_pi=6e-3,
                      linecenter=0.,
                      state_prep=False,
                      samples_per_point=100):
    """Get approximate lineshape via sampling.
    Parameters
    ----------
    lineshape_func : function
        Theoretical lineshape function. Must have form
        f(delta, tau_pi, state_prep, omega_0).¨
    delta : scalar or array
        Frequencies at which to sample.
    tau_pi : float, 6 us by default
    linecenter : float
        Frequency of clock transition.
    state_prep : boolean, False by default
        State preparation flag
    samples_per_point : float
        How many samples to take from binomial distribution to model probing
        cycles.
    Returns
    -------
    The sampled lineshape."""

    # quantum jump p from theory
    jump_probabilities = lineshape_func(delta, tau_pi, state_prep, linecenter)
    # draws from binomial
    measured_results = binom.rvs(n=samples_per_point, p=jump_probabilities)
    sample_shape = measured_results / samples_per_point
    return sample_shape


def sampling_cycle(f0,
                   T_s,
                   n_m,
                   theoretical_delta,
                   linecenter=0.,
                   tau_pi=6e-3,
                   laser_drift=20e-6,
                   state_prep=False):
    """Perform one cycle of sampling from data, including laser cavity
    drift.
    Parameters
    ----------
    f0 : scalar or np.array
        Initial laser detuning
    T_s : scalar
        Total sampling time (single sample time * samples)
    n_m : scalar
        Number of samples
    theoretical_delta: scalar or np.array
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
        detunings = np.multiply(np.ones(n_m), f0) + theoretical_delta
    else:
        detunings = np.linspace(start=f0 + laser_drift * time_step,
                                stop=f0 + (n_m * laser_drift * time_step),
                                num=n_m)
        detunings += theoretical_delta
    # quantum jump p from theory
    jump_probabilities = lineshape(detunings, tau_pi, state_prep, linecenter)
    # draws from binomial
    measured_results = binom.rvs(n=1, p=jump_probabilities)
    p_X = np.sum(measured_results, axis=0) / n_m
    center_f = detunings[-1] - theoretical_delta
    total_drift = f0 + T_s * laser_drift
    return p_X, center_f, total_drift


def sample_initial_values(center,
                          tau_pi,
                          sampling='adaptive',
                          d_sample=None,
                          state_prep=False,
                          N_s=100,
                          p0=None,
                          plot=False):
    """Set initial linecenter values"""
    if sampling == 'adaptive':
        fourier_limit = 0.8 / tau_pi
        d_sample = np.linspace(center-2.*fourier_limit,
                               center+2.*fourier_limit, 20)
    elif sampling != 'manual' or d_sample is None:
        raise ValueError('Sampling must be adaptive or manual.' +
                         ' In the latter case, ' +
                         'user must provide the sampling range.')
    l_sample = sampled_lineshape(lineshape,
                                 d_sample,
                                 tau_pi=tau_pi,
                                 state_prep=state_prep,
                                 samples_per_point=N_s,
                                 linecenter=center)
    # fit to sampled to simulate real measurement
    if p0 is None:
        p0 = [tau_pi, center]
    popt, pcov = curve_fit(lambda d, t, c: lineshape(d, t, False, c), d_sample,
                           l_sample, p0=p0)
    d = np.linspace(center-1000, center+1000, 10000)
    fit_shape = lineshape(d, popt[0], state_prep, popt[1])
    # calculate FWHM
    width = peak_widths(fit_shape, [np.argmax(fit_shape)], rel_height=0.5)
    fwhm = d[int(width[3][0])] - d[int(width[2][0])]
    # save results to dict
    out = {'fit_center': popt[1],
           'fit_tau': popt[0],
           'fit_FWHM': fwhm}
    if plot:
        f, ax = plt.subplots()
        ax.scatter(d_sample, l_sample, label='samples', marker='x')
        ax.plot(d, fit_shape, label='fit')
        ax.legend()
        out['plot'] = ax
    return out


def BC_servo_gains(centers, tau_pis, fwhms):
    # B servo
    eta_B = np.mean(np.abs(centers))
    seps, sep_step = np.linspace(0, 2*eta_B, 10000, retstep=True)
    pB_pB = lineshape(fwhms[1]/2.+seps, tau_pis[1], center=centers[1])
    pB_mB = lineshape(fwhms[0]/2.-seps, tau_pis[0], center=centers[0])
    pB_pR = lineshape(-fwhms[1]/2.+seps, tau_pis[1], center=centers[1])
    pB_mR = lineshape(-fwhms[0]/2.-seps, tau_pis[0], center=centers[0])
    p_sep1 = pB_pB + pB_mR
    p_sep2 = pB_mB + pB_pR
    k_B = p_sep1 - p_sep2
    dk_B = np.diff(k_B) / sep_step
    k_pB = dk_B[np.argmin(np.abs(seps-eta_B))]
    servo_gains = np.zeros(2)
    servo_gains[0] = -2 * 0.5/k_pB
    # set up LC servo
    d, step = np.linspace(-2*eta_B, 2*eta_B, 10000, retstep=True)
    pfb_pB = lineshape(centers[0] + fwhms[0]/2. + d, tau_pis[0],
                       center=centers[0])
    pfb_mB = lineshape(centers[1] + fwhms[1]/2. + d, tau_pis[1],
                       center=centers[1])
    pfb_pR = lineshape(centers[0] - fwhms[0]/2. + d, tau_pis[0],
                       center=centers[0])
    pfb_mR = lineshape(centers[1] - fwhms[1]/2. + d, tau_pis[1],
                       center=centers[1])
    pfb_1 = pfb_mR + pfb_pR
    pfb_2 = pfb_mB + pfb_pB
    k_C = pfb_2 - pfb_1
    dk_C = np.diff(k_C) / step
    k_p_lc = dk_C[np.argmin(np.abs(d))]
    servo_gains[1] = (-2 * 0.5)/k_p_lc
    return {'servo_gains': servo_gains,
            'k_p': [k_pB, k_p_lc]}


def run_BC_simulation(t,
                      centers,
                      tau_pi,
                      FWHMS,
                      servo_gains,
                      initial_vals,
                      laser_drift=0.,
                      B_drift=0.,
                      n_s=100):

    T_s = n_s * 2 * tau_pi
    eta_C = np.zeros(len(t))
    eta_C[0] = initial_vals['LC_servo']
    eta_B = np.zeros(len(t))
    eta_B[0] = initial_vals['B_servo']
    B_field = np.zeros(len(t))
    B_field[0] = initial_vals['B_field']
    z_s = linear_zeeman(initial_vals['B_field'])
    eta_cavity = np.zeros(len(t))
    eta_cavity[0] = f_cavity = initial_vals['Laser']
    ps = np.zeros((len(t), 6))
    for i in range(1, len(t)):
        # sample probabilities. Laser drifts only once per pair as two sites
        # are sampled simultaneously (is this a problem?)
        p_mB, f_cavity, delta_cavity = sampling_cycle(f_cavity, T_s, n_s,
                                                      eta_C[i-1]-eta_B[i-1]+fwhms[0]/2.,
                                                      linecenter=-z_s,
                                                      tau_pi=tau_pi,
                                                      laser_drift=0.)
        p_pR, f_cavity, delta_cavity = sampling_cycle(f_cavity, T_s, n_s,
                                                      eta_C[i-1]+eta_B[i-1]-fwhms[1]/2.,
                                                      linecenter=z_s,
                                                      tau_pi=tau_pi,
                                                      laser_drift=laser_drift)
        p_pB, f_cavity, delta_cavity = sampling_cycle(f_cavity, T_s, n_s,
                                                      eta_C[i-1]+eta_B[i-1]+fwhms[1]/2.,
                                                      linecenter=z_s,
                                                      tau_pi=tau_pi,
                                                      laser_drift=0.)
        p_mR, f_cavity, delta_cavity = sampling_cycle(f_cavity, T_s, n_s,
                                                      eta_C[i-1]-eta_B[i-1]-fwhms[0]/2.,
                                                      linecenter=-z_s,
                                                      tau_pi=tau_pi,
                                                      laser_drift=laser_drift)

        # Control B servo
        p_sep1 = p_pB + p_mR
        p_sep2 = p_mB + p_pR
        discriminant_B = np.divide(p_sep1 - p_sep2, p_sep1 + p_sep2)
        eta_B[i] = eta_B[i-1] + discriminant_B * servo_gains[0]

        # Control LC servo
        pfb_1 = p_mR+p_pR
        pfb_2 = p_mB+p_pB
        ps[i,:] = [p_mR, p_mB, p_pR, p_pB, pfb_1, pfb_2]
        discriminant_LC = (pfb_2 - pfb_1) / (pfb_2 + pfb_1)
        eta_C[i] = eta_C[i-1] + discriminant_LC * servo_gains[1]

        eta_cavity[i] = f_cavity

        # Drift mg-field
        B_field[i] = B_field[i-1] + B_drift
        z_s = linear_zeeman(B_field[i])

    return {'eta_C': eta_C,
            'eta_B': eta_B,
            'B_field': B_field,
            'eta_cavity': eta_cavity,
            'ps': ps}
