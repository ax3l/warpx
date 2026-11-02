#!/usr/bin/env python3
"""Check binary impact ionization of carbon by a monoenergetic 1 MeV electron beam.

- the charge state distribution follows the rate equations of the RBEB cross section
- each ionization creates one electron of the weight of the ion (exact charge conservation)
- the beam loses the mean energy that the cross section predicts per event

The cross section and its mean energies are evaluated directly, without the tables of WarpX.
"""

import numpy as np
from openpmd_viewer import OpenPMDTimeSeries
from scipy.constants import c, e, m_e, physical_constants
from scipy.linalg import expm

r_e = physical_constants["classical electron radius"][0]
mc2_eV = m_e * c**2 / e

density = 1.0e28
beam_energy_eV = 1.0e6
t_final = 300 * 1.0e-16

# Ionization energies [eV] of carbon from NIST (as in IonizationEnergiesTable.H)
ionization_energies = np.array(
    [11.2602880, 24.383154, 47.88778, 64.49352, 392.090515, 489.993194]
)
# Orbital binding energies [eV] of neutral carbon from NIST (as in atomic_orbital_data.txt), which
# set the binding energies of the ions, see Perez et al., Phys. Plasmas 19, 083104 (2012), Eq. (30)
HARTREE_EV = 27.211386245988
neutral_binding = (
    np.array([9.946149, 9.946149, 0.501090, 0.501090, 0.199111, 0.199111]) * HARTREE_EV
)


def ion_binding_energies(state):
    """Return the binding energies [eV] of all electrons bound to this charge state."""
    n_bound = len(neutral_binding) - state
    shift = ionization_energies[state] - neutral_binding[n_bound - 1]
    return shift + neutral_binding[:n_bound]


def rbeb(tau, b):
    """Return sigma [m^2] and mean energies [mc^2] of the new electron, per bound electron."""

    def beta2(x):
        return 1.0 - 1.0 / (1.0 + x) ** 2

    t = tau / b
    scale = 1.0 + 0.5 * tau
    prefactor = 2.0 * np.pi * r_e**2 / (b * (beta2(tau) + 2.0 * beta2(b)))
    log_term = np.log(beta2(tau) / (1.0 - beta2(tau))) - beta2(tau) - np.log(2.0 * b)
    interference = (1.0 + 2.0 * tau) / scale**2
    constant = b**2 / scale**2
    log_mid = np.log(0.5 * (t + 1.0))
    zeroth = (
        0.5 * log_term * (1.0 - 1.0 / t**2)
        + 1.0
        - 1.0 / t
        - interference * np.log(t) / (t + 1.0)
        + 0.5 * constant * (t - 1.0)
    )
    first = (
        0.5 * log_term * (t - 1.0) ** 2 / (t * (t + 1.0))
        + 2.0 * log_mid
        - np.log(t)
        - interference * (t * np.log(t) / (t + 1.0) - log_mid)
        + 0.125 * constant * (t - 1.0) ** 2
    )
    return prefactor * zeroth, first / zeroth * b


tau = beam_energy_eV / mc2_eV
n_states = len(ionization_energies) + 1
sigma = np.zeros(n_states - 1)
mean_binding = np.zeros(n_states - 1)
mean_W = np.zeros(n_states - 1)
for state in range(n_states - 1):
    bindings = ion_binding_energies(state)
    s, w = zip(*(rbeb(tau, b / mc2_eV) for b in bindings))
    s, w = np.array(s), np.array(w)
    sigma[state] = s.sum()
    mean_W[state] = (s * w).sum() / s.sum()
    mean_binding[state] = (s * bindings).sum() / s.sum() / mc2_eV

# Rate equations of the charge states at a constant beam energy
velocity = c * np.sqrt(1.0 - 1.0 / (1.0 + tau) ** 2)
rates = density * sigma * velocity
matrix = np.zeros((n_states, n_states))
for state in range(n_states - 1):
    matrix[state, state] -= rates[state]
    matrix[state + 1, state] += rates[state]
expected = expm(matrix * t_final)[:, 0]

ts = OpenPMDTimeSeries("./diags/diag/")
iteration = ts.iterations[-1]

level, w_ion = ts.get_particle(
    ["ionizationLevel", "w"], species="carbon", iteration=iteration
)
w_electrons, uz_electrons = ts.get_particle(
    ["w", "uz"], species="electrons", iteration=iteration
)
w_new, ux_new, uy_new, uz_new = ts.get_particle(
    ["w", "ux", "uy", "uz"], species="secondaries", iteration=iteration
)


def kinetic_energy(ux, uy, uz):
    """Normalized to m c^2, as openPMD-viewer returns the momentum normalized to m c."""
    return np.sqrt(1.0 + ux**2 + uy**2 + uz**2) - 1.0


fractions = (
    np.array([w_ion[level == state].sum() for state in range(n_states)]) / w_ion.sum()
)
n_ion_macro = w_ion.size
print("charge state fractions, simulation:", np.round(fractions, 4))
print("charge state fractions, theory    :", np.round(expected, 4))

# Multinomial statistics of the macroparticles (4 sigma), and at least 0.5%
tolerance = np.maximum(4.0 * np.sqrt(expected * (1.0 - expected) / n_ion_macro), 0.005)
assert np.all(np.abs(fractions - expected) < tolerance), (
    fractions,
    expected,
    tolerance,
)

# Each ionization creates one electron of the weight of the ion, at any weight ratio
new_charge = (w_ion * level).sum()
print("ionized charge / secondary charge:", new_charge / w_new.sum())
assert np.isclose(new_charge, w_new.sum(), rtol=1.0e-10)

# Secondaries of ions at rest move along the beam
assert np.all(uz_new > 0.0)
assert np.allclose(ux_new, 0.0) and np.allclose(uy_new, 0.0)

# The beam loses the mean energy per event of the charge state that was ionized
events = np.array([w_ion[level > state].sum() for state in range(n_states - 1)])
expected_loss = (events * (mean_W + mean_binding)).sum()
beam_loss = (w_electrons * (tau - kinetic_energy(0.0, 0.0, uz_electrons))).sum()
print("beam energy loss, simulation / expected:", beam_loss / expected_loss)
assert np.isclose(beam_loss, expected_loss, rtol=0.05)

# The new electrons carry the mean kinetic energy of the rate-weighted charge states
expected_new_energy = (events * mean_W).sum()
new_energy = (w_new * kinetic_energy(ux_new, uy_new, uz_new)).sum()
print("secondary energy, simulation / expected:", new_energy / expected_new_energy)
assert np.isclose(new_energy, expected_new_energy, rtol=0.05)
