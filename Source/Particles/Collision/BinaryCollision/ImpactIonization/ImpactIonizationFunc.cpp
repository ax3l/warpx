/* Copyright 2026 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#include "ImpactIonizationFunc.H"

#include "Utils/Physics/IonizationEnergiesTable.H"
#include "Utils/TextMsg.H"
#include "Utils/WarpXConst.H"

#include <AMReX_GpuContainers.H>
#include <AMReX_GpuDevice.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Vector.H>

#include <cmath>
#include <string>
#include <vector>

namespace
{
    // Energy range of the tables, in eV, with `points_per_decade` points per factor of ten in
    // energy. The lowest first ionization energy of any element is 3.9 eV (Cs).
    constexpr double table_energy_min_eV = 1.0;
    constexpr double table_energy_max_eV = 1.0e10;
    constexpr int points_per_decade = 40;

    /** Whether a species has the mass and the charge of an electron. */
    bool IsElectron (WarpXParticleContainer const& pc)
    {
        using namespace amrex::literals;

        return (pc.getCharge() < 0.0_prt) &&
               (std::abs(pc.getMass() - PhysConst::m_e) < 1.0e-3_prt * PhysConst::m_e);
    }

    /**
     * \brief Add one bound electron to the moments of the cross section.
     *
     * This is the relativistic binary-encounter-Bethe cross section, per bound electron, of
     * Kim, Santos, Parente, Phys. Rev. A 62, 052710 (2000), with the kinetic energy of the bound
     * electron set to its binding energy, as tables of the former are not available for all ions.
     *
     * In the reduced energies t = T/B and w = W/B, the first moment follows from integrating
     * w dsigma/dw from 0 to (t - 1)/2, i.e., the new electron is the slower one of the two.
     *
     * @param[in] tau kinetic energy of the incident electron, normalized to m_e c^2
     * @param[in] b binding energy, normalized to m_e c^2
     * @param[in,out] sigma cross section [m^2]
     * @param[in,out] moment_W sigma times the mean energy of the new electron [m^2 m_e c^2]
     * @param[in,out] moment_E sigma times the mean energy lost by the incident electron [m^2 m_e c^2]
     */
    void AddBoundElectron (double const tau, double const b,
                           double& sigma, double& moment_W, double& moment_E)
    {
        // Below threshold
        if (tau <= b) { return; }

        auto const beta_squared = [] (double const x) {
            return 1.0 - 1.0 / ((1.0 + x) * (1.0 + x));
        };

        constexpr double r_e = PhysConst::r_e_v<double>;
        constexpr double pi = MathConst::pi_v<double>;

        double const t = tau / b;
        double const log_t = std::log(t);
        double const log_mid = std::log(0.5 * (t + 1.0));
        double const beta_t2 = beta_squared(tau);
        double const beta_b2 = beta_squared(b);
        double const scale = 1.0 + 0.5 * tau;

        double const prefactor = 2.0 * pi * r_e * r_e / (b * (beta_t2 + 2.0 * beta_b2));
        double const relativistic_log = std::log(beta_t2 / (1.0 - beta_t2)) - beta_t2 - std::log(2.0 * b);
        double const interference = (1.0 + 2.0 * tau) / (scale * scale);
        double const constant = b * b / (scale * scale);

        double const zeroth =
            0.5 * relativistic_log * (1.0 - 1.0 / (t * t))
            + 1.0 - 1.0 / t
            - interference * log_t / (t + 1.0)
            + 0.5 * constant * (t - 1.0);

        // Close to threshold, the sum of the terms can round to a negative value.
        if (zeroth <= 0.0) { return; }

        double const first =
            0.5 * relativistic_log * (t - 1.0) * (t - 1.0) / (t * (t + 1.0))
            + 2.0 * log_mid - log_t
            - interference * (t * log_t / (t + 1.0) - log_mid)
            + 0.125 * constant * (t - 1.0) * (t - 1.0);

        sigma += prefactor * zeroth;
        moment_W += prefactor * first * b;
        moment_E += prefactor * (first + zeroth) * b;
    }
}

ImpactIonizationFunc::ImpactIonizationFunc (const std::string& collision_name,
                                            MultiParticleContainer const * const mypc,
                                            bool const isSameSpecies)
{
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(!isSameSpecies,
        "Impact ionization collision " + collision_name + " needs two different species: electrons and ions.");

    const amrex::ParmParse pp_collision_name(collision_name);

    amrex::Vector<std::string> species_names;
    pp_collision_name.getarr("species", species_names);
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(species_names.size() == 2u,
        "Impact ionization collision " + collision_name + " must have exactly two species.");

    auto const& electrons = mypc->GetParticleContainerFromName(species_names[0]);
    auto const& ions = mypc->GetParticleContainerFromName(species_names[1]);

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(IsElectron(electrons),
        "The first species of impact ionization collision " + collision_name + " must be electrons.");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(ions.DoFieldIonization(),
        "The second species of impact ionization collision " + collision_name +
        " must track its charge state: set " + species_names[1] + ".do_field_ionization = 1.");

    amrex::Vector<std::string> product_names;
    pp_collision_name.getarr("product_species", product_names);
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(product_names.size() == 1u,
        "Impact ionization collision " + collision_name + " must have exactly one product species.");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(IsElectron(mypc->GetParticleContainerFromName(product_names[0])),
        "The product species of impact ionization collision " + collision_name + " must be electrons.");

    std::string physical_element;
    const amrex::ParmParse pp_ions(species_names[1]);
    pp_ions.get("physical_element", physical_element);
    auto const element = utils::physics::ion_map_ids.find(physical_element);
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(element != utils::physics::ion_map_ids.end(),
        "Unknown physical element '" + physical_element + "' of species " + species_names[1] + ".");

    BuildTables(element->second);
}

void
ImpactIonizationFunc::SetupSpecies (WarpXParticleContainer& /*species1*/, WarpXParticleContainer& species2)
{
    m_exe.m_ion_level_comp = species2.GetIntCompIndex("ionizationLevel") - WarpXParticleContainer::NArrayInt;
}

void
ImpactIonizationFunc::BuildTables (int const ion_element_id)
{
    constexpr double mc2_eV = PhysConst::m_e_v<double> * PhysConst::c2_v<double> / PhysConst::q_e_v<double>;

    int const atomic_number = utils::physics::ion_atomic_numbers[ion_element_id];
    int const energies_offset = utils::physics::ion_energy_offsets[ion_element_id];

    double const decades = std::log10(table_energy_max_eV / table_energy_min_eV);
    int const table_size = static_cast<int>(std::lround(decades * points_per_decade)) + 1;
    double const log_step = std::log(10.0) / points_per_decade;
    double const log_tau_min = std::log(table_energy_min_eV / mc2_eV);

    auto const total_size = static_cast<std::size_t>(atomic_number) * table_size;
    amrex::Gpu::HostVector<double> h_sigma(total_size, 0.0);
    amrex::Gpu::HostVector<double> h_moment_W(total_size, 0.0);
    amrex::Gpu::HostVector<double> h_moment_E(total_size, 0.0);

    for (int charge_state = 0; charge_state < atomic_number; ++charge_state)
    {
        for (int i = 0; i < table_size; ++i)
        {
            double const tau = std::exp(log_tau_min + i * log_step);
            auto const index = static_cast<std::size_t>(charge_state) * table_size + i;

            // The electrons that are bound to an ion of this charge state are the ones that
            // successive ionization would remove, each with the corresponding ionization energy.
            for (int level = charge_state; level < atomic_number; ++level)
            {
                double const binding = static_cast<double>(
                    utils::physics::table_ionization_energies[energies_offset + level]) / mc2_eV;
                AddBoundElectron(tau, binding, h_sigma[index], h_moment_W[index], h_moment_E[index]);
            }
        }
    }

    m_sigma.resize(total_size);
    m_moment_W.resize(total_size);
    m_moment_E.resize(total_size);
    amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, h_sigma.begin(), h_sigma.end(), m_sigma.begin());
    amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, h_moment_W.begin(), h_moment_W.end(), m_moment_W.begin());
    amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, h_moment_E.begin(), h_moment_E.end(), m_moment_E.begin());
    amrex::Gpu::streamSynchronize();

    m_exe.m_atomic_number = atomic_number;
    m_exe.m_table_size = table_size;
    m_exe.m_log_tau_min = log_tau_min;
    m_exe.m_inverse_log_step = 1.0 / log_step;
    m_exe.m_sigma = m_sigma.data();
    m_exe.m_moment_W = m_moment_W.data();
    m_exe.m_moment_E = m_moment_E.data();
}
