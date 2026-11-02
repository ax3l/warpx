/* Copyright 2026 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_IMPACT_IONIZATION_FUNC_H_
#define WARPX_IMPACT_IONIZATION_FUNC_H_

#include "Particles/Collision/CollisionFuncBase.H"
#include "Particles/MultiParticleContainer.H"
#include "Particles/Pusher/GetAndSetPosition.H"
#include "Particles/WarpXParticleContainer.H"
#include "Utils/WarpXConst.H"

#include <AMReX_Algorithm.H>
#include <AMReX_DenseBins.H>
#include <AMReX_GpuContainers.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_MFIter.H>
#include <AMReX_Random.H>
#include <AMReX_REAL.H>

#include <cmath>
#include <string>

/**
 * \brief Binary electron-impact ionization of ions that track their charge state per macroparticle
 * (``ionizationLevel``), following the approach of Perez et al., Phys. Plasmas 19, 083104 (2012).
 *
 * Electron and ion macroparticles are paired in each cell. Each pair ionizes with a probability
 * set by the relativistic binary-encounter-Bethe (RBEB) cross section of the current charge state
 * at the kinetic energy of the electron in the ion rest frame. On an event, the ion charge state
 * increases by one, a new electron is created at the position of the ion, and the incident electron
 * loses energy. Energies are the cross-section averages over all bound electrons of the ion.
 *
 * Macroparticles are not split. If the weights differ, the ion is only ionized with probability
 * min(1, w_e/w_i) and the electron only slows down with probability min(1, w_i/w_e), so that the
 * number of physical events min(w_e, w_i) per pair is conserved on average.
 *
 * Differences to Perez et al.: energies are applied in the ion rest frame instead of the
 * center-of-mass frame, and the probability of each pair follows from its own weights (as in
 * Higginson et al., J. Comput. Phys. 413, 109450 (2020)) instead of a density factor of the cell,
 * which does not need the exclusion of electrons below threshold in their Sec. IV C.
 *
 * Limits: the probability of a pair must be small, i.e., max(n_e, n_i) sigma v dt << 1 (see
 * ndt_subcycle): a pair ionizes at most once per step, and a pair changes the ion by its whole
 * weight, so an electron of much larger weight ionizes too little if it saturates, see Morris et
 * al., Phys. Plasmas 29, 123907 (2022), Fig. 2. There is no recombination, excited state, ionization
 * potential depression, recoil of the ion, or spectrum of the new electrons (only their mean energy).
 */
class ImpactIonizationFunc
    : public CollisionFuncBase
{
    using ParticleTileType = WarpXParticleContainer::ParticleTileType;
    using ParticleTileDataType = ParticleTileType::ParticleTileDataType;
    using ParticleBins = amrex::DenseBins<ParticleTileDataType>;
    using index_type = ParticleBins::index_type;
    using SoaData_type = WarpXParticleContainer::ParticleTileType::ParticleTileDataType;

public:

    /** Momentum (x, y, z) of the new electron, stored per pair for the creation functor. */
    static constexpr int n_product_data_per_pair = 3;

    ImpactIonizationFunc () = default;

    /**
     * \brief Check the species and tabulate the cross sections of the target element.
     *
     * @param[in] collision_name the name of the collision
     * @param[in] mypc pointer to the MultiParticleContainer
     * @param[in] isSameSpecies whether the two colliding species are the same
     */
    ImpactIonizationFunc (const std::string& collision_name,
                          MultiParticleContainer const * mypc,
                          bool isSameSpecies);

    /** Look up ``ionizationLevel``, which is only added once the ionization module is initialized. */
    void SetupSpecies (WarpXParticleContainer& species1, WarpXParticleContainer& species2) override;

    struct Executor {
        /**
         * \brief Collide the electrons (species 1) and ions (species 2) of one cell.
         *
         * This updates the charge state of the ions and the momentum of the incident electrons.
         * The new electrons are created afterwards by ImpactIonizationCreationFunc.
         *
         * Pairs that are handled by different calls of this function do not share macroparticles.
         *
         * @param[in] I1s,I2s start index for I1,I2 (inclusive)
         * @param[in] I1e,I2e stop index for I1,I2 (exclusive)
         * @param[in] I1,I2 index arrays of the particles in this cell
         * @param[in,out] soa_1,soa_2 struct of array data of the two species
         * @param[in] dt time step length between two collision calls
         * @param[in] dV volume of the cell
         * @param[in] coll_idx collision index offset
         * @param[in] cell_start_pair start index of the pairs in this cell
         * @param[out] p_mask for each pair: 1 if a new electron is created, 0 otherwise
         * @param[out] p_pair_indices_1,p_pair_indices_2 indices of the particles of each pair
         * @param[out] p_pair_reaction_weight weight of the new electron of each pair
         * @param[out] p_product_data momentum of the new electron of each pair
         * @param[in] engine the random engine
         */
        AMREX_GPU_HOST_DEVICE AMREX_INLINE
        void operator() (
            index_type const I1s, index_type const I1e,
            index_type const I2s, index_type const I2e,
            index_type const* AMREX_RESTRICT I1,
            index_type const* AMREX_RESTRICT I2,
            const SoaData_type& soa_1, const SoaData_type& soa_2,
            GetParticlePosition<PIdx> /*get_position_1*/, GetParticlePosition<PIdx> /*get_position_2*/,
            amrex::ParticleReal const /*n1*/, amrex::ParticleReal const /*n2*/,
            amrex::ParticleReal const /*T1*/, amrex::ParticleReal const /*T2*/,
            amrex::Real const /*global_lamdb*/,
            amrex::ParticleReal const /*q1*/, amrex::ParticleReal const /*q2*/,
            amrex::ParticleReal const /*m1*/, amrex::ParticleReal const /*m2*/,
            amrex::Real const dt, amrex::Real const dV, index_type coll_idx,
            index_type const cell_start_pair, amrex::IntVect const /*global_index*/,
            index_type* AMREX_RESTRICT p_mask,
            index_type* AMREX_RESTRICT p_pair_indices_1, index_type* AMREX_RESTRICT p_pair_indices_2,
            amrex::ParticleReal* AMREX_RESTRICT p_pair_reaction_weight,
            amrex::ParticleReal* AMREX_RESTRICT p_product_data,
            amrex::RandomEngine const& engine) const
        {
            amrex::ParticleReal * const AMREX_RESTRICT w1 = soa_1.m_rdata[PIdx::w];
            amrex::ParticleReal * const AMREX_RESTRICT u1x = soa_1.m_rdata[PIdx::ux];
            amrex::ParticleReal * const AMREX_RESTRICT u1y = soa_1.m_rdata[PIdx::uy];
            amrex::ParticleReal * const AMREX_RESTRICT u1z = soa_1.m_rdata[PIdx::uz];

            amrex::ParticleReal const * const AMREX_RESTRICT w2 = soa_2.m_rdata[PIdx::w];
            amrex::ParticleReal const * const AMREX_RESTRICT u2x = soa_2.m_rdata[PIdx::ux];
            amrex::ParticleReal const * const AMREX_RESTRICT u2y = soa_2.m_rdata[PIdx::uy];
            amrex::ParticleReal const * const AMREX_RESTRICT u2z = soa_2.m_rdata[PIdx::uz];
            int * const AMREX_RESTRICT ion_level = soa_2.m_runtime_idata[m_ion_level_comp];

            const index_type NI1 = I1e - I1s;
            const index_type NI2 = I2e - I2s;
            const index_type max_N = amrex::max(NI1, NI2);
            const index_type min_N = amrex::min(NI1, NI2);

            // Each macroparticle of the larger set collides once per step, with one of the possible
            // partners. Scale the probability such that every macroparticle of the smaller set has
            // the correct rate: if max_N is not a multiple of min_N, these have either
            // floor(max_N/min_N) or floor(max_N/min_N) + 1 pairs, so this depends on coll_idx.
            index_type const n_pairs_of_partner = (max_N - 1 - coll_idx) / min_N + 1;
            double const multiplier = static_cast<double>(max_N) / static_cast<double>(n_pairs_of_partner);

            index_type pair_index = cell_start_pair + coll_idx;
            index_type i1 = I1s + coll_idx;
            index_type i2 = I2s + coll_idx;

            // Start from collision number = coll_idx and then add stride (smaller set size)
            // until we do all collisions (larger set size)
            for (index_type k = coll_idx; k < max_N; k += min_N)
            {
                index_type const ip1 = I1[i1];
                index_type const ip2 = I2[i2];

                p_mask[pair_index] = index_type(0);
                p_pair_indices_1[pair_index] = ip1;
                p_pair_indices_2[pair_index] = ip2;

                PairCollision(u1x[ip1], u1y[ip1], u1z[ip1], w1[ip1],
                              u2x[ip2], u2y[ip2], u2z[ip2], w2[ip2],
                              ion_level[ip2], multiplier, dt, dV,
                              p_mask[pair_index], p_pair_reaction_weight[pair_index],
                              &p_product_data[n_product_data_per_pair*pair_index], engine);

                if (max_N == NI1) { i1 += min_N; }
                if (max_N == NI2) { i2 += min_N; }
                pair_index += min_N;
            }
        }

        /**
         * \brief Interpolate the moments of the cross section of a charge state.
         *
         * Between grid points, this interpolates the moments linearly in the logarithm of the
         * energy, because mean energies interpolated toward 0 below a threshold would remove less
         * than the binding energy. Above the table, the cross section keeps its logarithmic slope
         * (the relativistic rise) and the mean energies stay constant.
         *
         * @param[in] charge_state the charge state of the ion before ionization
         * @param[in] tau kinetic energy of the electron in the ion frame, normalized to m_e c^2
         * @param[out] sigma cross section [m^2]
         * @param[out] mean_W mean kinetic energy of the new electron, normalized to m_e c^2
         * @param[out] mean_E mean energy lost by the incident electron, normalized to m_e c^2
         */
        AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
        void Interpolate (int const charge_state, double const tau,
                          double& sigma, double& mean_W, double& mean_E) const
        {
            sigma = 0.0;
            mean_W = 0.0;
            mean_E = 0.0;

            double const x = (std::log(tau) - m_log_tau_min) * m_inverse_log_step;
            if (x < 0.0) { return; }

            int const i = amrex::min(static_cast<int>(x), m_table_size - 2);
            int const offset = charge_state * m_table_size + i;
            double const a = x - static_cast<double>(i);

            sigma = m_sigma[offset] + a * (m_sigma[offset + 1] - m_sigma[offset]);
            if (sigma <= 0.0) { sigma = 0.0; return; }

            // Clamp the moments above the end of the table, and evaluate them with the matching
            // cross section, so that the mean energies stay constant.
            double const a_clamped = amrex::min(a, 1.0);
            double const sigma_clamped = m_sigma[offset] + a_clamped * (m_sigma[offset + 1] - m_sigma[offset]);
            mean_W = (m_moment_W[offset] + a_clamped * (m_moment_W[offset + 1] - m_moment_W[offset])) / sigma_clamped;
            mean_E = (m_moment_E[offset] + a_clamped * (m_moment_E[offset + 1] - m_moment_E[offset])) / sigma_clamped;
        }

        /**
         * \brief Perform the possible ionization of one electron-ion pair.
         *
         * The energies of the outgoing electrons are set in the ion rest frame, where both move
         * along the direction of the incident electron. In the lab frame, with the normalized
         * momenta u_e and u_i of the electron and the ion, and gamma* the Lorentz factor of the
         * electron in the ion frame, an electron that leaves with the kinetic energy tau' has
         *     u' = alpha u_e + (1 + tau' - alpha gamma*) u_i
         * with alpha = sqrt(tau' (tau' + 2)) / sqrt(gamma*^2 - 1).
         */
        AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
        void PairCollision (amrex::ParticleReal& u1x, amrex::ParticleReal& u1y, amrex::ParticleReal& u1z,
                            amrex::ParticleReal const w1,
                            amrex::ParticleReal const u2x, amrex::ParticleReal const u2y, amrex::ParticleReal const u2z,
                            amrex::ParticleReal const w2,
                            int& ion_level, double const multiplier,
                            amrex::Real const dt, amrex::Real const dV,
                            index_type& mask, amrex::ParticleReal& reaction_weight,
                            amrex::ParticleReal* AMREX_RESTRICT product_data,
                            amrex::RandomEngine const& engine) const
        {
            // Fully ionized
            if (ion_level >= m_atomic_number) { return; }

            constexpr double c = PhysConst::c_v<double>;
            constexpr double inv_c = PhysConst::inv_c_v<double>;

            // All energies and momenta are in double precision: the kinetic energy near thresholds
            // is a tiny fraction of the rest energy, and is lost to round-off in single precision.
            double const uex = static_cast<double>(u1x) * inv_c;
            double const uey = static_cast<double>(u1y) * inv_c;
            double const uez = static_cast<double>(u1z) * inv_c;
            double const uix = static_cast<double>(u2x) * inv_c;
            double const uiy = static_cast<double>(u2y) * inv_c;
            double const uiz = static_cast<double>(u2z) * inv_c;

            double const ue2 = uex*uex + uey*uey + uez*uez;
            double const ui2 = uix*uix + uiy*uiy + uiz*uiz;
            double const gamma_e = std::sqrt(1.0 + ue2);
            double const gamma_i = std::sqrt(1.0 + ui2);

            // Kinetic energy of the electron in the ion frame, gamma* - 1 = gamma_e gamma_i - u_e.u_i - 1,
            // written to avoid the cancellation of the leading 1.
            double const tau_e = ue2 / (gamma_e + 1.0);
            double const tau_i = ui2 / (gamma_i + 1.0);
            double const tau = tau_e + tau_i + tau_e * tau_i - (uex*uix + uey*uiy + uez*uiz);
            if (tau <= 0.0) { return; }

            double sigma, mean_W, mean_E;
            Interpolate(ion_level, tau, sigma, mean_W, mean_E);
            if (sigma <= 0.0) { return; }

            // gamma* beta* = sqrt(gamma*^2 - 1)
            double const gamma_beta = std::sqrt(tau * (tau + 2.0));

            // Event rate per target density: the Lorentz-invariant flux of the pair, with the
            // velocity in the rest frame of the ion, is c gamma* beta* / (gamma_e gamma_i).
            double const w_max = amrex::max(static_cast<double>(w1), static_cast<double>(w2));
            double const exponent = multiplier * w_max * sigma * c * gamma_beta / (gamma_e * gamma_i)
                                    * static_cast<double>(dt) / static_cast<double>(dV);
            double const probability = -std::expm1(-exponent);

            if (amrex::Random(engine) >= probability) { return; }

            // The cross section at the upper end of a grid interval can be non-zero below a
            // threshold, at energies from which this mean energy cannot be taken.
            if (mean_E >= tau) { return; }

            double const gamma_star = 1.0 + tau;

            // Macroparticles are not split: each of the two changes with the probability that
            // conserves the number of physical events, min(w_e, w_i).
            auto const random_weight = static_cast<double>(amrex::Random(engine));
            bool const ionize = random_weight * static_cast<double>(w2) < static_cast<double>(w1);
            bool const slow_down = random_weight * static_cast<double>(w1) < static_cast<double>(w2);

            if (ionize)
            {
                ion_level += 1;

                double const alpha = std::sqrt(mean_W * (mean_W + 2.0)) / gamma_beta;
                double const beta_i = 1.0 + mean_W - alpha * gamma_star;
                mask = index_type(1);
                reaction_weight = w2;
                product_data[0] = static_cast<amrex::ParticleReal>((alpha * uex + beta_i * uix) * c);
                product_data[1] = static_cast<amrex::ParticleReal>((alpha * uey + beta_i * uiy) * c);
                product_data[2] = static_cast<amrex::ParticleReal>((alpha * uez + beta_i * uiz) * c);
            }

            if (slow_down)
            {
                double const tau_after = tau - mean_E;
                double const alpha = std::sqrt(tau_after * (tau_after + 2.0)) / gamma_beta;
                double const beta_i = 1.0 + tau_after - alpha * gamma_star;
                u1x = static_cast<amrex::ParticleReal>((alpha * uex + beta_i * uix) * c);
                u1y = static_cast<amrex::ParticleReal>((alpha * uey + beta_i * uiy) * c);
                u1z = static_cast<amrex::ParticleReal>((alpha * uez + beta_i * uiz) * c);
            }
        }

        bool m_computeSpeciesDensities = false;
        bool m_computeSpeciesTemperatures = false;
        bool m_need_product_data = true;

        int m_atomic_number = 0;
        int m_ion_level_comp = -1;

        int m_table_size = 0;
        double m_log_tau_min = 0.0;
        double m_inverse_log_step = 0.0;
        double const* m_sigma = nullptr;
        double const* m_moment_W = nullptr;
        double const* m_moment_E = nullptr;
    };

    [[nodiscard]] Executor const& executor (int const /*lev*/, amrex::MFIter const& /*mfi*/) const { return m_exe; }

    bool use_global_debye_length () { return false; }

private:
    /** Fill the tables of the moments of the cross section for all charge states of an element. */
    void BuildTables (int ion_element_id);

    amrex::Gpu::DeviceVector<double> m_sigma;
    amrex::Gpu::DeviceVector<double> m_moment_W;
    amrex::Gpu::DeviceVector<double> m_moment_E;

    Executor m_exe;
};

#endif // WARPX_IMPACT_IONIZATION_FUNC_H_
