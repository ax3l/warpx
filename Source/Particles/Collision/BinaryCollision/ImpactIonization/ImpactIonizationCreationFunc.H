/* Copyright 2026 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_IMPACT_IONIZATION_CREATION_FUNC_H_
#define WARPX_IMPACT_IONIZATION_CREATION_FUNC_H_

#include "ImpactIonizationFunc.H"

#include "Particles/MultiParticleContainer.H"
#include "Particles/ParticleCreation/DefaultInitialization.H"
#include "Particles/ParticleCreation/SmartCopy.H"
#include "Particles/WarpXParticleContainer.H"

#include <AMReX_DenseBins.H>
#include <AMReX_GpuContainers.H>
#include <AMReX_GpuDevice.H>
#include <AMReX_Random.H>
#include <AMReX_REAL.H>
#include <AMReX_Scan.H>
#include <AMReX_Vector.H>

#include <string>

/**
 * \brief Create the new electrons of impact ionization.
 *
 * One electron is created per ionized ion macroparticle, at the position of the ion and with the
 * weight of the ion. This balances the additional charge of the ion exactly, at the same position.
 */
class ImpactIonizationCreationFunc
{
    using ParticleTileType = typename WarpXParticleContainer::ParticleTileType;
    using ParticleTileDataType = typename ParticleTileType::ParticleTileDataType;
    using ParticleBins = amrex::DenseBins<ParticleTileDataType>;
    using index_type = typename ParticleBins::index_type;

public:
    ImpactIonizationCreationFunc () = default;

    ImpactIonizationCreationFunc (const std::string& /*collision_name*/,
                                  MultiParticleContainer const * const /*mypc*/) {}

    /**
     * \brief Create the new electrons.
     *
     * @param[in] n_total_pairs how many binary collisions have been performed in this tile
     * @param[in] ptile2 the particle tile of the ions
     * @param[in] pc_products the container of the new electrons (first entry)
     * @param[in,out] tile_products the tile of the new electrons (first entry)
     * @param[in] p_mask 1 if the pair created a new electron, 0 otherwise
     * @param[in] products_np the number of existing electrons in the tile of the new electrons
     * @param[in] copy_species2 functors that copy from the ions to the new electrons
     * @param[in] p_pair_indices_2 the index of the ion of each pair
     * @param[in] p_pair_reaction_weight the weight of the new electron of each pair
     * @param[in] p_product_data the momentum of the new electron of each pair
     * @return the number of particles that were added
     */
    AMREX_INLINE
    amrex::Vector<int> operator() (
        const index_type& n_total_pairs,
        ParticleTileType& /*ptile1*/, ParticleTileType& ptile2,
        const amrex::Vector<WarpXParticleContainer*>& pc_products,
        ParticleTileType** AMREX_RESTRICT tile_products,
        const amrex::ParticleReal& /*m1*/, const amrex::ParticleReal& /*m2*/,
        const amrex::Vector<amrex::ParticleReal>& /*products_mass*/,
        const index_type* AMREX_RESTRICT p_mask,
        const amrex::Vector<index_type>& products_np,
        const SmartCopy* /*copy_species1*/,
        const SmartCopy* AMREX_RESTRICT copy_species2,
        const index_type* /*p_pair_indices_1*/,
        const index_type* AMREX_RESTRICT p_pair_indices_2,
        const amrex::ParticleReal* AMREX_RESTRICT p_pair_reaction_weight,
        const amrex::ParticleReal* AMREX_RESTRICT p_product_data) const
    {
        if (n_total_pairs == 0) { return {0}; }

        amrex::Gpu::DeviceVector<index_type> offsets(n_total_pairs);
        const index_type total = amrex::Scan::ExclusiveSum(n_total_pairs, p_mask, offsets.data());
        const index_type* AMREX_RESTRICT p_offsets = offsets.dataPtr();

        const index_type np_before = products_np[0];
        tile_products[0]->resize(np_before + total);

        // Get the tile data after the resize, as the new electrons may be added to the
        // species of the incident electrons.
        const auto soa_ions = ptile2.getParticleTileData();
        auto soa_new = tile_products[0]->getParticleTileData();

        constexpr int n_data = ImpactIonizationFunc::n_product_data_per_pair;

        amrex::ParallelForRNG(n_total_pairs,
        [=] AMREX_GPU_DEVICE (int i, amrex::RandomEngine const& engine) noexcept
        {
            if (p_mask[i])
            {
                const index_type product_index = np_before + p_offsets[i];
                copy_species2[0](soa_new, soa_ions, static_cast<int>(p_pair_indices_2[i]),
                                 static_cast<int>(product_index), engine);
                soa_new.m_rdata[PIdx::w][product_index] = p_pair_reaction_weight[i];
                soa_new.m_rdata[PIdx::ux][product_index] = p_product_data[n_data*i];
                soa_new.m_rdata[PIdx::uy][product_index] = p_product_data[n_data*i + 1];
                soa_new.m_rdata[PIdx::uz][product_index] = p_product_data[n_data*i + 2];
            }
        });

        ParticleCreation::DefaultInitializeRuntimeAttributes(*tile_products[0], *pc_products[0],
            static_cast<int>(np_before), static_cast<int>(np_before + total));

        amrex::Gpu::synchronize();

        return {static_cast<int>(total)};
    }
};

#endif // WARPX_IMPACT_IONIZATION_CREATION_FUNC_H_
