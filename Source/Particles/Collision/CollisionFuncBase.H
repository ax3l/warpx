/* Copyright 2026 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#ifndef WARPX_COLLISION_FUNC_BASE_H_
#define WARPX_COLLISION_FUNC_BASE_H_

#include "Particles/WarpXParticleContainer_fwd.H"

class CollisionFuncBase
{
public:

    //! Number of values per colliding pair in the product data of a binary collision, if needed.
    static constexpr int n_product_data_per_pair = 1;

    CollisionFuncBase () = default;
    virtual ~CollisionFuncBase () = default;

    CollisionFuncBase (const CollisionFuncBase&) = default;
    CollisionFuncBase& operator= (const CollisionFuncBase&) = default;

    CollisionFuncBase (CollisionFuncBase&&) noexcept = default;
    CollisionFuncBase& operator= (CollisionFuncBase&&) noexcept = default;

    //! Optional hook to allocate additional data structures.
    virtual void AllocData () {}

    //! Optional hook, called before each collision step, to look up data of the colliding species
    //! that does not exist yet when the collision is constructed (e.g., runtime attributes).
    virtual void SetupSpecies (WarpXParticleContainer& /*species1*/, WarpXParticleContainer& /*species2*/) {}

};

#endif /* WARPX_COLLISION_FUNC_BASE_H_ */
