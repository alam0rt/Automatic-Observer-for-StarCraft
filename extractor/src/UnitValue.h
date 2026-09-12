#pragma once

#include <BWAPI.h>

// Combat value of units, ported from Stardust's CombatSim scoring: resource cost
// (gas counted double, morphs include what they morphed from) with adjustments
// for units whose fighting value differs from their price. A quarter of the value
// is fixed and the rest scales with remaining hit points and shields.
namespace UnitValue
{
    void initialize();

    int full(BWAPI::UnitType type);

    int scaled(BWAPI::UnitType type, int health, int shields);

    // For FAP units, whose health and shields are fixed point: pass FAP's own maxima
    int scaled(BWAPI::UnitType type, int health, int shields, int maxHealth, int maxShields);
}
