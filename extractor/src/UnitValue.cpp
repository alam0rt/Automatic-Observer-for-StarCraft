#include "UnitValue.h"

namespace
{
    int baseScore[BWAPI::UnitTypes::Enum::MAX];
    int scaledScore[BWAPI::UnitTypes::Enum::MAX];

    std::pair<BWAPI::UnitType, int> morphsFrom(BWAPI::UnitType type)
    {
        // Anything built by a drone is a morph
        if (type.whatBuilds().first == BWAPI::UnitTypes::Zerg_Drone) return type.whatBuilds();

        // A building "built" by another building is a morph, unless it is an add-on
        if (type.isBuilding() && !type.isAddon() && type.whatBuilds().first.isBuilding()) return type.whatBuilds();

        // A unit "built" by another unit is a morph unless it is an interceptor or scarab
        if (!type.isBuilding() && !type.whatBuilds().first.isBuilding() &&
            type != BWAPI::UnitTypes::Protoss_Interceptor && type != BWAPI::UnitTypes::Protoss_Scarab)
        {
            return type.whatBuilds();
        }

        return std::make_pair(BWAPI::UnitTypes::None, 0);
    }

    int mineralCost(BWAPI::UnitType type)
    {
        if (type == BWAPI::UnitTypes::None || type == BWAPI::UnitTypes::Unknown) return 0;

        int minerals = type.mineralPrice();

        // BWAPI lists some units as having mineral cost 1 instead of 0, e.g. Larva
        if (minerals == 1) minerals = 0;

        if (type.isTwoUnitsInOneEgg()) minerals /= 2;

        auto from = morphsFrom(type);
        if (from.second > 0) minerals += from.second * mineralCost(from.first);

        return minerals;
    }

    int gasCost(BWAPI::UnitType type)
    {
        if (type == BWAPI::UnitTypes::None || type == BWAPI::UnitTypes::Unknown || type == BWAPI::UnitTypes::Zerg_Larva) return 0;

        int gas = type.gasPrice();

        // BWAPI lists some units as having gas cost 1 instead of 0, e.g. Larva
        if (gas == 1) gas = 0;

        if (type.isTwoUnitsInOneEgg()) gas /= 2;

        auto from = morphsFrom(type);
        if (from.second > 0) gas += from.second * gasCost(from.first);

        return gas;
    }

    int adjustment(BWAPI::UnitType type)
    {
        switch (type)
        {
            // Cost of built / loaded units
            case BWAPI::UnitTypes::Terran_Bunker:
                return 4 * mineralCost(BWAPI::UnitTypes::Terran_Marine);
            case BWAPI::UnitTypes::Protoss_Carrier:
                return 8 * mineralCost(BWAPI::UnitTypes::Protoss_Interceptor);

            // Units whose combat value differs from their cost
            case BWAPI::UnitTypes::Zerg_Sunken_Colony:
                return 125;
            case BWAPI::UnitTypes::Zerg_Spore_Colony:
                return 100;
            case BWAPI::UnitTypes::Zerg_Creep_Colony:
                return 50;
            case BWAPI::UnitTypes::Protoss_Photon_Cannon:
                return 100;
            case BWAPI::UnitTypes::Terran_Missile_Turret:
                return 150;
            case BWAPI::UnitTypes::Protoss_Shuttle:
            case BWAPI::UnitTypes::Terran_Dropship:
                return 200;
            case BWAPI::UnitTypes::Zerg_Zergling:
                return 15;
            case BWAPI::UnitTypes::Terran_Vulture:
                return 50;
            case BWAPI::UnitTypes::Terran_Siege_Tank_Siege_Mode:
            case BWAPI::UnitTypes::Terran_Vulture_Spider_Mine:
                return 100;
            default:
                return 0;
        }
    }
}

namespace UnitValue
{
    void initialize()
    {
        for (auto type : BWAPI::UnitTypes::allUnitTypes())
        {
            int score = mineralCost(type) + gasCost(type) * 2 + adjustment(type);
            baseScore[type] = score >> 2U;
            scaledScore[type] = score - baseScore[type];
        }
    }

    int full(BWAPI::UnitType type)
    {
        return baseScore[type] + scaledScore[type];
    }

    int scaled(BWAPI::UnitType type, int health, int shields)
    {
        return scaled(type, health, shields, type.maxHitPoints(), type.maxShields());
    }

    int scaled(BWAPI::UnitType type, int health, int shields, int maxHealth, int maxShields)
    {
        int64_t max = (int64_t)maxHealth * 3 + maxShields;
        if (max <= 0) return full(type);

        return baseScore[type] + (int)((scaledScore[type] * ((int64_t)health * 3 + shields)) / max);
    }
}
