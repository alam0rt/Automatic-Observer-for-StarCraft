#include "Fights.h"

#include "UnitValue.h"

#include <fap.h>

#include <algorithm>
#include <climits>
#include <numeric>
#include <unordered_map>

namespace
{
    bool isSimUnit(BWAPI::Unit unit)
    {
        if (!unit->exists() || !unit->isCompleted() || unit->isLoaded()) return false;
        if (unit->getPlayer()->isNeutral()) return false;

        switch (unit->getType())
        {
            case BWAPI::UnitTypes::Protoss_Interceptor:
            case BWAPI::UnitTypes::Protoss_Scarab:
            case BWAPI::UnitTypes::Zerg_Larva:
            case BWAPI::UnitTypes::Zerg_Egg:
            case BWAPI::UnitTypes::Zerg_Lurker_Egg:
            case BWAPI::UnitTypes::Zerg_Cocoon:
                return false;
            default:
                return true;
        }
    }

    BWAPI::WeaponType groundWeapon(BWAPI::UnitType type)
    {
        if (type == BWAPI::UnitTypes::Protoss_Reaver) return BWAPI::UnitTypes::Protoss_Scarab.groundWeapon();
        if (type == BWAPI::UnitTypes::Protoss_Carrier) return BWAPI::UnitTypes::Protoss_Interceptor.groundWeapon();
        if (type == BWAPI::UnitTypes::Terran_Bunker) return BWAPI::UnitTypes::Terran_Marine.groundWeapon();
        return type.groundWeapon();
    }

    BWAPI::WeaponType airWeapon(BWAPI::UnitType type)
    {
        if (type == BWAPI::UnitTypes::Protoss_Carrier) return BWAPI::UnitTypes::Protoss_Interceptor.airWeapon();
        if (type == BWAPI::UnitTypes::Terran_Bunker) return BWAPI::UnitTypes::Terran_Marine.airWeapon();
        return type.airWeapon();
    }

    bool canAttack(BWAPI::Unit unit)
    {
        auto type = unit->getType();
        return groundWeapon(type) != BWAPI::WeaponTypes::None
               || airWeapon(type) != BWAPI::WeaponTypes::None
               || type == BWAPI::UnitTypes::Terran_Medic;
    }

    int groundDamage(BWAPI::Unit unit)
    {
        auto type = unit->getType();

        // Lurkers only attack when burrowed, and nothing else attacks while burrowed
        if (type != BWAPI::UnitTypes::Terran_Vulture_Spider_Mine && unit->isBurrowed() != (type == BWAPI::UnitTypes::Zerg_Lurker))
        {
            return 0;
        }

        auto weapon = groundWeapon(type);
        return weapon == BWAPI::WeaponTypes::None ? 0 : unit->getPlayer()->damage(weapon);
    }

    int airDamage(BWAPI::Unit unit)
    {
        auto weapon = airWeapon(unit->getType());
        return weapon == BWAPI::WeaponTypes::None ? 0 : unit->getPlayer()->damage(weapon);
    }

    int groundRange(BWAPI::Unit unit)
    {
        auto type = unit->getType();
        if (type == BWAPI::UnitTypes::Protoss_Carrier || type == BWAPI::UnitTypes::Protoss_Reaver) return 256;
        if (type == BWAPI::UnitTypes::Terran_Bunker) return unit->getPlayer()->weaponMaxRange(groundWeapon(type)) + 48;
        return unit->getPlayer()->weaponMaxRange(groundWeapon(type));
    }

    int airRange(BWAPI::Unit unit)
    {
        auto type = unit->getType();
        if (type == BWAPI::UnitTypes::Protoss_Carrier) return 256;
        if (type == BWAPI::UnitTypes::Terran_Bunker) return unit->getPlayer()->weaponMaxRange(airWeapon(type)) + 48;
        return unit->getPlayer()->weaponMaxRange(airWeapon(type));
    }

    int groundCooldown(BWAPI::Unit unit)
    {
        auto type = unit->getType();
        if (type == BWAPI::UnitTypes::Protoss_Reaver) return 60;
        if (type == BWAPI::UnitTypes::Protoss_Carrier) return 38;
        if (type == BWAPI::UnitTypes::Terran_Bunker) return unit->getPlayer()->weaponDamageCooldown(BWAPI::UnitTypes::Terran_Marine);
        return unit->getPlayer()->weaponDamageCooldown(type);
    }

    int airCooldown(BWAPI::Unit unit)
    {
        auto type = unit->getType();
        if (type == BWAPI::UnitTypes::Protoss_Carrier) return 38;
        return airWeapon(type).damageCooldown();
    }

    // Ported from Stardust: melee units have less room to manoeuvre, so they block each other
    // more. Fights are simulated without choke geometry, so only the open-terrain value is used.
    int collisionValue(BWAPI::Unit unit)
    {
        if (unit->isFlying()) return 0;

        int range = groundRange(unit);
        if (range > 128) return 3;
        if (range > 32) return 4;
        return 6;
    }

    int attackerCount(BWAPI::Unit unit)
    {
        if (unit->getType() == BWAPI::UnitTypes::Protoss_Carrier) return unit->getInterceptorCount();
        if (unit->getType() == BWAPI::UnitTypes::Terran_Bunker) return (int)unit->getLoadedUnits().size();
        return 0;
    }

    // Replays are played with complete map information, so BWAPI's own detection
    // flag is not meaningful; check for an enemy detector in sight range instead.
    bool isUndetected(BWAPI::Unit unit, const std::vector<BWAPI::Unit> &group)
    {
        if (!unit->isCloaked() && !unit->isBurrowed() && !unit->getType().hasPermanentCloak()) return false;

        return std::none_of(group.begin(), group.end(), [&](BWAPI::Unit other)
        {
            if (!other->getType().isDetector() || !other->getPlayer()->isEnemy(unit->getPlayer())) return false;
            return other->getDistance(unit) <= other->getPlayer()->sightRange(other->getType());
        });
    }

    auto makeUnit(BWAPI::Unit unit, const std::vector<BWAPI::Unit> &group)
    {
        auto target = unit->getTarget() ? unit->getTarget() : unit->getOrderTarget();
        auto targetPosition = unit->getOrderTargetPosition();
        if (!targetPosition.isValid()) targetPosition = unit->getPosition();

        return FAP::makeUnit<>()
                .setUnitType(unit->getType())
                .setPosition(unit->getPosition())
                .setTargetPosition(targetPosition)
                .setHealth(unit->getHitPoints())
                .setShields(unit->getShields())
                .setFlying(unit->isFlying())

                        // Stardust's FAP takes upgraded values rather than upgrade levels
                .setSpeed((float)unit->getPlayer()->topSpeed(unit->getType()))
                .setArmor(unit->getPlayer()->armor(unit->getType()))
                .setGroundCooldown(groundCooldown(unit))
                .setGroundDamage(groundDamage(unit))
                .setGroundMaxRange(groundRange(unit))
                .setAirCooldown(airCooldown(unit))
                .setAirDamage(airDamage(unit))
                .setAirMaxRange(airRange(unit))

                .setElevation(BWAPI::Broodwar->getGroundHeight(unit->getTilePosition()))

                .setAttackerCount(attackerCount(unit))
                .setAttackCooldownRemaining(std::max(unit->getGroundWeaponCooldown(), unit->getAirWeaponCooldown()))

                .setSpeedUpgrade(false)
                .setRangeUpgrade(false)
                .setShieldUpgrades(unit->getPlayer()->getUpgradeLevel(BWAPI::UpgradeTypes::Protoss_Plasma_Shields))

                .setStimmed(unit->isStimmed())
                .setUndetected(isUndetected(unit, group))

                .setID(unit->getID())
                .setTarget(target ? target->getID() : 0)

                .setCollisionValues(collisionValue(unit), collisionValue(unit))

                .setData({});
    }

    // Value per unit ID. A bunker that dies in the simulation becomes marines that keep
    // its ID, so they are summed back into it.
    std::unordered_map<int, int> valueById(const std::vector<FAP::FAPUnit<>> &units)
    {
        std::unordered_map<int, int> result;
        for (auto &unit : units)
        {
            result[unit.id] += UnitValue::scaled(unit.unitType, unit.health, unit.shields, unit.maxHealth, unit.maxShields);
        }
        return result;
    }

    int total(const std::unordered_map<int, int> &values)
    {
        int result = 0;
        for (auto &[id, value] : values) result += value;
        return result;
    }

    // Value destroyed, counted per unit and only where a unit's value went down. FAP also
    // simulates shield and HP regeneration, medic healing and bunker repair; netting those
    // against a side's damage made its total rise and reported negative losses.
    int lost(const std::unordered_map<int, int> &before, const std::unordered_map<int, int> &after)
    {
        int result = 0;
        for (auto &[id, value] : before)
        {
            auto it = after.find(id);
            result += std::max(0, value - (it == after.end() ? 0 : it->second));
        }
        return result;
    }

    struct DisjointSet
    {
        std::vector<int> parent;

        explicit DisjointSet(size_t n) : parent(n)
        {
            std::iota(parent.begin(), parent.end(), 0);
        }

        int find(int i)
        {
            while (parent[i] != i) i = parent[i] = parent[parent[i]];
            return i;
        }

        void unite(int a, int b)
        {
            parent[find(a)] = find(b);
        }
    };
}

FightFinder::FightFinder(int linkRadius, int simFrames)
        : linkRadius(linkRadius)
        , simFrames(simFrames)
{
    // FAP's collision grid has 2x2 cells per build tile
    collisionPlayer1.resize(BWAPI::Broodwar->mapWidth() * BWAPI::Broodwar->mapHeight() * 4, 0);
    collisionPlayer2.resize(BWAPI::Broodwar->mapWidth() * BWAPI::Broodwar->mapHeight() * 4, 0);
}

std::vector<Fight> FightFinder::find()
{
    std::vector<BWAPI::Unit> units;
    std::vector<bool> armed;
    for (auto unit : BWAPI::Broodwar->getAllUnits())
    {
        if (!isSimUnit(unit)) continue;
        units.push_back(unit);
        armed.push_back(canAttack(unit));
    }

    // Bucket units on a grid of linkRadius-sized cells so each unit is only compared
    // against its own and neighbouring cells
    auto cellKey = [&](int cx, int cy)
    {
        return ((int64_t)cx << 32) | (uint32_t)cy;
    };
    std::unordered_map<int64_t, std::vector<int>> grid;
    for (int i = 0; i < (int)units.size(); i++)
    {
        auto pos = units[i]->getPosition();
        grid[cellKey(pos.x / linkRadius, pos.y / linkRadius)].push_back(i);
    }

    DisjointSet groups(units.size());
    int radiusSquared = linkRadius * linkRadius;
    for (int i = 0; i < (int)units.size(); i++)
    {
        auto pos = units[i]->getPosition();
        int cx = pos.x / linkRadius;
        int cy = pos.y / linkRadius;
        for (int dx = -1; dx <= 1; dx++)
        {
            for (int dy = -1; dy <= 1; dy++)
            {
                auto it = grid.find(cellKey(cx + dx, cy + dy));
                if (it == grid.end()) continue;

                for (int j : it->second)
                {
                    if (j <= i || (!armed[i] && !armed[j])) continue;
                    if (pos.getDistance(units[j]->getPosition()) * pos.getDistance(units[j]->getPosition()) > radiusSquared) continue;
                    groups.unite(i, j);
                }
            }
        }
    }

    std::unordered_map<int, std::vector<BWAPI::Unit>> byGroup;
    for (int i = 0; i < (int)units.size(); i++)
    {
        byGroup[groups.find(i)].push_back(units[i]);
    }

    std::vector<Fight> fights;
    for (auto &[root, group] : byGroup)
    {
        // Only groups where opposing players meet and someone can fight
        bool hasEnemies = std::any_of(group.begin(), group.end(), [&](BWAPI::Unit unit)
        {
            return unit->getPlayer()->isEnemy(group.front()->getPlayer());
        });
        bool hasArmed = std::any_of(group.begin(), group.end(), canAttack);
        if (!hasEnemies || !hasArmed) continue;

        fights.push_back(simulate(group));
    }

    return fights;
}

Fight FightFinder::simulate(const std::vector<BWAPI::Unit> &group)
{
    // Side A is the player with the most value here; side B is everyone hostile to them
    std::unordered_map<BWAPI::Player, int> valueByPlayer;
    for (auto unit : group)
    {
        valueByPlayer[unit->getPlayer()] += UnitValue::scaled(unit->getType(), unit->getHitPoints(), unit->getShields());
    }
    auto playerA = std::max_element(valueByPlayer.begin(), valueByPlayer.end(), [](auto &a, auto &b)
    {
        return a.second < b.second;
    })->first;

    std::fill(collisionPlayer1.begin(), collisionPlayer1.end(), 0);
    std::fill(collisionPlayer2.begin(), collisionPlayer2.end(), 0);
    FAP::FastAPproximation sim(collisionPlayer1, collisionPlayer2);

    Fight fight{};
    fight.playerA = playerA->getID();
    fight.playerB = INT_MIN;
    fight.left = fight.top = INT_MAX;
    fight.right = fight.bottom = INT_MIN;

    int64_t weightedX = 0, weightedY = 0, totalWeight = 0;
    for (auto unit : group)
    {
        auto player = unit->getPlayer();
        if (player != playerA && !player->isEnemy(playerA)) continue;

        if (player == playerA)
        {
            sim.addPlayer1(makeUnit(unit, group));
            fight.unitsA++;
        }
        else
        {
            sim.addPlayer2(makeUnit(unit, group));
            fight.unitsB++;
            if (fight.playerB == INT_MIN) fight.playerB = player->getID();
            else if (fight.playerB != player->getID()) fight.playerB = -1;
        }

        auto pos = unit->getPosition();
        int weight = UnitValue::full(unit->getType()) + 1;
        weightedX += (int64_t)pos.x * weight;
        weightedY += (int64_t)pos.y * weight;
        totalWeight += weight;
        fight.left = std::min(fight.left, unit->getLeft());
        fight.top = std::min(fight.top, unit->getTop());
        fight.right = std::max(fight.right, unit->getRight());
        fight.bottom = std::max(fight.bottom, unit->getBottom());
    }

    fight.x = (int)(weightedX / totalWeight);
    fight.y = (int)(weightedY / totalWeight);

    auto state = sim.getState();
    auto beforeA = valueById(*state.first);
    auto beforeB = valueById(*state.second);
    fight.valueA = total(beforeA);
    fight.valueB = total(beforeB);

    sim.simulate(simFrames);

    state = sim.getState();
    fight.lossA = lost(beforeA, valueById(*state.first));
    fight.lossB = lost(beforeB, valueById(*state.second));

    return fight;
}
