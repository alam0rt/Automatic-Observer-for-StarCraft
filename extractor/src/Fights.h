#pragma once

#include <BWAPI.h>

#include <vector>

// A group of units from opposing players close enough to interact, with FAP's
// estimate of how much value each side loses over the next simFrames frames.
struct Fight
{
    int x, y;                       // value-weighted centre, pixels
    int left, top, right, bottom;   // bounding box, pixels
    int playerA;                    // the player with the most value in the group
    int playerB;                    // their enemy, or -1 if several enemies are merged
    int unitsA, unitsB;
    int valueA, valueB;             // value before simulating
    int lossA, lossB;               // value lost after simulating
};

class FightFinder
{
public:
    // linkRadius: units within this many pixels of each other are grouped, as
    // long as at least one of the pair can attack.
    FightFinder(int linkRadius, int simFrames);

    std::vector<Fight> find();

private:
    int linkRadius;
    int simFrames;

    std::vector<unsigned char> collisionPlayer1;
    std::vector<unsigned char> collisionPlayer2;

    Fight simulate(const std::vector<BWAPI::Unit> &group);
};
