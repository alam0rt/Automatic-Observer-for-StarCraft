// sc-extract: plays a Brood War replay headlessly in OpenBW and writes per-frame
// game state for training an automatic observer.
//
//   sc-extract <replay.rep> <out-dir> [--interval N] [--sim-frames N] [--link-radius PX] [--no-fights]
//
// Output files in <out-dir>:
//   meta.json   map, players and extraction settings
//   unit_types.json  BWAPI unit type table (name, race, building, worker, flyer, value)
//   map.json    tile grids (ground height, walkability, buildability) and BWEM areas, chokepoints and bases
//   units.csv   every non-neutral unit, every --interval frames
//   events.csv  unit create / destroy / merge / morph / renegade events, every frame
//               (merge: a templar absorbed into an archon, which BWAPI reports as destroyed)
//   fights.csv  groups of opposing units with FAP's predicted value loss over --sim-frames

#include "Fights.h"
#include "UnitValue.h"

#include <BWAPI.h>
#include "BWAPI/GameImpl.h"
#include "BW/BWData.h"
#include <bwem.h>
#include <nlohmann/json.hpp>

#include <iconv.h>

#include <chrono>
#include <climits>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <optional>
#include <set>
#include <unordered_map>

namespace
{
    struct Options
    {
        std::string replay;
        std::filesystem::path outDir;
        int interval = 8;
        int simFrames = 96;
        int linkRadius = 224;
        bool fights = true;
    };

    void usage()
    {
        std::cerr << "usage: sc-extract <replay.rep> <out-dir> [--interval N] [--sim-frames N] [--link-radius PX] [--no-fights]\n"
                  << "  --interval N      write units and fights every N frames (default 8)\n"
                  << "  --sim-frames N    frames FAP simulates ahead for each fight (default 96)\n"
                  << "  --link-radius PX  units closer than this are grouped into one fight (default 224)\n"
                  << "  --no-fights       skip fight detection and simulation\n"
                  << "OPENBW_MPQ_PATH must point at a directory containing StarDat.mpq, BrooDat.mpq and Patch_rt.mpq.\n";
    }

    std::optional<Options> parseArgs(int argc, char **argv)
    {
        Options options;
        std::vector<std::string> positional;
        for (int i = 1; i < argc; i++)
        {
            std::string arg = argv[i];
            auto intValue = [&](int &out)
            {
                if (i + 1 >= argc) return false;
                out = std::stoi(argv[++i]);
                return out > 0;
            };

            if (arg == "--interval")
            {
                if (!intValue(options.interval)) return std::nullopt;
            }
            else if (arg == "--sim-frames")
            {
                if (!intValue(options.simFrames)) return std::nullopt;
            }
            else if (arg == "--link-radius")
            {
                if (!intValue(options.linkRadius)) return std::nullopt;
            }
            else if (arg == "--no-fights")
            {
                options.fights = false;
            }
            else if (arg == "-h" || arg == "--help" || arg.rfind("--", 0) == 0)
            {
                return std::nullopt;
            }
            else
            {
                positional.push_back(arg);
            }
        }

        if (positional.size() != 2) return std::nullopt;
        options.replay = positional[0];
        options.outDir = positional[1];
        return options;
    }

    int coord(int value, bool valid)
    {
        return valid ? value : -1;
    }

    bool isUtf8(const std::string &text)
    {
        for (size_t i = 0; i < text.size();)
        {
            auto byte = (unsigned char)text[i];
            size_t length = byte < 0x80 ? 1 : (byte >> 5) == 0x6 ? 2 : (byte >> 4) == 0xE ? 3 : (byte >> 3) == 0x1E ? 4 : 0;
            if (length == 0 || i + length > text.size()) return false;
            for (size_t k = 1; k < length; k++)
            {
                if (((unsigned char)text[i + k] >> 6) != 0x2) return false;
            }
            i += length;
        }
        return true;
    }

    // Map and player names in Korean ladder replays (most of StarData) are CP949, and
    // nlohmann::json throws on anything that isn't UTF-8 when dumping. Convert them,
    // replacing whatever still doesn't decode with '?'.
    std::string toUtf8(const std::string &text)
    {
        if (isUtf8(text)) return text;

        std::string result;
        iconv_t cd = iconv_open("UTF-8", "CP949");
        if (cd != (iconv_t)-1)
        {
            char *in = const_cast<char *>(text.data());
            size_t inLeft = text.size();
            std::string buffer(text.size() * 4 + 4, '\0');
            while (inLeft > 0)
            {
                char *out = buffer.data();
                size_t outLeft = buffer.size();
                size_t converted = iconv(cd, &in, &inLeft, &out, &outLeft);
                result.append(buffer.data(), out);
                if (converted == (size_t)-1 && inLeft > 0)
                {
                    result += '?';
                    in++;
                    inLeft--;
                }
            }
            iconv_close(cd);
        }
        if (isUtf8(result)) return result;

        result.clear();
        for (char c : text) result += (unsigned char)c < 0x80 ? c : '?';
        return result;
    }

    nlohmann::json pixel(BWAPI::WalkPosition walk)
    {
        auto pos = BWAPI::Position(walk) + BWAPI::Position(4, 4);
        return {pos.x, pos.y};
    }

    class ExtractorModule : public BWAPI::AIModule
    {
    public:
        explicit ExtractorModule(const Options &options)
                : options(options)
                , units(options.outDir / "units.csv")
                , events(options.outDir / "events.csv")
        {
            units << "frame,id,player,type,x,y,hp,shields,energy,value,flying,burrowed,cloaked,attacking,under_attack,completed,order,order_x,order_y\n";
            events << "frame,event,id,player,type,x,y,value\n";
            if (options.fights)
            {
                fights.open(options.outDir / "fights.csv");
                fights << "frame,x,y,left,top,right,bottom,player_a,player_b,units_a,units_b,value_a,value_b,loss_a,loss_b\n";
            }
        }

        void onStart() override
        {
            UnitValue::initialize();
            writeUnitTypes();
            writeMap();
            if (options.fights) fightFinder = std::make_unique<FightFinder>(options.linkRadius, options.simFrames);
        }

        void onFrame() override
        {
            int frame = BWAPI::Broodwar->getFrameCount();
            lastFrame = frame;
            trackUnits(frame);
            if (frame % options.interval != 0) return;

            std::map<int, int> unitsPerPlayer;
            for (auto unit : BWAPI::Broodwar->getAllUnits())
            {
                if (!unit->exists() || unit->getPlayer()->isNeutral()) continue;
                playersSeen.insert(unit->getPlayer());
                unitsPerPlayer[unit->getPlayer()->getID()]++;

                auto pos = unit->getPosition();
                auto order = unit->getOrderTargetPosition();
                units << frame << ','
                      << unit->getID() << ','
                      << unit->getPlayer()->getID() << ','
                      << unit->getType().getID() << ','
                      << coord(pos.x, pos.isValid()) << ','
                      << coord(pos.y, pos.isValid()) << ','
                      << unit->getHitPoints() << ','
                      << unit->getShields() << ','
                      << unit->getEnergy() << ','
                      << UnitValue::scaled(unit->getType(), unit->getHitPoints(), unit->getShields()) << ','
                      << unit->isFlying() << ','
                      << unit->isBurrowed() << ','
                      << unit->isCloaked() << ','
                      << unit->isAttacking() << ','
                      << underAttack(unit, frame) << ','
                      << unit->isCompleted() << ','
                      << unit->getOrder().getID() << ','
                      << coord(order.x, order.isValid()) << ','
                      << coord(order.y, order.isValid()) << '\n';
            }
            for (auto &[player, count] : unitsPerPlayer)
            {
                peakUnits[player] = std::max(peakUnits[player], count);
            }

            if (fightFinder)
            {
                for (auto &fight : fightFinder->find())
                {
                    fights << frame << ','
                           << fight.x << ',' << fight.y << ','
                           << fight.left << ',' << fight.top << ',' << fight.right << ',' << fight.bottom << ','
                           << fight.playerA << ',' << fight.playerB << ','
                           << fight.unitsA << ',' << fight.unitsB << ','
                           << fight.valueA << ',' << fight.valueB << ','
                           << fight.lossA << ',' << fight.lossB << '\n';
                }
            }

            if (frame % (24 * 60) == 0)
            {
                std::cerr << "\r  " << options.replay << ": " << frame << " / " << BWAPI::Broodwar->getReplayFrameCount() << " frames" << std::flush;
            }
        }

        void onUnitCreate(BWAPI::Unit unit) override
        {
            event("create", unit);
        }

        void onUnitDestroy(BWAPI::Unit unit) override
        {
            // Unit events are handled before onFrame, so this is the templar's state on
            // the frame before it disappeared
            auto it = tracked.find(unit->getID());
            bool merged = it != tracked.end() && it->second.merging;
            event(merged ? "merge" : "destroy", unit);
            if (it != tracked.end()) tracked.erase(it);
        }

        void onUnitMorph(BWAPI::Unit unit) override
        {
            event("morph", unit);
        }

        void onUnitRenegade(BWAPI::Unit unit) override
        {
            event("renegade", unit);
        }

        int framesPlayed() const
        {
            return lastFrame;
        }

        void writeMeta(double elapsedSeconds)
        {
            nlohmann::json players = nlohmann::json::array();
            for (auto player : playersSeen)
            {
                auto start = player->getStartLocation();
                players.push_back({
                                          {"id",    player->getID()},
                                          {"name",  toUtf8(player->getName())},
                                          {"race",  player->getRace().getName()},
                                          {"start", {start.x, start.y}},
                                          {"peak_units", peakUnits[player->getID()]},
                                  });
            }

            nlohmann::json meta = {
                    {"replay",             std::filesystem::absolute(options.replay).string()},
                    {"map_file",           toUtf8(BWAPI::Broodwar->mapFileName())},
                    {"map_name",           toUtf8(BWAPI::Broodwar->mapName())},
                    {"map_hash",           BWAPI::Broodwar->mapHash()},
                    {"map_width_tiles",    BWAPI::Broodwar->mapWidth()},
                    {"map_height_tiles",   BWAPI::Broodwar->mapHeight()},
                    {"replay_frame_count", BWAPI::Broodwar->getReplayFrameCount()},
                    {"last_frame",         lastFrame},
                    {"interval",           options.interval},
                    {"sim_frames",         options.fights ? options.simFrames : 0},
                    {"link_radius",        options.fights ? options.linkRadius : 0},
                    {"players",            players},
                    // When OpenBW desyncs from a replay, both sides stall with their starting
                    // units: a handful of creates over the whole game. See observer.features.played_out
                    {"event_counts",       eventCounts},
                    {"elapsed_seconds",    elapsedSeconds},
            };
            std::ofstream(options.outDir / "meta.json") << meta.dump(2) << '\n';
        }

    private:
        // A unit counts as under attack for this many frames after it last lost hit
        // points or shields (one game second)
        static constexpr int underAttackFrames = 24;

        struct TrackedUnit
        {
            BWAPI::UnitType type;
            int hitPointsAndShields;
            int lastDamagedFrame = INT_MIN / 2;
            bool merging = false;
        };

        Options options;
        std::ofstream units;
        std::ofstream events;
        std::ofstream fights;
        std::unique_ptr<FightFinder> fightFinder;
        std::set<BWAPI::Player> playersSeen;
        std::unordered_map<int, TrackedUnit> tracked;
        std::map<std::string, int> eventCounts;
        std::map<int, int> peakUnits;  // most units a player had at once, by player ID
        int lastFrame = 0;

        // OpenBW's BWAPI never sets isUnderAttack (UnitUpdate.cpp hard-codes
        // recentlyAttacked = false), so watch every unit's hit points and shields each
        // frame instead. Also remembers which templars are merging into archons.
        void trackUnits(int frame)
        {
            for (auto unit : BWAPI::Broodwar->getAllUnits())
            {
                if (!unit->exists() || unit->getPlayer()->isNeutral()) continue;

                auto type = unit->getType();
                int current = unit->getHitPoints() + unit->getShields();
                auto [it, inserted] = tracked.try_emplace(unit->getID(), TrackedUnit{type, current});
                auto &state = it->second;

                // Terran buildings below a third of their hit points burn down on their own
                bool burning = type.getRace() == BWAPI::Races::Terran && type.isBuilding()
                               && unit->getHitPoints() < type.maxHitPoints() / 3;
                if (!inserted && state.type == type && current < state.hitPointsAndShields && !burning)
                {
                    state.lastDamagedFrame = frame;
                }

                state.type = type;
                state.hitPointsAndShields = current;
                state.merging = unit->getOrder() == BWAPI::Orders::ArchonWarp || unit->getOrder() == BWAPI::Orders::DarkArchonMeld;
            }
        }

        bool underAttack(BWAPI::Unit unit, int frame) const
        {
            auto it = tracked.find(unit->getID());
            return it != tracked.end() && frame - it->second.lastDamagedFrame <= underAttackFrames;
        }

        void event(const char *name, BWAPI::Unit unit)
        {
            if (unit->getPlayer()->isNeutral()) return;
            eventCounts[name]++;

            auto pos = unit->getPosition();
            events << BWAPI::Broodwar->getFrameCount() << ','
                   << name << ','
                   << unit->getID() << ','
                   << unit->getPlayer()->getID() << ','
                   << unit->getType().getID() << ','
                   << coord(pos.x, pos.isValid()) << ','
                   << coord(pos.y, pos.isValid()) << ','
                   << UnitValue::full(unit->getType()) << '\n';
        }

        void writeUnitTypes()
        {
            nlohmann::json types = nlohmann::json::object();
            for (auto type : BWAPI::UnitTypes::allUnitTypes())
            {
                types[std::to_string(type.getID())] = {
                        {"name",       type.getName()},
                        {"race",       type.getRace().getName()},
                        {"building",   type.isBuilding()},
                        {"worker",     type.isWorker()},
                        {"flyer",      type.isFlyer()},
                        {"can_attack", type.canAttack() || type == BWAPI::UnitTypes::Terran_Bunker || type == BWAPI::UnitTypes::Protoss_Carrier || type == BWAPI::UnitTypes::Protoss_Reaver},
                        {"value",      UnitValue::full(type)},
                };
            }
            std::ofstream(options.outDir / "unit_types.json") << types.dump() << '\n';
        }

        void writeMap()
        {
            int width = BWAPI::Broodwar->mapWidth();
            int height = BWAPI::Broodwar->mapHeight();

            // Row-major tile grids
            std::vector<int> groundHeight, walkable, buildable;
            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    groundHeight.push_back(BWAPI::Broodwar->getGroundHeight(x, y));
                    buildable.push_back(BWAPI::Broodwar->isBuildable(x, y));

                    int walkableMiniTiles = 0;
                    for (int wy = 0; wy < 4; wy++)
                    {
                        for (int wx = 0; wx < 4; wx++)
                        {
                            walkableMiniTiles += BWAPI::Broodwar->isWalkable(x * 4 + wx, y * 4 + wy);
                        }
                    }
                    walkable.push_back(walkableMiniTiles);
                }
            }

            nlohmann::json map = {
                    {"width_tiles",  width},
                    {"height_tiles", height},
                    {"ground_height", groundHeight},
                    {"walkable_minitiles", walkable},
                    {"buildable", buildable},
            };

            try
            {
                auto &bwem = BWEM::Map::Instance();
                bwem.Initialize(BWAPI::BroodwarPtr);
                bwem.EnableAutomaticPathAnalysis();
                bwem.FindBasesForStartingLocations();

                nlohmann::json areas = nlohmann::json::array();
                nlohmann::json chokepoints = nlohmann::json::array();
                nlohmann::json bases = nlohmann::json::array();
                std::set<const BWEM::ChokePoint *> seenChokes;

                for (auto &area : bwem.Areas())
                {
                    areas.push_back({
                                            {"id",           area.Id()},
                                            {"top_left",     {area.TopLeft().x, area.TopLeft().y}},
                                            {"bottom_right", {area.BottomRight().x, area.BottomRight().y}},
                                            {"top",          pixel(area.Top())},
                                            {"max_altitude", area.MaxAltitude()},
                                    });

                    for (auto choke : area.ChokePoints())
                    {
                        if (!seenChokes.insert(choke).second) continue;
                        chokepoints.push_back({
                                                      {"areas",  {choke->GetAreas().first->Id(), choke->GetAreas().second->Id()}},
                                                      {"center", pixel(choke->Center())},
                                                      {"end1",   pixel(choke->Pos(BWEM::ChokePoint::end1))},
                                                      {"end2",   pixel(choke->Pos(BWEM::ChokePoint::end2))},
                                              });
                    }

                    for (auto &base : area.Bases())
                    {
                        bases.push_back({
                                                {"area",     area.Id()},
                                                {"location", {base.Location().x, base.Location().y}},
                                                {"center",   {base.Center().x, base.Center().y}},
                                        });
                    }
                }

                map["areas"] = areas;
                map["chokepoints"] = chokepoints;
                map["bases"] = bases;
            }
            catch (std::exception &ex)
            {
                std::cerr << "BWEM analysis failed: " << ex.what() << std::endl;
                map["bwem_error"] = ex.what();
            }

            std::ofstream(options.outDir / "map.json") << map.dump() << '\n';
        }
    };
}

int extract(const Options &options)
{
    std::filesystem::create_directories(options.outDir);

    auto start = std::chrono::steady_clock::now();

    BW::GameOwner gameOwner;
    BWAPI::BroodwarImpl_handle h(gameOwner.getGame());
    BWAPI::BroodwarImpl.bwgame.setMapFileName(options.replay);
    h->createSinglePlayerGame([]()
                              {});

    if (!gameOwner.getGame().InReplay())
    {
        std::cerr << "not a replay: " << options.replay << std::endl;
        return 1;
    }

    ExtractorModule module(options);
    module.afterOnStart = [&]()
    {
        h->setLocalSpeed(0);
    };
    h->setAIModule(&module);

    try
    {
        while (!gameOwner.getGame().gameOver())
        {
            gameOwner.getGame().nextFrame();
            h->update();
        }
        h->onGameEnd();
    }
    catch (std::exception &ex)
    {
        std::cerr << std::endl << "replay aborted at frame " << h->getFrameCount() << ": " << ex.what() << std::endl;
        module.writeMeta(std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count());
        return 1;
    }

    double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    module.writeMeta(elapsed);
    std::cerr << "\r  " << options.replay << ": done, " << module.framesPlayed() << " frames in " << elapsed << "s" << std::endl;
    return 0;
}

int main(int argc, char **argv)
{
    auto options = parseArgs(argc, argv);
    if (!options)
    {
        usage();
        return 2;
    }
    if (!std::filesystem::is_regular_file(options->replay))
    {
        std::cerr << "replay not found: " << options->replay << std::endl;
        return 1;
    }

    // Anything escaping here used to abort the process (and dump core) after the whole
    // replay had been played; fail with a message instead
    try
    {
        return extract(*options);
    }
    catch (std::exception &ex)
    {
        std::cerr << std::endl << "extraction failed: " << ex.what() << std::endl;
    }
    catch (...)
    {
        std::cerr << std::endl << "extraction failed: unknown exception" << std::endl;
    }
    return 1;
}
