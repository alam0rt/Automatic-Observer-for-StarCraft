from observer.extract import find_replays


def touch(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")
    return path


def test_sc_docker_game_dirs_use_first_player_replay(tmp_path):
    first = touch(tmp_path / "GAME_A" / "player_0.rep")
    touch(tmp_path / "GAME_A" / "player_1.rep")

    assert find_replays(tmp_path) == {"GAME_A": first}


def test_stardata_shards_keep_every_replay(tmp_path):
    a = touch(tmp_path / "stardata_original_replays" / "0" / "IC_PvP_IC323654.rep")
    b = touch(tmp_path / "stardata_original_replays" / "0" / "IC_PvP_IC348188.rep")
    c = touch(tmp_path / "stardata_original_replays" / "1" / "IC_TvZ_IC100001.rep")

    assert find_replays(tmp_path) == {"IC_PvP_IC323654": a, "IC_PvP_IC348188": b, "IC_TvZ_IC100001": c}


def test_loose_replays_are_named_by_stem(tmp_path):
    replay = touch(tmp_path / "some_game.rep")

    assert find_replays(tmp_path) == {"some_game": replay}
