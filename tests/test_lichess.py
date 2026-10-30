import json

import pandas as pd
import pytest
import requests

from src.features import GAME_COLUMNS, build_player_profiles
from src.lichess import LichessError, fetch_games, parse_ndjson, validate_username


@pytest.fixture(autouse=True)
def lichess_token(monkeypatch):
    monkeypatch.setenv("LICHESS_TOKEN", "lip_test_token")


def game(winner="white", status="resign", moves="e4 e5 Nf3 Nc6", with_clock=True):
    g = {"players": {"white": {"user": {"id": "Alice"}, "rating": 1800}, "black": {"user": {"id": "Bob"}, "rating": 1750}},
         "status": status, "moves": moves, "opening": {"name": "Italian Game"}}
    if winner:
        g["winner"] = winner
    if with_clock:
        g["clock"] = {"initial": 180, "increment": 2}
    return json.dumps(g)


class FakeResponse:
    def __init__(self, status=200, text=""):
        self.status_code, self.text = status, text


class FakeSession:
    def __init__(self, response=None, exc=None):
        self.response, self.exc, self.calls = response, exc, []

    def get(self, url, **kw):
        self.calls.append((url, kw))
        if self.exc:
            raise self.exc
        return self.response


def test_parse_ndjson_shapes_rows_like_training_data():
    df = parse_ndjson("\n".join([game(), game(winner=None, status="draw"), game(status="noStart"), game(with_clock=False), "not json"]))
    assert list(df.columns) == GAME_COLUMNS and len(df) == 2          # unusable games skipped, not invented
    assert df.loc[0, "turns"] == 4 and df.loc[0, "increment_code"] == "3+2" and df.loc[0, "white_id"] == "alice"
    assert df.loc[1, "winner"] == "draw"


def test_profile_from_parsed_games():
    df = parse_ndjson("\n".join([game()] * 12))
    p = build_player_profiles(df, min_games=10, players=["alice"])
    assert p.loc[0, "n_games"] == 12 and p.loc[0, "win_rate"] == 1.0 and p.loc[0, "fav_speed"] == "blitz"


@pytest.mark.parametrize("name", ["", "a", "bad name", "../etc", "x" * 31, "user;rm"])
def test_invalid_usernames_rejected(name):
    with pytest.raises(LichessError):
        validate_username(name)


def test_fetch_error_messages_and_params():
    with pytest.raises(LichessError, match="does not exist or their games"):
        fetch_games("alice", session=FakeSession(FakeResponse(404)))
    with pytest.raises(LichessError, match="429"):
        fetch_games("alice", session=FakeSession(FakeResponse(429)))
    with pytest.raises(LichessError, match="in time"):
        fetch_games("alice", session=FakeSession(exc=requests.exceptions.Timeout()))
    with pytest.raises(LichessError, match="No usable"):
        fetch_games("alice", session=FakeSession(FakeResponse(200, "")))
    ok = FakeSession(FakeResponse(200, game()))
    assert len(fetch_games("alice", "rapid", 50, session=ok)) == 1
    url, kw = ok.calls[0]
    assert url.endswith("/alice") and kw["params"]["perfType"] == "rapid" and kw["params"]["rated"] == "true"


def test_token_required_and_sent(monkeypatch):
    monkeypatch.delenv("LICHESS_TOKEN", raising=False)
    with pytest.raises(LichessError, match="LICHESS_TOKEN"):
        fetch_games("alice", session=FakeSession(FakeResponse(200)))

    ok = FakeSession(FakeResponse(200, game()))
    fetch_games("alice", session=ok, token="lip_explicit")
    assert ok.calls[0][1]["headers"]["Authorization"] == "Bearer lip_explicit"


@pytest.mark.parametrize("code", [401, 403])
def test_rejected_token(code):
    with pytest.raises(LichessError, match="rejected"):
        fetch_games("alice", session=FakeSession(FakeResponse(code)))
