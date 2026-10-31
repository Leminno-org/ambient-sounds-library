"""Verify original procedural sound identity, byte integrity, loops, and release separation.

No media gets published by this test. Human playback/listening remains a separate
acceptance gate: numerical checks cannot prove realistic environments or pleasant loops.
"""
import hashlib
import json
import math
import sys
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from generate_original_sounds import APP_CATALOG, DURATION, RATE, RECIPES, RECIPE_VERSION, STAGING

def main():
    inventory = json.loads((ROOT / "staged-originals.json").read_text(encoding="utf-8"))
    sounds = inventory["sounds"]
    planned = {x["id"]: x for x in json.loads(APP_CATALOG.read_text(encoding="utf-8"))["sounds"]}
    published = json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))["sounds"]
    locked_originals = json.loads((ROOT / "legacy-release-lock.json").read_text(encoding="utf-8"))["sounds"]
    assert len(locked_originals) == 7, "Seven legacy pinned release assets must stay present"
    assert all("DNAABR/ambient-sounds-library/releases/download/" in x["url"] for x in locked_originals)
    published_map = {x["sound_id"]: x for x in published}
    assert len(published_map) == len(published), "Duplicate published ID"
    for row in locked_originals:
        assert published_map[row["sound_id"]] == row, f"Legacy release metadata changed: {row['sound_id']}"
    originally_recorded = {x["sound_id"] for x in locked_originals}
    bundled_recordings = json.loads((ROOT / "bundled-cc0-recordings.json").read_text(encoding="utf-8"))["sounds"]
    previously_recorded = originally_recorded | {x["sound_id"] for x in bundled_recordings}
    assert len(previously_recorded) == 17
    assert len(sounds) == inventory["sound_count"] == len(RECIPES), "All distinct recipes must render"
    assert {x["sound_id"] for x in sounds} == set(RECIPES)
    assert not (set(RECIPES) & previously_recorded), "Do not relabel CC0 recordings as original procedural sounds"
    assert inventory["generation_version"] == RECIPE_VERSION
    assert inventory["status"] == "staged_unpublished"
    app = json.loads((ROOT.parent / "ambient-sounds" / "app" / "src" / "main" / "assets" / "catalog.json").read_text(encoding="utf-8"))
    runtime = {x["id"]: x for x in app["sounds"]}
    assert previously_recorded <= runtime.keys(), "App lost previously recorded IDs"
    for sound_id in set(RECIPES) & runtime.keys():
        staged_row = next(x for x in sounds if x["sound_id"] == sound_id)
        assert runtime[sound_id]["assetSha256"] == staged_row["asset_sha256"], f"Runtime promoted obsolete PCM: {sound_id}"
    original_engine_generators = {x["id"] for x in planned.values() if x["playback_kind"] == "generator"}
    assert len(original_engine_generators) == 5
    baseline_ids = previously_recorded | original_engine_generators

    keys, fingerprints, lengths = set(), [], []
    for entry in sounds:
        sid = entry["sound_id"]
        assert sid in planned, f"Unknown planned ID {sid}"
        assert sid not in keys, f"Duplicate inventory entry {sid}"
        keys.add(sid)
        assert entry["source_type"] == "original_procedural_audio"
        assert entry["recording_claim"] is False and entry["url"] is None
        assert entry["distribution_status"] == "staged_unpublished"
        assert entry["license"] == "CC0-1.0"
        assert entry["technical_measurement"]["listening_approved"] is False
        assert entry["recipe_id"] == RECIPES[sid][0]
        assert entry["playback_kind"] == planned[sid]["playback_kind"]
        src = STAGING / entry["file"]
        assert src.is_file(), f"Missing WAV: {src}"
        assert src.stat().st_size == entry["asset_bytes"]
        with src.open("rb") as source:
            sha = hashlib.file_digest(source, "sha256").hexdigest()
        assert sha == entry["asset_sha256"]
        with wave.open(str(src), "rb") as wavefile:
            assert (wavefile.getframerate(),wavefile.getnchannels(),wavefile.getsampwidth(),
                    wavefile.getcomptype(),wavefile.getnframes()) == (RATE,2,2,"NONE",RATE*DURATION)
            pcm = np.frombuffer(wavefile.readframes(RATE*DURATION), dtype="<i2")
        assert pcm.size == RATE*DURATION*2
        data = pcm.reshape(-1, 2).astype(np.float64) / 32768
        peak = float(np.max(np.abs(data)))
        rms = float(np.sqrt(np.mean(data ** 2)))
        boundary = float(np.max(np.abs(data[0] - data[-1])))
        dc = float(np.max(np.abs(data.mean(axis=0))))
        assert math.isfinite(rms) and .001 < rms < .11, f"{sid}: unusable loudness {rms}"
        assert 0.11 <= peak <= .1401, f"{sid}: peak or silent {peak}"
        assert boundary < .00012, f"{sid}: loop join is not sample-continuous {boundary}"
        assert dc < .002, f"{sid}: DC offset {dc}"
        assert np.mean(np.abs(np.diff(data, axis=0))) > .00005, f"{sid}: static output"
        # Decimated fingerprints catch renamed identical sounds and excessive
        # linear correlation without gigabytes of pairwise array materialization.
        finger = data[80000::96, 0]
        fingerprints.append(finger)
        lengths.append(src.stat().st_size)
        print(f"{sid:25} RMS={20*math.log10(rms):6.1f}dBFS "
              f"seam={boundary:.5f} SHA256={sha[:14]}")
    assert len(set(x["asset_sha256"] for x in sounds)) == len(sounds)
    strongest = (0., "", "")
    for i in range(len(sounds)):
        for j in range(i + 1, len(sounds)):
            a, b = fingerprints[i], fingerprints[j]
            correlation = float(abs(np.corrcoef(a,b)[0,1]))
            if correlation > strongest[0]:
                strongest = (correlation, sounds[i]["sound_id"], sounds[j]["sound_id"])
            assert correlation < .96, f"Likely recycled/pitched source: {sounds[i]['sound_id']} vs {sounds[j]['sound_id']} ({correlation:.3f})"

    published_total = len(published)
    beta = sum(planned[sid]["beta"] for sid in baseline_ids | set(RECIPES))
    starter = set(x["id"] for x in planned.values() if x.get("bundled_starter"))
    assert starter <= (baseline_ids | set(RECIPES)), "24 planned starter IDs must be backed"
    assert beta == 60, f"Beta candidate ID coverage incomplete: {beta}/60"
    print(f"PASS: {len(sounds)} distinct staged procedural outputs, {len(locked_originals)} protected legacy CC0 release URLs, {published_total} currently published entries, "
          f"{len(starter)} candidate starter IDs backed, {beta} beta IDs backed (including existing app catalog)")
    print(f"PCM staging: {sum(lengths):,} bytes, strongest pair correlation {strongest}")
    print("NOT a release approval: synthetic impressions remain pending listening review and app integration")

if __name__ == "__main__":
    main()
