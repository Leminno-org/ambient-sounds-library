"""Promote verified GitHub release WAVs into the published-only public manifest.

This tool refuses to invent URLs, alter legacy release entries, or promote
partially uploaded assets. It requires GitHub's own computed asset SHA-256.
"""
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = "Leminno-org/ambient-sounds-library"
TAG = "v0.4.0-preview"
URL_ROOT = f"https://github.com/{REPO}/releases/download/{TAG}/"

def read(path):
    return json.loads(path.read_text(encoding="utf-8"))

def main():
    staged = read(ROOT / "staged-originals.json")["sounds"]
    locked = read(ROOT / "legacy-release-lock.json")["sounds"]
    old = read(ROOT / "manifest.json")
    old_rows = {x["sound_id"]: x for x in old["sounds"]}
    for row in locked:
        assert old_rows[row["sound_id"]] == row, f"Historical published URL or hash changed: {row['sound_id']}"
    result = subprocess.run([
        "gh", "release", "view", TAG, "--repo", REPO,
        "--json", "assets,isDraft,isPrerelease,tagName",
    ], check=True, capture_output=True, text=True)
    release = json.loads(result.stdout)
    assert release["tagName"] == TAG and not release["isDraft"] and release["isPrerelease"]
    assets = {x["name"]: x for x in release["assets"]}
    assert len(assets) == len(staged) == 41
    out = list(locked)
    for row in staged:
        asset = assets[row["file"]]
        assert asset["digest"] == "sha256:" + row["asset_sha256"]
        assert asset["size"] == row["asset_bytes"] and asset["state"] == "uploaded"
        assert row["sound_id"] not in old_rows or old_rows[row["sound_id"]]["asset_sha256"] == row["asset_sha256"]
        url = URL_ROOT + row["file"]
        out.append({
            "sound_id": row["sound_id"],
            "source_type": "original_procedural_audio",
            "creator": row["creator"],
            "actual_source": row["actual_source"],
            "sounds_like": row["sounds_like"],
            "recording_claim": False,
            "recipe_id": row["recipe_id"],
            "recipe_version": row["recipe_version"],
            "seed_derivation": row["seed_derivation"],
            "license": row["license"],
            "license_url": row["license_url"],
            "asset_sha256": row["asset_sha256"],
            "asset_bytes": row["asset_bytes"],
            "duration_seconds": row["duration_seconds"],
            "sample_rate": row["sample_rate"],
            "channels": row["channels"],
            "encoding": row["encoding"],
            "file": row["file"],
            "distribution_status": "published_preview",
            "url": url,
            "listening_approved": False,
            "technical_measurement": row["technical_measurement"],
        })
    assert len({x["sound_id"] for x in out}) == len(out) == 48
    manifest = {
        "version": "0.4.0-preview",
        "status": "engineering_preview",
        "content_warning": "Procedural audio is an artificial sound-design impression, not an environmental field recording; listening approval is pending.",
        "sounds": out,
    }
    target = ROOT / "manifest.json"
    target.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Promoted 41 independently verified hosted original WAVs; published manifest has {len(out)} entries.")
    print(f"All 7 historical pinned CC0 entries unchanged: {target}")

if __name__ == "__main__":
    main()
