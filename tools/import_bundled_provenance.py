"""Mirror only safe source facts for existing app-bundled CC0 recordings.

These files already exist in the app, and this command does NOT create/rename,
replace or newly publish their audio. Licensing is read from recorded prior
source evidence; no claim is made about access to any additional sound.
"""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT.parent / "ambient-sounds"

def read(p):
    return json.loads(p.read_text(encoding="utf-8"))

def main():
    published = {x["sound_id"] for x in read(ROOT / "manifest.json")["sounds"]}
    catalog = read(APP / "app" / "src" / "main" / "assets" / "catalog.json")
    rows = []
    for sound in catalog["sounds"]:
        sid = sound["id"]
        if sound["kind"] == "generator" or sid in published: continue
        assert sound.get("bundled", True), f"Missing source for optional non-published recording: {sid}"
        source = read(APP / "catalog" / "source-evidence" / (sid + ".json"))
        assert source["sound_id"] == sid and source["license"] == "CC0-1.0"
        audio = APP / "app" / "src" / "main" / "assets" / "audio" / (sid + ".wav")
        with audio.open("rb") as stream:
            sha = hashlib.file_digest(stream, "sha256").hexdigest()
        assert sha == source["asset_sha256"] and audio.stat().st_size == source["asset_bytes"]
        rows.append({
            "sound_id": sid,
            "source_type": "third_party_cc0_recording",
            "creator": source["creator"],
            "original_title": source["original_title"],
            "source_url": source["source_url"],
            "license": source["license"],
            "license_url": source["license_url"],
            "asset_sha256": sha,
            "asset_bytes": audio.stat().st_size,
            "sample_rate": source["sample_rate"],
            "channels": source["channels"],
            "encoding": source["encoding"],
            "duration_seconds": source["duration_seconds"],
            "distribution_status": "already_bundled_inside_android_app",
            "public_release_url": None,
            "listening_approved": bool(source["listening_approval"]),
        })
    target = ROOT / "bundled-cc0-recordings.json"
    target.write_text(json.dumps({"schema_version":1,"status":"metadata_only","count":len(rows),"sounds":sorted(rows,key=lambda x:x["sound_id"])}, indent=2)+"\n", encoding="utf-8")
    print(f"Preserved source evidence for {len(rows)} existing bundled third-party CC0 recordings: {target}")

if __name__ == "__main__":
    main()
