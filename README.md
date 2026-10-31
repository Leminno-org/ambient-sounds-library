# Ambient Sounds by Leminno — public sound library

Public audio resources for [Ambient Sounds by Leminno](https://github.com/Leminno-org/ambient-sounds).
Every hosted sound has a stable catalog ID, verified PCM byte count, SHA-256
and a pinned versioned GitHub Release link. Downloads are free.

## Distinguish the types of sounds

- **CC0 recordings**: The seven previously released audio assets are individual
  recordings by artists who published them under CC0 1.0. Their creator, title,
  original Freesound URL, license and immutable download URL remain in
  [`manifest.json`](manifest.json). The earlier `DNAABR` GitHub links are
  deliberate existing release URLs and are never rewritten or silently
  replaced. Ten other separately sourced CC0 recordings are already bundled
  in the Android app and have verified metadata in
  [`bundled-cc0-recordings.json`](bundled-cc0-recordings.json).
- **Leminno original procedural audio**: Independently synthesized sound-design
  impressions, built deterministically from mathematical noise, physical
  resonance approximations, oscillators and time envelopes. They contain
  **no captured field recordings**, speech or music. The sound ID describes
  a stylized *impression* of rain, a train cabin, a keyboard, and so on.
  [`staged-originals.json`](staged-originals.json) records their source recipe,
  generation version/seed, PCM checks, SHA-256 and publication state.

Both types are CC0 1.0 for audio use. The original procedural outputs are
dedicated to the public domain in [`ORIGINAL_AUDIO_LICENSE.md`](ORIGINAL_AUDIO_LICENSE.md).
No app trademarks, logos or interfaces are included in the audio license.

## Reproduce original sound files without committing WAVs to Git

Requires Python 3.11+ and NumPy 2.x, plus the adjacent
`ambient-sounds/catalog/planned-sounds.json` repo checkout for ID validation:

```powershell
python tools/generate_original_sounds.py --all
python tools/verify_staged_audio.py
```

This creates 24-second 48 kHz stereo PCM16 WAVs under ignored
`.artifacts/v0.4.0-preview/`. The outputs have crossfaded sample-continuous
loop joins. Individual files are about 4.6 MB, so generated WAVs belong in
versioned **GitHub Release assets**, not Git history. Sound-design validation
is limited to file integrity, levels, DC offset, distinct samples and seam
measurements. Human listening and content-label approval are separate gates.

## Manifest policy and compatibility

The [published manifest](manifest.json) lists only files already uploaded to
immutable release tags. `staged-originals.json` is a deterministic build
provenance snapshot of local files *before publication*; its null URLs and
`staged_unpublished` values describe that generation stage. The separate
published manifest now has the verified
[`v0.4.0-preview`](https://github.com/Leminno-org/ambient-sounds-library/releases/tag/v0.4.0-preview)
links for 41 new procedural files and seven unchanged historical CC0 files.
Later app catalogs must pin the published URL alongside byte count and
SHA-256. Never reuse a release tag for changed audio, reassign a sound ID
or fabricate recording licenses.

The planned 120-sound content register remains a target, not a production
claim. Stage and preview availability does not imply the Android app has
integrated the new sounds or that the listener has approved them.
