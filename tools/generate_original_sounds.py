"""Render original deterministic *synthetic impressions* of ambient environments.

Every recipe has an original procedural model (no reused recordings). These are
sound-design approximations, not claims of field recordings or verified wildlife.
24-second stereo 48 kHz PCM16 WAVs stay in ignored .artifacts; manifests carry
the recipe, SHA-256, format, technical levels, and unpublished distribution state.

Requires Python 3.11+ and NumPy 2.x. To regenerate:
  python tools/generate_original_sounds.py --all
  python tools/generate_original_sounds.py --sound fire-camp
"""
import argparse
import hashlib
import json
import math
import struct
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
APP_CATALOG = ROOT.parent / "ambient-sounds" / "catalog" / "planned-sounds.json"
STAGING = ROOT / ".artifacts" / "v0.4.0-preview"
RATE = 48_000
DURATION = 24
CROSSFADE = int(RATE * 1.25)
FRAMES = RATE * DURATION
WORK_FRAMES = FRAMES + CROSSFADE
RECIPE_VERSION = "original-sound-design-1"

# Distinct physical/acoustic models, not parameter aliases for one noise bed.
# The display titles are taken from the stable cross-repo planned-sounds JSON.
RECIPES = {
    "fire-camp": ("campfire_crackle", "A low air wash and irregular dry wood snaps"),
    "travel-airplane": ("cabin_turbine", "Layered high-altitude ventilation, turbine and cabin vibration"),
    "city-distant-traffic": ("passing_traffic", "Independent moving vehicle low-frequency passes and road hiss"),
    "texture-paper": ("paper_turns", "Intermittent granular page turns with thin flutter"),
    "texture-keyboard": ("keyboard_keys", "Distinct tactile key downstrokes and quieter release clicks"),
    "space-room": ("quiet_room", "Subdued room-tone ventilation with a barely audible electrical resonance"),
    "space-library": ("library_room", "Quiet large-room air and a few distant paper movements"),
    "rain-leaves": ("leaf_rain", "Broad leaf rustle with isolated soft droplet impacts"),
    "rain-tent": ("tent_rain", "Fabric membrane patter and soft tension resonance"),
    "rain-tin-roof": ("metal_roof_rain", "Metallic resonant drops over a dense roof patter"),
    "rain-puddle": ("puddle_rain", "Wet splashes and low soft broadband rain"),
    "rain-heavy": ("downpour", "Dense broad-spectrum rain with slow surges"),
    "rain-car": ("rain_inside_car", "Damped rain against a car body with rhythmic wiper motion"),
    "rain-porch": ("covered_porch", "Sheltered rain with roof-edge drips and open air"),
    "water-brook": ("pebble_brook", "Fast granular water turbulence and hollow bubble tones"),
    "water-surf": ("rolling_surf", "Long periodic breaking wave envelopes and foam wash"),
    "water-lake": ("lake_lapping", "Short soft laps with sparse small droplets"),
    "wind-pines": ("pine_wind", "Tall narrow wind harmonics and intermittent needle hiss"),
    "wind-grass": ("grass_wind", "Open gusts with many fine short grass blade rustles"),
    "nature-cricket-field": ("crickets", "Multiple overlapping synthetic chirping insects"),
    "nature-cicadas": ("cicada_chorus", "Rhythmic raspy synthetic insect chorusing"),
    "nature-jungle": ("jungle_dusk", "Layered stylized insect trills, foliage and distant low calls"),
    "nature-wetland": ("wetland_night", "Wet soft air with pulsed low amphibian-like calls and high insects"),
    "nature-meadow": ("morning_meadow", "Soft grass air and light staggered original chirping tones"),
    "fire-stove": ("stove_fire", "Enclosed low fire whoosh and occasional damped metal ticks"),
    "animals-frogs": ("frog_chorus", "Synthetic low-frequency frog-like calls of varying lengths"),
    "animals-owl": ("owl_calls", "Isolated two-note synthetic owl-like hoots over subdued night air"),
    "home-ceiling-fan": ("ceiling_fan", "Three-blade periodic air pulses and motor harmonics"),
    "home-air-purifier": ("air_purifier", "Steady filtered air through a grille and a soft fan motor"),
    "home-dryer": ("tumble_dryer", "Rotating fabric tumbles, drum thuds and ventilator noise"),
    "home-washing-machine": ("washing_machine", "Cyclic agitated water sloshing, spin and motor"),
    "travel-car": ("car_cabin", "Rolling tire hiss, low engine hum and sparse road expansion joints"),
    "travel-bus": ("bus_cabin", "Heavy idle vibration, engine harmonics and road rattles"),
    "travel-boat": ("boat_engine", "Low diesel pulses and irregular water hull slaps"),
    "city-plaza": ("open_plaza", "Open air with sparse distant footsteps and moving city wash"),
    "city-station": ("station_hall", "Large indoor hall ventilation and periodic synthetic rail rumble"),
    "texture-pencil": ("pencil_writing", "Short discontinuous graphite scratches on paper"),
    "texture-brush": ("brush_sweeps", "Broad soft bristle sweeps with variable stroke duration"),
    "texture-wood": ("wood_taps", "Separated dry resonant wood impacts"),
    "space-attic": ("attic_wind", "Draft through wooden gaps with subtle wooden creaks"),
    "texture-clock": ("clock_ticks", "Alternating pair of clock escapement ticks"),
}

def rng_for(sound_id: str) -> np.random.Generator:
    seed = int.from_bytes(hashlib.sha256(f"{RECIPE_VERSION}:{sound_id}".encode()).digest()[:8], "little")
    return np.random.default_rng(seed)

def smooth_random(rng: np.random.Generator, seconds: float, size: int = WORK_FRAMES):
    spacing = max(3, int(seconds * RATE))
    knots = rng.uniform(-1.0, 1.0, size=(size // spacing + 3))
    t = np.arange(size, dtype=np.float64) / spacing
    idx = np.minimum(t.astype(np.int64), len(knots) - 2)
    f = t - idx
    f = f * f * (3 - 2 * f)
    return knots[idx] * (1 - f) + knots[idx + 1] * f

def band_noise(rng, lo, hi, alpha=0.0):
    """FFT band-limited noise with spectral exponent; normalized for layering."""
    noise = rng.standard_normal(WORK_FRAMES)
    spectrum = np.fft.rfft(noise)
    hz = np.fft.rfftfreq(WORK_FRAMES, 1.0 / RATE)
    low_edge = max(1.0, float(lo))
    spectral = np.maximum(hz, low_edge) ** (-alpha / 2)
    low_slope = np.clip((hz - max(0, lo * 0.65)) / max(1.0, lo * 0.35), 0, 1)
    high_slope = np.clip((hi * 1.35 - hz) / max(1.0, hi * 0.35), 0, 1)
    spectrum *= spectral * low_slope * high_slope
    result = np.fft.irfft(spectrum, n=WORK_FRAMES)
    return result / max(np.std(result), 1e-10)

def tone(freq, amp=1.0, phase=0.0, vibrato=0.0):
    t = np.arange(WORK_FRAMES) / RATE
    return amp * np.sin(2 * np.pi * freq * t + phase + vibrato * np.sin(2 * np.pi * .21 * t))

def pulses(rng, period, width, pitch, decay=8.0, variation=.12, start=.7, end=DURATION+.05):
    """Original independently triggered impacts with modeled resonances."""
    out = np.zeros(WORK_FRAMES)
    moment = start
    while moment < end:
        moment += max(.045, period * (1 + variation * rng.standard_normal()))
        if moment >= end:
            break
        length = min(int(width * RATE), WORK_FRAMES - int(moment * RATE))
        if length < 12: continue
        t = np.arange(length) / RATE
        envelope = np.exp(-decay * t / width)
        pitch_shift = float(rng.uniform(.84, 1.2))
        burst = np.sin(2 * np.pi * pitch * pitch_shift * t)
        burst += .23 * np.sin(2 * np.pi * 2.15 * pitch * pitch_shift * t)
        envelope *= (1 - np.exp(-t * 550))
        i = int(moment * RATE)
        out[i:i+length] += envelope * burst * rng.uniform(.55, 1)
    return out

def granular(rng, interval, width, band=(300, 6500), attack=.015, density=1.0):
    """Irregular transient grain splashes, rustles or friction strokes."""
    out = np.zeros(WORK_FRAMES)
    t = .35
    while t < DURATION:
        t += max(.024, interval * rng.uniform(.45, 1.65))
        if t >= DURATION: break
        length = min(int(RATE * width * rng.uniform(.65, 1.3)), WORK_FRAMES-int(RATE*t))
        if length < 6:continue
        hz = float(rng.uniform(*band))
        time = np.arange(length) / RATE
        hiss = rng.standard_normal(length)
        # FIR comb-like local resonator for a grain; not recycled across stems.
        filter_gain = np.sin(2 * np.pi * hz * time + rng.uniform(0, 6.283))
        envelope = (1-np.exp(-time/max(attack, .001))) * np.exp(-time/max(width*.24, .01))
        grain = (.55*hiss+.45*filter_gain) * envelope * rng.uniform(.45, 1.1) * density
        pos = int(t * RATE)
        out[pos:pos+length] += grain
    return out

def swell(freq, phase=0, lower=0.2, upper=1.0):
    t = np.arange(WORK_FRAMES) / RATE
    return lower + (upper-lower) * (.5+.5*np.sin(2*np.pi*freq*t+phase))**2

def render(sound_id):
    model = RECIPES[sound_id][0]
    rng = rng_for(sound_id)
    bed = np.zeros(WORK_FRAMES)
    motion = smooth_random(rng, 1.8)

    if model == "campfire_crackle":
        bed = .4*band_noise(rng, 65, 1800, 1.1)*(1+.25*motion) + .19*granular(rng, .095, .085, (900, 6100))
        bed += .13*pulses(rng, .37, .045, 580, decay=5)
    elif model == "stove_fire":
        bed = .57*band_noise(rng, 40, 500, 1.6)*(1+.1*motion)+.15*band_noise(rng, 700, 2300)
        bed += .07*pulses(rng, 1.45, .16, 930)
    elif model == "cabin_turbine":
        bed = .26*band_noise(rng, 180, 3200, .9)+.6*band_noise(rng, 25, 230, 1.8)
        bed += .2*tone(63, vibrato=.07)+.15*tone(125.5)+.07*tone(460, vibrato=.2)
    elif model == "passing_traffic":
        bed = .25*band_noise(rng, 120, 1800) + .12*band_noise(rng, 1000, 5500)
        for start in (1.8, 7.2, 12.5, 19.8, 23):
            secs = np.arange(WORK_FRAMES)/RATE
            travel = np.exp(-.5*((secs-start)/1.55)**2)
            bed += travel*(.38*band_noise(rng, 33, 400, 1.4)+.32*tone(rng.uniform(45, 85),vibrato=.3))
    elif model == "paper_turns":
        bed = .03*band_noise(rng, 100, 800) + .54*granular(rng, 2.9, .58, (950, 6700), attack=.009)
        bed += .06*granular(rng, 1.25, .12, (2100, 9000))
    elif model == "keyboard_keys":
        bed = .015*band_noise(rng, 60, 500) + .75*pulses(rng, .20, .035, 1800, decay=12, variation=.48)
        bed += .2*pulses(rng, .205, .026, 450, decay=16, variation=.51, start=.73)
    elif model == "quiet_room":
        bed = .55*band_noise(rng, 58, 450, 2) + .12*band_noise(rng, 600, 1450)
        bed += .08*tone(50)+.04*tone(119)
    elif model == "library_room":
        bed = .28*band_noise(rng, 35, 260, 2)+.17*band_noise(rng, 800, 3000)
        bed += .075*granular(rng, 4.4, .55, (1500, 5300))
    elif model == "leaf_rain":
        bed = .32*band_noise(rng, 500, 7000, .1)*(1+.24*motion)
        bed += .31*granular(rng, .055, .040, (750, 4800))
    elif model == "tent_rain":
        bed = .23*band_noise(rng, 280, 2100, 1)
        bed += .63*pulses(rng, .065, .07, 280, 6, .45)+.19*granular(rng, .095, .045, (300, 1600))
    elif model == "metal_roof_rain":
        bed = .31*band_noise(rng, 900, 7400, .1)
        bed += .65*pulses(rng, .084, .13, 2100, 3.1, .38)+.22*pulses(rng, .23, .23, 770, 3.0)
    elif model == "puddle_rain":
        bed = .25*band_noise(rng, 500, 4100) + .29*granular(rng, .085, .17, (280, 3200), attack=.018)
        bed += .22*pulses(rng, .29, .11, 190)
    elif model == "downpour":
        bed = (.51*band_noise(rng, 110, 8200,.1)+.31*band_noise(rng, 50, 1200, 1))*(1+.13*motion)
        bed += .16*granular(rng, .06, .07, (500, 3400))
    elif model == "rain_inside_car":
        bed = .36*band_noise(rng, 90, 1600, 1.5) + .27*granular(rng, .14, .08, (450, 2700))
        bed += .24*pulses(rng, 1.67, .22, 90, 3, .01)
    elif model == "covered_porch":
        bed = .40*band_noise(rng, 300, 4900, .35)*swell(.09, lower=.35)
        bed += .32*pulses(rng, .74, .19, 370, 3.5, .55)
        bed += .18*band_noise(rng, 55, 700, 1.3)
    elif model == "pebble_brook":
        bed = .30*band_noise(rng, 80, 1900,.7) + .32*band_noise(rng, 250, 5500)
        bed += .43*pulses(rng, .33, .19, 350, 3,.5)
    elif model == "rolling_surf":
        bed = .63*band_noise(rng, 75, 2600,.6)*swell(.074,phase=.5,lower=.15)
        bed += .35*band_noise(rng, 1700, 9600)*swell(.074,phase=-.8,lower=.05)
    elif model == "lake_lapping":
        bed = .18*band_noise(rng, 280, 2100,.8)*swell(.23,.6,.2)
        bed += .26*granular(rng, 1.65, .45, (600, 4200), attack=.05)
    elif model == "pine_wind":
        bed = .47*band_noise(rng, 55, 650,1.5)*(1+.35*motion)
        bed += .21*band_noise(rng, 2200, 6700)*swell(.12,phase=.2,lower=.12)
        bed += .16*tone(170, vibrato=2)*swell(.058,lower=.02)
    elif model == "grass_wind":
        bed = .22*band_noise(rng, 70, 600,1.3)*(1+.33*motion)
        bed += .4*band_noise(rng, 1300, 9500)*swell(.16,lower=.08)
        bed += .09*granular(rng, .12, .10, (2300, 6900))
    elif model == "crickets":
        bed = .055*band_noise(rng, 120, 1100) + .26*band_noise(rng, 2600, 6000)
        bed += .46*tone(4100)*swell(7.1,lower=0)
        bed += .43*tone(3500,phase=1.2)*swell(5.7,phase=1.1,lower=0)
    elif model == "cicada_chorus":
        bed = .10*band_noise(rng, 320, 2000)+.29*band_noise(rng, 3700, 10500)
        bed += .48*tone(6800,vibrato=4)*swell(12.5,lower=0)
        bed += .22*tone(4900,phase=2)*swell(.37,lower=0)
    elif model == "jungle_dusk":
        bed = .14*band_noise(rng, 160, 3700, .4)+.25*band_noise(rng, 2100, 8000)
        bed += .27*tone(2900, vibrato=3.3)*swell(5.3, lower=0)
        bed += .18*pulses(rng, 2.4, .26, 170, 3.2, .45)
        bed += .13*granular(rng, 1.35, .12, (1300, 5000))
    elif model == "wetland_night":
        bed = .20*band_noise(rng, 220, 2100, .5)+.27*band_noise(rng, 3900, 8500)
        bed += .41*pulses(rng, 1.65, .46, 185, 1.9, .4)
        bed += .22*tone(3700,vibrato=.35)*swell(8.9, lower=0)
    elif model == "morning_meadow":
        bed = .17*band_noise(rng, 80, 950, 1.1)*swell(.14, lower=.45)
        bed += .23*band_noise(rng, 1300, 5300)
        bed += .31*pulses(rng, 1.9, .19, 1850, 2.2, .5)
        bed += .16*tone(2600,vibrato=1.5)*swell(2.4, lower=0)
    elif model == "frog_chorus":
        bed = .065*band_noise(rng, 80, 550, 1.7)
        bed += .43*pulses(rng, 1.15, .37, 220, 1.85, .5)
        bed += .33*pulses(rng, 2.4, .54, 125, 1.5, .2, start=2.4)
    elif model == "owl_calls":
        bed = .042*band_noise(rng, 85, 520, 1.6)+.055*band_noise(rng, 2100, 6500)
        bed += .63*pulses(rng, 5.8, 1.05, 460, 1.35, .16)
        bed += .27*pulses(rng, 5.8, .87, 335, 1.6, .16, start=1.3)
    elif model == "ceiling_fan":
        bed = .33*band_noise(rng, 90, 1100, 1.3)*swell(2.6,lower=.65)
        bed += .41*tone(73)+.1*tone(146)+.1*band_noise(rng, 1800, 4400)
    elif model == "air_purifier":
        bed = .47*band_noise(rng, 700, 5600,.3)+.17*band_noise(rng, 120, 580, 1.3)
        bed += .23*tone(110)+.075*tone(220)
    elif model == "tumble_dryer":
        bed = .36*band_noise(rng, 90, 2200,.8)*swell(.85,lower=.35)
        bed += .24*tone(89)+.29*pulses(rng, 1.35, .19, 260, 2)
    elif model == "washing_machine":
        bed = .32*band_noise(rng, 70, 700,.9)*swell(.48,lower=.25)
        bed += .25*tone(117,vibrato=1)+.33*granular(rng,.3,.25,(150,1250),attack=.05)
    elif model == "car_cabin":
        bed = .36*band_noise(rng, 40, 900, 1)*swell(.25,lower=.7)
        bed += .24*band_noise(rng, 1400, 4500,.2)+.26*tone(83,vibrato=.9)
        bed += .18*pulses(rng, 3.1,.14,180, 4,.12)
    elif model == "bus_cabin":
        bed = .55*band_noise(rng, 30, 420,1.6)*swell(1.6,lower=.7)
        bed += .28*tone(54,vibrato=3.3)+.19*pulses(rng,1.4,.18,280,3,.35)
    elif model == "boat_engine":
        bed = .48*band_noise(rng, 30, 400,2)*swell(6.3,lower=.65)
        bed += .27*tone(49,vibrato=1.3)+.25*granular(rng,1.7,.37,(280,3200),attack=.025)
    elif model == "open_plaza":
        bed = .30*band_noise(rng, 110, 2100, .9)*swell(.11,lower=.55)
        bed += .22*band_noise(rng, 2300, 6700)*swell(.05,lower=.2)
        bed += .29*pulses(rng, 1.9, .09, 220, 4, .4)
    elif model == "station_hall":
        bed = .32*band_noise(rng, 60, 720, 1.3) + .24*band_noise(rng, 900, 4000)
        bed += .35*tone(72)*swell(.07,lower=.13)
        bed += .34*band_noise(rng, 38, 450, 1.1)*swell(.10,phase=1.2,lower=.08)
        bed += .11*pulses(rng, 2.6, .13, 420, 3.8)
    elif model == "pencil_writing":
        bed = .028*band_noise(rng, 70, 300)+.59*granular(rng,.72,.36,(1150,6500),attack=.013)
        bed += .07*pulses(rng,2.8,.05,900)
    elif model == "brush_sweeps":
        bed = .015*band_noise(rng, 60, 350)+.46*granular(rng,1.4,.86,(500,4100),attack=.12)
        bed += .09*band_noise(rng, 1000,5500)*swell(.68,lower=.01)
    elif model == "wood_taps":
        bed = .022*band_noise(rng, 70,280)+.8*pulses(rng,1.3,.28,360,decay=5.5, variation=.35)
        bed += .16*pulses(rng,2.3,.19,540, decay=8, start=2.1)
    elif model == "attic_wind":
        bed = .28*band_noise(rng, 60,1000,1.2)*swell(.09,lower=.2)
        bed += .3*band_noise(rng,1900,5000)*swell(.09,phase=.8,lower=.05)
        bed += .17*pulses(rng,4.4,.75,140,1.1)
    elif model == "clock_ticks":
        bed = .018*band_noise(rng,70,250)+.67*pulses(rng,.995,.033,2700,12,.0)
        bed += .35*pulses(rng,1.99,.038,1580,11,.0,start=1.3)
    else:
        raise ValueError(f"No unique generator model for {sound_id}")

    # Different stereo field per recipe; no source samples are mixed or duplicated.
    side = band_noise(rng, 180, 7200) * .032
    left = bed + side * (.7 + .3 * motion)
    right = bed - side * (.65 - .15 * motion)
    sound = np.stack((left, right), axis=1)
    # Overlap-add the first 1.25 seconds against the future source tail; seam
    # is continuous without a fade to audible silence or an inserted pause.
    fade = np.linspace(0, 1, CROSSFADE, endpoint=True)[:, None]
    sound[:CROSSFADE] = (1-fade)*sound[FRAMES:FRAMES+CROSSFADE] + fade*sound[:CROSSFADE]
    sound = sound[:FRAMES]
    # A 12ms C1-smoothed seam bridge eliminates the remaining tiny high-band
    # single-sample jump at the wrap. Unlike an amplitude fade this bridges
    # directly into the first sample instead of inserting an audible gap.
    edge = int(RATE * .012)
    eased = np.linspace(0, 1, edge, endpoint=True)
    eased = (eased * eased * (3 - 2 * eased))[:, None]
    sound[-edge:] = sound[-edge:] * (1-eased) + sound[0] * eased
    sound -= sound.mean(axis=0)
    sound /= max(np.max(np.abs(sound)), 1e-8)
    sound *= .14
    pcm = np.round(sound * 32767).astype('<i2')
    return pcm

def evidence(sound_id, recipe, pcm, outfile):
    raw = outfile.read_bytes()
    f = pcm.astype(np.float64) / 32768.0
    peak = float(np.max(np.abs(f)))
    rms = float(np.sqrt(np.mean(f**2)))
    return {
        "sound_id": sound_id,
        "source_type": "original_procedural_audio",
        "creator": "Leminno original procedural sound design",
        "actual_source": "Synthesized entirely from deterministic mathematical noise, oscillators and envelope/event models; no field recordings or sampled source files",
        "sounds_like": recipe[1],
        "recording_claim": False,
        "recipe_id": recipe[0],
        "recipe_version": RECIPE_VERSION,
        "playback_kind": "event" if sound_id in {"texture-paper", "texture-wood"} else "loop",
        "seed_derivation": "first 8 bytes little-endian of SHA-256(recipe_version + ':' + sound_id), PCG64 NumPy default_rng",
        "license": "CC0-1.0",
        "license_url": "https://creativecommons.org/publicdomain/zero/1.0/",
        "copyright_dedication": "Leminno dedicates its original procedural audio in this file to the public domain under CC0 1.0",
        "asset_sha256": hashlib.sha256(raw).hexdigest(),
        "asset_bytes": len(raw),
        "duration_seconds": DURATION,
        "sample_rate": RATE,
        "channels": 2,
        "encoding": "PCM16 WAV",
        "file": sound_id + ".wav",
        "distribution_status": "staged_unpublished",
        "url": None,
        "technical_measurement": {
            "peak_dbfs": round(20 * math.log10(max(peak, 1e-12)), 3),
            "rms_dbfs": round(20 * math.log10(max(rms, 1e-12)), 3),
            "dc_per_channel": [round(float(x), 8) for x in f.mean(axis=0)],
            "loop_boundary_step": round(float(np.max(np.abs(f[0]-f[-1]))), 8),
            "clipped_samples": int(np.sum(np.abs(pcm.astype(np.int32)) >= 32767)),
            "listening_approved": False,
            "heard_content_labels_verified": False,
        },
    }

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--sound", action="append", choices=sorted(RECIPES))
    args = parser.parse_args()
    if not args.all and not args.sound:
        parser.error("Choose --all or --sound ID")
    if not APP_CATALOG.is_file():
        parser.error(f"Missing planned source-of-truth catalog: {APP_CATALOG}")
    planned = {row["id"]:row for row in json.loads(APP_CATALOG.read_text(encoding="utf-8"))["sounds"]}
    selected = sorted(RECIPES if args.all else set(args.sound))
    for sound_id in selected:
        if sound_id not in planned or planned[sound_id]["playback_kind"] not in {"loop", "event"}:
            raise ValueError(f"Generator ID is not a planned playback entry: {sound_id}")
    STAGING.mkdir(parents=True, exist_ok=True)
    output = {}
    previous = ROOT / "staged-originals.json"
    if previous.is_file():
        output = {r["sound_id"]:r for r in json.loads(previous.read_text(encoding="utf-8"))["sounds"]}
    for sound_id in selected:
        pcm = render(sound_id)
        target = STAGING / f"{sound_id}.wav"
        with wave.open(str(target), "wb") as out:
            out.setparams((2, 2, RATE, 0, "NONE", "not compressed"))
            out.writeframes(pcm.tobytes())
        info = evidence(sound_id, RECIPES[sound_id], pcm, target)
        info["playback_kind"] = planned[sound_id]["playback_kind"]
        if info["technical_measurement"]["rms_dbfs"] < -58:
            raise AssertionError(f"Too quiet for usable playback: {sound_id}")
        output[sound_id] = info
        print(f"{sound_id:24s} {info['asset_bytes']/1e6:.2f}MB "
              f"{info['technical_measurement']['rms_dbfs']} dBFS SHA256={info['asset_sha256'][:16]}")
    inventory = {
        "schema_version": 1,
        "generation_version": RECIPE_VERSION,
        "release_candidate": "v0.4.0-preview",
        "status": "staged_unpublished",
        "generation_command": "python tools/generate_original_sounds.py --all",
        "sound_count": len(output),
        "sounds": [output[k] for k in sorted(output)],
    }
    previous.write_text(json.dumps(inventory, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(f"Generated {len(selected)}; staged inventory tracks {len(output)}: {previous}")

if __name__ == "__main__":
    main()
