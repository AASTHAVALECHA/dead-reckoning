# DEAD RECKONING
### an interactive installation navigating toward loss
#### CVPR AI Art Gallery 2026

---

> *"Dead reckoning: to calculate where you are by using knowledge of your speed,  
> direction of travel, and time elapsed since your last known position.  
> You are always wrong. You are always drifting. The error accumulates."*  
> — nautical terminology

---

## What it is

A participant types one sentence about something they have lost.

The system navigates backward through latent space from that input — generating four frames that trace a path from photorealistic present toward pure colour field. The output is printed as a 2×2 grid on heavy matte paper and given to the participant to keep.

**The four frames:**

```
I    PRESENT          Kodak Portra 400. Film grain, halation, chromatic aberration.
                      Evidence without presence.

II   FIVE YEARS AGO   Kodachrome fade. Cyan drains first. The photograph
                      you find in a drawer you don't usually open.

III  A DECADE BACK    Edward Hopper geometry. Hard light bar, shadow diagonal.
                      The room has become a diagram of itself.

IV   AT ONE LIMIT     Rothko colour fields. Two temperatures extracted from
                      the scene's own palette. The model has run out of information.
```

The installation wall accumulates prints across three conference days. By the final day it is dense with other people's inference errors. The wall is the piece.

> *"The model has never lost anything.  
> It cannot navigate toward absence.  
> Everything you see is inference."*

---

## Samples

All 17 scenes, rendered at full resolution:

| Scene | Participant text |
|-------|-----------------|
| Two Glasses, Late Afternoon *(primary)* | my father's laugh at the end of a joke I can't remember |
| Empty Swing | something left swinging in the yard after |
| Phone Booth | her number still in my head after six years |
| Hospital Window | the last thing she looked at from that room |
| Lighthouse | the summer we found it and thought we'd come back |
| Train Window | I don't remember when I stopped turning to look |
| Diving Board | we never went back after that summer |
| Murmuration | how they would move through a room |
| First Snow | the morning before everything changed |
| Carnival Closing | sitting on the hood of the car watching |
| Eclipse | we drove four hours to stand in a field |
| Greenhouse | she could name every plant in there |
| Empty Cinema | every film we saw in that seat, row F |
| Glacier | the scale of what we cannot hold onto |
| Operating Theatre | what the ceiling looks like from that table |
| Highway 3AM | driving back when there was nothing left to say |
| Tide Pool | the things that live in that small world |

---

## Running the proof-of-concept

This repository contains the full procedural renderer — no ML dependencies, runs on CPU.

```bash
pip install numpy pillow scipy
python dead_reckoning.py
```

Renders all 17 scenes to `./outputs/`. Each scene takes ~6 seconds. Full batch ~2 minutes.

**To render a single scene:**

```python
from dead_reckoning import scene_empty_swing, make_frames, compose_2x2

base   = scene_empty_swing()
frames = make_frames(base)
compose_2x2(frames, "Empty Swing", "something left swinging in the yard after",
            "outputs/my_scene.jpg")
```

**To add your own scene:**

```python
def my_scene():
    # Return an (H, W, 3) float32 numpy array, values in [0, 1]
    # This is your "present image" — the four frames are derived from it
    img = np.zeros((720, 1200, 3), np.float32)
    # ... build your scene ...
    return img
```

---

## Production installation pipeline

The proof-of-concept renders everything procedurally. The full interactive installation uses:

| Stage | Process |
|-------|---------|
| Generate | Text → present image via Stable Diffusion XL (SDXL) |
| Invert | DDIM inversion recovers full latent trajectory from image back toward noise |
| Direction | Aesthetic direction vector: learned offset in latent space from photographic clarity toward painterly uncertainty — the artistic core of the work |
| Decode | Four equidistant latent points decoded with exponentially increasing aesthetic displacement |
| Compose | 2×2 grid at 3200px, warm linen ground, temporal labels embedded in each frame |
| Print | 300 DPI, heavy matte paper, ~10" × 7" |

Processing time per print: 2–4 minutes on a 24GB GPU.

---

## Installation requirements

- 1× GPU workstation (24GB VRAM)
- Inkjet printer
- Touchscreen kiosk
- 6ft × 8ft wall for print accumulation

---

## Files

```
dead_reckoning.py       Full renderer — all 17 scenes, 4 frame transforms, compositor
requirements.txt        numpy, pillow, scipy
README.md               This file
docs/
  statement.docx        Full artist statement and submission document
  submission.md         CVPR submission form text
```

---

## License

Code: MIT  
Artwork and artist statement: All rights reserved
