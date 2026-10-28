# ComfyUI-Illustrious-Selector

> A ComfyUI custom node pack for Illustrious / SDXL anime-style workflows,
> providing tools for **expression variations + hand & face repair**.
>
> The whole thing comes down to one sentence:
> **when switching expression variations, not a single pixel outside the face may move.**
> We didn't get it to stay perfectly still in the end... but it's very close.

---

## Three concrete problems this pack solves

### 1. Expression variations shouldn't make the hair jitter

When you build a variation mask from a bbox, the "box" inevitably touches the hair,
the ears and the background plate around them. Once the edited region goes out of bounds,
putting several variations side by side makes it obvious that the hair, the background,
the hair ornaments — and even the neck and part of the clothing — are shifting slightly.

The approach here is **segmentation-driven, eye-point-oriented**:

- Mask = face region ∪ brow/eye band ∪ eye circles, with the upper edge cut off by a
  superellipse "ceiling" (so it can't crawl up along the bangs)
- The direction of the brow/eye band comes from the **line between the eye points**,
  so it rotates with the head tilt
- The face segmentation only provides size, never angle (the segmentation mask is
  nearly circular, and its second-moment major axis has no anatomical meaning)

An expression without eyebrows looks stiff, so eyebrows are mandatory —
which means the bangs have to move along with them.

### 2. Hand bbox sometimes locks onto a knee

Both the bbox detector and the segm edge detector occasionally mistake toes, hair,
or other things next to the body for a hand.
Hand repair is a high-denoise repaint, so a wrong box simply destroys the image.

The **Detection Guard** node takes bboxes from one detector and validates them with
a segmentation detector: if the area inside the box isn't sufficiently covered by the
segmentation mask, the box is rejected.
The check is coverage-based (real hands are usually 40–70%, false positives near 0%).
The output is still a bbox, so the repaint region is unchanged.

> ⚠️ Note: segm is only used for **validation**. Do not use it as the repaint mask.
> When the segmentation binds the finger boundaries too tightly, the repaint has no room
> to correct the finger shapes and the result gets much worse.

### 3. Core only saves PNG, but a 7MP image is 10MB

The **Save JPG/WebP** node adds webp/jpg saving.
A 7MP image at webp q93 is only 1.2 MB, and the workflow is embedded in the EXIF —
drag the output back onto the ComfyUI canvas and the node graph is restored.
This solves the disk-space problem when you have many expression variations.

---

## Node list (23)

### Prompt

| Node | Description |
|---|---|
| **Character / Action Selector** | Character table + action table; outputs `prompt` and `lora_spec`, drives LoRA loading automatically |
| **Prompt Display** | Read-only display of the current prompt |
| **LoRA Stack** | Batch-loads LoRAs from `lora_spec`; outputs model / clip |
| **Scene Tag Filter** | Filters Danbooru tags by category whitelist; by default keeps only scene / object / pose |
| **Scene Prompt Concat** | Encodes scene words and concatenates them into the conditioning, with a switch (pass-through when off) |

<details>
<summary><b>Character / Action Selector</b> — adding characters, favourites, overrides</summary>

**Adding a character**

1. Fill in the name and the character prompt, and link the corresponding LoRA
2. Click save
3. **Refresh the page**

The new character shows up in the candidate list after the refresh.
The search bar may freeze for a moment before the refresh — that's normal,
the data is being rebuilt.

**Favourites**

Favouriting adds the character **currently shown in the character selector bar**,
not the current prompt.

**Overrides (does not modify the stored record)**

- `character_prompt_override`: temporarily change this character's prompt
- `prefix` / `suffix`: append text before / after the prompt

Neither is written back to the character table, so selecting the same character
again gives you the original content.
</details>

<details>
<summary><b>Scene Tag Filter</b> — how to tune the whitelist</summary>

The node's parameters are four layers, from loose to strict:

| Parameter | Purpose |
|---|---|
| `include_categories` | **Whitelist.** Only tags that hit these categories are kept |
| `exclude_categories` | Removed by category *before* the whitelist is applied |
| `extra_exclude` | Removed by tag name directly (comma separated) |
| `max_tags` | Maximum number of tags output |

Category names come from `tag_category_v2.json` (a mapping of 11099 tags → categories).
Available categories include `background` / `nature` / `location` / `setting` / `object` /
`props` / `furniture` / `plant` / `weather` / `pose` / `action` / `gesture` / `camera` and more;
the default whitelist is exactly these.

The node also has **a built-in forced exclusion** (not exposed as a parameter and not
changeable): character names, animal ears/tails, body parts, hair, face, clothing,
accessories, headwear, shoes, patterns, colours.
These are "the character's own traits" and shouldn't leak into the scene prompt.

**How to tell whether the tuning is right**

Look at it together with the two `PreviewAny` nodes:

| Symptom | Cause | What to do |
|---|---|---|
| Present in raw, absent after filtering | Category not in the whitelist | Add that category to `include_categories` |
| Absent from raw too | The tagger didn't recognise it | Lower `general_threshold` on `WDTimmTagger` |
| Filtered result is empty | `enable` is off, or `max_tags` is too small | Turn enable on / raise the limit |
</details>

### Detection

| Node | Description |
|---|---|
| **SAM/FV+Brow Band V10** | Main detector. Face segmentation defines the boundary, eye points define the direction; outputs the mask used for variations |
| **Detection Guard** | bbox provides the box, segm validates it; rejected if coverage is insufficient. Usually used for hand repair |
| **SAM Brow Band Detector** | The original version, SAM only. Mediocre but usable; fails easily on large-angle side profiles |
| **Pose Preview** / **Pose Preview V10** | Draws POSE_KEYPOINT onto the image, for checking detection quality |
| **Mask Colour Debug V10** | Displays the face / brow / eye masks in separate colours |

<details>
<summary><b>SAM/FV+Brow Band V10</b> — where it applies</summary>

For jobs like expression variations, where subject consistency matters a lot,
I'm fairly happy with the results myself.

The key principle is **segmentation provides size only, never angle**:

- Width comes from the spread of the face segmentation mask
- Angle always comes from the eye points (Eyeful boxes → openpose eye points)
- The second-moment major axis of the segmentation mask **takes no part in the angle**

`face_mask` is an optional input: leave it unconnected and the node falls back to SAM;
it also falls back automatically if FV fails.
</details>

### Image

| Node | Description |
|---|---|
| **Expression Sheet** | Combines N variations into a picker panel and outputs the selected one |
| **Variation Switch** | Outputs a blank image when off, so the whole variation branch is skipped downstream |
| **Variation Router** | Outputs image + expression CLIP in pairs (experimental) |
| **Face Crop** | Crops a tight face close-up using the mask from FaceDetailer |
| **Face Tuner** | Interactive face box adjustment |
| **Save JPG/WebP** | Adds webp/jpg saving, and can embed the workflow in EXIF |
| **Seed Hub** | Change the seed in one place and control any number of samplers at once |

<details>
<summary><b>Expression Sheet</b> — some difficulties?</summary>

The biggest regret about the sheet node is that it cannot output the selected
variation on its own.

| Requirement | Works? | How / why |
|---|---|---|
| Still **visible** after switching workflows | ✅ | Assets are written to the `temp` directory; the node's `properties` keeps only a filename manifest and reloads by URL |
| Still **clickable** after switching, and switching the image immediately | ✅ | What's stored isn't just a screenshot but the **separated assets**: main image + each variation + each thumbnail + layout rectangles |
| Click and **write a file without running** | ❌ | ComfyUI's outputs must come from an execution; a node has no ability to "write files" |

The first two are already implemented. The approach is **assets on disk, manifest in the JSON**:

- The backend writes the main image / variations / thumbnails as JPEG into `temp/illustrious_sheet/<node id>/`
- `properties` keeps only a manifest of about **0.6 KB**
- On reload the images are fetched via `/api/view?filename=...&type=temp`

So clarity is no longer limited by workflow size: the main image and the variations are both
stored at a **1536 px long side (~1.4 MP)**, which means switching a variation looks exactly
as sharp as the main image.

The trade-off: once the `temp` directory is cleaned up (usually after restarting ComfyUI) the
snapshot is gone and the node falls back to the schematic layout — one run restores it.
(An earlier version embedded base64 directly in the JSON: 85 KB for 640px, 180 KB for 1MP,
easily over 1 MB with five variations — that's why it was replaced.)

A freshly dropped node that hasn't run yet displays a **schematic layout**
(main image on the left / a column of 5 cells on the right, one cell highlighted
with a yellow box) instead of a single "waiting for run" line.

### Why the third one can't be done

This isn't an implementation problem of this node — it's ComfyUI's execution model:

```
Clicking only changes a parameter (selected)
selected_image, and the files that get saved, are products of "that execution"
```

A workflow only works when the signal propagates from upstream to downstream —
no execution, no output.
To make a "click" write to disk directly there is only one way: build your own
server route, POST the base64 of the selected image, and have the backend write it.

But the cost isn't small:

- It bypasses the user's **save node**, so filename / format / directory are no longer
  under the user's control
- The visible save node on the canvas and this implicit write create **two sources of truth**,
  which is confusing for anyone taking over
- You need to cache the current run's images on the server, or have the browser upload
  a few hundred KB back

This pack **does not** do that. So the positioning is:

> **Visible when you come back, clickable, and the image switches immediately on click;
> writing to disk still needs one run.**

You can choose...

**A. Set it up before running**

Just fill 1–6 into the `selected` parameter and run once.

**B. Don't use "selected" at all (most robust)**

Connect the variation batch directly to a **Save JPG/WebP** node,
save every variation, and pick the one you want in your file manager.
The sheet is then only for *viewing* and takes no part in the output logic.
</details>

<details>
<summary><b>Variation Switch</b> — just a switch</summary>

Yes, that's all it does: with `enable=False` it outputs a blank image,
so the downstream FaceDetailer finds no face and the entire variation branch idles.

It doesn't control the count, doesn't control the strength, and takes no part
in mask computation. It's placed at the very front of the variation branch so that
"turn off the variations when you don't need them" is a single switch and the
variation sampling is skipped.
</details>

### Utility

| Node | Description |
|---|---|
| **Normalised Resize** | Scales by a pixel budget while keeping the aspect ratio |
| **Edge Selector** | canny / lineart / softedge in one node, outputs an edge image |
| **Latent Ratio** | Computes the latent size ratio from a preset or the reference image ratio, with a customisable pixel budget |
| **Mask Overlay Preview** | Overlays the mask on the image |
| **Variation Mask Hook** | detailer hook for the variation branch |

<details>
<summary><b>Edge Selector</b> — you must change the ControlNet model after switching modes</summary>

This node only produces the **edge image**; it has no say in which model the ControlNet loads.
After switching modes, the model in the downstream `ControlNetLoader` must change too,
otherwise the lines go grey and soft:

| mode | Edge characteristics | Suggested ControlNet |
|---|---|---|
| `canny` | Hard edges, even thickness | canny family, or lineart-anime works too |
| `lineart` | Thin lines, more internal structure preserved | **lineart family** (do not use a canny model) |
| `softedge` | Soft edges, forgiving | softedge family, or lineart-anime |

The node exposes a second output `controlnet_hint` (STRING) with the recommended model
name for the current mode, and also prints a line to the server log.
You can connect it to a `PreviewAny` to see it.

`low_threshold` / `high_threshold` only apply to canny mode.
</details>

<details>
<summary><b>Latent Ratio</b> — multiple of is a trade-off</summary>

Computes the latent size from a preset or the reference image ratio.
Lowering `multiple_of` gives a more precise ratio, but SDXL's "bucket" design handles
some pixel sizes poorly, which shows up as a clear drop in quality or a shifted
composition — you have to trade it off yourself.
32 is a conservative value, 64 is safer.
</details>

---

## Example workflows

`examples/` contains 6 minimal workflows that **run on their own** (I've run all of them
locally), showing how the nodes are wired:

| File | Demonstrates |
|---|---|
| `01_最简生图_角色与LoRA.json` | Selector → LoRA Stack → prompt → sample → save |
| `02_场景标签提取.json` | Reference image tagging → category whitelist filtering → dual preview comparison |
| `03_场景条件拼接.json` | Scene words encoded and concatenated into conditioning (the switch really turns it off) |
| `04_手部检测双重约束.json` | bbox provides the box + segm validates → mask preview |
| `05_眉眼带V10检测器.json` | FV face segmentation + eye points → brow band mask |
| `06_差分拼版与保存.json` | Multi-image sheet picking → save selected / all separately |

Start with **03** (no extra dependencies) to confirm the nodes load correctly.
See [`examples/README.md`](examples/README.md) for details.

---

## Installation

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/jing83506-lab/ComfyUI-Illustrious-Selector
```

**No extra Python dependencies** (only torch / numpy / Pillow that ship with ComfyUI).

### Optional external extensions

The nodes themselves have no hard dependencies, but the examples use these:

| Extension | Where it's used |
|---|---|
| `comfyui-impact-pack` | FaceDetailer / SAM / SEGS (examples 04, 05) |
| `ComfyUI-Forbidden-Vision` | Face segmentation input for the V10 detector (example 05) |
| `ComfyUI-WD-Timm-Tagger` | Source of the scene tags (example 02) |
| `comfyui_controlnet_aux` | Depth preprocessing, lineart / softedge preprocessing |
| `Nvidia_RTX_Nodes_ComfyUI` | RTX VSR upscaling (requires an NVIDIA GPU) |
| `rgthree-comfy` | Before/after image comparison preview |

---

## Data files

| File | Shipped | Description |
|---|---|---|
| `characters.json` | ✅ | Built-in character table |
| `characters_i18n.json` | ✅ | Chinese name → English tag mapping |
| `action.json` | ✅ | Action → tag |
| `character_loras.json` | ✅ | Character → LoRA filename |
| `tag_category_v2.json` | ✅ | 11099 Danbooru tags → categories (used by the scene filter) |
| `char_thumbs.json` | ❌ | Character thumbnail cache, 34 MB; without it no thumbnails are shown, functionality is fine |
| `favorites.json` | ❌ | Personal favourites |
| `characters_custom.json` | ❌ | Self-made characters |
| `custom_action.json` | ❌ | Custom actions |
| `custom_character.json` | ❌ | Custom characters |

All data files **return an empty dict** when missing; nothing errors out.

---

## Legacy detectors

The V7 / V7.1 / V8 / V8.2 / V9 detectors live in `legacy/` and are **not registered
by default**. To open old workflows, set this before starting ComfyUI:

```bash
set ILLUSTRIOUS_ENABLE_LEGACY=1      # Windows
export ILLUSTRIOUS_ENABLE_LEGACY=1   # Linux / macOS
```

That registers 7 additional legacy nodes (30 in total).

---

## Version history: why so many detector generations

Brow-band localisation went through a dozen-plus versions, each fixing one specific failure:

| Version | Problem it solved | Failure it introduced |
|---|---|---|
| V7 | Band didn't rotate with head tilt | Parallelogram forced onto any pose |
| V8 | Parallelogram geometry | "False concavity" from bangs caused false rejections |
| V9 | Span / area criteria + Eyeful cross-check | SAM follows edges and gets dragged off by bangs |
| **V10** | **Face segmentation (Forbidden Vision) replaces SAM** | —— |

The key principle of V10: **segmentation provides size only, never angle**.

Along the way a bug that had been lurking since V9 was fixed: `_lm_plausible` was
defined twice in the file, patches landed on the first (dead) copy, and criteria A/B/C
had in fact never run. V10 deleted those 109 lines.

---

## Known limitations

- Companion workflows depend on **exact widget order**; a major plugin update may shift them
- `char_thumbs.json` is not shipped, so the selector shows character names only
- Legacy nodes may behave differently from the current version; they are archived only
- The sheet's workflow snapshot is only a "placeholder view"; picking still requires a run

---

## Credits

- [Forbidden Vision](https://github.com/numz/ComfyUI-Forbidden-Vision) — face segmentation, replacing the earlier SAM approach
- [comfyui_tag_filter](https://github.com/sugarkwork/comfyui_tag_filter) — the tag category table `tag_category_v2.json`
- [Impact Pack](https://github.com/ltdrdata/ComfyUI-Impact-Pack)

---

## License

MIT (see [LICENSE](LICENSE))
`tag_category_v2.json` follows the original project's MIT license.
The built-in character / action tables are self-made data; please verify the contents
before use.
