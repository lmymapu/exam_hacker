# Project Brief: Exam Question Extractor
> **Version:** 0.2  
> **Last Updated:** 2026-06-22  
> **Coordinator:** [Your Name]

---

## 1. Project Goal

Build a **local, offline, GPU-accelerated desktop application** that:
1. Takes a photo of a German school exam sheet as input
2. Detects and distinguishes **printed text** (questions) from **handwritten text** (student answers)
3. Removes the handwritten answers from the image
4. Outputs a clean image containing **only the printed questions and figures**, so the child can reuse it for practice

---

## 2. Target User

A parent whose child takes German school exams. The parent photographs the completed exam and runs this tool to generate a blank exercise sheet for repeat practice.

---

## 3. Hardware & Host Environment

| Item | Detail |
|---|---|
| OS | Ubuntu 24.04 LTS |
| GPU | NVIDIA RTX 3060 |
| Driver | NVIDIA (CUDA 12.3 compatible) |
| IDE | Visual Studio Code |
| Development shell | Inside Docker container |

---

## 4. Development Environment (Toolchain)

All development happens **inside a Docker container**. The host machine is only used for VSCode (via Dev Containers extension) and file storage.

### Docker Image
- **Base:** `nvidia/cuda:12.3.2-cudnn9-devel-ubuntu22.04`
- **Image name:** `exam-extractor-dev`
- **Container name:** `exam-hacker-2026-04-20` *(or as created by the user)*

### Key container properties
- Home folder (`$HOME` on host) is mounted to `/workspace` inside the container
- X11 forwarding enabled → GUI applications run inside container, display on host desktop
- `--network host` → full access to all host network ports
- `--privileged` + `--gpus all` → full RTX 3060 access via CUDA
- X auth file stored at `$HOME/.docker.xauth` (persists across reboots)

### Helper scripts (located in `$HOME`)
| Script | Purpose |
|---|---|
| `run_exam_extractor.sh` | First-time container creation and launch |
| `reattach_container.sh <name>` | Re-attach to an existing stopped container |

### Software stack inside container
| Tool | Purpose |
|---|---|
| GCC / G++ | C++ compiler |
| CMake | Build system |
| Qt6 | GUI framework |
| OpenCV 4.x (CUDA build) | Image processing |
| Tesseract 5 + `deu` lang pack | Printed text OCR fallback |
| PaddlePaddle GPU | Deep learning inference runtime |
| PaddleOCR | Main OCR + layout analysis engine |
| Python 3 | PaddleOCR runtime (called from C++) |
| nlohmann/json | JSON parsing in C++ |

---

## 5. Software Architecture (Overview)

The pipeline has two parallel tracks that split after layout analysis and converge again at the image compositor.

```
Input Photo (JPEG/PNG)
        │
        ▼
┌───────────────────┐
│   Preprocessor    │  OpenCV: deskew, denoise, normalize contrast
└───────────────────┘
        │
        ▼
┌───────────────────┐
│  Layout Analysis  │  PPStructureV3 → coarse region boxes with type labels
│                   │  (text / title / table / figure)
└───────────────────┘
        │
        ├─────────────────────────────────────┐
        │  type = text / title / table        │  type = figure
        ▼                                     │
┌───────────────────┐                         │  (bypass: no OCR,
│    OCR Engine     │  server_det →            │   no detection,
│                   │  fine text line boxes    │   preserve as-is)
│                   │  server_rec →            │
│                   │  text content +          │
│                   │  confidence scores       │
└───────────────────┘                         │
        │                                     │
        ▼                                     │
┌───────────────────┐                         │
│ Handwriting       │  Per text line crop:     │
│ Detector          │  · stroke width CV       │
│                   │  · row valley regularity │
│                   │  · contour smoothness    │
│                   │  → weighted score        │
└───────────────────┘                         │
        │                                     │
        ▼                                     │
┌───────────────────┐                         │
│ Region Classifier │  score vs threshold →    │
│                   │  label: printed or       │
│                   │  handwritten             │
└───────────────────┘                         │
        │                                     │
        │  printed boxes only                 │  figure regions
        │  (handwritten discarded)            │  (unconditionally kept)
        └──────────────────┬──────────────────┘
                           ▼
                ┌───────────────────┐
                │ Image Compositor  │  printed boxes → keep (no-op)
                │                   │  figure regions → keep (no-op)
                │                   │  handwritten boxes → erase (inpaint)
                └───────────────────┘
                           │
                           ▼
              Output Image (questions + figures only)
```

### Pipeline stage summary

| Stage | Tool / Model | Input | Output |
|---|---|---|---|
| Preprocessor | OpenCV | Raw photo | Deskewed, denoised image |
| Layout analysis | PPStructureV3 | Full image | Coarse region boxes + type labels |
| OCR engine | server_det + server_rec | Text/title/table region crops | Fine text line boxes + content + confidence |
| Handwriting detector | OpenCV (feature extraction) | Per text line crop | Weighted score (stroke CV, valley regularity, contour smoothness) |
| Region classifier | Heuristic threshold | Weighted score | Label: `printed` or `handwritten` |
| Image compositor | OpenCV inpaint | All labelled regions | Output image with handwriting erased |

### Region type routing

| PPStructureV3 type | OCR engine | Handwriting detector | Compositor action |
|---|---|---|---|
| `text` / `title` / `table` | ✅ Run | ✅ Run per line box | Keep if printed, erase if handwritten |
| `figure` | ⛔ Skip | ⛔ Skip | Always keep unconditionally |

### Handwriting detector — feature algorithm

Three OpenCV features are extracted per text line crop and combined into a single weighted score:

| Feature | Method | Weight | Printed | Handwritten |
|---|---|---|---|---|
| Stroke width CV | `distanceTransform` on binarised crop; CV = std/mean of distance values | 0.5 | CV ≈ 0.05–0.15 | CV ≈ 0.35–0.60 |
| Row valley regularity | Horizontal projection profile; std of local minima depths | 0.3 | Low std (even gaps) | High std (irregular gaps) |
| Contour smoothness | `findContours`; variance of local curvature angles | 0.2 | Low variance | High variance |

**Score formula:** `score = 0.5 × CV_stroke + 0.3 × valley_irregularity + 0.2 × contour_variance`

**Threshold:** `score < threshold` → printed; `score ≥ threshold` → handwritten.
Starting value: **0.30**. Tuned on real exam photos. Biased conservatively toward keeping regions (false negative = leave some handwriting visible; false positive = erase a printed question — far worse).

### JSON output schema (ocr_bridge.py → C++)

Each detected region is emitted as a JSON object. The `type` field drives compositor routing:

```json
{
  "regions": [
    {
      "type": "printed",
      "bbox": [x, y, w, h],
      "text": "1a. Berechne die Fläche ...",
      "confidence": 0.97
    },
    {
      "type": "handwritten",
      "bbox": [x, y, w, h],
      "text": "",
      "confidence": 0.0
    },
    {
      "type": "figure",
      "bbox": [x, y, w, h],
      "text": "",
      "confidence": 0.0
    }
  ]
}
```

**Coordinate system:** all `bbox` values are in full-image pixel coordinates.
`server_det` returns coordinates relative to the region crop — these must be offset by the region's top-left corner before writing to JSON.

### C++ Module Breakdown

| Module | File(s) | Responsibility |
|---|---|---|
| GUI | `MainWindow.cpp/h` | Qt6 window, buttons, file dialogs, preview panels |
| OCR Engine | `OcrEngine.cpp/h` | Calls `ocr_bridge.py` via `QProcess`, parses JSON result |
| Layout Analyser | *(part of ocr_bridge.py)* | PPStructureV3 region segmentation |
| Handwriting Detector | *(part of ocr_bridge.py)* | Per-crop feature extraction and scoring |
| Region Classifier | `RegionClassifier.cpp/h` | Applies threshold to score; assigns `printed` / `handwritten` / `figure` label |
| Image Compositor | `ImageCompositor.cpp/h` | Routes each region: keep or inpaint |
| Preprocessor | `Preprocessor.cpp/h` | Deskew, denoise, contrast normalization |
| Entry point | `main.cpp` | Qt application init |

### C++ ↔ Python Bridge Strategy
- C++ calls a Python script (`ocr_bridge.py`) via `QProcess`
- `ocr_bridge.py` runs the full layout analysis + OCR + handwriting scoring pipeline
- Results written as JSON to stdout (schema above)
- C++ reads stdout and parses JSON using `nlohmann/json`
- This avoids the complexity of the PaddlePaddle C++ inference API at this stage

---

## 6. AI Models Used

| Model | Source | License | Purpose |
|---|---|---|---|
| `PP-OCRv5_server_det` | [github.com/PaddlePaddle/PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) | Apache 2.0 | Text line detection — fine bounding boxes within region crops (~84 MB) |
| `PP-OCRv5_server_rec` | Same repo | Apache 2.0 | Text recognition — reads content + confidence per line box (~81 MB) |
| `PPStructureV3` | Same repo | Apache 2.0 | Layout analysis — coarse region segmentation, type labelling (~35 MB) |
| Tesseract 5 `deu` | [github.com/tesseract-ocr](https://github.com/tesseract-ocr/tesseract) | Apache 2.0 | Printed text OCR fallback (German Umlauts) |

**Note:** No dedicated handwriting recognition model is needed. Handwriting only needs to be *detected* (classified), not *read*. Detection is performed by OpenCV feature extraction (stroke CV, projection profile, contour analysis) — no additional model download required.

All models are **downloaded once** to the bind-mounted host path and stored locally. **No internet connection required at runtime.**

---

## 7. GUI Design

Built with **Qt6**. Single main window with:

| Element | Function |
|---|---|
| 📁 Load Photo button | Opens file dialog, loads exam image |
| 🖼️ Input preview panel | Displays the loaded exam photo |
| 📂 Output Folder button | Selects destination folder for result |
| ⚡ Extract Questions button | Triggers the full AI pipeline |
| 🔄 Progress bar | Shows processing status |
| ✅ Output preview panel | Shows the cleaned output image side-by-side |

---

## 8. Project Folder Structure

```
$HOME/exam-extractor/
│
├── docs/
│   ├── project_brief.md          ← THIS FILE — paste into every new agent
│   ├── architecture.md           ← Detailed architecture decisions
│   └── toolchain.md              ← Docker setup log, known issues
│
├── src/
│   ├── main.cpp
│   ├── MainWindow.cpp
│   ├── MainWindow.h
│   ├── OcrEngine.cpp
│   ├── OcrEngine.h
│   ├── RegionClassifier.cpp
│   ├── RegionClassifier.h
│   ├── ImageCompositor.cpp
│   ├── ImageCompositor.h
│   ├── Preprocessor.cpp
│   └── Preprocessor.h
│
├── scripts/
│   └── ocr_bridge.py             ← Python script called by OcrEngine
│
├── resources/
│   └── icons/                    ← PNG icons for toolbar buttons
│
├── models/                       ← PaddleOCR model files (downloaded at build)
│
├── CMakeLists.txt
├── Dockerfile
├── run_exam_extractor.sh
└── reattach_container.sh
```

---

## 9. Development Roadmap

| Phase | Task | Status |
|---|---|---|
| 1 | Docker image build + GPU verification | ✅ Done |
| 2 | Container run script + X11 forwarding | ✅ Done |
| 2.5 | PaddleOCR model download + cache setup | ✅ Done — models at `/workspace/proj/AI_models/.paddleocr/`, env var `PADDLE_PDX_CACHE_HOME` set in `~/.bashrc` |
| 2.6 | Pipeline architecture design | ✅ Done — five-stage pipeline with figure bypass confirmed (see §5) |
| 3 | `ocr_bridge.py` — full pipeline implementation | ⬜ Todo |
| 4 | C++ project skeleton + CMake + Qt6 window | ⬜ Todo |
| 5 | C++ ↔ Python OCR bridge + JSON parsing | ⬜ Todo |
| 6 | Region classifier logic (threshold tuning) | ⬜ Todo |
| 7 | OpenCV image compositor | ⬜ Todo |
| 8 | GUI polish + icons + file dialogs | ⬜ Todo |
| 9 | End-to-end testing with real exam photos | ⬜ Todo |

---

## 10. Known Issues & Decisions Log

| Date | Issue / Decision | Resolution |
|---|---|---|
| 2026-04-20 | X auth file `/tmp/.docker.xauth` deleted on reboot | Moved to `$HOME/.docker.xauth` — persists across reboots |
| 2026-04-20 | `/tmp/.docker.xauth` became a directory after reboot | `reattach_container.sh` does `rm -rf` before `touch` |
| 2026-04-20 | C++ ↔ PaddleOCR integration strategy | Use Python subprocess bridge (QProcess) for now; migrate to C++ inference API later if needed |
| 2026-06-07 | Do not bake model weights into Docker image | Models are large (~200 MB). Use bind-mount persistence instead: download once via `scripts/download_models.py`, models persist on host at `/workspace/proj/AI_models/.paddleocr/` across container restarts |
| 2026-06-07 | PaddleX model cache env var — wrong variable names | `PADDLEX_HOME`, `PADDLE_HOME`, `PADDLEOCR_HOME` are all ignored by PaddleX. The correct variable (confirmed from PaddleX 3.5.2 source `paddlex/utils/cache.py` line 29) is **`PADDLE_PDX_CACHE_HOME`** |
| 2026-06-07 | `PADDLE_PDX_CACHE_HOME` must be set in shell, not in Python | PaddleX evaluates `CACHE_DIR = os.environ.get("PADDLE_PDX_CACHE_HOME", ...)` as a module-level constant at import time. Setting it via `os.environ` inside a Python script has no effect. Must be in `~/.bashrc` inside the container |
| 2026-06-07 | Model cache path inside container | Set `export PADDLE_PDX_CACHE_HOME=/workspace/proj/AI_models/.paddleocr` in `~/.bashrc`. Models stored here persist on the host filesystem via the `$HOME → /workspace` bind mount |
| 2026-06-07 | PaddleX version in container | 3.5.2. Models used: `PP-OCRv5_server_det`, `PP-OCRv5_server_rec`, `PPStructureV3` (layout analysis) |
| 2026-06-22 | Pipeline order: layout analysis before OCR | PPStructureV3 must run first to produce region crops. OCR and handwriting detection operate on individual crops, not the raw full image. Running handwriting detection on the full image without crops would be unguided and fragile. |
| 2026-06-22 | Figure regions must bypass OCR and handwriting detection | Figures (geometry diagrams, graphs, maps) are part of the printed question, not student answers. PPStructureV3 `figure`-typed regions skip OCR and handwriting detection entirely and are passed unconditionally to the compositor as keep. |
| 2026-06-22 | No dedicated handwriting recognition model needed | The project requires handwriting *detection* only, not recognition. Detection is done by OpenCV feature extraction (stroke width CV, horizontal projection valley regularity, contour curvature variance) with a weighted score threshold. No additional model download is required. |
| 2026-06-22 | Bounding box coordinate offset | `server_det` returns coordinates relative to each region crop. These must be offset by adding the region's top-left `(x, y)` before writing full-image coordinates to the JSON output. Failure to do this is the most common multi-stage OCR bug. |
| 2026-06-22 | Compositor receives three region types | The compositor handles: `printed` (keep, no-op), `figure` (keep, no-op), `handwritten` (erase via OpenCV inpaint). The `type` field in the JSON output drives this routing. |
| 2026-06-22 | Classification threshold bias | Threshold set conservatively toward retaining regions. False negative (missed handwriting left visible) is acceptable. False positive (printed question erased) is a serious user-facing failure. Starting threshold: 0.30. |

---

## 11. How to Use This Brief (For New Agents)

If you are a Claude agent joining this project, start your first message with:

> *"I have read the project brief. I understand this is a C++ / Qt6 / PaddleOCR exam question extractor running inside a Docker container on Ubuntu 24.04 with an RTX 3060. My role in this session is: [YOUR ROLE]. What do you need from me?"*

Then ask the coordinator which **phase** or **module** you are responsible for in this session.