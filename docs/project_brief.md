# Project Brief: Exam Question Extractor
> **Version:** 0.1  
> **Last Updated:** 2026-04-20  
> **Coordinator:** [Your Name]

---

## 1. Project Goal

Build a **local, offline, GPU-accelerated desktop application** that:
1. Takes a photo of a German school exam sheet as input
2. Detects and distinguishes **printed text** (questions) from **handwritten text** (student answers)
3. Removes the handwritten answers from the image
4. Outputs a clean image containing **only the printed questions**, so the child can reuse it for practice

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
│    OCR Engine     │  PaddleOCR (GPU) → bounding boxes + confidence scores
│                   │  Tesseract 5 → fallback for edge cases
└───────────────────┘
        │
        ▼
┌───────────────────┐
│Region Classifier  │  Distinguishes printed (questions) vs handwritten (answers)
│                   │  Rules: confidence score, stroke uniformity, font regularity
└───────────────────┘
        │
        ▼
┌───────────────────┐
│Image Compositor   │  OpenCV: fills handwritten regions with background color
└───────────────────┘
        │
        ▼
   Output Image (questions only)
```

### C++ Module Breakdown

| Module | File(s) | Responsibility |
|---|---|---|
| GUI | `MainWindow.cpp/h` | Qt6 window, buttons, file dialogs, preview panels |
| OCR Engine | `OcrEngine.cpp/h` | Calls PaddleOCR Python script via `QProcess`, parses JSON result |
| Region Classifier | `RegionClassifier.cpp/h` | Classifies bounding boxes as printed or handwritten |
| Image Compositor | `ImageCompositor.cpp/h` | Blanks out handwritten regions using OpenCV |
| Preprocessor | `Preprocessor.cpp/h` | Deskew, denoise, contrast normalization |
| Entry point | `main.cpp` | Qt application init |

### C++ ↔ Python Bridge Strategy
- C++ calls a Python script (`ocr_bridge.py`) via `QProcess`
- Python script runs PaddleOCR and writes results as JSON to stdout
- C++ reads stdout and parses JSON using `nlohmann/json`
- This avoids the complexity of the PaddlePaddle C++ inference API at this stage

---

## 6. AI Models Used

| Model | Source | License | Purpose |
|---|---|---|---|
| PaddleOCR (multilingual) | [github.com/PaddlePaddle/PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) | Apache 2.0 | Main OCR + layout detection |
| PaddleOCR handwriting model | Same repo | Apache 2.0 | Handwritten text recognition |
| Tesseract 5 `deu` | [github.com/tesseract-ocr](https://github.com/tesseract-ocr/tesseract) | Apache 2.0 | Printed text fallback |

All models are **downloaded once** during Docker image build and stored locally. **No internet connection required at runtime.**

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
| 3 | PaddleOCR Python prototype (test on one photo) | ⬜ Todo |
| 4 | C++ project skeleton + CMake + Qt6 window | ⬜ Todo |
| 5 | C++ ↔ Python OCR bridge + JSON parsing | ⬜ Todo |
| 6 | Region classifier logic | ⬜ Todo |
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

---

## 11. How to Use This Brief (For New Agents)

If you are a Claude agent joining this project, start your first message with:

> *"I have read the project brief. I understand this is a C++ / Qt6 / PaddleOCR exam question extractor running inside a Docker container on Ubuntu 24.04 with an RTX 3060. My role in this session is: [YOUR ROLE]. What do you need from me?"*

Then ask the coordinator which **phase** or **module** you are responsible for in this session.