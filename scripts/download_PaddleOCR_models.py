#!/usr/bin/env python3
"""
download_models.py
==================
One-time setup script for the Exam Question Extractor project.

Run this ONCE inside the container after first start to pre-download all
PaddleOCR models needed by the project. Models are saved to:

    /workspace/proj/AI_models/.paddleocr/

This path lives on the HOST filesystem (via the $HOME -> /workspace bind
mount) and persists across container restarts and full container recreation.

IMPORTANT — the correct shell variable MUST be exported BEFORE running
this script (or added to ~/.bashrc so it is always set):

    export PADDLE_PDX_CACHE_HOME=/workspace/proj/AI_models/.paddleocr

WHY this variable and not others:
  PaddleX (the backend PaddleOCR 3.x wraps) reads PADDLE_PDX_CACHE_HOME
  as a module-level constant at import time (paddlex/utils/cache.py line 29):

      CACHE_DIR = os.environ.get("PADDLE_PDX_CACHE_HOME", DEFAULT_CACHE_DIR)

  This means the variable MUST exist in the shell environment BEFORE Python
  starts. Setting os.environ inside the script has no effect because by the
  time any line of code runs, the paddlex module is already imported and
  CACHE_DIR is already frozen.

  Variables that do NOT work (and why):
    PADDLEX_HOME    -> not read by PaddleX source at all
    PADDLE_HOME     -> not read by PaddleX source at all
    PADDLEOCR_HOME  -> not read by PaddleX source at all

Usage:
    export PADDLE_PDX_CACHE_HOME=/workspace/proj/AI_models/.paddleocr
    python3 scripts/download_models.py

Expected runtime: 2-5 minutes depending on internet speed (~200 MB total).
"""

import sys
import os

# ---------------------------------------------------------------------------
# Verify PADDLE_PDX_CACHE_HOME is set in the shell environment.
#
# We cannot set it here and have it take effect — PaddleX reads it at
# import time as a module-level constant. If it is missing we abort early
# with clear instructions rather than silently downloading to the wrong place.
# ---------------------------------------------------------------------------

MODEL_CACHE_DIR = "/workspace/proj/AI_models/.paddleocr"
REQUIRED_ENV_VAR = "PADDLE_PDX_CACHE_HOME"

actual_cache = os.environ.get(REQUIRED_ENV_VAR, "")

if not actual_cache:
    print()
    print(f"  [FAIL] ${REQUIRED_ENV_VAR} is not set in your shell.")
    print()
    print("  Fix: run these two commands, then re-run this script:")
    print()
    print(f"    export {REQUIRED_ENV_VAR}={MODEL_CACHE_DIR}")
    print(f"    echo 'export {REQUIRED_ENV_VAR}={MODEL_CACHE_DIR}' >> ~/.bashrc")
    print()
    sys.exit(1)

if actual_cache != MODEL_CACHE_DIR:
    print()
    print(f"  [WARN] ${REQUIRED_ENV_VAR} is set but points to a different path:")
    print(f"         Got     : {actual_cache}")
    print(f"         Expected: {MODEL_CACHE_DIR}")
    print()
    print("  Continuing with the path that is actually set in your shell.")
    print("  Models will be stored at:", actual_cache)
    MODEL_CACHE_DIR = actual_cache
    print()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def section(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")

def ok(msg: str) -> None:
    print(f"  [OK]  {msg}")

def info(msg: str) -> None:
    print(f"  [..]  {msg}")

def fail(msg: str) -> None:
    print(f"  [FAIL] {msg}", file=sys.stderr)

# ---------------------------------------------------------------------------
# Confirm effective cache dir by reading it back from PaddleX itself
# ---------------------------------------------------------------------------

section("0 / 4  Verifying effective PaddleX cache directory")

try:
    from paddlex.utils.cache import CACHE_DIR
    if CACHE_DIR == MODEL_CACHE_DIR:
        ok(f"PaddleX CACHE_DIR confirmed: {CACHE_DIR}")
    else:
        fail(f"PaddleX CACHE_DIR mismatch!")
        fail(f"  PaddleX sees : {CACHE_DIR}")
        fail(f"  Expected     : {MODEL_CACHE_DIR}")
        fail("")
        fail("  This means $PADDLE_PDX_CACHE_HOME was not set when Python started.")
        fail("  Make sure it is in ~/.bashrc, then open a NEW shell and retry.")
        sys.exit(1)
except Exception as e:
    fail(f"Could not import paddlex.utils.cache: {e}")
    sys.exit(1)

# ---------------------------------------------------------------------------
# 1. Verify GPU is visible to PaddlePaddle
# ---------------------------------------------------------------------------

section("1 / 4  GPU verification")

try:
    import paddle
    paddle.utils.run_check()

    if paddle.device.cuda.device_count() > 0:
        ok(f"PaddlePaddle detected {paddle.device.cuda.device_count()} GPU(s)")
        ok(f"CUDA device: {paddle.device.get_device()}")
        DEVICE = "gpu:0"
    else:
        fail("No CUDA GPU detected by PaddlePaddle.")
        fail("Check that the container was started with --gpus all and that")
        fail("the NVIDIA driver is installed on the host.")
        sys.exit(1)

except Exception as e:
    fail(f"PaddlePaddle import or GPU check failed: {e}")
    sys.exit(1)

# ---------------------------------------------------------------------------
# 2. Set model source to Baidu Object Storage
#
#    PADDLE_PDX_MODEL_SOURCE=bos routes downloads to Baidu's servers instead
#    of HuggingFace — faster and avoids rate-limiting during bulk downloads.
#    Remove this line to use HuggingFace instead.
# ---------------------------------------------------------------------------

os.environ["PADDLE_PDX_MODEL_SOURCE"] = "bos"
info("Model source set to Baidu Object Storage (bos).")

# ---------------------------------------------------------------------------
# 3. Download models (instantiating each class triggers the download + cache)
# ---------------------------------------------------------------------------

section("2 / 4  Downloading Text Detection model (PP-OCRv5_server_det ~84 MB)")

try:
    from paddleocr import TextDetection
    info("Initialising TextDetection — download will start now ...")
    TextDetection(model_name="PP-OCRv5_server_det", device=DEVICE)
    ok("PP-OCRv5_server_det downloaded and cached.")
except Exception as e:
    fail(f"TextDetection download failed: {e}")
    sys.exit(1)

section("3 / 4  Downloading Text Recognition model (PP-OCRv5_server_rec ~81 MB)")

try:
    from paddleocr import TextRecognition
    info("Initialising TextRecognition — download will start now ...")
    TextRecognition(model_name="PP-OCRv5_server_rec", device=DEVICE)
    ok("PP-OCRv5_server_rec downloaded and cached.")
except Exception as e:
    fail(f"TextRecognition download failed: {e}")
    sys.exit(1)

section("4 / 4  Downloading PP-StructureV3 layout analysis models (~35 MB)")

try:
    from paddleocr import PPStructureV3
    info("Initialising PPStructureV3 — download will start now ...")
    PPStructureV3(
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        device=DEVICE,
    )
    ok("PPStructureV3 models downloaded and cached.")
except Exception as e:
    fail(f"PPStructureV3 download failed: {e}")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

section("Download complete")

ok("All models are cached and ready for offline use.")
ok(f"Cache location : {MODEL_CACHE_DIR}")
ok( "GPU device     : " + DEVICE)
print()
print("  Verify with:  ls", MODEL_CACHE_DIR)
print()
print("  Next step: run the PaddleOCR prototype on a test exam image.")
print("  See Phase 3 in the project brief.")
print()