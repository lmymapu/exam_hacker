#!/bin/bash
# =============================================================================
# run_container.sh
# Launches the Docker container with:
#   - Home folder mapped to /workspace
#   - X11 GUI forwarding
#   - Full GPU access (NVIDIA)
#   - Host network (all ports)
# =============================================================================

set -e  # Exit immediately on error

IMAGE_NAME="exam-hacker-dev:latest"
DATE=$(date +%Y-%m-%d)
CONTAINER_NAME="exam-hacker-${DATE}"
XAUTH_FILE="/tmp/.docker.xauth"

# -----------------------------------------------------------------------------
# 1. Sanity checks
# -----------------------------------------------------------------------------

# Check Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "[ERROR] Docker daemon is not running. Please start Docker first."
    exit 1
fi

# Check the image exists
if ! docker image inspect "$IMAGE_NAME" > /dev/null 2>&1; then
    echo "[ERROR] Docker image '$IMAGE_NAME' not found."
    echo "        Please build it first:  docker build -t $IMAGE_NAME ."
    exit 1
fi

# Check DISPLAY is set
if [ -z "$DISPLAY" ]; then
    echo "[ERROR] \$DISPLAY is not set. Are you running in a graphical session?"
    exit 1
fi

# -----------------------------------------------------------------------------
# 2. Prepare X11 authority file for the container
# -----------------------------------------------------------------------------

echo "[INFO] Preparing X11 auth file at $XAUTH_FILE ..."

# Remove stale auth file if it exists
rm -f "$XAUTH_FILE"
touch "$XAUTH_FILE"
chmod 600 "$XAUTH_FILE"

# Extract the current display's MIT-MAGIC-COOKIE and write it into the
# auth file with a wildcard hostname so the container can use it regardless
# of its internal hostname.
DISPLAY_NUM=$(echo "$DISPLAY" | cut -d: -f2 | cut -d. -f1)

xauth nlist "$DISPLAY" | while read -r line; do
    # Replace the connection family/address with "ffff" (FamilyWild) so the
    # cookie is accepted from any hostname, including the container.
    echo "ffff${line:4}" | xauth -f "$XAUTH_FILE" nmerge -
done

if [ ! -s "$XAUTH_FILE" ]; then
    echo "[WARN] X auth file is empty — X11 forwarding may not work."
    echo "       Make sure 'xauth' is installed:  sudo apt install xauth"
fi

# Allow the local Docker socket to connect to your X server (belt-and-suspenders)
xhost +local:docker > /dev/null 2>&1 || true

# -----------------------------------------------------------------------------
# 3. Stop and remove any existing container with the same name
# -----------------------------------------------------------------------------

if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "[INFO] Removing existing container '$CONTAINER_NAME' ..."
    docker rm -f "$CONTAINER_NAME" > /dev/null
fi

# -----------------------------------------------------------------------------
# 4. Run the container
# -----------------------------------------------------------------------------

echo "[INFO] Starting container '$CONTAINER_NAME' ..."
echo "       Image   : $IMAGE_NAME"
echo "       Workspace: $HOME  →  /workspace"
echo "       Display : $DISPLAY"
echo "       Network : host"
echo "       GPU     : all"
echo ""

docker run -it \
    --name "$CONTAINER_NAME" \
    \
    `# ── GPU access ──────────────────────────────────────────────────────` \
    --gpus all \
    \
    `# ── Host network (full port access) ─────────────────────────────────` \
    --network host \
    \
    `# ── Privileged mode (required for some GPU/device operations) ───────` \
    --privileged \
    \
    `# ── Home folder → /workspace ─────────────────────────────────────────` \
    -v "$HOME":/workspace \
    -w /workspace \
    \
    `# ── X11 GUI forwarding ───────────────────────────────────────────────` \
    -e DISPLAY="$DISPLAY" \
    -e XAUTHORITY="$XAUTH_FILE" \
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
    -v "$XAUTH_FILE":"$XAUTH_FILE":ro \
    \
    `# ── share Git config and SSH agent ───────────────────────────────────` \
    -v "$HOME/.gitconfig":/root/.gitconfig:ro \
    -v "$SSH_AUTH_SOCK":/run/ssh-agent:ro \
    -e SSH_AUTH_SOCK=/run/ssh-agent \
    \
    `# ── Timezone (set to Germany) ─────────────────────────────────────────` \
    -e TZ=Europe/Berlin \
    \
    `# ── Shared memory (prevents Qt/OpenCV crashes under load) ────────────` \
    --shm-size=4g \
    \
    "$IMAGE_NAME" \
    bash

# -----------------------------------------------------------------------------
# 5. Cleanup after container exits
# -----------------------------------------------------------------------------

echo ""
echo "[INFO] Container exited. Cleaning up X11 auth file ..."
rm -f "$XAUTH_FILE"
xhost -local:docker > /dev/null 2>&1 || true
echo "[INFO] Done."