#!/bin/bash
# =============================================================================
# reattach_container.sh
# Recreates the X11 auth file and re-attaches to a stopped Docker container.
#
# Usage:
#   ./reattach_container.sh <container_name>
#
# Example:
#   ./reattach_container.sh exam-hacker-2026-04-20
# =============================================================================

set -e

XAUTH_FILE="$HOME/.docker.xauth"

# -----------------------------------------------------------------------------
# 1. Argument check
# -----------------------------------------------------------------------------

if [ -z "$1" ]; then
    echo "[ERROR] No container name provided."
    echo "        Usage: $0 <container_name>"
    exit 1
fi

CONTAINER_NAME="$1"

# -----------------------------------------------------------------------------
# 2. Check the container exists
# -----------------------------------------------------------------------------

if ! docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "[ERROR] No container named '$CONTAINER_NAME' found."
    echo "        Available containers:"
    docker ps -a --format '    {{.Names}}  ({{.Status}})'
    exit 1
fi

# -----------------------------------------------------------------------------
# 3. Recreate the X11 auth file (safe across reboots, stored in $HOME)
# -----------------------------------------------------------------------------

echo "[INFO] Recreating X11 auth file at $XAUTH_FILE ..."

# Remove whatever was there before (file or directory)
rm -rf "$XAUTH_FILE"
touch "$XAUTH_FILE"
chmod 600 "$XAUTH_FILE"

if [ -z "$DISPLAY" ]; then
    echo "[WARN] \$DISPLAY is not set — X11 forwarding may not work inside the container."
else
    xauth nlist "$DISPLAY" | while read -r line; do
        echo "ffff${line:4}" | xauth -f "$XAUTH_FILE" nmerge -
    done
    xhost +local:docker > /dev/null 2>&1 || true
    echo "[INFO] X11 auth ready for display $DISPLAY"
fi

# -----------------------------------------------------------------------------
# 4. Start the container (if stopped) and attach
# -----------------------------------------------------------------------------

STATUS=$(docker inspect -f '{{.State.Status}}' "$CONTAINER_NAME")
echo "[INFO] Container '$CONTAINER_NAME' is currently: $STATUS"

if [ "$STATUS" != "running" ]; then
    echo "[INFO] Starting container ..."
    docker start "$CONTAINER_NAME"
else
    echo "[INFO] Container already running, opening new bash session ..."
fi

echo "[INFO] Attaching to '$CONTAINER_NAME' ..."
echo ""
docker exec -it "$CONTAINER_NAME" bash

# -----------------------------------------------------------------------------
# 5. Done
# -----------------------------------------------------------------------------

echo ""
echo "[INFO] Session ended. Container '$CONTAINER_NAME' is still running in the background."
echo "       To stop it:  docker stop $CONTAINER_NAME"