#!/data/data/com.termux/files/usr/bin/bash
# ==============================================================================
# Bonsai 2 27B Termux Orchestrator
# Platform: OnePlus 13 (Snapdragon 8 Elite / SM8750, Hexagon v79 HVX)
# ==============================================================================

# Keep CPU/NPU active in background if termux-wake-lock exists
if command -v termux-wake-lock >/dev/null 2>&1; then
    termux-wake-lock
    echo "[orchestrator] Termux wake lock acquired"
fi

BIN_DIR="/data/local/tmp"
MODEL_BIN="$BIN_DIR/bonsai1bit/bonsai2-27b.npubin"
TOK_BIN="$BIN_DIR/tok.bin"
EXEC_BIN="$BIN_DIR/bonsai_fwd"

# FastRPC & NPU Engine environment
export ADSP_LIBRARY_PATH="$BIN_DIR;$BIN_DIR/bonsai1bit;/vendor/dsp/cdsp;/vendor/lib/rfsa/adsp"
export BONSAI_NT_COPY=0
export BONSAI_STATIC_LAYERS=31
export BONSAI_STATIC_LM=1
export BONSAI_RING_LAYERS=3
export BONSAI_CTX=1024

MODE="${1:-server}"
PORT="${2:-8080}"

case "$MODE" in
    server|daemon)
        echo "===================================================================="
        echo " Starting Bonsai 2 27B OpenAI HTTP Server Daemon on port $PORT"
        echo "===================================================================="
        cd "$BIN_DIR" && exec "$EXEC_BIN" "$MODEL_BIN" "$TOK_BIN" --server "$PORT"
        ;;
    chat|repl)
        echo "===================================================================="
        echo " Starting Bonsai 2 27B Interactive Chat REPL"
        echo "===================================================================="
        cd "$BIN_DIR" && exec "$EXEC_BIN" "$MODEL_BIN" "$TOK_BIN" --chat
        ;;
    bench)
        PROMPT="${2:-The capital of France is}"
        STEPS="${3:-8}"
        echo "===================================================================="
        echo " Running Bonsai 2 27B Benchmark: \"$PROMPT\" ($STEPS steps)"
        echo "===================================================================="
        cd "$BIN_DIR" && exec "$EXEC_BIN" "$MODEL_BIN" "$TOK_BIN" "$PROMPT" "$STEPS"
        ;;
    *)
        echo "Usage: $0 {server [port]|chat|bench [\"prompt\"] [steps]}"
        exit 1
        ;;
esac
