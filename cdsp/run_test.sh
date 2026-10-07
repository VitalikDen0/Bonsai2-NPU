#!/system/bin/sh
cd /data/local/tmp/bonsai1bit
export LD_LIBRARY_PATH=/data/local/tmp/bonsai1bit
export ADSP_LIBRARY_PATH="/data/local/tmp/bonsai1bit;/vendor/dsp/cdsp;/vendor/lib/rfsa/adsp"
export BONSAI_FUSED_LIN=${1:-1}
export BONSAI_FUSED_LAYER=${2:-1}
export BONSAI_PROFILE_TREE=${3:-1}
export BONSAI_LM_CHUNK=${4:-31040}
export BONSAI_RING_LAYERS=${5:-2}
export BONSAI_STATIC_GROUP=${6:-16}
export BONSAI_NT_COPY=${7:-0}
export BONSAI_LM_PINGPONG=${8:-1}
./bonsai_fwd bonsai2-27b.npubin tok.bin "The capital of France is" 4
