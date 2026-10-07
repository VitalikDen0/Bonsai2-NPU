#!/system/bin/sh
cd /data/local/tmp/bonsai1bit
export BONSAI_PROFILE_TREE=1
export BONSAI_FUSED_LIN=1
./bonsai_fwd bonsai2-27b.npubin tok.bin "The capital of France is" 4
