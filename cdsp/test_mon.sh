#!/system/bin/sh
cd /data/local/tmp/bonsai1bit
(
  sleep 1
  for i in 1 2 3 4 5 6 7 8; do
    echo "[MON $i] DDR=$(cat /sys/devices/system/cpu/bus_dcvs/DDR/cur_freq) LLCC=$(cat /sys/devices/system/cpu/bus_dcvs/LLCC/cur_freq) QOS=$(cat /sys/devices/system/cpu/bus_dcvs/DDRQOS/cur_freq) CPU0=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq) CPU6=$(cat /sys/devices/system/cpu/cpu6/cpufreq/scaling_cur_freq)"
    sleep 0.3
  done
) &
MON_PID=$!
BONSAI_SKIP_MEMCPY=1 LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=. ./bonsai_fwd bonsai2-27b.npubin tok.bin "The capital of France is" 4
wait $MON_PID
