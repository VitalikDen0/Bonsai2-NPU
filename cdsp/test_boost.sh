#!/system/bin/sh
set_boost_qos0() {
  su -c '
    echo 4761000 > /sys/devices/system/cpu/bus_dcvs/DDR/boost_freq 2>/dev/null
    echo 1211000 > /sys/devices/system/cpu/bus_dcvs/LLCC/boost_freq 2>/dev/null
    for f in /sys/devices/system/cpu/bus_dcvs/LLCC/240b*bwmon*/max_freq; do echo 1211000 > "$f" 2>/dev/null; done
    echo 0 > /sys/devices/system/cpu/bus_dcvs/DDRQOS/boost_freq 2>/dev/null
    for f in /sys/devices/system/cpu/bus_dcvs/DDRQOS/*/min_freq; do echo 0 > "$f" 2>/dev/null; done
    for f in /sys/devices/system/cpu/bus_dcvs/DDRQOS/*/max_freq; do echo 0 > "$f" 2>/dev/null; done
  '
}

restore_stock() {
  su -c '
    echo 547000 > /sys/devices/system/cpu/bus_dcvs/DDR/boost_freq 2>/dev/null
    echo 350000 > /sys/devices/system/cpu/bus_dcvs/LLCC/boost_freq 2>/dev/null
    for f in /sys/devices/system/cpu/bus_dcvs/LLCC/240b*bwmon*/max_freq; do echo 806000 > "$f" 2>/dev/null; done
    echo 0 > /sys/devices/system/cpu/bus_dcvs/DDRQOS/boost_freq 2>/dev/null
    for f in /sys/devices/system/cpu/bus_dcvs/DDRQOS/*/min_freq; do echo 0 > "$f" 2>/dev/null; done
    for f in /sys/devices/system/cpu/bus_dcvs/DDRQOS/*/max_freq; do echo 1 > "$f" 2>/dev/null; done
  '
}

trap restore_stock EXIT INT TERM

cd /data/local/tmp/bonsai1bit
set_boost_qos0
echo "=== [BOOST + DDRQOS=0 E2E] DDR=$(cat /sys/devices/system/cpu/bus_dcvs/DDR/cur_freq) LLCC=$(cat /sys/devices/system/cpu/bus_dcvs/LLCC/cur_freq) QOS=$(cat /sys/devices/system/cpu/bus_dcvs/DDRQOS/cur_freq) ==="
./bonsai_fwd bonsai2-27b.npubin tok.bin "The capital of France is" 4

echo "=== [AFTER RUN] DDR=$(cat /sys/devices/system/cpu/bus_dcvs/DDR/cur_freq) LLCC=$(cat /sys/devices/system/cpu/bus_dcvs/LLCC/cur_freq) QOS=$(cat /sys/devices/system/cpu/bus_dcvs/DDRQOS/cur_freq) ==="
restore_stock
sleep 1
echo "=== [RESTORED] DDR=$(cat /sys/devices/system/cpu/bus_dcvs/DDR/cur_freq) LLCC=$(cat /sys/devices/system/cpu/bus_dcvs/LLCC/cur_freq) QOS=$(cat /sys/devices/system/cpu/bus_dcvs/DDRQOS/cur_freq) ==="
