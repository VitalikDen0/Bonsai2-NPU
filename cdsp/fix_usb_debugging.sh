#!/system/bin/sh
# OnePlus 13 UDC & ConfigFS Interface Order Watchdog (guarantees ffs.adb exists and stays at MI_00)

(
while true; do
    CFG=/config/usb_gadget/g1/configs/b.1
    UDC_FILE=/config/usb_gadget/g1/UDC
    UDC_VAL=$(/system/bin/cat "$UDC_FILE" 2>/dev/null)
    OLDEST_SYM=$(/system/bin/ls -U "$CFG" 2>/dev/null | /system/bin/grep -E "^(f1|hid\.gs)" | /system/bin/tail -n 1)

    if [ ! -d /dev/usb_hid_setup.lock ] && ( [ ! -L "$CFG/f1" ] || [ "$OLDEST_SYM" != "f1" ] || [ "$UDC_VAL" != "a600000.dwc3" ] ); then
        if [ -d /sys/class/udc/a600000.dwc3 ] && [ -d /config/usb_gadget/g1/functions/ffs.adb ]; then
            echo "" > "$UDC_FILE" 2>/dev/null
            /system/bin/sleep 0.5

            if [ "$(/system/bin/getprop init.svc.adbd)" != "running" ] || [ "$(/system/bin/getprop sys.usb.ffs.ready)" != "1" ]; then
                /system/bin/setprop sys.usb.config adb
                /system/bin/start adbd
                for _i in 1 2 3 4 5 6 7 8 9 10; do
                    [ "$(/system/bin/getprop sys.usb.ffs.ready)" = "1" ] && break
                    /system/bin/sleep 0.3
                done
            fi

            /system/bin/rm -f "$CFG/hid.gs1" "$CFG/hid.gs2" "$CFG/hid.gs3" 2>/dev/null
            if [ ! -L "$CFG/f1" ]; then
                /system/bin/ln -s /config/usb_gadget/g1/functions/ffs.adb "$CFG/f1" 2>/dev/null
            fi

            if [ ! -d /config/usb_gadget/g1/functions/hid.gs1 ] && [ -x /system/bin/hid-setup ]; then
                /system/bin/hid-setup >/dev/null 2>&1
                echo "" > "$UDC_FILE" 2>/dev/null
                /system/bin/sleep 0.3
                /system/bin/rm -f "$CFG/hid.gs1" "$CFG/hid.gs2" "$CFG/hid.gs3" 2>/dev/null
                [ ! -L "$CFG/f1" ] && /system/bin/ln -s /config/usb_gadget/g1/functions/ffs.adb "$CFG/f1" 2>/dev/null
            fi

            [ -d /config/usb_gadget/g1/functions/hid.gs1 ] && /system/bin/ln -s /config/usb_gadget/g1/functions/hid.gs1 "$CFG/hid.gs1" 2>/dev/null
            [ -d /config/usb_gadget/g1/functions/hid.gs2 ] && /system/bin/ln -s /config/usb_gadget/g1/functions/hid.gs2 "$CFG/hid.gs2" 2>/dev/null
            [ -d /config/usb_gadget/g1/functions/hid.gs3 ] && /system/bin/ln -s /config/usb_gadget/g1/functions/hid.gs3 "$CFG/hid.gs3" 2>/dev/null

            echo a600000.dwc3 > "$UDC_FILE" 2>/dev/null
            /system/bin/setprop sys.usb.state adb
        fi
    fi
    /system/bin/sleep 2
done
) >/dev/null 2>&1 </dev/null &
