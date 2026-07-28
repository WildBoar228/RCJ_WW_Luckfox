#!/bin/sh

DELAY=2

start() {
    export LD_LIBRARY_PATH=/oem/usr/lib:$LD_LIBRARY_PATH
    (
        sleep $DELAY &&
        echo "Start rcj_ww_vision" >> /var/log/rcj_init.log &&
        /root/rcj_ww_vision >> /var/log/rcj_ww_vision.log 2>&1
    ) &
}

case $1 in
        start)
                echo "start S99_rcj_ww" >> /var/log/rcj_init.log
                start
                ;;
        stop)
                echo "stop S99_rcj_ww" >> /var/log/rcj_init.log
                ;;
        *)
                exit 1
                ;;
esac
