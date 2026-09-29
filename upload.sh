#!/usr/bin/env bash
set -Eeuo pipefail

BIN_NAME="rcj_ww_vision"

WORK_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
INSTALL_DIR="$WORK_DIR/build"
TARGET="$INSTALL_DIR/$BIN_NAME"
ADB_BIN="${ADB_BIN:-adb}"
REMOTE_DIR="${LUCKFOX_REMOTE_DIR:-/root/$BIN_NAME}"
SCAN_TIMEOUT="${LUCKFOX_SCAN_TIMEOUT:-0.25}"

command -v "$ADB_BIN" >/dev/null 2>&1 || { echo "Ошибка: adb не найден в PATH." >&2; exit 2; }
[[ -x "$TARGET" ]] || { echo "Ошибка: сначала выполните ./build.sh." >&2; exit 1; }

mapfile -t DEVICES < <("$ADB_BIN" devices | awk '$2 == "device" { print $1 }')

is_luckfox() {
    local serial="$1"
    "$ADB_BIN" -s "$serial" shell 'test -d /userdata && test -d /oem' >/dev/null 2>&1
}

connect_candidate() {
    local ip="$1"
    local serial="${ip}:5555"

    "$ADB_BIN" connect "$serial" >/dev/null 2>&1 || return 1
    if is_luckfox "$serial"; then
        echo "$serial"
        return 0
    fi
    "$ADB_BIN" disconnect "$serial" >/dev/null 2>&1 || true
    return 1
}

discover_luckfox() {
    local network="${LUCKFOX_NETWORK:-}"
    local prefix
    local scan_dir
    local candidate
    local host
    local found=()

    if [[ -n "${LUCKFOX_IP:-}" ]]; then
        connect_candidate "$LUCKFOX_IP" || return 1
        return 0
    fi

    if [[ -z "$network" ]] && command -v ip >/dev/null 2>&1; then
        network="$(ip -4 route show scope link | awk '$1 ~ /^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+\/24$/ { print $1; exit }')"
    fi

    if [[ "$network" =~ ^([0-9]+\.[0-9]+\.[0-9]+)\.0/24$ ]]; then
        prefix="${BASH_REMATCH[1]}"
    else
        echo "Не удалось определить сеть /24. Задайте, например: LUCKFOX_NETWORK=192.168.0.0/24" >&2
        return 1
    fi

    scan_dir="$(mktemp -d "${TMPDIR:-/tmp}/luckfox-adb-scan.XXXXXX")"

    for host in $(seq 1 254); do
        (
            candidate="${prefix}.${host}"
            if timeout "$SCAN_TIMEOUT" bash -c ": </dev/tcp/$candidate/5555" >/dev/null 2>&1; then
                printf '%s\n' "$candidate" > "$scan_dir/$host"
            fi
        ) &
    done
    wait

    for candidate_file in "$scan_dir"/*; do
        [[ -f "$candidate_file" ]] || continue
        candidate="$(<"$candidate_file")"
        if serial="$(connect_candidate "$candidate")"; then
            found+=("$serial")
        fi
    done

    if [[ "${#found[@]}" -eq 1 ]]; then
        printf '%s\n' "${found[0]}"
        rm -f "$scan_dir"/*
        rmdir "$scan_dir" 2>/dev/null || true
        return 0
    fi
    if [[ "${#found[@]}" -gt 1 ]]; then
        echo "Найдено несколько Luckfox: ${found[*]}. Укажите ADB_DEVICE=<serial>." >&2
    else
        echo "Luckfox с ADB на порту 5555 в сети $network не найден." >&2
    fi
    rm -f "$scan_dir"/*
    rmdir "$scan_dir" 2>/dev/null || true
    return 1
}

if [[ -n "${ADB_DEVICE:-}" ]]; then
    DEVICE="$ADB_DEVICE"
elif [[ "${#DEVICES[@]}" -eq 1 ]]; then
    DEVICE="${DEVICES[0]}"
elif [[ "${#DEVICES[@]}" -eq 0 ]]; then
    echo "ADB-устройство не подключено, запускаю поиск Luckfox в локальной сети..." >&2
    DEVICE="$(discover_luckfox)" || {
        echo "Проверьте adb devices или задайте LUCKFOX_IP / LUCKFOX_NETWORK." >&2
        exit 1
    }
else
    echo "Ошибка: найдено несколько устройств. Используйте ADB_DEVICE=<serial>." >&2
    printf '  %s\n' "${DEVICES[@]}" >&2
    exit 1
fi

echo "Загрузка $INSTALL_DIR -> $DEVICE:$REMOTE_DIR"
"$ADB_BIN" -s "$DEVICE" shell mkdir -p "$REMOTE_DIR"
"$ADB_BIN" -s "$DEVICE" push "$INSTALL_DIR/." "$REMOTE_DIR/"
"$ADB_BIN" -s "$DEVICE" shell chmod 755 "$REMOTE_DIR/$BIN_NAME"

LOCAL_MD5="$(md5sum "$TARGET" | awk '{print $1}')"
REMOTE_MD5="$("$ADB_BIN" -s "$DEVICE" shell md5sum "$REMOTE_DIR/$BIN_NAME" | tr -d '\r' | awk '{print $1}')"
[[ "$LOCAL_MD5" == "$REMOTE_MD5" ]] || {
    echo "Ошибка: MD5 не совпадает: local=$LOCAL_MD5 remote=$REMOTE_MD5" >&2
    exit 1
}

echo "Загрузка завершена. MD5: $LOCAL_MD5"
echo "Запуск: $ADB_BIN -s $DEVICE shell $REMOTE_DIR/$BIN_NAME"
