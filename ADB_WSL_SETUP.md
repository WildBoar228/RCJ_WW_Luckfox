# Проброс ADB с Windows в WSL

Luckfox подключается к Windows по USB и определяется как `ADB Interface`.
Windows запускает ADB-сервер, работающий непосредственно с USB-устройством,
а `adb` внутри WSL подключается к этому серверу по TCP-порту `5037`.

В результате статический IP для USB/RNDIS-интерфейса и команда
`adb connect 172.32.0.93` не требуются.

```text
upload.sh → adb в WSL → Windows:5037 → USB ADB → Luckfox
```

## 1. Установка ADB на Windows

Скачайте [Android SDK Platform Tools](https://developer.android.com/tools/releases/platform-tools),
распакуйте архив и добавьте каталог с `adb.exe` в переменную `PATH` Windows.

Подключите Luckfox по USB и проверьте устройство в PowerShell:

```powershell
adb.exe devices -l
```

Устройство должно присутствовать в списке со статусом `device`.

## 2. Разрешение доступа из WSL

Откройте PowerShell **от имени администратора** и один раз добавьте правило
Windows Firewall:

```powershell
New-NetFirewallRule `
  -DisplayName "ADB server from WSL" `
  -Direction Inbound `
  -Action Allow `
  -Protocol TCP `
  -LocalPort 5037 `
  -InterfaceAlias "vEthernet (WSL (Hyper-V firewall))"
```

Если правило уже существует, повторно создавать его не нужно. Проверить его
можно командой:

```powershell
Get-NetFirewallRule -DisplayName "ADB server from WSL"
```

## 3. Запуск ADB-сервера на Windows

В обычном PowerShell остановите стандартный сервер и запустите сервер,
принимающий подключения не только через `localhost`:

```powershell
adb.exe kill-server
adb.exe -a nodaemon server
```

Команда работает в foreground-режиме, поэтому это окно PowerShell необходимо
оставить открытым на время загрузки файлов.

## 4. Подключение ADB-клиента из WSL

В WSL определите адрес Windows-хоста и укажите его Linux-версии `adb`:

```bash
WINDOWS_HOST="$(ip route show default | awk '{print $3; exit}')"
export ADB_SERVER_SOCKET="tcp:${WINDOWS_HOST}:5037"
```

Проверьте подключение:

```bash
adb devices -l
```

В списке должно появиться то же USB-устройство, которое видно через
`adb.exe devices -l` в Windows.

## 5. Сборка и загрузка

Из корня проекта выполните:

```bash
./build.sh
./upload.sh
```

Переменная `ADB_SERVER_SOCKET` наследуется скриптом `upload.sh`, поэтому все
его вызовы `adb` будут выполняться через сервер Windows. Файлы при этом
читаются непосредственно из файловой системы WSL; копировать каталог `build`
на диск Windows не требуется.

Без отдельного `export` загрузку можно выполнить одной командой:

```bash
ADB_SERVER_SOCKET="tcp:$(ip route show default | awk '{print $3; exit}'):5037" ./upload.sh
```

## Диагностика

### Windows не видит Luckfox

Проверьте:

```powershell
adb.exe devices -l
```

Если список пуст, убедитесь, что в диспетчере устройств присутствует
`ADB Interface`, USB-кабель поддерживает передачу данных и установлен подходящий
ADB-драйвер.

### WSL не видит Windows ADB-сервер

Узнайте адрес Windows и проверьте порт `5037`:

```bash
WINDOWS_HOST="$(ip route show default | awk '{print $3; exit}')"
timeout 2 bash -c ": </dev/tcp/${WINDOWS_HOST}/5037" && echo "ADB server доступен"
```

Если порт недоступен:

1. Убедитесь, что `adb.exe -a nodaemon server` продолжает работать.
2. Проверьте правило Windows Firewall.
3. Повторно определите `WINDOWS_HOST` после перезапуска WSL — адрес может измениться.

### Найдено несколько устройств

Посмотрите их идентификаторы:

```bash
adb devices -l
```

Передайте нужный идентификатор скрипту:

```bash
ADB_DEVICE="серийный-номер" ./upload.sh
```

## Завершение работы

Остановите foreground-процесс ADB-сервера сочетанием `Ctrl+C`. При
необходимости затем можно вернуть обычный локальный сервер Windows:

```powershell
adb.exe start-server
```
