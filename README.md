# RCJ_WW_Luckfox

Программное обеспечение для камеры Luckfox Pico Pro/Max, разработанное командой White Wings для соревнований в категории RCJ Soccer Lightweight. Написано с поддержкой C++17, используется [opencv-mobile](https://github.com/nihui/opencv-mobile), [примеры Luckfox](https://github.com/LuckfoxTECH/luckfox_pico_rkmpi_example) и [Rockchip](https://github.com/airockchip/rknn_model_zoo).

## Сборка для отладки на десктопе

В директории проекта:

`cmake -B build-debug -DBUILD_DESKTOP_DEBUG=ON`

`cmake --build build-debug --target rcj_ww_vision`

После этого исполняемый файл находится в `build/rcj_ww_vision`.

**Замечание**: убедитесь, что у вас установлена полная библиотека OpenCV и её находит `find_package(OpenCV REQUIRED)`. В отладочной версии для десктопа используется она, т.к. в opencv-mobile не реализован заголовок `<opencv2/videoio.hpp>` и полностью отключена возможность работать с графической оболочкой устройства.

## Сборка под Luckfox

Создайте `.env` файл, аналогичный `.env.example`, и сохраните в нём свой путь до **opencv-mobile**, до кросс-компиляторов и RKMPI (подойдёт папка с [этим репозиторием](https://github.com/LuckfoxTECH/luckfox_pico_rkmpi_example)). Затем загрузите файл в переменные среды (для сохранения в пределах текущей сессии терминала на устройствах с Linux достаточно `. .env`).

`cmake -B build -DBUILD_DESKTOP_DEBUG=OFF`

`cmake --build build --target rcj_ww_vision`

Подразумевается, что:
- бинарник `rcj_ww_vision` лежит в `/root/`
- рантайм-настроки хранятся в файле `/userdata/runtime.cfg`
- yolo (если используется) находится в `/root/best-int8.rknn`
- результаты передаются по UART через `/dev/ttyS3`

## Запуск тестов

Некоторые библиотеки покрыты тестами с помощью Google Test (геометрия и т.д.). Они доступны в десктопной версии:

`cmake -B build -DBUILD_DESKTOP_DEBUG=ON -DBUILD_VISION_TESTS=ON`

`cmake --build build --target rcj_vision_tests --parallel`

`build/tests/rcj_vision_tests`

## Параметры бинарника

- `--stream`/`--no-stream`: добавить/убрать rtsp-стрим (по умолчанию true). В "боевом" режиме лучше отключить для максимальной производительности
- `--draw-blobs`/`--no-draw-blobs`: отрисовывать объекты на изображении после детекции (по умолчанию true). Регулируется в рантайме, можно менять через UI в [UniKostyl](https://github.com/WildBoar228/UniKostyl)
- `--runtime-cfg`/`--no-runtime-cfg`: обновлять настройки из файла в рантайме (по умолчанию true), см. далее
- `--detect-mode=<mode>`: режим детектора, доступно 2:
  - `thr-blobs` (по умолчанию): поиск блобов по цветовым диапазонам
  - `yolo-pose`: модель yolo11n-pose, предсказывает ворота с 2 ключевыми точками (левый/правый угол). В режиме DESKTOP_DEBUG на данный момент ничего не делает

## Изменение настроек в рантайме

Запустите на Luckfox [server.py](https://github.com/WildBoar228/RCJ_WW_Luckfox/blob/main/py-server/server.py). По адресу `172.32.0.93:8000` поднимется HTTP-сервер, принимающий JSON-запросы на изменение настроек. JSON должен содержать следующие поля (все обязательные):

```json
{
    "draw_blobs": true,
    "thresholds": [0, 100, -128, 127, -128, 127]
}
```

## Автостарт

Поместите скрипт [S99_rcj_ww.sh](https://github.com/WildBoar228/RCJ_WW_Luckfox/blob/main/S99_rcj_ww.sh) в директорию `/etc/init.d/` на Luckfox. В скрипте можно менять параметры запуска.

Для отладки: вывод скрипта перенаправлен в `/var/log/rcj_init.log`, вывод бинарника - в `/var/log/rcj_ww_vision.log`
