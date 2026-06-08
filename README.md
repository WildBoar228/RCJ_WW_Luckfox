# RCJ_WW_Luckfox

Программное обеспечение для камеры Luckfox Pico Pro/Max, разработанное командой White Wings для соревнований в категории RCJ Soccer Lightweight. Написано с поддержкой C++17, используется [opencv-mobile](https://github.com/nihui/opencv-mobile).

## Сборка для отладки на десктопе

В директории проекта:

`cmake -B build -DBUILD_DESKTOP_DEBUG=ON`

`cmake --build build --target rcj_ww_vision`

После этого исполняемый файл находится в `build/rcj_ww_vision`.

**Замечание**: убедитесь, что у вас установлена полная библиотека OpenCV и её находит `find_package(OpenCV REQUIRED)`. В отладочной версии для десктопа используется она, т.к. в opencv-mobile не реализован заголовок `<opencv2/videoio.hpp>` и полностью отключена возможность работать с графической оболочкой устройства.

## Сборка под Luckfox

Создайте `.env` файл, аналогичный `.env.example`, и сохраните в нём свой путь до **opencv-mobile**. Затем загрузите файл в переменные среды (для сохранения в пределах текущей сессии терминала на усройствах с Linux достаточно `. .env`).

`cmake -B build -DBUILD_DESKTOP_DEBUG=OFF`

`cmake --build build --target rcj_ww_vision`

## Запуск тестов

Некоторые библиотеки покрыты тестами с помощью Google Test (геометрия и т.д.). Они доступны в десктопной версии:

`cmake -B build -DBUILD_DESKTOP_DEBUG=ON -DBUILD_VISION_TESTS=ON`

`cmake --build build --target rcj_vision_tests --parallel`

`build/tests/rcj_vision_tests`
