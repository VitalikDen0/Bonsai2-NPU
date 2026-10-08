# Bonsai-NPU

[![Hardware](https://img.shields.io/badge/Аппаратная%20платформа-Snapdragon%208%20Elite%20(SM8750)-blue.svg)](#аппаратные-требования)
[![NPU Backend](https://img.shields.io/badge/Бэкенд-Qualcomm%20QNN%20%2F%20Hexagon%20HVX-green.svg)](#архитектурный-обзор)
[![License](https://img.shields.io/badge/Лицензия-Некоммерческая%20(VitalikDen0)-red.svg)](LICENSE)
[![Language](https://img.shields.io/badge/Язык-C99%20%2F%20Hexagon%20Assembly-orange.svg)](#)

**Bonsai-NPU** — высокопроизводительный низкоуровневый runtime инференса для запуска 27-миллиардной языковой модели ([Ternary-Bonsai-2-27B](https://huggingface.co/collections/Q-Bonsai/bonsai-2-models)) на мобильных SoC Qualcomm Snapdragon. Движок спроектирован для ультранизкобитного квантования (1-битные и тернарные веса) с прямой утилизацией аппаратного NPU и векторного сопроцессора Hexagon DSP (HVX).

Исполнение модели осуществляется через монолитный SMMU-маппинг объемом 3.4 ГБ, полностью исключая накладные расходы ONNX/TFLite и выполняя гибридные слои трансформера (32 Linear Attention / Delta-Net + 16 Full Attention / GQA) в сквозных слитых вычислительных проходах DSP и NPU.

> **Примечание об окружении тестирования:**  
> Эталонные бенчмарки снимались на смартфоне **OnePlus 13 (Snapdragon 8 Elite / SM8750, 16 ГБ LPDDR5X)** с включенными правами **Root**. Root-доступ применялся исключительно для низкоуровневой отладки: фиксации частот CPU/GPU/DRAM через узлы ядра `/sys/` (для исключения троттлинга) и прямого деплоя через ADB. Для пользовательской эксплуатации root-права **не требуются**.

---

## Статус проекта и физический предел (Завершено на 99.9%)

Проект **Bonsai-NPU** доведен до своего логического, архитектурного и физического завершения на текущем поколении мобильного кремния:

* [x] **Физический потолок шины памяти ($B = 1$)**: Сквозная задержка одиночного шага доведена до **341–349 мс/токен (2.87–2.93 tok/s)** — это **>81% от абсолютного теоретического предела пропускной способности LPDDR5X** (`279.8 мс`), при **0.00 мс** вычислений на CPU, **0.00 мс** накладных расходов SMMU и ровно **66 вызовах FastRPC** на всю 27-миллиардную модель.
* [x] **Спекулятивное декодирование MTP (`B = 2..4`) и Rank-1 Inverse Rollback**: Реализована однопроходная пакетная верификация `1..3` драфт-токенов с удержанием матриц рекуррентности DeltaNet в регистрах HVX и 6-поточным аналитическим откатом состояния (`10.16 МБ`), обеспечивающая **4.9–6.54 tok/s (153–203 мс/токен)** при пакетном принятии и **6.01 tok/s** сквозной скорости.
* [x] **TurboQuant и режим Turbo4 (4-битный KV-кэш)**: Векторное 4-битное квантование KV-кэша на HVX (`16 KiB/токен`, сжатие в 4 раза) и пакетный prefill со скоростью **6.01–6.25 tok/s (160–166 мс/токен)**.

> **Итог разработки:**  
> Архитектура движка и низкоуровневые векторные ядра HVX завершены на **99.9%**. Поскольку скорость инференса уперлась непосредственно в физическую пропускную способность памяти LPDDR5X смартфона, дальнейшие обновления репозитория будут связаны исключительно с исправлением возможных багов, стабильностью и мелкой поддержкой.

---

## Ключевые технические особенности

* **Монолитный стриминг весов без ремаппинга**: На этапе генерации накладные расходы SMMU сведены к нулю. Все веса модели (~3.4 ГБ) удерживаются в непрерывном DMA-BUF буфере и адресуются NPU напрямую без межпроцессных задержек.
* **Векторный сопроцессор Hexagon DSP HVX**: 48 слоев (32 Delta-Net Linear Attention + 16 GQA) выполняются на DSP Hexagon v79 с использованием 1024-битных векторных регистров (128 байт) с аппаратным слиянием SwiGLU, RMSNorm, NeoX RoPE и FWHT-1024.
* **Физический предел скорости генерации (B = 1)**: Достигнуто время **345 мс на токен** (2.90 tok/s) на OnePlus 13 (Snapdragon 8 Elite), что составляет **81.1% от абсолютного физического предела пропускной способности памяти LPDDR5X** (279.8 мс).
* **Пакетная верификация MTP (Multi-Token Prediction)**: Аппаратная верификация батчей размера B = 2..4. Проверка пакета из 4 токенов занимает **380 мс** (95 мс/токен ≈ 10.5 tok/s), обеспечивая результирующую скорость **5.0–8.0 tok/s** в режиме спекулятивного декодирования.
* **Архитектура долгого контекста Turbo4**: Динамический виртуальный адресный пул до **262k токенов контекста** (`mmap(MAP_NORESERVE)`). 4-битное квантование KV-кэша сокращает потребление до **16 KiB на токен** (всего 1.05 ГБ при 65k токенах).
* **Работа без Root-прав**: Вызовы CDSP происходят через штатные узлы Android HAL (`/dev/fastrpc-cdsp`) и системные библиотеки FastRPC. Для запуска и работы root-права **не требуются**.
* **Встроенный SSE-демон**: HTTP/1.1 сервер с поддержкой Server-Sent Events, реализующий совместимый с OpenAI API эндпоинт `/v1/chat/completions` прямо из Termux или внутри автономного Android-приложения.

---

## Сводка производительности (Snapdragon 8 Elite / SM8750)

| Режим работы | Размер батча (B) | Задержка шага | Эффективная скорость | Утилизация кремния |
| :--- | :---: | :---: | :---: | :---: |
| **Базовый Decode** | B = 1 | 345–349 мс | 2.87–2.90 tok/s | 81.1% физического предела шины DRAM |
| **MTP 2x Принятый шаг** | B = 2 | 407–418 мс (203.5 мс/ток) | **4.79–4.91 tok/s** | Двухтокенное регистровое ядро HVX |
| **MTP 4x Принятый шаг** | B = 4 | 612 мс (153.0 мс/ток) | **6.54 tok/s** | Веса + матрица $S_h$ читаются 1 раз на 4 токена |
| **MTP Сквозная генерация** | B = 1..4 | 166.4 мс/ток (средн.) | **6.01 tok/s** | Прогон 20 токенов (`avg 3.33 tok/step`) |
| **Prefill Throughput** | B = 4 | 160.1–166.5 мс/ток | **6.01–6.25 tok/s** | Слитый 64-слойный пакетный GEMM + DeltaNet |

*Тестовое устройство: OnePlus 13 (Snapdragon 8 Elite, 16 ГБ LPDDR5X @ 106.7 ГБ/с пиковой пропускной способности, DSP Hexagon v79). Полная хронология оптимизаций от скалярного C++ (52 сек/GEMV) до 6.54 tok/s приведена в [OPTIMIZATION_HISTORY.md](OPTIMIZATION_HISTORY.md).*

### Подтвержденный лог замера на устройстве (`OnePlus 13`, `--turbo4 --temp 0.6 --top-p 0.9`)

```text
[fwd] TurboQuant / Turbo4 mode ENABLED (4-bit KV Cache: 16 KiB/token)
[fwd] Stochastic Sampling ACTIVE: Temp=0.60, Top-P=0.90
[fwd] Hybrid Engine Ready: 30 Static Layers (2.82 GiB) + Static LM Head (322 MB) + 34 Streamed Layers (Ring Arena 194 MB)
prompt tokens=19
prefill 18 toks in 2997.1 ms (166.5 ms/tok)
[fwd] Multi-Token Prediction (MTP / Speculative Decoding, max_drafts=3) ENABLED
step 0..3  [MTP 4x MATCH!] tok0=11751( Paris) tok1=13(.) tok2=561( The) tok3=6511( capital) total=665ms (166.3 ms/tok = 6.01 tok/s | NPU_RPC=660ms [66 calls])
step 4..5  [MTP 2x MATCH!] tok0=314( of) tok1=9564( Germany) total=418ms (208.9 ms/tok = 4.79 tok/s | NPU_RPC=406ms [66 calls])
step 6..9  [MTP 4x MATCH!] tok0=369( is) tok1=19241( Berlin) tok2=13(.) tok3=561( The) total=612ms (153.0 ms/tok = 6.54 tok/s | NPU_RPC=609ms [66 calls])
step 10..13 [MTP 4x MATCH!] tok0=6511( capital) tok1=314( of) tok2=9338( France) tok3=369( is) total=612ms (153.1 ms/tok = 6.53 tok/s | NPU_RPC=610ms [66 calls])
step 14..17 [MTP 4x MATCH!] tok0=11751( Paris) tok1=13(.) tok2=561( The) tok3=6511( capital) total=614ms (153.6 ms/tok = 6.51 tok/s | NPU_RPC=612ms [66 calls])
step 18..19 [MTP 2x MATCH!] tok0=314( of) tok1=9564( Germany) total=407ms (203.5 ms/tok = 4.91 tok/s | NPU_RPC=401ms [66 calls])

[SUMMARY] Generated 20 tokens in 6 NPU steps (3328.9 ms total = 166.4 ms/tok = 6.01 tok/s | avg 3.33 tok/step)
```

---

## Структура репозитория

```
├── cdsp/
│   ├── bonsai_fwd_main.c     # Хост-движок, оркестратор, MTP верификация, управление SMMU
│   ├── bonsai_hvx.c          # 1024-битные ядра HVX (Delta-Net, GQA, RoPE, RMSNorm)
│   ├── bonsai_imp.c          # Реализация скелета FastRPC на DSP
│   ├── bonsai_server.h       # HTTP/1.1 SSE сервер с API OpenAI
│   ├── bonsai.idl            # Описание интерфейса FastRPC IDL
│   ├── tensors.h             # Описатели тензоров и раскладка памяти
│   ├── tok.c                 # Быстрый BPE токенизатор на C
│   └── ucat.c                # Таблицы категорий Unicode
├── repack_bonsai2_npu.py     # Сериализатор весов и упаковщик NPU-бинарника
├── OPTIMIZATION_HISTORY.md   # Инженерная история оптимизаций от скалярного C++ до 6.54 tok/s (EN)
├── ARCHITECTURE.md           # Детальный технический разбор архитектуры (EN)
├── ARCHITECTURE_RU.md        # Детальный технический разбор архитектуры (RU)
├── README.md                 # Документация проекта (EN)
└── README_RU.md              # Документация проекта (RU)
```

---

## Совместимость с чипами, доступ к путям и работа без Root

### 1. Архитектура Hexagon и БОЛЕЕ новые процессоры
* **Snapdragon 8 Elite (SM8750)**: Hexagon v79 (Базовая аппаратная цель).
* **Предыдущие поколения**: Hexagon v75 (8 Gen 3) и Hexagon v73 (8 Gen 2) поддерживаются перекомпиляцией (`-mv75` / `-mv73`).
* **БОЛЕЕ новые поколения**: Hexagon v81 / v83 (Snapdragon 8 Elite Gen 2, Snapdragon 8 Gen 5 и др.) сохраняют обратную совместимость с набором инструкций HVX v79 и получат автоматический прирост скорости от более быстрой памяти LPDDR5X/LPDDR6 (>130 ГБ/с).

### 2. Права доступа и работа без Root
* **Root НЕ нужен**: Подсистема Qualcomm FastRPC создана для стандартных пользователей Android. Узел `/dev/fastrpc-cdsp` открыт для любых приложений.
* **Доступ к `/data/local/tmp`**: При подключении через ADB (с ПК) доступ открыт полностью (`rwx`, пользователь `shell`, UID 2000) на всех телефонах без рута.
* **Автономная работа на телефоне (без ПК)**: Запуск выполняется из пользовательской директории Termux (`$HOME`) или из стандартного APK приложения (`lib/arm64-v8a`).
* Максимальная производительность форсируется штатным userspace API FastRPC (`HAP_power_set`) без необходимости рутования.

---

## Быстрый запуск (Termux / ADB)

### Требования к сборке
* Qualcomm Hexagon SDK 5.x / 6.x (или компилятор Hexagon LLVM Clang)
* Android NDK (r25c или новее, `aarch64-linux-android`)
* Python 3.10+ (для сериализации весов)

### Компиляция
```bash
# 1. Компиляция DSP-скелета (.so)
hexagon-clang -mv79 -O3 -fvectorize -mhvx -mhvx-length=128b \
  -shared -fPIC -o cdsp/libbonsai_q1_skel.so cdsp/bonsai_hvx.c cdsp/bonsai_imp.c cdsp/gen/bonsai_skel.c

# 2. Компиляция хост-бинарника ARM64
$NDK/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android34-clang \
  -O3 -march=armv8.7-a -pthread -o cdsp/bonsai_fwd cdsp/bonsai_fwd_main.c cdsp/tok.c cdsp/ucat.c cdsp/gen/bonsai_stub.c -ldl
```

### Запуск на устройстве
```bash
# Загрузка бинарников и библиотек (работает без root через ADB)
adb push cdsp/bonsai_fwd /data/local/tmp/
adb push cdsp/libbonsai_q1_skel.so /data/local/tmp/

# Запуск текстовой генерации
adb shell "export ADSP_LIBRARY_PATH=/data/local/tmp; cd /data/local/tmp && ./bonsai_fwd bonsai27b-1bit.npubin tok.bin \"Привет, расскажи о себе\" 32"
```

Запуск в режиме фонового HTTP-сервера:
```bash
adb shell "export ADSP_LIBRARY_PATH=/data/local/tmp; cd /data/local/tmp && ./bonsai_fwd bonsai27b-1bit.npubin tok.bin --server 8080"
```

---

## Документация

* [История оптимизаций: от скалярного C++ (52 сек) до 6.54 tok/s (English)](OPTIMIZATION_HISTORY.md)
* [Полный технический разбор архитектуры (Русский)](ARCHITECTURE_RU.md)
* [Архитектурная спецификация (English)](ARCHITECTURE.md)
* [English README](README.md)

---

## Лицензия

Кодовая база распространяется по лицензии **Bonsai-NPU Source-Available Non-Commercial & Anti-Enterprise-R&D License v1.2** (см. [LICENSE](LICENSE)).
* **Разрешено**: Исключительно персональное, любительское и неспонсируемое академическое использование физическими лицами.
* **Запрещено**: Любое коммерческое использование, внедрение в платные продукты, а также **внутренний корпоративный R&D / бенчмаркинг / заимствование архитектурных решений** коммерческими юридическими лицами без прямого коммерческого соглашения с **VitalikDen0**.
* **Коммерческое лицензирование**: Связь с автором по email (**me@zepmoriq.com**) или через GitHub (https://github.com/VitalikDen0). Веса модели подпадают под лицензионные условия авторов архитектуры Bonsai / Q-Bonsai.
