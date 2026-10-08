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

## 🔬 Аппаратный Roofline-аудит: Почему 81.1% предела кремния — это непреодолимый потолок

### 1. Откуда взялась цифра «81.1% от физического предела кремния»?
Чтобы понять, почему дальнейшая оптимизация поштучного декодирования токенов ($B = 1$) физически невозможна на платформе Snapdragon 8 Elite, необходимо сопоставить время работы с абсолютным физическим порогом чипа.

При авторегрессионном декодировании одиночного токена ($B = 1$) без спекуляций веса всех 64 слоёв модели и выходной проекции LM Head обязаны быть считаны из системной памяти DRAM во внутреннюю векторную память NPU (VTCM) ровно один раз на каждый генерируемый токен:

| Компонент модели | Архитектура и вычислительная нагрузка | Минимальное время на кристалле Hexagon v79 | Поток весов из DRAM |
| :--- | :--- | :---: | :---: |
| **48 слоёв Linear Attention** | Gated DeltaNet (FWHT + GEMV + Recurrence) | **205.4 мс** (4.28 мс/слой) | ~4.53 ГБ |
| **16 слоёв Full Attention** | GQA Attention + Fused SwiGLU MLP | **62.4 мс** (3.90 мс/слой) | ~1.51 ГБ |
| **Static LM Head** | 248 320 строк словаря $\times$ 5120 dim (W8A16/Q2) | **12.0 мс** | 322 МБ |
| **Интерфейс NPU / Хост** | Финальный RMSNorm + аппаратный HVX Argmax | **0.2 мс** | В регистрах |
| **Абсолютный потолок кремния ($T_{\text{floor}}$)** | **Полный проход 27B модели** | **279.8 мс / токен** (**3.57 tok/s**) | **6.36 ГБ суммарно** |

На смартфоне OnePlus 13 установившееся сквозное время одиночного токена, достигнутое движком, составляет **345.0 мс / токен** (2.90 tok/s).
$$\text{Утилизация физического предела} = \frac{T_{\text{floor}}}{T_{\text{actual}}} = \frac{279.8\text{ мс}}{345.0\text{ мс}} = \mathbf{81.1\%}$$

### 2. Эксперимент: Почему Pipelined SMMU Zero-Copy (0 memcpy) дал только 411 мс (68.1%) вместо 292 мс (95.8%)?
Закономерный вопрос: *Можно ли полностью убрать CPU-копирование памяти, предварительно выделив все 64 слоя (6.02 ГБ) напрямую в DMA-BUF `rpcmem` и динамически переключая их в 32-битном SMMU-окне CDSP через `fastrpc_mmap` / `fastrpc_munmap`?*

Мы реализовали и детально протестировали эту архитектуру на реальном железе:
* Все 4 группы слоёв были заранее аллоцированы в `/dev/dma_heap/system`.
* Внутри активной группы NPU исполнял слои со 100% скоростью чистого кремния: **4.28 мс** (Linear) и **3.90 мс** (Full Attention).
* **Однако суммарное время токена выросло с 345 мс до 411–416 мс!**

#### Корневая причина: сериализация на мьютексе ядра Linux FastRPC (`fl->map_mutex`)
С помощью аппаратного зонда `test_mmap_contention` было доказано поведение драйвера ядра Qualcomm (`adsprpc.c`):
1. **Удержание мьютекса ядра**: Системный вызов `fastrpc_mmap` для группы весом 1.54 ГБ (394 240 страниц) обновляет таблицы трансляции страниц в ОС QuRT на DSP. Драйвер ядра удерживает мьютекс сессии `fl->map_mutex` непрерывно в течение **30–45 мс**.
2. **Блокировка вычислительного потока NPU**: Каждый запуск вычислений NPU (`ioctl(FASTRPC_IOCTL_INVOKE)`) внутри ядра также требует захвата этого же самого мьютекса `fl->map_mutex`.
3. **Аппаратный затор**: На стыках групп (слои 0, 16, 32) поток вычислений NPU физически засыпал в ожидании, пока фоновый поток закончит обновление MMU в ядре.
4. На 4 переключениях групп за 1 токен суммарная задержка блокировок составила:
   $$44.6\text{ мс} + 40.0\text{ мс} + 29.2\text{ мс} + 7.4\text{ мс} = \mathbf{121.2\text{ мс чистых блокировок ядра Linux}}.$$
   $$\text{Полное время} = 289.8\text{ мс (кремний)} + 121.2\text{ мс (мьютекс ядра)} = \mathbf{411.0\text{ мс}}.$$

### 3. Почему Streaming Hybrid Engine (345 мс / 81.1%) быстрее SMMU-переключений?
В архитектуре **Streaming Hybrid Engine**:
1. **Ноль SMMU-вызовов на шаге инференса**: Первые 30 статических слоёв и LM Head (3.14 ГБ) маппятся в SMMU один раз при старте процесса. Во время генерации происходит ровно **0 вызовов `fastrpc_mmap` и 0 вызовов `fastrpc_munmap`**, что на 100% исключает блокировки ядра.
2. **Налог шины памяти**: Оставшиеся 34 слоя (2.88 ГБ) копируются фоновым потоком CPU через кольцевой буфер объемом 194 МБ по шине LPDDR5X (106.7 ГБ/с).
3. Фоновое копирование создаёт небольшую конкуренцию за шину памяти, добавляя $+1.8\text{ мс}$ на каждый стриминг-слой, что суммарно составляет **$+61.4\text{ мс}$ накладных расходов**.
4. **Физический факт**:
   $$\mathbf{+61.4\text{ мс (конкуренция шины LPDDR5X)}} < \mathbf{+121.2\text{ мс (блокировка ядра Linux)}}.$$
   Копирование через CPU по шине памяти работает в **2 раза быстрее системных вызовов ядра Linux**!

Эти $+61.4\text{ мс}$ — фундаментальный, математически непреодолимый налог на разделение шины LPDDR5X между ядрами CPU Oryon и DSP Hexagon v79.

### 4. Выход за физический потолок: пакетное декодирование MTP
Поскольку авторегрессионный шаг $B = 1$ физически упёрт в потолок чтения весов за 279.8 мс, единственный математический способ превысить 3.57 tok/s — **амортизировать чтение весов сразу на несколько токенов**.

В режиме **пакетной верификации MTP ($B = 4$)**:
* 6.36 ГБ весов модели и 151 МБ матрицы рекуррентности DeltaNet считываются из памяти **один раз на 4 токена-кандидата**.
* 4 токена одновременно проверяются в 1024-битных регистрах HVX через пакетные ядра `process_slice_q2_pair`.
* Верификация 4 токенов занимает всего **612–632 мс**, что даёт **153.0–158.0 мс на токен (6.33–6.54 tok/s)**!

### 5. Живой лог аппаратной телеметрии (OnePlus 13 / SM8750)

Свежий лог, снятый напрямую с процессора Snapdragon 8 Elite, наглядно подтверждает эти физические границы:

```text
================================================================================
  OnePlus 13 (Snapdragon 8 Elite / Hexagon v79 HTP) — Живая телеметрия железа
================================================================================
  [fwd] TurboQuant / Turbo4 mode ENABLED (4-битный KV Cache: 16 KiB/токен)
  [fwd] Thermal-safe bus & CPU boost acquired (DDR=4761M, LLCC=1211M, CPU=3532M/4320M)
  [fwd] Hybrid Engine Ready: 30 Static Layers (2.82 GiB) + Static LM Head (322 MB)
                            + 34 Streamed Layers (Ring Arena 194 MB) staged in 2860.9 ms!
  prompt tokens=12
  prefill 11 toks in 1766.9 ms (160.6 ms/tok = 6.23 tok/s)
  
  [fwd] Multi-Token Prediction (MTP / Speculative Decoding, max_drafts=3) ENABLED:
  step 0..2 [MTP 3x MATCH!] tok0=11751( Paris) tok1=13(.) tok2=271(\n\n)
            total=654ms (218.1 ms/tok = 4.58 tok/s | NPU_RPC=642ms [66 calls])
            
  step 3    [Одиночный B=1 Decode] tok=760(The) logit=12.918
            total=363ms (NPU_RPC=359ms [66 calls], trans=0ms, memcpy=4ms, CPU_math=1ms)
            --> 77.1% от физического потолка кремния (279.8ms)
            
  step 4..7 [MTP 4x MATCH!] tok0=6511( capital) tok1=314( of) tok2=9338( France) tok3=369( is)
            total=632ms (158.0 ms/tok = 6.33 tok/s | NPU_RPC=626ms [66 calls])
            --> Превышение предела одиночного токена в 1.77 раза!
            
  [SUMMARY] Generated 8 tokens in 3 NPU steps (1661.8 ms total = 207.7 ms/tok = 4.81 tok/s)
================================================================================
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

## Установка на телефон и запуск OpenAI-сервера (Termux / ADB / APK)

Репозиторий предоставляет полностью готовую к работе оболочку (`bonsai_fwd` + `libbonsai_q1_skel.so` + встроенный OpenAI-совместимый HTTP/SSE сервер и консольный REPL-чат). Вы получаете готовый бэкенд, а куда его встроить — запускать напрямую в **Termux**, подключать к нему локальные клиенты (SillyTavern, OpenWebUI, Python-агентов) или упаковать внутрь собственного **Android APK** — решаете сами.

> **О возможном выходе Debug APK:**  
> Возможно, в будущем будет сделан базовый набросок Android APK (Debug UI), который позволит визуально на экране телефона отслеживать телеметрию NPU, тайминги слоев и процесс генерации, чтобы наглядно понимать, что происходит внутри движка. Однако это задача второстепенного приоритета, и неизвестно, когда именно дойдут руки её реализовать. На данный момент в репозитории уже есть вся готовая серверная и консольная начинка для самостоятельного использования.

### 1. Подготовка файла модели и токенизатора (`bonsai2-27b.npubin` и `tok.bin`)

1. Скачайте исходные веса [Ternary-Bonsai-2-27B GGUF](https://huggingface.co/collections/Q-Bonsai/bonsai-2-models) и перепакуйте их в единый монолитный NPU-файл (`6.74 GiB`):
   ```bash
   # При необходимости укажите свои пути GGUF_PATH и OUT_PATH в repack_bonsai2_npu.py, затем запустите:
   python repack_bonsai2_npu.py
   ```
2. Соберите компактный бинарный BPE-токенизатор (`tok.bin`) из метаданных репозитория:
   ```bash
   python cdsp/gen_tok.py --pack Ternary-Bonsai-2-27B-mlx-2bit --output tok.bin
   ```

### 2. Готовые бинарники или самостоятельная сборка

Готовые скомпилированные бинарники под **Snapdragon 8 Elite (Hexagon v79)** уже лежат в папке [`cdsp/`](cdsp/):
* `cdsp/bonsai_fwd` — хост-движок ARM64, интерактивный консольный чат и HTTP/SSE сервер OpenAI API
* `cdsp/libbonsai_q1_skel.so` — векторное ядро Hexagon v79 HVX (1024-бит) для CDSP
* `cdsp/libcdsprpc.so` — системная библиотека транспорта FastRPC
* `cdsp/start_bonsai.sh` — готовый скрипт-оркестратор для запуска из Termux

*(Опционально)* Самостоятельная пересборка из исходников (требуется Qualcomm Hexagon SDK 6.x и Android NDK r26+):
```bash
# 1. Компиляция DSP-скелета (.so)
hexagon-clang -mv79 -O3 -fvectorize -mhvx -mhvx-length=128b \
  -shared -fPIC -o cdsp/libbonsai_q1_skel.so cdsp/bonsai_hvx.c cdsp/bonsai_imp.c cdsp/gen/bonsai_skel.c

# 2. Компиляция хост-бинарника ARM64
$NDK/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android35-clang \
  -O3 -march=armv8.7-a -fopenmp -static-openmp \
  cdsp/bonsai_fwd_main.c cdsp/bonsai_ops.c cdsp/ucat.c cdsp/gen/bonsai_stub.c \
  -o cdsp/bonsai_fwd -Icdsp -Icdsp/gen -Lcdsp -lcdsprpc -lm -ldl
```

### 3. Загрузка файлов на смартфон (через ADB)

Скопируйте файлы движка и модель в `/data/local/tmp/bonsai1bit` (работает без Root через обычный ADB, либо положите файлы напрямую в домашнюю директорию Termux `$HOME`):
```bash
adb shell "mkdir -p /data/local/tmp/bonsai1bit"
adb push cdsp/bonsai_fwd /data/local/tmp/bonsai1bit/
adb push cdsp/libbonsai_q1_skel.so /data/local/tmp/bonsai1bit/
adb push cdsp/libcdsprpc.so /data/local/tmp/bonsai1bit/
adb push cdsp/start_bonsai.sh /data/local/tmp/bonsai1bit/
adb push tok.bin /data/local/tmp/bonsai1bit/
adb push bonsai2-27b.npubin /data/local/tmp/bonsai1bit/
adb shell "chmod 755 /data/local/tmp/bonsai1bit/bonsai_fwd /data/local/tmp/bonsai1bit/start_bonsai.sh"
```

### 4. Запуск OpenAI-совместимого HTTP/SSE сервера (`--server`)

Запустите встроенный сервер OpenAI API на порту `8080` с поддержкой 4-битного KV-кэша (`--turbo4`), спекулятивного декодирования MTP и сэмплирования:
```bash
adb shell "cd /data/local/tmp/bonsai1bit && \
  export LD_LIBRARY_PATH=/data/local/tmp/bonsai1bit:/vendor/lib64 && \
  export ADSP_LIBRARY_PATH='/data/local/tmp/bonsai1bit;/vendor/dsp/cdsp;/vendor/lib/rfsa/adsp' && \
  ./bonsai_fwd bonsai2-27b.npubin tok.bin --server 8080 4096 --turbo4 --temp 0.6 --top-p 0.9"
```
*(Либо прямо из Termux готовым скриптом: `./start_bonsai.sh server 8080`)*

После запуска сервер поднимает стандартные эндпоинты OpenAI (`POST /v1/chat/completions` с поддержкой потокового SSE `"stream": true` и обычного JSON `"stream": false`, а также `GET /v1/models`):

```bash
# Пример запроса из Termux, с ПК (через adb forward tcp:8080 tcp:8080) или любого HTTP-клиента:
curl http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "bonsai-2-27b",
    "stream": true,
    "messages": [
      {"role": "user", "content": "Коротко объясни, как работает квантовое туннелирование."}
    ]
  }'
```

### 5. Интерактивный консольный чат (`--chat`) и бенчмарк

* **Интерактивный чат прямо в консоли (`--chat`)** — общение с моделью в терминале без прослойки HTTP:
  ```bash
  adb shell "cd /data/local/tmp/bonsai1bit && \
    export LD_LIBRARY_PATH=/data/local/tmp/bonsai1bit:/vendor/lib64 && \
    export ADSP_LIBRARY_PATH='/data/local/tmp/bonsai1bit;/vendor/dsp/cdsp;/vendor/lib/rfsa/adsp' && \
    ./bonsai_fwd bonsai2-27b.npubin tok.bin --chat 4096 --turbo4 --temp 0.6 --top-p 0.9"
  ```
* **Одиночный прогон / бенчмарк скорости**:
  ```bash
  adb shell "cd /data/local/tmp/bonsai1bit && \
    export LD_LIBRARY_PATH=/data/local/tmp/bonsai1bit:/vendor/lib64 && \
    export ADSP_LIBRARY_PATH='/data/local/tmp/bonsai1bit;/vendor/dsp/cdsp;/vendor/lib/rfsa/adsp' && \
    ./bonsai_fwd bonsai2-27b.npubin tok.bin 'The capital of France is' 20 --turbo4 --temp 0.6 --top-p 0.9"
  ```

### 6. Куда встроить готовую оболочку (Termux или свой APK)

* **В Termux**: Запустите `bonsai_fwd --server 8080` в фоне (например, через `tmux` с `termux-wake-lock`) и подключайте любые CLI-утилиты, Python-скрипты с библиотекой `openai` (`base_url="http://127.0.0.1:8080/v1"`) или локальные веб-интерфейсы к `127.0.0.1:8080`.
* **В собственный Android APK**: Положите `libbonsai_q1_skel.so` и `bonsai_fwd` в `jniLibs/arm64-v8a/` вашего приложения, укажите `ADSP_LIBRARY_PATH` на `context.applicationInfo.nativeLibraryDir` и обращайтесь к локальному серверу по `http://127.0.0.1:8080/v1/chat/completions` из UI приложения (или вызывайте движок напрямую через JNI).

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
