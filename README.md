# Проект HeadSwapDiT

## Возможности
*   Приложение Gradio для InfiniteYou (`infiniteyou_app.py`)
*   Приложение Gradio для KV-Edit (`kvedit_app.py`)

## Требования
*   **GPU:** Рекомендуется NVIDIA A40 45GB (InfiniteYou требует мин. 43GB).
*   **Хранилище:** ~120GB для чекпоинтов.
*   Окружение Python (см. `requirements.txt`).

## Установка и Настройка

1.  **Клонировать:**
    ```bash
    git clone --recurse-submodules https://github.com/Kiberchaika/HeadSwapDiT && cd HeadSwapDiT
    # Если клонировали без --recurse-submodules, выполните: git submodule update --init --recursive
    ```
2.  **(Только Linux) Системные зависимости:** (например, для OpenCV)
    ```bash
    sudo apt-get update && sudo apt-get install -y libgl1-mesa-glx
    ```
3.  **Окружение Python:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # Windows: venv\Scripts\activate
    pip install -r requirements.txt
    ```
4.  **Настройка модели (InfiniteYou - FLUX.1-dev):**
    *   **Доступ:** Запросите доступ на [huggingface.co/black-forest-labs/FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev).
    *   **Аутентификация:** Войдите через CLI с помощью `huggingface-cli login`, используя токен чтения ([сгенерировать здесь](https://huggingface.co/settings/tokens)).
    *   **Альтернатива:** Скачайте файлы модели вручную после получения доступа и поместите их в ожидаемый каталог (например, `./models/FLUX.1-dev`).

## Использование

1.  **Активировать окружение:** `source venv/bin/activate`
2.  **Запустить приложения Gradio:**
    *   InfiniteYou: `python infiniteyou_app.py`
    *   KV-Edit: `python kvedit_app.py`

## Примечания
*   **Vast.ai Termux:** Отключите автозапуск с помощью `touch ~/.no_auto_tmux` в домашнем каталоге и переподключитесь.

## Подмодули
*   [InfiniteYou](https://github.com/bytedance/InfiniteYouu)
*   [KV-Edit](https://github.com/Xilluill/KV-Edit) 
