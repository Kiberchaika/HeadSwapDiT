# Проект HeadSwapDiT

## Возможности

*   Приложение Gradio для InfiniteYou (`infiniteyou_app.py`)
*   Приложение Gradio для KV-Edit (`kvedit_app.py`)

## Требования

*   **GPU:** Рекомендуется NVIDIA A40 45GB (для InfiniteYou требуется минимум 43GB).
*   **Хранилище:** ~120GB для чекпоинтов.
*   Окружение Python (см. `requirements.txt`)

## Установка

1.  **Клонируйте репозиторий:**
    ```bash
    git clone --recurse-submodules https://github.com/Kiberchaika/HeadSwapDiT
    cd HeadSwapDiT
    ```
    *(Если вы клонировали без `--recurse-submodules`, выполните `git submodule update --init --recursive`)*

2.  **Настройте окружение Python:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # В Windows используйте `venv\Scripts\activate`
    pip install -r requirements.txt
    ```

## Использование

1.  **Активируйте окружение:**
    ```bash
    source venv/bin/activate
    ```

2.  **Запустите приложения Gradio:**
    *   Для InfiniteYou:
        ```bash
        python infiniteyou_app.py
        ```
    *   Для KV-Edit:
        ```bash
        python kvedit_app.py
        ```

## Примечания

*   **Vast.ai Termux:** Чтобы отключить автоматический запуск Termux на Vast.ai, выполните `touch ~/.no_auto_tmux` в вашей домашней директории и переподключитесь.

## Подмодули

Этот проект использует следующие подмодули:
*   [InfiniteYou](https://github.com/bytedance/InfiniteYouu)
*   [KV-Edit](https://github.com/Xilluill/KV-Edit) 
