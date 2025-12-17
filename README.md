# Multi-Agent System for Scientific Publications

Эта версия проекта упрощена: **без FastAPI и Docker**.
Запуск через `main.py` в режиме непрерывного диалога (REPL) в консоли.

## Структура

```text
  app.py              # Flask сервис
  docker-compose.yaml # docker compose файл для запуска Postgres
  templates/
    index.html        # UI Flask приложения
  scipubs_mas_cli/
    app/
      config.py       # Конфигурация LLM и PostgreSQL
      db.py           # Подключение к БД и выполнение SQL (только SELECT)
      memory.py       # Долговременная память (таблица agent_memory)
      llm.py          # Обёртка над ChatOpenAI (LiteLLM / OpenAI-совместимый endpoint)
      safety.py       # Простые проверки безопасности SQL и запросов
      graph_app.py    # LangGraph-граф: Planner → Classifier → Collector → Analyst
      agents/
        planner.py    # Планировщик анализа
        classifier.py # Выделение предметных тем (без расширения операциями)
        collector.py  # Детерминированная сборка SQL + (опционально) сбор агрегатов из OpenAlex
        analyst.py    # Аналитический агент + построение графиков
      tools/
        openalex.py   # Интеграция с OpenAlex API
        plotting.py   # Построение графиков и возврат в base64
    data/                       # Черновики проекта в ipynb
      __init__.py
      init_db_from_notebook.py  
    notebooks/
      Planner_test.ipynb        
    main.py
    requirements.txt
```

## Установка

1. Создайте и активируйте виртуальное окружение (по желанию).
2. Установите зависимости:

```bash
cd scipubs_mas_cli
pip install -r requirements.txt
```

3. Убедитесь, что PostgreSQL запущен и доступен с параметрами из `app/config.py`
   (по умолчанию: `host=localhost`, `db=papers_db`, `user=postgres`, `password=postgres`).
   При необходимости поменяйте настройки в `config.py`.

## Запуск

### Запуск Flask приложения

```bash
python app.py
```

### Интерактивный режим (REPL)

```bash
cd scipubs_mas_cli
python main.py
```

Примеры запросов:

- "Опиши, какие данные есть в нашей базе научных публикаций"
- "Сравни динамику числа публикаций по темам computer vision и natural language processing"
- "Покажи таблицу статей по теме graph neural networks"

Команды в REPL:

- `/exit` — выход
- `/memory` или `/memory N` — показать последние записи long-term memory (`agent_memory`)

Программа при каждом запросе:

1. Запустит конвейер Planner → Classifier → Collector → Analyst (через LangGraph).
2. Выведет исходный запрос.
3. Покажет сгенерированный SQL-запрос.
4. Покажет количество строк результата.
5. Напечатает аналитический комментарий Analyst.
6. Если строился график, сообщит его тип и выведет первые символы base64-кодировки.

### Одноразовый запуск (один запрос → один ответ)

Если передать запрос аргументом, будет выполнен один прогон и программа завершится:

```bash
python main.py "Построй динамику публикаций по теме deep learning 2010–2024"
```

## Требования к окружению

- Python 3.10+
- PostgreSQL 13+ (локально или в контейнере)
- Доступ к LLM через OpenAI-совместимый endpoint
  (адрес и ключ настраиваются в `app/config.py`)

## Конфигурация

Параметры задаются через переменные окружения (с дефолтами для локального запуска):

- `POSTGRES_HOST` (по умолчанию `localhost`)
- `POSTGRES_PORT` (по умолчанию `5432`)
- `POSTGRES_DB` (по умолчанию `papers_db`)
- `POSTGRES_USER` (по умолчанию `postgres`)
- `POSTGRES_PASSWORD` (по умолчанию `postgres`)
- `LITELLM_BASE_URL` (по умолчанию `http://localhost:34000/v1`)
- `LITELLM_API_KEY` (по умолчанию пусто)
- `MODEL_NAME` (по умолчанию `qwen3-32b`)

Дополнительно:

- `OPENALEX_MAILTO` — рекомендованный параметр OpenAlex `mailto` (можно не задавать).

### Важно про OpenAlex

По умолчанию система использует **только локальную БД (SQL)**.
Обращение к OpenAlex выполняется **только если пользователь явно попросил об этом в текущем запросе**.

Способы явно запросить OpenAlex:

- префикс `@openalex ...` — использовать OpenAlex как источник данных Collector;
- префикс `@both ...` — собрать и вывести данные из SQL и OpenAlex (для поддерживаемых типов задач);
- явная фраза в тексте запроса (например: "используй OpenAlex").

## Reasoning loop

Пайплайн выполняется итеративно: если по итогам первой попытки выборка из БД пустая,
система делает повторную попытку с обратной связью для Planner (до 2 попыток).
