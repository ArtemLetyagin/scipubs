import psycopg2
from psycopg2.extras import execute_values

# ---- Параметры подключения к серверу PostgreSQL ----
DB_USER = "postgres"
DB_PASSWORD = "postgres"
DB_HOST = "localhost"
DB_PORT = "5432"
TARGET_DB = "papers_db"


# ---------- 1. Создать БД, если её ещё нет ----------
def ensure_database():
    conn = psycopg2.connect(
        dbname="postgres",
        user=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=DB_PORT,
    )
    conn.autocommit = True
    with conn.cursor() as cur:
        cur.execute("SELECT 1 FROM pg_database WHERE datname = %s;", (TARGET_DB,))
        exists = cur.fetchone() is not None
        if not exists:
            print(f"Создаю базу данных {TARGET_DB}...")
            cur.execute(f"CREATE DATABASE {TARGET_DB};")
        else:
            print(f"База данных {TARGET_DB} уже существует.")
    conn.close()


# ---------- 2. Подключение к целевой БД ----------
def get_target_connection():
    return psycopg2.connect(
        dbname=TARGET_DB,
        user=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=DB_PORT,
    )


# ---------- 3. Таблица articles ----------
def create_table(conn):
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS articles (
        id SERIAL PRIMARY KEY,
        doi VARCHAR(255) UNIQUE NOT NULL,
        abstract TEXT,
        title TEXT NOT NULL,
        publication_year INT,
        cited_by_count INT,
        journal TEXT,
        domain TEXT,
        field TEXT,
        subfield TEXT,
        topic TEXT
    );
    """
    with conn.cursor() as cur:
        cur.execute(create_table_sql)
    conn.commit()
    print("Таблица articles готова.")



# ---------- 3b. Создать совместимый VIEW articles_cast (для кода MAS) ----------
def ensure_articles_cast_view(conn):
    """
    Код MAS ожидает таблицу/представление `articles_cast` с колонкой `cited_by_count`.
    Для совместимости создаём VIEW поверх `articles`.
    """
    with conn.cursor() as cur:
        # Пытаемся создать VIEW, используя cited_by_count (актуальная схема).
        try:
            cur.execute("""
                CREATE OR REPLACE VIEW articles_cast AS
                SELECT
                    id, doi, abstract, title, publication_year,
                    cited_by_count,
                    journal, domain, field, subfield, topic
                FROM articles;
            """)
        except Exception:
            # Если таблица очень старая и есть только cted_by_count — создадим VIEW с алиасом.
            conn.rollback()
            cur.execute("""
                CREATE OR REPLACE VIEW articles_cast AS
                SELECT
                    id, doi, abstract, title, publication_year,
                    cted_by_count AS cited_by_count,
                    journal, domain, field, subfield, topic
                FROM articles;
            """)
    conn.commit()
    print("VIEW articles_cast готово (совместимость с пайплайном).")

# ---------- 3a. Гарантированно добавить новые колонки (на случай старой версии таблицы) ----------
def migrate_articles_add_new_columns(conn):
    with conn.cursor() as cur:
        cur.execute("""
            ALTER TABLE articles
                ADD COLUMN IF NOT EXISTS domain   TEXT,
                ADD COLUMN IF NOT EXISTS field    TEXT,
                ADD COLUMN IF NOT EXISTS subfield TEXT,
                ADD COLUMN IF NOT EXISTS topic    TEXT;
        """)
    conn.commit()


# ---------- 3c. Миграция: гарантировать cited_by_count и перенос значений из старого cted_by_count ----------
def migrate_articles_fix_cited_by(conn):
    with conn.cursor() as cur:
        # Добавим правильную колонку, если её нет
        cur.execute("ALTER TABLE articles ADD COLUMN IF NOT EXISTS cited_by_count INT;")
        # Попробуем скопировать из старой колонки, если она существует
        try:
            cur.execute("""
                UPDATE articles
                SET cited_by_count = cted_by_count
                WHERE cited_by_count IS NULL AND cted_by_count IS NOT NULL;
            """)
        except Exception:
            # старой колонки может не быть — это ок
            conn.rollback()
    conn.commit()
    print("Миграция cited_by_count выполнена (если была нужна).")
    print("Колонки domain, field, subfield, topic добавлены (если их не было).")
    
# ---------- 4. Заполнение / обновление данных ----------
def insert_sample_data(conn):
    # Базовые записи
    base_rows = [
        (
            "10.1000/xyz123",
            "We propose a new numerical method for solving large sparse linear systems arising in scientific computing.",
            "A Novel Iterative Method for Sparse Linear Systems",
            2020,
            35,
            "Journal of Computational Mathematics",
            "Mathematics",
            "Applied Mathematics",
            "Numerical Linear Algebra",
            "Sparse Linear Systems",
        ),
        (
            "10.1016/j.apm.2021.005",
            "The paper studies optimization algorithms for training deep neural networks with constraints.",
            "Constrained Optimization Methods in Deep Learning",
            2021,
            52,
            "Applied Mathematical Modelling",
            "Computer Science",
            "Artificial Intelligence",
            "Optimization for Deep Learning",
            "Constrained Optimization",
        ),
        (
            "10.1145/3394486.3403211",
            "We introduce an efficient graph-based recommendation system and evaluate it on real-world datasets.",
            "Graph-Based Recommender Systems at Scale",
            2019,
            120,
            "ACM Transactions on Information Systems",
            "Computer Science",
            "Information Retrieval",
            "Recommender Systems",
            "Graph-Based Recommendation",
        ),
        (
            "10.1109/TIT.2022.3141234",
            "This work analyzes error-correcting codes for modern communication channels with fading.",
            "Error-Correcting Codes for Fading Channels",
            2022,
            18,
            "IEEE Transactions on Information Theory",
            "Engineering",
            "Electrical Engineering",
            "Coding Theory",
            "Error-Correcting Codes",
        ),
        (
            "10.1093/biomet/asab012",
            "We discuss Bayesian approaches to modeling high-dimensional time series in econometrics.",
            "Bayesian Models for High-Dimensional Time Series",
            2021,
            44,
            "Biometrika",
            "Statistics",
            "Bayesian Statistics",
            "Time Series Analysis",
            "Bayesian Time Series",
        ),
        (
            "10.1137/19M1276416",
            "The article explores fast algorithms for large-scale eigenvalue problems in scientific computing.",
            "Fast Algorithms for Large-Scale Eigenvalue Problems",
            2019,
            67,
            "SIAM Journal on Scientific Computing",
            "Mathematics",
            "Numerical Analysis",
            "Eigenvalue Problems",
            "Large-Scale Eigenproblems",
        ),
    ]
    # Метаданные по тематикам
    topic_meta = {
        "Machine Learning": (
            "Computer Science",
            "Artificial Intelligence",
            "Machine Learning",
            "Supervised and Unsupervised Learning",
        ),
        "Numerical Analysis": (
            "Mathematics",
            "Applied Mathematics",
            "Numerical Analysis",
            "Numerical Methods",
        ),
        "Graph Theory": (
            "Mathematics",
            "Discrete Mathematics",
            "Graph Theory",
            "Graph Algorithms",
        ),
        "Optimization": (
            "Mathematics",
            "Optimization",
            "Convex and Nonlinear Optimization",
            "Optimization Methods",
        ),
        "Data Mining": (
            "Computer Science",
            "Data Science",
            "Data Mining",
            "Pattern Discovery",
        ),
        "Computer Vision": (
            "Computer Science",
            "Artificial Intelligence",
            "Computer Vision",
            "Image Recognition",
        ),
        "Reinforcement Learning": (
            "Computer Science",
            "Artificial Intelligence",
            "Reinforcement Learning",
            "Sequential Decision Making",
        ),
        "Cryptography": (
            "Computer Science",
            "Security",
            "Cryptography",
            "Modern Cryptographic Protocols",
        ),
        "Complex Networks": (
            "Physics",
            "Statistical Physics",
            "Complex Networks",
            "Network Dynamics",
        ),
        "Statistical Inference": (
            "Statistics",
            "Statistical Inference",
            "Parametric and Nonparametric Inference",
            "Estimation and Hypothesis Testing",
        ),
    }

    topics = list(topic_meta.keys())

    journals = [
        "Journal of Machine Learning Research",
        "Numerical Algorithms",
        "Discrete Mathematics",
        "Optimization Letters",
        "Data Mining and Knowledge Discovery",
        "IEEE Transactions on Neural Networks",
        "Pattern Recognition Letters",
        "Journal of Cryptology",
        "Network Science",
        "Annals of Statistics",
    ]

    generated_rows = []
    for i in range(1, 51):
        topic_name = topics[(i - 1) % len(topics)]
        domain, field, subfield, topic_label = topic_meta[topic_name]
        journal = journals[(i - 1) % len(journals)]
        publication_year = 2005 + (i % 20)
        cited_by_count = (i * 7) % 200

        doi = f"10.1234/sim.{publication_year}.{i:03d}"
        title = f"Simulated Study {i} on {topic_name}"
        abstract = (
            f"This simulated article ({i}) discusses methods and experiments related to "
            f"{topic_name.lower()} with applications in applied mathematics and computer science."
        )

        generated_rows.append(
            (
                doi,
                abstract,
                title,
                publication_year,
                cited_by_count,
                journal,
                domain,
                field,
                subfield,
                topic_label,
            )
        )

    rows = base_rows + generated_rows

    # ВАЖНО: используем DO UPDATE, чтобы заполнить колонки для уже существующих строк
    insert_sql = """
    INSERT INTO articles (
        doi,
        abstract,
        title,
        publication_year,
        cited_by_count,
        journal,
        domain,
        field,
        subfield,
        topic
    )
    VALUES %s
    ON CONFLICT (doi) DO UPDATE SET
        abstract         = EXCLUDED.abstract,
        title            = EXCLUDED.title,
        publication_year = EXCLUDED.publication_year,
        cited_by_count    = EXCLUDED.cited_by_count,
        journal          = EXCLUDED.journal,
        domain           = EXCLUDED.domain,
        field            = EXCLUDED.field,
        subfield         = EXCLUDED.subfield,
        topic            = EXCLUDED.topic;
    """

    with conn.cursor() as cur:
        execute_values(cur, insert_sql, rows)
    conn.commit()
    print(f"Добавлено/обновлено {len(rows)} записей.")


# ---------- 5. Проверочный вывод ----------
def show_data(conn, limit=15):
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, doi, title, publication_year, cited_by_count, journal,
                   domain, field, subfield, topic
            FROM articles
            ORDER BY id
            LIMIT %s;
            """,
            (limit,),
        )
        for row in cur.fetchall():
            print(row)


ensure_database()
conn = get_target_connection()
try:
    create_table(conn)
    migrate_articles_add_new_columns(conn)
    insert_sample_data(conn)
    print("Первые строки в таблице:")
    show_data(conn, limit=15)
finally:
    conn.close()
