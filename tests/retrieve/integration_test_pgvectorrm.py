"""Instructions:
Add to dev container features:
    "ghcr.io/itsmechlark/features/postgresql:1": {},
    "ghcr.io/robbert229/devcontainer-features/postgresql-client:1": {}
Add to .personalization.sh:
    sudo apt install -y postgresql-16-pgvector

    sudo /etc/init.d/postgresql restart

    psql -v ON_ERROR_STOP=1 --user ${PGUSER} <<EOF
    create extension if not exists vector;
    EOF
poetry install -E postgres
"""
from dspy.primitives.example import Example
import pytest
import psycopg2
from dspy.retrieve.pgvector_rm import PgVectorRM

DB_URL = "postgresql://postgres:password@localhost/postgres"
PG_TABLE_NAME = "test_table"


def get_pgvectorrm():
    openai_client = None  # Mock or use a real OpenAI client
    pgvectorrm = PgVectorRM(DB_URL, PG_TABLE_NAME, openai_client=openai_client, embedding_func=lambda x: '[2,3,4]', include_similarity=True)
    return pgvectorrm


@pytest.fixture
def setup_pgvectorrm():
    pgvectorrm = get_pgvectorrm()
    conn = psycopg2.connect(DB_URL)
    cursor = conn.cursor()
    cursor.execute(f'DROP TABLE IF EXISTS {PG_TABLE_NAME}')
    conn.commit()

    cursor.execute(f"CREATE TABLE IF NOT EXISTS {PG_TABLE_NAME} (id SERIAL PRIMARY KEY, text TEXT, embedding VECTOR)")
    cursor.execute(f"INSERT INTO {PG_TABLE_NAME} (text, embedding) VALUES ('Dummy text1', '[1,2,3]'), ('Dummy text2', '[4,5,6]')")
    conn.commit()

    yield pgvectorrm

    cursor.execute(f'TRUNCATE TABLE {PG_TABLE_NAME}')
    conn.commit()

    cursor.close()
    conn.close()


def test_pgvectorrm_retrieve(setup_pgvectorrm):
    pgvectorrm = setup_pgvectorrm
    query = "test query"
    results = pgvectorrm(query)
    assert len(results) == 2
    assert results == [
        Example(text='Dummy text2', similarity=0.9946115458726394),
        Example(text='Dummy text1', similarity=0.9925833339709302)
    ]


@pytest.mark.parametrize("k, expected", [
    (1, 1),
    (2, 2),
    (3, 2),  # Assuming only 2 entries exist
])
def test_pgvectorrm_retrieve_diff_k(setup_pgvectorrm, k, expected):
    setup_pgvectorrm.k = k
    query = "test query"
    results = setup_pgvectorrm(query)
    assert len(results) == expected


def test_empty_table():
    # Assuming setup_pgvectorrm cleans up after yielding
    query = "test query"
    results = get_pgvectorrm()(query)
    assert len(results) == 0


def test_retrieval_without_similarity(setup_pgvectorrm):
    setup_pgvectorrm.include_similarity = False
    query = "test query"
    results = setup_pgvectorrm(query)
    # Ensure 'similarity' key is not in results
    assert all('similarity' not in result for result in results)
