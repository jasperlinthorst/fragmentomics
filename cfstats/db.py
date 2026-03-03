import clickhouse_connect
import pandas as pd
import io


DEFAULT_HOST = 'localhost'
DEFAULT_PORT = 8123
DATABASE_NAME = 'fragmentomics'

CORE_TABLE = 'core'
META_TABLE = 'meta'


def get_client(host=DEFAULT_HOST, port=DEFAULT_PORT):
    client = clickhouse_connect.get_client(host=host, port=port)
    client.command(f'CREATE DATABASE IF NOT EXISTS {DATABASE_NAME}')
    client.command(f'USE {DATABASE_NAME}')
    return client


def ensure_meta_table(client):
    client.command(f'''
        CREATE TABLE IF NOT EXISTS {DATABASE_NAME}.{META_TABLE} (
            sample_id String,
            feature_name String,
            value String
        ) ENGINE = ReplacingMergeTree()
        ORDER BY (sample_id, feature_name)
    ''')


def ensure_core_table(client, columns):
    """Create the core table in wide format based on the provided column names.
    
    columns: list of (column_name, clickhouse_type) tuples, excluding sample_id.
    """
    existing = _get_existing_columns(client, CORE_TABLE)

    if not existing:
        col_defs = ',\n            '.join(
            [f'`{name}` {ctype}' for name, ctype in columns]
        )
        client.command(f'''
            CREATE TABLE IF NOT EXISTS {DATABASE_NAME}.{CORE_TABLE} (
                sample_id String,
                {col_defs}
            ) ENGINE = ReplacingMergeTree()
            ORDER BY (sample_id)
        ''')
    else:
        # Add any missing columns
        for name, ctype in columns:
            if name not in existing:
                client.command(
                    f'ALTER TABLE {DATABASE_NAME}.{CORE_TABLE} ADD COLUMN IF NOT EXISTS `{name}` {ctype}'
                )


def _get_existing_columns(client, table):
    try:
        result = client.query(
            f"SELECT name FROM system.columns WHERE database='{DATABASE_NAME}' AND table='{table}'"
        )
        return {row[0] for row in result.result_rows}
    except Exception:
        return set()


def _infer_clickhouse_type(series):
    if pd.api.types.is_integer_dtype(series):
        return 'Float64'
    elif pd.api.types.is_float_dtype(series):
        return 'Float64'
    else:
        return 'Float64'


def upload_core_tsv(client, filepath_or_df):
    """Upload a core TSV file (wide format) into the core table.
    
    The first column is the sample identifier.
    All other columns are stored as Float64.
    """
    if isinstance(filepath_or_df, pd.DataFrame):
        df = filepath_or_df
    else:
        df = pd.read_csv(filepath_or_df, sep='\t', index_col=0)
    
    df.index = df.index.astype(str)
    df.index.name = 'sample_id'

    feature_columns = [(str(c), 'Float64') for c in df.columns]
    ensure_core_table(client, feature_columns)

    df_insert = df.reset_index()
    df_insert.columns = ['sample_id'] + [str(c) for c in df.columns]

    # Convert all feature columns to numeric
    for col in df_insert.columns[1:]:
        df_insert[col] = pd.to_numeric(df_insert[col], errors='coerce')

    df_insert = df_insert.fillna(0.0)

    client.insert_df(f'{DATABASE_NAME}.{CORE_TABLE}', df_insert)
    return len(df_insert)


def upload_meta_tsv(client, filepath_or_df):
    """Upload a meta TSV file into the meta table in long format.
    
    The first column ('filename') is the sample identifier.
    All other columns are melted into (sample_id, feature_name, value) rows.
    """
    if isinstance(filepath_or_df, pd.DataFrame):
        df = filepath_or_df
    else:
        df = pd.read_csv(filepath_or_df, sep='\t')
    
    ensure_meta_table(client)

    # First column is the sample identifier
    id_col = df.columns[0]
    df[id_col] = df[id_col].astype(str)

    # Melt to long format
    feature_cols = [c for c in df.columns if c != id_col]
    df_long = df.melt(
        id_vars=[id_col],
        value_vars=feature_cols,
        var_name='feature_name',
        value_name='value'
    )
    df_long = df_long.rename(columns={id_col: 'sample_id'})
    df_long['value'] = df_long['value'].fillna('').astype(str)

    # Insert in batches to avoid memory issues
    batch_size = 100000
    total = len(df_long)
    for start in range(0, total, batch_size):
        batch = df_long.iloc[start:start + batch_size]
        client.insert_df(f'{DATABASE_NAME}.{META_TABLE}', batch[['sample_id', 'feature_name', 'value']])

    return total


def load_core_df(client):
    """Load the core table as a wide DataFrame with sample_id as index."""
    result = client.query_df(f'SELECT * FROM {DATABASE_NAME}.{CORE_TABLE}')
    if result.empty:
        return result
    result = result.set_index('sample_id')
    result.index = result.index.astype(str)
    return result


def load_meta_df(client):
    """Load the meta table (long format) and pivot back to wide DataFrame."""
    result = client.query_df(f'SELECT * FROM {DATABASE_NAME}.{META_TABLE}')
    if result.empty:
        return pd.DataFrame()
    
    # Pivot from long to wide format
    df_wide = result.pivot_table(
        index='sample_id',
        columns='feature_name',
        values='value',
        aggfunc='first'
    )
    df_wide.columns.name = None
    df_wide.index.name = 'filename'
    df_wide.index = df_wide.index.astype(str)

    # Try to convert numeric columns back to numeric types
    for col in df_wide.columns:
        converted = pd.to_numeric(df_wide[col], errors='coerce')
        non_empty = df_wide[col].replace('', pd.NA).dropna()
        if len(non_empty) > 0 and converted.notna().sum() >= len(non_empty) * 0.5:
            df_wide[col] = converted

    return df_wide


def get_sample_count(client, table):
    """Return the number of distinct samples in a table."""
    try:
        if table == META_TABLE:
            result = client.query(f'SELECT count(DISTINCT sample_id) FROM {DATABASE_NAME}.{META_TABLE}')
        else:
            result = client.query(f'SELECT count() FROM {DATABASE_NAME}.{CORE_TABLE}')
        return result.result_rows[0][0]
    except Exception:
        return 0


def drop_tables(client):
    """Drop both core and meta tables (useful for re-upload)."""
    client.command(f'DROP TABLE IF EXISTS {DATABASE_NAME}.{CORE_TABLE}')
    client.command(f'DROP TABLE IF EXISTS {DATABASE_NAME}.{META_TABLE}')


def upload_core_from_bytes(client, content_bytes):
    """Upload core TSV from raw bytes (for Dash upload component)."""
    text = content_bytes.decode('utf-8')
    df = pd.read_csv(io.StringIO(text), sep='\t', index_col=0)
    return upload_core_tsv(client, df)


def upload_meta_from_bytes(client, content_bytes):
    """Upload meta TSV from raw bytes (for Dash upload component)."""
    text = content_bytes.decode('utf-8')
    df = pd.read_csv(io.StringIO(text), sep='\t')
    return upload_meta_tsv(client, df)
