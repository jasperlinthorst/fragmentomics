import clickhouse_connect
import pandas as pd
import io
import os
import logging

DEFAULT_HOST = 'localhost'
DEFAULT_PORT = 8123
DATABASE_NAME = 'fragmentomics'

CORE_TABLE = 'core'
META_TABLE = 'meta'
META_FEATURES_TABLE = 'meta_features'
CORE_STATS_TABLE = 'core_stats'

log = logging.getLogger(__name__)

def get_client(host=DEFAULT_HOST, port=DEFAULT_PORT):
    client = clickhouse_connect.get_client(host=host, port=port)
    client.command(f'CREATE DATABASE IF NOT EXISTS {DATABASE_NAME}')
    client.command(f'USE {DATABASE_NAME}')
    return client


def get_fresh_client(host=DEFAULT_HOST, port=DEFAULT_PORT):
    """Get a fresh ClickHouse client instance for concurrent use."""
    return get_client(host=host, port=port)


def ensure_meta_table(client):
    client.command(f'''
        CREATE TABLE IF NOT EXISTS {DATABASE_NAME}.{META_TABLE} (
            sample_id String,
            feature_name String,
            value String
        ) ENGINE = ReplacingMergeTree()
        ORDER BY (sample_id, feature_name)
    ''')


def ensure_meta_features_table(client):
    client.command(f'''
        CREATE TABLE IF NOT EXISTS {DATABASE_NAME}.{META_FEATURES_TABLE} (
            feature_name String,
            data_type String
        ) ENGINE = ReplacingMergeTree()
        ORDER BY (feature_name)
    ''')


def ensure_core_stats_table(client):
    client.command(f'''
        CREATE TABLE IF NOT EXISTS {DATABASE_NAME}.{CORE_STATS_TABLE} (
            feature_name String,
            mean Float64,
            std Float64
        ) ENGINE = ReplacingMergeTree()
        ORDER BY (feature_name)
    ''')


def ensure_core_table(client, columns, recreate=False):
    """Create the core table in wide format based on the provided column names.
    
    columns: list of (column_name, clickhouse_type) tuples, excluding sample_id.
    """
    if recreate:
        client.command(f'DROP TABLE IF EXISTS {DATABASE_NAME}.{CORE_TABLE}')

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
        return 'UInt32'
    elif pd.api.types.is_float_dtype(series):
        return 'Float64'
    else:
        return 'String'

def infer_core_column_types(df, sample_rows=100):
    """Infer ClickHouse types for core TSV columns based on a sample of rows.

    Returns dict: column_name -> ClickHouse type.
    """
    if df is None or df.empty:
        return {}

    df_sample = df.head(sample_rows)
    types = {}
    for col in df_sample.columns:
        s = pd.to_numeric(df_sample[col], errors='coerce')
        s = s.dropna()
        if s.empty:
            types[str(col)] = 'UInt32'
            continue

        # Check if all values are integer-like
        is_int_like = (s % 1 == 0).all()
        if not is_int_like:
            types[str(col)] = 'Float64'
            continue

        min_v = int(s.min())
        max_v = int(s.max())
        if min_v >= 0:
            if max_v <= 255:
                types[str(col)] = 'UInt8'
            elif max_v <= 65535:
                types[str(col)] = 'UInt16'
            elif max_v <= 4294967295:
                types[str(col)] = 'UInt32'
            else:
                types[str(col)] = 'UInt64'
        else:
            # Signed
            if -128 <= min_v and max_v <= 127:
                types[str(col)] = 'Int8'
            elif -32768 <= min_v and max_v <= 32767:
                types[str(col)] = 'Int16'
            elif -2147483648 <= min_v and max_v <= 2147483647:
                types[str(col)] = 'Int32'
            else:
                types[str(col)] = 'Int64'

    return types


def upload_core_tsv(client, filepath_or_df, column_types=None, recreate_table=False):
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

    if column_types is None:
        feature_columns = [(str(c), 'Float64') for c in df.columns]
    else:
        feature_columns = [(str(c), str(column_types.get(str(c), 'Float64'))) for c in df.columns]
    ensure_core_table(client, feature_columns, recreate=recreate_table)

    df_insert = df.reset_index()
    df_insert.columns = ['sample_id'] + [str(c) for c in df.columns]

    # Convert all feature columns to numeric
    for col in df_insert.columns[1:]:
        df_insert[col] = pd.to_numeric(df_insert[col], errors='coerce')

    df_insert = df_insert.fillna(0.0)

    client.insert_df(f'{DATABASE_NAME}.{CORE_TABLE}', df_insert)
    
    # Update core stats after upload
    try:
        update_core_stats(client)
    except Exception:
        pass  # Don't fail upload if stats computation fails
    
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

    # Drop empty values (including NaN) to keep the meta table compact
    df_long = df_long[df_long['value'].notna()]
    df_long['value'] = df_long['value'].astype(str)
    df_long['value'] = df_long['value'].str.strip()
    df_long = df_long[(df_long['value'] != '') & (df_long['value'].str.lower() != 'nan')]

    # Insert in batches to avoid memory issues
    batch_size = 100000
    total = len(df_long)
    for start in range(0, total, batch_size):
        batch = df_long.iloc[start:start + batch_size]
        client.insert_df(f'{DATABASE_NAME}.{META_TABLE}', batch[['sample_id', 'feature_name', 'value']])

    # Update meta features after upload
    try:
        update_meta_features(client)
    except Exception:
        pass  # Don't fail upload if features computation fails
    
    return total


def delete_empty_meta_values(client):
    """Remove rows from meta where value is empty string.

    ClickHouse doesn't have UPDATE-in-place; this uses a mutation.
    """
    client.command(
        f"ALTER TABLE {DATABASE_NAME}.{META_TABLE} DELETE WHERE (value = '') OR (lower(value) = 'nan')"
    )


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
    client.command(f'DROP TABLE IF EXISTS {DATABASE_NAME}.{META_FEATURES_TABLE}')
    client.command(f'DROP TABLE IF EXISTS {DATABASE_NAME}.{CORE_STATS_TABLE}')


def upload_core_from_bytes(client, content_bytes, column_types=None, recreate_table=False):
    """Upload core TSV from raw bytes (for Dash upload component)."""
    text = content_bytes.decode('utf-8')
    df = pd.read_csv(io.StringIO(text), sep='\t', index_col=0)
    return upload_core_tsv(client, df, column_types=column_types, recreate_table=recreate_table)


def upload_meta_from_bytes(client, content_bytes):
    """Upload meta TSV from raw bytes (for Dash upload component)."""
    text = content_bytes.decode('utf-8')
    df = pd.read_csv(io.StringIO(text), sep='\t')
    return upload_meta_tsv(client, df)


def update_meta_features(client):
    """Update the meta_features table with current feature names from meta table."""
    ensure_meta_features_table(client)
    
    # Clear existing entries
    client.command(f'TRUNCATE TABLE {DATABASE_NAME}.{META_FEATURES_TABLE}')
    
    # Get distinct feature names and their data types
    result = client.query(f'''
        SELECT 
            feature_name,
            any(CASE 
                WHEN toFloat64OrNull(value) IS NOT NULL AND value NOT LIKE '%.%' AND value NOT LIKE '%e%' 
                THEN 'numeric' 
                ELSE 'categorical' 
            END) as data_type
        FROM {DATABASE_NAME}.{META_TABLE} 
        WHERE value != '' AND lower(value) != 'nan'
        GROUP BY feature_name
        ORDER BY feature_name
    ''')
    
    if result.result_rows:
        df_features = pd.DataFrame(result.result_rows, columns=['feature_name', 'data_type'])
        client.insert_df(f'{DATABASE_NAME}.{META_FEATURES_TABLE}', df_features)
    
    return len(result.result_rows) if result.result_rows else 0


def update_core_stats(client):
    """Update the core_stats table with mean and std dev for each core feature."""
    ensure_core_stats_table(client)
    
    # Get column names (excluding sample_id)
    columns_result = client.query(
        f"SELECT name FROM system.columns WHERE database='{DATABASE_NAME}' AND table='{CORE_TABLE}' AND name != 'sample_id'"
    )

    if not columns_result.result_rows:
        return 0
    
    # Clear existing entries
    client.command(f'TRUNCATE TABLE {DATABASE_NAME}.{CORE_STATS_TABLE}')
    
    # Compute stats for each column
    stats_data = []
    for row in columns_result.result_rows:
        col_name = row[0]
        try:
            # Compute mean and std using ClickHouse aggregation functions
            stats_result = client.query(f'''
                SELECT 
                    avg(`{col_name}`) as mean_val,
                    stddevSamp(`{col_name}`) as std_val
                FROM {DATABASE_NAME}.{CORE_TABLE}
                WHERE `{col_name}` IS NOT NULL
            ''')

            if stats_result.result_rows:
                mean_val, std_val = stats_result.result_rows[0]
                if mean_val is not None:  # Only add if we have data
                    stats_data.append([col_name, float(mean_val), float(std_val) if std_val is not None else 0.0])
        except Exception:
            continue  # Skip columns that can't be aggregated
    
    if stats_data:
        df_stats = pd.DataFrame(stats_data, columns=['feature_name', 'mean', 'std'])
        client.insert_df(f'{DATABASE_NAME}.{CORE_STATS_TABLE}', df_stats)
    
    return len(stats_data)


def get_meta_features(client):
    """Get list of available meta feature names."""
    try:
        result = client.query(f'SELECT feature_name FROM {DATABASE_NAME}.{META_FEATURES_TABLE} ORDER BY feature_name')
        return [row[0] for row in result.result_rows]
    except Exception:
        return []


def get_core_stats(client):
    """Get core feature statistics as a DataFrame."""
    try:
        return client.query_df(f'SELECT * FROM {DATABASE_NAME}.{CORE_STATS_TABLE} ORDER BY feature_name')
    except Exception:
        return pd.DataFrame()


def lazy_load_meta_samples(client, sample_ids=None, features=None):
    """Lazily load meta data for specific samples and features."""
    conditions = []
    
    log.info('lazy_load_meta_samples (%d,%d)',len(sample_ids),len(features))

    # Handle large sample ID lists by chunking
    if sample_ids and len(sample_ids) > 1000:
        # Process in chunks to avoid query size limits
        all_results = []
        chunk_size = 1000
        for i in range(0, len(sample_ids), chunk_size):
            chunk_ids = sample_ids[i:i + chunk_size]
            chunk_result = _lazy_load_meta_samples_chunk(client, chunk_ids, features)
            if not chunk_result.empty:
                all_results.append(chunk_result)
        
        if not all_results:
            return pd.DataFrame()
        
        # Combine all chunks
        df_combined = pd.concat(all_results, ignore_index=False)
        # Remove duplicates (in case of overlapping chunks)
        df_combined = df_combined[~df_combined.index.duplicated(keep='first')]
        
        return df_combined
    else:
        return _lazy_load_meta_samples_chunk(client, sample_ids, features)


def _lazy_load_meta_samples_chunk(client, sample_ids=None, features=None):
    """Load a chunk of meta data for specific samples and features."""
    conditions = []
    if sample_ids:
        sample_ids_str = "', '".join(str(sid) for sid in sample_ids)
        conditions.append(f"sample_id IN ('{sample_ids_str}')")
    
    if features:
        features_str = "', '".join(str(f) for f in features)
        conditions.append(f"feature_name IN ('{features_str}')")
    
    where_clause = f" WHERE {' AND '.join(conditions)}" if conditions else ""
    
    query = f'SELECT * FROM {DATABASE_NAME}.{META_TABLE}{where_clause}'
    result = client.query_df(query)
    
    if result.empty:
        return pd.DataFrame()
    
    log.info('lazy_load_meta_samples result: %d rows', len(result))
    
    # Pivot to wide format
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


def lazy_load_core_samples(client, sample_ids=None, features=None):
    """Lazily load core data for specific samples and features."""
    
    log.info('lazy_load_core_samples (%d,%d)',len(sample_ids),len(features))

    # Handle large sample ID lists by chunking
    if sample_ids and len(sample_ids) > 1000:
        # Process in chunks to avoid query size limits
        all_results = []
        chunk_size = 1000
        for i in range(0, len(sample_ids), chunk_size):
            chunk_ids = sample_ids[i:i + chunk_size]
            chunk_result = _lazy_load_core_samples_chunk(client, chunk_ids, features)
            if not chunk_result.empty:
                all_results.append(chunk_result)
        
        if not all_results:
            return pd.DataFrame()
        
        # Combine all chunks
        df_combined = pd.concat(all_results, ignore_index=False)
        # Remove duplicates (in case of overlapping chunks)
        df_combined = df_combined[~df_combined.index.duplicated(keep='first')]
        
        return df_combined
    else:
        return _lazy_load_core_samples_chunk(client, sample_ids, features)


def _lazy_load_core_samples_chunk(client, sample_ids=None, features=None):
    """Load a chunk of core data for specific samples and features."""
    conditions = []
    if sample_ids:
        sample_ids_str = "', '".join(str(sid) for sid in sample_ids)
        conditions.append(f"sample_id IN ('{sample_ids_str}')")
    
    select_cols = 'sample_id'
    if features:
        features_str = ", ".join(f"`{f}`" for f in features)
        select_cols = f"sample_id, {features_str}"
    
    where_clause = f" WHERE {' AND '.join(conditions)}" if conditions else ""
    
    query = f'SELECT {select_cols} FROM {DATABASE_NAME}.{CORE_TABLE}{where_clause}'
    result = client.query_df(query)
    
    if result.empty:
        return result
    
    result = result.set_index('sample_id')
    result.index = result.index.astype(str)
    return result


def apply_filters_clickhouse(client, filters):
    """Apply filters directly in ClickHouse and return matching sample_ids."""
    if not filters:
        # Return all sample_ids from meta table
        result = client.query(f'SELECT DISTINCT sample_id FROM {DATABASE_NAME}.{META_TABLE}')
        return [row[0] for row in result.result_rows]
    
    conditions = []
    for f in filters:
        col = f.get('column')
        op = f.get('op')
        if not col:
            continue
        
        if op == 'range':
            lo = f.get('lo')
            hi = f.get('hi')
            if lo is not None and hi is not None:
                conditions.append(f"toFloat64OrNull(value) BETWEEN {lo} AND {hi} AND feature_name = '{col}'")
        elif op in ('==', '!='):
            expr = f.get('expr')
            if expr and str(expr).strip():
                if op == '==':
                    conditions.append(f"match(value, '{expr}') AND feature_name = '{col}'")
                else:
                    conditions.append(f"NOT match(value, '{expr}') AND feature_name = '{col}'")
    
    if not conditions:
        return []
    
    where_clause = " AND ".join(conditions)
    query = f'''
        SELECT sample_id FROM {DATABASE_NAME}.{META_TABLE}
        WHERE {where_clause}
        GROUP BY sample_id
        HAVING count(*) = {len(conditions)}
    '''
    
    try:
        result = client.query(query)
        return [row[0] for row in result.result_rows]
    except Exception:
        return []


def insert_alignment_results(client, filename, features, umap1, umap2):
    """Insert alignment upload results into ClickHouse tables.
    
    Args:
        client: ClickHouse client
        filename: Original filename (used as base for sample_id)
        features: Feature vector (numpy array) matching core table structure
        umap1: UMAP coordinate 1
        umap2: UMAP coordinate 2
    """
    import uuid
    
    # Generate unique sample_id from filename and UUID
    base_name = os.path.splitext(os.path.basename(filename))[0]
    unique_id = f"{base_name}_{uuid.uuid4().hex[:8]}"
    
    # Insert core features
    # First, get the core table structure
    columns_result = client.query(
        f"SELECT name FROM system.columns WHERE database='{DATABASE_NAME}' AND table='{CORE_TABLE}' AND name != 'sample_id' ORDER BY position"
    )
    
    if columns_result.result_rows and len(features) == len(columns_result.result_rows):
        # Build insert data for core table
        core_data = {'sample_id': [unique_id]}
        for i, row in enumerate(columns_result.result_rows):
            col_name = row[0]
            core_data[col_name] = [float(features[i])]
        
        df_core = pd.DataFrame(core_data)
        client.insert_df(f'{DATABASE_NAME}.{CORE_TABLE}', df_core)
        
        # Update core stats
        try:
            update_core_stats(client)
        except Exception:
            pass
        
        # Insert UMAP coordinates into meta table
        meta_data = [
            {'sample_id': unique_id, 'feature_name': 'umap1', 'value': str(umap1)},
            {'sample_id': unique_id, 'feature_name': 'umap2', 'value': str(umap2)},
            {'sample_id': unique_id, 'feature_name': 'filename', 'value': filename},
        ]
        df_meta = pd.DataFrame(meta_data)
        client.insert_df(f'{DATABASE_NAME}.{META_TABLE}', df_meta)
        
        # Update meta features
        try:
            update_meta_features(client)
        except Exception:
            pass
        
        return unique_id
    
    return None
