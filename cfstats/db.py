import clickhouse_connect
import numpy as np
import pandas as pd
import io
import os
import logging
import re
import time
import gzip
import threading

DEFAULT_HOST = 'localhost'
DEFAULT_PORT = 8123
DATABASE_NAME = 'fragmentomics'

META_TABLE = 'meta'
META_FEATURES_TABLE = 'meta_features'

# Regex for bincount feature names: chr1_0_1000000, chrX_500_1500, etc.
BINCOUNT_PATTERN = re.compile(r'^(chr)?[\dXYMT]+_\d+_\d+$', re.IGNORECASE)

# Supported typed meta tables: data_type -> table_name
TYPED_META_TABLES = {
    'UInt8':    'meta_uint8',
    'UInt16':   'meta_uint16',
    'UInt32':   'meta_uint32',
    'Int8':     'meta_int8',
    'Int16':    'meta_int16',
    'Int32':    'meta_int32',
    'Int64':    'meta_int64',
    'Float32':  'meta_float32',
    'Float64':  'meta_float64',
    'Bool':     'meta_bool',
    'Date':     'meta_date',
    'DateTime': 'meta_datetime',
    'String':   'meta',
}

# ClickHouse column type for the value column in each typed table
CH_VALUE_TYPES = {
    'UInt8':    'UInt8',
    'UInt16':   'UInt16',
    'UInt32':   'UInt32',
    'Int8':     'Int8',
    'Int16':    'Int16',
    'Int32':    'Int32',
    'Int64':    'Int64',
    'Float32':  'Float32',
    'Float64':  'Float64',
    'Bool':     'UInt8',
    'Date':     'Date',
    'DateTime': 'DateTime',
    'String':   'String',
}

SUPPORTED_DATA_TYPES = list(TYPED_META_TABLES.keys())

# ── Upload progress tracking ──
_upload_progress = {}
_upload_progress_lock = threading.Lock()


def _set_upload_progress(token, current, total, rows_inserted, status='running'):
    with _upload_progress_lock:
        _upload_progress[token] = {
            'current': current, 'total': total,
            'rows_inserted': rows_inserted, 'status': status,
        }


def get_upload_progress(token):
    """Return progress dict for a given upload token."""
    with _upload_progress_lock:
        return dict(_upload_progress.get(token, {
            'current': 0, 'total': 0, 'rows_inserted': 0, 'status': 'unknown',
        }))


def clear_upload_progress(token):
    with _upload_progress_lock:
        _upload_progress.pop(token, None)


log = logging.getLogger(__name__)

def get_client(host=DEFAULT_HOST, port=DEFAULT_PORT):
    client = clickhouse_connect.get_client(host=host, port=port, send_receive_timeout=6000000)
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
            value String,
            version UInt64
        ) ENGINE = ReplacingMergeTree(version)
        ORDER BY (sample_id, feature_name)
    ''')

def ensure_typed_tables(client):
    """Create all typed meta tables if they don't exist."""
    ensure_meta_table(client)
    for dtype, table_name in TYPED_META_TABLES.items():
        if dtype == 'String':
            continue  # Already handled by ensure_meta_table
        ch_type = CH_VALUE_TYPES[dtype]
        client.command(f'''
            CREATE TABLE IF NOT EXISTS {DATABASE_NAME}.{table_name} (
                sample_id String,
                feature_name String,
                value {ch_type},
                version UInt64
            ) ENGINE = ReplacingMergeTree(version)
            ORDER BY (sample_id, feature_name)
        ''')


def ensure_meta_features_table(client):
    client.command(f'''
        CREATE TABLE IF NOT EXISTS {DATABASE_NAME}.{META_FEATURES_TABLE} (
            feature_name String,
            data_type String,
            mean Nullable(Float64),
            std Nullable(Float64)
        ) ENGINE = ReplacingMergeTree()
        ORDER BY (feature_name)
    ''')
    _migrate_meta_features_schema(client)


def _migrate_meta_features_schema(client):
    """Add feature_type and description columns if they don't exist yet."""
    existing = {row[0] for row in client.query(
        f"SELECT name FROM system.columns "
        f"WHERE database='{DATABASE_NAME}' AND table='{META_FEATURES_TABLE}'"
    ).result_rows}
    if 'feature_type' not in existing:
        client.command(
            f"ALTER TABLE {DATABASE_NAME}.{META_FEATURES_TABLE} "
            f"ADD COLUMN feature_type String DEFAULT 'meta'"
        )
        log.info('Added feature_type column to %s', META_FEATURES_TABLE)
    if 'description' not in existing:
        client.command(
            f"ALTER TABLE {DATABASE_NAME}.{META_FEATURES_TABLE} "
            f"ADD COLUMN description String DEFAULT ''"
        )
        log.info('Added description column to %s', META_FEATURES_TABLE)


def _is_core_feature(name):
    """Return True if feature_name looks like a core feature (not shown in dropdowns)."""
    name = str(name)
    # fszd: purely numeric 0-999
    try:
        ci = int(name)
        if 0 <= ci <= 999:
            return True
    except (TypeError, ValueError):
        pass
    # csm / 5p suffixes
    if name.endswith('-csm') or name.endswith('-5p'):
        return True
    # bincount pattern
    if BINCOUNT_PATTERN.match(name):
        return True
    return False


def _get_table_for_type(data_type):
    """Return the ClickHouse table name for a given data type."""
    if data_type not in TYPED_META_TABLES:
        raise ValueError(f"Unsupported data type: {data_type!r}. "
                         f"Supported: {', '.join(TYPED_META_TABLES)}")
    return TYPED_META_TABLES[data_type]


def _detect_column_type(series):
    """Auto-detect the smallest possible ClickHouse data type for a pandas Series."""
    s = series.dropna()
    str_vals = s.astype(str).str.strip()
    str_vals = str_vals[(str_vals != '') & (str_vals.str.lower() != 'nan')]

    if len(str_vals) == 0:
        return 'String'

    # Try boolean (only when actual boolean words are present, not just 0/1)
    lower_vals = str_vals.str.lower()
    if lower_vals.isin({'true', 'false', '0', '1', 'yes', 'no'}).all():
        if lower_vals.isin({'true', 'false', 'yes', 'no'}).any():
            return 'Bool'

    # Try numeric
    numeric = pd.to_numeric(str_vals, errors='coerce')
    valid_count = numeric.notna().sum()

    if valid_count >= len(str_vals) * 0.9:
        non_null = numeric.dropna()
        if len(non_null) > 0:
            # Check if all values are integers
            try:
                is_int = (non_null == non_null.round(0)).all()
            except (ValueError, OverflowError):
                is_int = False

            if is_int:
                vmin, vmax = non_null.min(), non_null.max()
                if vmin >= 0:
                    if vmax <= 255:
                        return 'UInt8'
                    if vmax <= 65535:
                        return 'UInt16'
                    if vmax <= 4294967295:
                        return 'UInt32'
                    return 'Int64'
                else:
                    if vmin >= -128 and vmax <= 127:
                        return 'Int8'
                    if vmin >= -32768 and vmax <= 32767:
                        return 'Int16'
                    if vmin >= -2147483648 and vmax <= 2147483647:
                        return 'Int32'
                    return 'Int64'
            else:
                # Floating point — check if Float32 is sufficient
                f32 = non_null.astype('float32')
                if np.allclose(non_null.values, f32.values, rtol=1e-6, atol=1e-9, equal_nan=True):
                    return 'Float32'
                return 'Float64'

    # Try Date / DateTime
    try:
        parsed = pd.to_datetime(str_vals, errors='coerce', infer_datetime_format=True)
        if parsed.notna().sum() >= len(str_vals) * 0.9:
            # Check if any have a time component
            has_time = (parsed.dropna().dt.hour != 0).any() or \
                       (parsed.dropna().dt.minute != 0).any() or \
                       (parsed.dropna().dt.second != 0).any()
            return 'DateTime' if has_time else 'Date'
    except Exception:
        pass

    return 'String'


def _get_feature_data_type(client, feature_name):
    """Look up the data_type for a feature from meta_features.

    Returns the stored data type, or 'String' if the feature is not yet
    registered in meta_features.
    """
    try:
        result = client.query(
            f"SELECT data_type FROM {DATABASE_NAME}.{META_FEATURES_TABLE} "
            f"WHERE feature_name = '{feature_name}' LIMIT 1"
        )
        if result.result_rows:
            return result.result_rows[0][0]
    except Exception:
        pass
    return 'String'


def _get_feature_data_types_bulk(client, feature_names):
    """Look up data_types for multiple features from meta_features."""
    if not feature_names:
        return {}
    try:
        names_str = "', '".join(str(f) for f in feature_names)
        result = client.query(
            f"SELECT feature_name, data_type FROM {DATABASE_NAME}.{META_FEATURES_TABLE} "
            f"WHERE feature_name IN ('{names_str}')"
        )
        return {row[0]: row[1] for row in result.result_rows}
    except Exception:
        return {}


def analyze_tsv_columns(client, content_bytes_or_path):
    """Analyze a TSV and return per-column type info.

    Reads up to 1000 rows for type detection.  For each feature column returns
    its auto-detected type, the previously stored type (if any), and up to 25
    distinct sample values.

    Returns:
        (columns, id_col) where *columns* is a list of dicts with keys
        ``name, detected_type, existing_type, sample_values, has_more``
        and *id_col* is the name of the sample-id column (first column).
    """
    if isinstance(content_bytes_or_path, (bytes, bytearray)):
        raw = bytes(content_bytes_or_path)
        if raw[:2] == b'\x1f\x8b':
            raw = gzip.decompress(raw)
        df = pd.read_csv(io.BytesIO(raw), sep='\t', nrows=1000)
    else:
        df = pd.read_csv(content_bytes_or_path, sep='\t', nrows=1000)

    id_col = df.columns[0]
    feature_cols = [c for c in df.columns if c != id_col]

    detected = {c: _detect_column_type(df[c]) for c in feature_cols}

    try:
        existing = _get_feature_data_types_bulk(client, feature_cols)
    except Exception:
        existing = {}

    columns = []
    for c in feature_cols:
        uniq = df[c].dropna().astype(str).str.strip()
        uniq = uniq[uniq != ''].unique()
        has_more = len(uniq) > 25
        sample_vals = list(uniq[:25])
        columns.append({
            'name': c,
            'detected_type': detected[c],
            'existing_type': existing.get(c),
            'sample_values': sample_vals,
            'has_more': has_more,
        })

    return columns, id_col

_PANDAS_DTYPE_MAP = {
    'UInt8': 'uint8', 'UInt16': 'uint16', 'UInt32': 'uint32',
    'Int8': 'int8', 'Int16': 'int16', 'Int32': 'int32', 'Int64': 'int64',
    'Float32': 'float32', 'Float64': 'float64',
}


def _melt_and_insert_chunk(client, chunk_df, id_col, feature_cols, version, col_types=None):
    """Melt a wide-format chunk to long format and insert into typed meta tables.

    Args:
        col_types: dict mapping feature_name -> data_type. If None, all go to String table.
    """
    chunk_df = chunk_df.copy()
    chunk_df[id_col] = chunk_df[id_col].astype(str)

    if col_types is None:
        col_types = {c: 'String' for c in feature_cols}

    # Group columns by their target data type
    type_groups = {}
    for col in feature_cols:
        dtype = col_types.get(col, 'String')
        type_groups.setdefault(dtype, []).append(col)

    total = 0
    for dtype, cols in type_groups.items():
        table = _get_table_for_type(dtype)

        df_long = chunk_df.melt(
            id_vars=[id_col],
            value_vars=cols,
            var_name='feature_name',
            value_name='value'
        )
        df_long = df_long.rename(columns={id_col: 'sample_id'})

        # Drop empty/NaN values
        df_long = df_long[df_long['value'].notna()]
        df_long['value'] = df_long['value'].astype(str)
        df_long['value'] = df_long['value'].str.strip()
        df_long = df_long[(df_long['value'] != '') & (df_long['value'].str.lower() != 'nan')]

        if df_long.empty:
            continue

        # Convert value column to target type
        if dtype in _PANDAS_DTYPE_MAP:
            # Numeric types: parse to float64 first, then safely downcast
            df_long['value'] = pd.to_numeric(df_long['value'], errors='coerce')
            df_long = df_long.dropna(subset=['value'])
            if not df_long.empty:
                target_pd = _PANDAS_DTYPE_MAP[dtype]
                if 'int' in target_pd or 'uint' in target_pd:
                    # Round first to avoid float→int precision issues
                    df_long['value'] = df_long['value'].round(0).astype('int64').astype(target_pd)
                else:
                    df_long['value'] = df_long['value'].astype(target_pd)
        elif dtype == 'Bool':
            bool_map = {'true': 1, 'false': 0, '1': 1, '0': 0, 'yes': 1, 'no': 0}
            df_long['value'] = df_long['value'].str.lower().map(bool_map)
            df_long = df_long.dropna(subset=['value'])
            if not df_long.empty:
                df_long['value'] = df_long['value'].astype('uint8')
        elif dtype == 'Date':
            df_long['value'] = pd.to_datetime(df_long['value'], errors='coerce').dt.date
            df_long = df_long.dropna(subset=['value'])
        elif dtype == 'DateTime':
            df_long['value'] = pd.to_datetime(df_long['value'], errors='coerce')
            df_long = df_long.dropna(subset=['value'])
        # String: keep as-is

        df_long['version'] = version

        if not df_long.empty:
            client.insert_df(f'{DATABASE_NAME}.{table}', df_long[['sample_id', 'feature_name', 'value', 'version']])
            total += len(df_long)

    return total


def _count_file_lines(filepath):
    """Fast line count (excluding header)."""
    with open(filepath, 'rb') as f:
        count = sum(1 for _ in f) - 1
    return max(count, 0)


def upload_meta_tsv(client, filepath_or_df, feature_type=None, description=None,
                    data_type=None, col_types=None, include_columns=None,
                    rename_columns=None, progress_token=None):
    """Upload a meta TSV file into typed meta tables in long format.
    
    The first column ('filename') is the sample identifier.
    All other columns are melted into (sample_id, feature_name, value) rows.
    
    When a file path is given, reads in streaming chunks of 1000 rows to keep
    memory usage bounded.  When a DataFrame is passed, processes it directly.
    
    Args:
        feature_type: 'core' or 'meta' — applied to new features in meta_features.
        description: Free text describing how these features were calculated.
        data_type: Override data type for all columns. None or 'auto' for auto-detect.
        col_types: Dict mapping original_feature_name -> data_type.  Takes
                   precedence over *data_type* for any column present in the dict.
        include_columns: Optional list/set of column names (original) to import.
        rename_columns: Optional dict mapping original_name -> new_name.  Applied
                        before insertion so features are stored under the new name.
        progress_token: Optional string token for progress tracking via
                        get_upload_progress().
    """
    ensure_typed_tables(client)
    version = int(time.time())
    total = 0
    rows_done = 0

    include_set = set(include_columns) if include_columns else None
    rename_map = rename_columns or {}

    def _remap(name):
        """Return the (potentially renamed) feature name."""
        return rename_map.get(name, name)

    def _resolve_col_types(feature_cols, sample_df=None):
        """Build final col_types dict keyed by *renamed* feature names."""
        resolved = {}
        for c in feature_cols:
            new_c = _remap(c)
            if col_types and c in col_types:
                resolved[new_c] = col_types[c]
            elif data_type and data_type != 'auto':
                resolved[new_c] = data_type
            elif sample_df is not None:
                resolved[new_c] = _detect_column_type(sample_df[c])
            else:
                resolved[new_c] = 'String'
        return resolved

    def _report(current, total_rows, status='running'):
        if progress_token:
            _set_upload_progress(progress_token, current, total_rows, total, status)

    if isinstance(filepath_or_df, pd.DataFrame):
        df = filepath_or_df
        id_col = df.columns[0]
        feature_cols = [c for c in df.columns if c != id_col]
        if include_set:
            feature_cols = [c for c in feature_cols if c in include_set]
        resolved = _resolve_col_types(feature_cols, df)
        # Rename columns in the dataframe
        df_renamed = df.rename(columns={c: _remap(c) for c in feature_cols})
        renamed_features = [_remap(c) for c in feature_cols]
        total_rows = len(df)
        _report(0, total_rows)
        total = _melt_and_insert_chunk(client, df_renamed, id_col,
                                        renamed_features, version, resolved)
        _report(total_rows, total_rows, 'done')
    else:
        # Count total rows for progress
        total_rows = _count_file_lines(filepath_or_df)
        _report(0, total_rows)

        reader = pd.read_csv(filepath_or_df, sep='\t', chunksize=1000)
        id_col = None
        feature_cols = None
        resolved = None
        renamed_features = None
        for chunk in reader:
            if id_col is None:
                id_col = chunk.columns[0]
                feature_cols = [c for c in chunk.columns if c != id_col]
                if include_set:
                    feature_cols = [c for c in feature_cols if c in include_set]
                resolved = _resolve_col_types(feature_cols, chunk)
                renamed_features = [_remap(c) for c in feature_cols]
            chunk_renamed = chunk.rename(columns={c: _remap(c) for c in feature_cols})
            total += _melt_and_insert_chunk(client, chunk_renamed, id_col,
                                             renamed_features, version, resolved)
            rows_done += len(chunk)
            _report(rows_done, total_rows)

    # Signal "updating features" phase
    _report(total_rows, total_rows, 'updating_features')

    # Update meta features after upload
    try:
        update_meta_features(client, feature_type=feature_type, description=description,
                             imported_features=renamed_features)
    except Exception:
        pass  # Don't fail upload if features computation fails

    _report(total_rows, total_rows, 'done')
    
    return total


def delete_empty_meta_values(client):
    """Remove rows from meta where value is empty string.

    ClickHouse doesn't have UPDATE-in-place; this uses a mutation.
    """
    client.command(
        f"ALTER TABLE {DATABASE_NAME}.{META_TABLE} DELETE WHERE (value = '') OR (lower(value) = 'nan')"
    )


def get_sample_count(client):
    """Return the number of distinct samples across all typed meta tables."""
    try:
        # Build a UNION ALL across all typed tables to count distinct samples
        subqueries = []
        for table_name in set(TYPED_META_TABLES.values()):
            subqueries.append(f'SELECT DISTINCT sample_id FROM {DATABASE_NAME}.{table_name}')
        union_query = ' UNION ALL '.join(subqueries)
        result = client.query(f'SELECT count(DISTINCT sample_id) FROM ({union_query})')
        return result.result_rows[0][0]
    except Exception:
        return 0


def drop_tables(client):
    """Drop all meta tables (typed + features) for a clean re-upload."""
    for table_name in set(TYPED_META_TABLES.values()):
        client.command(f'DROP TABLE IF EXISTS {DATABASE_NAME}.{table_name}')
    client.command(f'DROP TABLE IF EXISTS {DATABASE_NAME}.{META_FEATURES_TABLE}')


def upload_tsv_from_bytes(client, content_bytes, feature_type=None, description=None,
                          data_type=None, col_types=None, include_columns=None,
                          rename_columns=None, progress_token=None):
    """Upload a TSV from raw bytes (for Dash upload component).
    
    Writes bytes to a temporary file and streams via upload_meta_tsv
    to keep memory usage bounded for large uploads.
    """
    import tempfile as _tempfile
    raw = content_bytes
    if raw and raw[:2] == b'\x1f\x8b':
        raw = gzip.decompress(raw)
    tmp = _tempfile.NamedTemporaryFile(prefix='tsv_bytes_', suffix='.tsv', delete=False)
    tmp_path = tmp.name
    try:
        tmp.write(raw)
        tmp.close()
        return upload_meta_tsv(client, tmp_path, feature_type=feature_type,
                               description=description, data_type=data_type,
                               col_types=col_types, include_columns=include_columns,
                               rename_columns=rename_columns,
                               progress_token=progress_token)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def update_meta_features(client, feature_type=None, description=None,
                         imported_features=None):
    """Update the meta_features table with current feature names, data types,
    and mean/std for numeric features from all typed meta tables.
    
    Preserves existing feature_type and description values. If feature_type or
    description are provided, they are applied to *new* features and to any
    features listed in *imported_features*.  Existing features whose stored
    type is 'meta' but match ``_is_core_feature()`` patterns are auto-corrected
    to 'core'.

    Args:
        imported_features: Optional set/list of feature names that were just
            imported.  ``feature_type`` and ``description`` are applied to
            these even if they already existed in meta_features.
    """
    imported_set = set(imported_features) if imported_features else set()
    ensure_meta_features_table(client)
    
    # 1. Read existing feature_type / description so we can preserve them.
    try:
        existing_df = client.query_df(
            f"SELECT feature_name, feature_type, description "
            f"FROM {DATABASE_NAME}.{META_FEATURES_TABLE}"
        )
        if existing_df is not None and not existing_df.empty:
            existing_map = existing_df.set_index('feature_name')[['feature_type', 'description']].to_dict('index')
        else:
            existing_map = {}
    except Exception:
        existing_map = {}
    
    # 2. Clear existing entries
    client.command(f'TRUNCATE TABLE {DATABASE_NAME}.{META_FEATURES_TABLE}')
    
    # 3. Collect features and stats from all typed tables.
    _NUMERIC_TYPES = {'UInt8', 'UInt16', 'UInt32', 'Int8', 'Int16', 'Int32',
                      'Int64', 'Float32', 'Float64'}
    all_features = []

    for dtype, table_name in TYPED_META_TABLES.items():
        try:
            if dtype in _NUMERIC_TYPES:
                result = client.query(f'''
                    SELECT
                        feature_name,
                        avg(value) as mean_val,
                        stddevSamp(value) as std_val
                    FROM {DATABASE_NAME}.{table_name} FINAL
                    GROUP BY feature_name
                    ORDER BY feature_name
                ''')
                for row in (result.result_rows or []):
                    all_features.append({
                        'feature_name': row[0],
                        'data_type': dtype,
                        'mean': row[1],
                        'std': row[2],
                    })
            else:
                # Bool, Date, DateTime, String — no mean/std
                result = client.query(f'''
                    SELECT feature_name
                    FROM {DATABASE_NAME}.{table_name} FINAL
                    GROUP BY feature_name
                    ORDER BY feature_name
                ''')
                for row in (result.result_rows or []):
                    all_features.append({
                        'feature_name': row[0],
                        'data_type': dtype,
                        'mean': None,
                        'std': None,
                    })
        except Exception as e:
            log.warning('Failed to query table %s for feature stats: %s', table_name, e)

    if all_features:
        df_features = pd.DataFrame(all_features)
        df_features['mean'] = pd.to_numeric(df_features['mean'], errors='coerce')
        df_features['std'] = pd.to_numeric(df_features['std'], errors='coerce')
        
        # 5. Merge back preserved feature_type / description.
        ft_list = []
        desc_list = []
        for fname in df_features['feature_name']:
            # Explicitly imported features: use the caller-supplied type
            if fname in imported_set and feature_type is not None:
                ft_list.append(feature_type)
                desc_list.append(description or (
                    existing_map[fname].get('description', '') if fname in existing_map else ''))
            elif fname in existing_map:
                existing_ft = existing_map[fname].get('feature_type', 'meta')
                # Auto-correct: existing 'meta' that match core patterns
                if existing_ft == 'meta' and _is_core_feature(fname):
                    existing_ft = 'core'
                ft_list.append(existing_ft)
                desc_list.append(existing_map[fname].get('description', ''))
            else:
                if feature_type is not None:
                    ft_list.append(feature_type)
                else:
                    ft_list.append('core' if _is_core_feature(fname) else 'meta')
                desc_list.append(description or '')
        
        df_features['feature_type'] = ft_list
        df_features['description'] = desc_list
        
        client.insert_df(f'{DATABASE_NAME}.{META_FEATURES_TABLE}',
                         df_features[['feature_name', 'data_type', 'mean', 'std', 'feature_type', 'description']])
    
    return len(all_features)


def get_meta_features(client, feature_type='meta'):
    """Get list of available feature names, optionally filtered by feature_type.
    
    Args:
        client: ClickHouse client
        feature_type: 'meta', 'core', or None for all features
    """
    try:
        where = ''
        if feature_type is not None:
            where = f" WHERE feature_type = '{feature_type}'"
        result = client.query(
            f'SELECT feature_name FROM {DATABASE_NAME}.{META_FEATURES_TABLE}{where} ORDER BY feature_name'
        )
        return [row[0] for row in result.result_rows]
    except Exception:
        return []


def get_feature_stats(client):
    """Get feature statistics (mean, std) as a DataFrame from meta_features."""
    try:
        return client.query_df(
            f'SELECT feature_name, mean, std FROM {DATABASE_NAME}.{META_FEATURES_TABLE} '
            f'WHERE mean IS NOT NULL ORDER BY feature_name'
        )
    except Exception:
        return pd.DataFrame()


def get_feature_groups(client):
    """Classify features from meta_features into display groups by name pattern.
    
    Returns dict with keys 'fszd', 'csm', '5p', 'bincount', each containing:
        'col_names': list of feature names
        'x_vals': list of display x-axis values
    The 'bincount' group additionally has 'chromosomes', 'starts', 'ends' lists.
    """
    features = get_meta_features(client, feature_type=None)
    
    fszd_cols, fszd_x = [], []
    csm_cols, csm_x = [], []
    p5_cols, p5_x = [], []
    bin_cols, bin_chroms, bin_starts, bin_ends = [], [], [], []
    
    for f in features:
        if str(f).endswith('-csm'):
            csm_cols.append(f)
            csm_x.append(str(f)[:-4])
        elif str(f).endswith('-5p'):
            p5_cols.append(f)
            p5_x.append(str(f)[:-3])
        elif BINCOUNT_PATTERN.match(str(f)):
            parts = str(f).split('_')
            if len(parts) >= 3:
                chrom = parts[0]
                try:
                    start = int(parts[1])
                    end = int(parts[2])
                    bin_cols.append(f)
                    bin_chroms.append(chrom)
                    bin_starts.append(start)
                    bin_ends.append(end)
                except (TypeError, ValueError):
                    pass
        else:
            try:
                ci = int(f)
                if 1 <= ci <= 1000:
                    fszd_cols.append(f)
                    fszd_x.append(ci)
            except (TypeError, ValueError):
                pass
    
    return {
        'fszd': {'col_names': fszd_cols, 'x_vals': fszd_x},
        'csm': {'col_names': csm_cols, 'x_vals': csm_x},
        '5p': {'col_names': p5_cols, 'x_vals': p5_x},
        'bincount': {
            'col_names': bin_cols,
            'chromosomes': bin_chroms,
            'starts': bin_starts,
            'ends': bin_ends,
        },
    }


def load_meta_feature_series(client, feature_name):
    """Load one meta feature for all samples as a Series indexed by sample_id.

    Routes to the correct typed table based on data_type stored in meta_features.
    """
    if feature_name is None:
        return pd.Series(dtype=object)

    feature_name = str(feature_name)
    data_type = _get_feature_data_type(client, feature_name)
    table = _get_table_for_type(data_type)

    query = (
        f"SELECT sample_id, argMax(value, version) AS value FROM {DATABASE_NAME}.{table} "
        f"WHERE feature_name = '{feature_name}' GROUP BY sample_id"
    )
    df = client.query_df(query)

    if df is None or df.empty:
        return pd.Series(dtype=object, name=feature_name)

    if 'sample_id' in df.columns and df['sample_id'].duplicated().any():
        df = df.drop_duplicates(subset=['sample_id'], keep='last')

    s = df.set_index('sample_id')['value']
    s.index = s.index.astype(str)
    s.name = feature_name

    return s


def apply_filters_pandas(df_wide, filters):
    """Apply filter specs to a wide DataFrame and return a boolean mask."""
    if df_wide is None or df_wide.empty:
        return pd.Series([], dtype=bool, index=getattr(df_wide, 'index', None))
    if not filters:
        return pd.Series(True, index=df_wide.index)

    mask = pd.Series(True, index=df_wide.index)
    for f in filters:
        col = f.get('column') if isinstance(f, dict) else None
        op = f.get('op') if isinstance(f, dict) else None
        if not col or not op or col not in df_wide.columns:
            mask &= False
            continue

        s = df_wide[col]
        if op == 'range':
            lo = f.get('lo')
            hi = f.get('hi')
            s_num = pd.to_numeric(s, errors='coerce')
            cond = pd.Series(True, index=df_wide.index)
            if lo is not None:
                cond &= s_num >= float(lo)
            if hi is not None:
                cond &= s_num <= float(hi)
            cond &= s_num.notna()
            mask &= cond
        elif op in ('==', '!='):
            expr = f.get('expr')
            if expr is None or not str(expr).strip():
                mask &= False
                continue
            try:
                re.compile(str(expr))
            except re.error:
                mask &= False
                continue
            m = s.astype(str).str.contains(str(expr), regex=True, na=False)
            mask &= m if op == '==' else ~m
        else:
            mask &= False

    return mask


def lazy_load_meta_samples(client, sample_ids=None, features=None):
    """Lazily load meta data for specific samples and features."""
    conditions = []
    
    log.info('lazy_load_meta_samples (n_samples=%s, n_features=%s)',
             len(sample_ids) if sample_ids is not None else None,
             len(features) if features is not None else None)

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
    """Load a chunk of meta data for specific samples and features.

    Routes queries to the correct typed tables based on feature data_types,
    then combines results into a single wide DataFrame.
    """
    sample_cond = ''
    if sample_ids:
        sample_ids_str = "', '".join(str(sid) for sid in sample_ids)
        sample_cond = f"sample_id IN ('{sample_ids_str}')"

    # Group features by their typed table
    if features:
        feature_types = _get_feature_data_types_bulk(client, features)
        table_features = {}
        for f in features:
            dtype = feature_types.get(f, 'String')
            table = _get_table_for_type(dtype)
            table_features.setdefault(table, []).append(f)
    else:
        table_features = {table: None for table in set(TYPED_META_TABLES.values())}

    all_wide = []

    for table, feats in table_features.items():
        conditions = []
        if sample_cond:
            conditions.append(sample_cond)
        if feats:
            features_str = "', '".join(str(f) for f in feats)
            conditions.append(f"feature_name IN ('{features_str}')")

        where_clause = f" WHERE {' AND '.join(conditions)}" if conditions else ""

        query = f'SELECT sample_id, feature_name, toString(value) as value FROM {DATABASE_NAME}.{table}{where_clause}'
        result = client.query_df(query)

        if result.empty:
            continue

        df_part = result.pivot_table(
            index='sample_id',
            columns='feature_name',
            values='value',
            aggfunc='first'
        )
        df_part.columns.name = None
        df_part.index = df_part.index.astype(str)

        # Convert numeric columns from typed tables back to native types
        if table != META_TABLE:
            for col in df_part.columns:
                converted = pd.to_numeric(df_part[col], errors='coerce')
                if converted.notna().any():
                    df_part[col] = converted

        all_wide.append(df_part)

    if not all_wide:
        return pd.DataFrame()

    df_wide = pd.concat(all_wide, axis=1)
    df_wide.index.name = 'filename'

    return df_wide


def insert_alignment_results(client, filename, features, feature_names, umap1, umap2):
    """Insert alignment upload results into typed meta tables.
    
    Numeric features and UMAP coordinates go to meta_float64.
    The filename goes to the String meta table.
    
    Args:
        client: ClickHouse client
        filename: Original filename (used as base for sample_id)
        features: Feature vector (numpy array)
        feature_names: List of feature names matching the features vector
        umap1: UMAP coordinate 1
        umap2: UMAP coordinate 2
    """
    import uuid
    
    ensure_typed_tables(client)
    
    # Generate unique sample_id from filename and UUID
    base_name = os.path.splitext(os.path.basename(filename))[0]
    unique_id = f"{base_name}_{uuid.uuid4().hex[:8]}"
    
    version = int(time.time())
    
    # All computed features + UMAP coords are Float64
    float_rows = []
    for i, fname in enumerate(feature_names):
        val = float(features[0, i]) if hasattr(features, 'shape') and len(features.shape) > 1 else float(features[i])
        float_rows.append({
            'sample_id': unique_id,
            'feature_name': str(fname),
            'value': val,
            'version': version
        })
    float_rows.append({'sample_id': unique_id, 'feature_name': 'umap1', 'value': float(umap1), 'version': version})
    float_rows.append({'sample_id': unique_id, 'feature_name': 'umap2', 'value': float(umap2), 'version': version})
    
    df_float = pd.DataFrame(float_rows)
    client.insert_df(f'{DATABASE_NAME}.{TYPED_META_TABLES["Float64"]}',
                     df_float[['sample_id', 'feature_name', 'value', 'version']])
    
    # Filename is a String feature
    df_str = pd.DataFrame([{
        'sample_id': unique_id,
        'feature_name': 'filename',
        'value': filename,
        'version': version
    }])
    client.insert_df(f'{DATABASE_NAME}.{META_TABLE}', df_str[['sample_id', 'feature_name', 'value', 'version']])
    
    # Update meta features — all alignment features are core
    all_imported = [str(f) for f in feature_names] + ['umap1', 'umap2', 'filename']
    try:
        update_meta_features(client, feature_type='core', imported_features=all_imported)
    except Exception:
        pass
    
    return unique_id


def upload_plink_bed(client, bed_path, bim_path, fam_path, progress_token=None):
    """Upload PLINK binary genotype data into meta_uint8.

    Each variant becomes a feature (feature_name from BIM variant_id or chr:pos).
    Each sample (from FAM IID) gets a UInt8 genotype value per variant:
        0 = homozygous A1, 1 = heterozygous, 2 = homozygous A2, 255 = missing.

    Args:
        client: ClickHouse client
        bed_path: Path to .bed file
        bim_path: Path to .bim file
        fam_path: Path to .fam file
        progress_token: Optional token for progress tracking

    Returns:
        Total number of value rows inserted.
    """
    ensure_typed_tables(client)

    def _report(current, total, rows_ins, status='running'):
        if progress_token:
            _set_upload_progress(progress_token, current, total, rows_ins, status)

    # ── Parse FAM (sample IDs) ──
    fam_df = pd.read_csv(fam_path, sep=r'\s+', header=None,
                         names=['fid', 'iid', 'father', 'mother', 'sex', 'phenotype'],
                         dtype=str)
    sample_ids = fam_df['iid'].astype(str).tolist()
    n_samples = len(sample_ids)

    # ── Parse BIM (variant names) ──
    bim_df = pd.read_csv(bim_path, sep='\t', header=None,
                         names=['chrom', 'variant_id', 'cm', 'pos', 'a1', 'a2'],
                         dtype=str)
    variant_names = []
    for _, row in bim_df.iterrows():
        vid = row['variant_id']
        if vid == '.' or not vid.strip():
            variant_names.append(f"{row['chrom']}:{row['pos']}")
        else:
            variant_names.append(vid)
    n_variants = len(variant_names)

    log.info('PLINK upload: %d samples, %d variants from %s', n_samples, n_variants, bed_path)
    _report(0, n_variants, 0)

    # ── Validate BED magic bytes ──
    with open(bed_path, 'rb') as f:
        magic = f.read(3)
    if magic[:2] != b'\x6c\x1b':
        raise ValueError('Invalid PLINK BED file: bad magic bytes.')
    if magic[2:3] != b'\x01':
        raise ValueError('PLINK BED file is in individual-major mode; only SNP-major (0x01) is supported.')

    # ── Read BED genotypes in variant chunks ──
    # In SNP-major mode, each variant uses ceil(n_samples / 4) bytes.
    bytes_per_snp = (n_samples + 3) // 4
    version = int(time.time())
    total_inserted = 0
    chunk_size = 5000  # variants per chunk

    # 2-bit genotype lookup: PLINK encodes 00=hom A1(0), 01=missing, 10=het(1), 11=hom A2(2)
    _GENO_LOOKUP = np.array([0, 255, 1, 2], dtype=np.uint8)

    table_name = TYPED_META_TABLES['UInt8']

    with open(bed_path, 'rb') as bed_fh:
        bed_fh.seek(3)  # skip magic header

        for chunk_start in range(0, n_variants, chunk_size):
            chunk_end = min(chunk_start + chunk_size, n_variants)
            chunk_n = chunk_end - chunk_start

            # Read raw bytes for this chunk of variants
            raw = bed_fh.read(bytes_per_snp * chunk_n)
            if len(raw) < bytes_per_snp * chunk_n:
                log.warning('PLINK BED: unexpected EOF at variant %d', chunk_start + len(raw) // bytes_per_snp)
                chunk_n = len(raw) // bytes_per_snp
                if chunk_n == 0:
                    break

            raw_arr = np.frombuffer(raw, dtype=np.uint8).reshape(chunk_n, bytes_per_snp)

            # Decode 2-bit packed genotypes → (chunk_n, n_samples) uint8 array
            # Each byte holds 4 samples: bits 0-1 = sample 0, bits 2-3 = sample 1, etc.
            geno = np.zeros((chunk_n, n_samples), dtype=np.uint8)
            for bit_idx in range(4):
                sample_start = bit_idx
                two_bits = (raw_arr >> (bit_idx * 2)) & 0x03
                # Map the columns to the right sample indices
                indices = np.arange(sample_start, n_samples, 4)
                if len(indices) > 0:
                    geno[:, indices] = _GENO_LOOKUP[two_bits[:, :len(indices)]]

            # Build long-format rows for insertion
            rows = []
            chunk_variant_names = variant_names[chunk_start:chunk_start + chunk_n]
            for vi in range(chunk_n):
                for si in range(n_samples):
                    val = int(geno[vi, si])
                    if val == 255:
                        continue  # skip missing genotypes
                    rows.append({
                        'sample_id': sample_ids[si],
                        'feature_name': chunk_variant_names[vi],
                        'value': val,
                        'version': version,
                    })

            if rows:
                df_chunk = pd.DataFrame(rows)
                df_chunk['value'] = df_chunk['value'].astype('uint8')
                client.insert_df(
                    f'{DATABASE_NAME}.{table_name}',
                    df_chunk[['sample_id', 'feature_name', 'value', 'version']]
                )
                total_inserted += len(rows)

            _report(chunk_end, n_variants, total_inserted)

    # ── Update meta_features ──
    _report(n_variants, n_variants, total_inserted, 'updating_features')
    try:
        update_meta_features(client, feature_type='core', imported_features=variant_names)
    except Exception:
        log.warning('update_meta_features failed after PLINK upload (non-fatal)')

    _report(n_variants, n_variants, total_inserted, 'done')
    log.info('PLINK upload complete: %d value rows inserted for %d variants x %d samples',
             total_inserted, n_variants, n_samples)
    return total_inserted


def drop_features(client, feature_names):
    """Delete specific features from all typed meta tables and meta_features.

    Args:
        client: ClickHouse client
        feature_names: List of feature names to delete.

    Returns:
        Number of features dropped from meta_features.
    """
    if not feature_names:
        return 0

    # Look up which typed tables contain these features
    feature_types = _get_feature_data_types_bulk(client, feature_names)

    # Group by table
    table_features = {}
    for fname in feature_names:
        dtype = feature_types.get(fname, 'String')
        table = _get_table_for_type(dtype)
        table_features.setdefault(table, []).append(fname)

    # Delete from each typed table
    for table, feats in table_features.items():
        chunk_size = 5000
        for i in range(0, len(feats), chunk_size):
            chunk = feats[i:i + chunk_size]
            names_str = "', '".join(str(f) for f in chunk)
            client.command(
                f"ALTER TABLE {DATABASE_NAME}.{table} "
                f"DELETE WHERE feature_name IN ('{names_str}')"
            )

    # Delete from meta_features
    chunk_size = 5000
    for i in range(0, len(feature_names), chunk_size):
        chunk = feature_names[i:i + chunk_size]
        names_str = "', '".join(str(f) for f in chunk)
        client.command(
            f"ALTER TABLE {DATABASE_NAME}.{META_FEATURES_TABLE} "
            f"DELETE WHERE feature_name IN ('{names_str}')"
        )

    log.info('Dropped %d features from database', len(feature_names))
    return len(feature_names)
