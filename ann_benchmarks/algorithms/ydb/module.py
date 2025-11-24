"""
This module supports connecting to a YDB instance and performing vector
indexing and search.

For authentication the module uses the standard YDB environment variables:
https://ydb.tech/docs/en/recipes/ydb-sdk/auth-env
"""

import concurrent.futures
import datetime
import json
import math
import multiprocessing as mp
import numpy as np
import os
import random
import shutil
import subprocess
import sys
import time
import traceback
import uuid
import ydb

from typing import Dict, Any, Optional

from urllib.parse import urlparse, parse_qs

from ..base.module import BaseANN


TABLE_NAME = "items"
INDEX_BASE_NAME = "idx_vector_items"

MIN_SHARDS = 100

BATCH_SIZE = 1000

MAX_RETRIES = 100
BACKOFF_MILLIS = 10
BACKOFF_CEILING = 5

DEFAULT_MEANS_TOP_SIZE = 3

MAX_BATCH_QUERY_THREADS = 32


def query_impl(pool, database, use_stale_reads, index_name, metric, means_top_size, v, n):
    start = time.perf_counter()
    binary_embedding = float_embedding_to_binary(v)

    if metric == "angular":
        distance_func = "Knn::CosineDistance"
    elif metric == "euclidean":
        distance_func = "Knn::EuclideanDistance"
    else:
        print(f"Unsupported metric: {metric}", file=sys.stderr)
        sys.exit(1)

    query = f"""
        PRAGMA TablePathPrefix("{database}");

        pragma ydb.KMeansTreeSearchTopSize = "{means_top_size}";

        DECLARE $embedding as String;

        SELECT id, {distance_func}(embedding, $embedding) as dist
        FROM `{TABLE_NAME}`
        VIEW `{index_name}`
        ORDER BY dist ASC
        LIMIT {n};
    """

    params = {
        "$embedding": (float_embedding_to_binary(v), ydb.PrimitiveType.String),
    }

    def ydb_to_primitive_types(
        ydb_row: dict[str, Any]
    ) -> dict[str, Any]:
        prepared_row = {}
        for key, value in ydb_row.items():
            if isinstance(value, (datetime.datetime, datetime.date)):
                value = value.replace(tzinfo=datetime.timezone.utc)
                # convert to microseconds
                prepared_row[key] = int(value.timestamp() * 1e6)
            elif isinstance(value, uuid.UUID):
                prepared_row[key] = str(value)
            elif isinstance(value, datetime.timedelta):
                prepared_row[key] = value.microseconds
            else:
                prepared_row[key] = value
        return prepared_row

    def ydb_to_primitive_types_iter(stream):
        yield from map(ydb_to_primitive_types, stream)

    def iter_ydb_rows(stream):
        for stream_part in stream:
            if isinstance(stream_part, ydb.ScanQueryResult):
                result_set = stream_part.result_set
            else:
                result_set = stream_part
            for row in ydb_to_primitive_types_iter(result_set.rows):
                yield row

    def callee(session: ydb.QuerySession):
        response = session.transaction(ydb.QueryStaleReadOnly()).execute(query, params, commit_tx=True)
        return iter_ydb_rows(response)

    try:
        if use_stale_reads:
            rows = pool.retry_operation_sync(callee, ydb.RetrySettings(max_retries=10, idempotent=True))
            ids = [row["id"] for row in rows]
            elapsed = time.perf_counter() - start
            return ids, elapsed
        else:
            result_sets = pool.execute_with_retries(query, params)
            rows = result_sets[0].rows
            ids = [row.id for row in rows]
            elapsed = time.perf_counter() - start
        return ids, elapsed
    except Exception as e:
        print("Query failed: ", e)
        traceback.print_exc()
        raise e


def proc_execute_sub_batch(database,
                           metric,
                           use_stale_reads,
                           base_index_name,
                           index_count,
                           means_top_size,
                           X_chunk: np.ndarray,  # THIS IS COPIED to the child
                           n: int):
    """
    Executes queries for the rows in X_chunk and returns (results_sub, latencies_sub).
    """

    driver_config = ydb.DriverConfig.default_from_connection_string(
        os.environ["YDB_CONNECTION_STRING"],
        credentials=ydb.credentials_from_env_variables(),
        use_all_nodes=True
    )

    driver = ydb.Driver(driver_config=driver_config)

    # Wait for the driver to become active
    driver.wait(timeout=5)

    pool = ydb.QuerySessionPool(driver)

    results_sub   = np.empty((len(X_chunk), n), dtype=int)
    latencies_sub = np.empty(len(X_chunk), dtype=float)

    for j, v in enumerate(X_chunk):
        use_index_name = base_index_name
        if index_count > 1:
            idx = random.randrange(index_count) + 1
            if idx > 1:
                use_index_name = base_index_name + f"_i{idx}"

        t0 = time.perf_counter()
        result = query_impl(
            pool, database, use_stale_reads, use_index_name, metric, means_top_size, v, n)[0]
        results_sub[j, :] = result
        latencies_sub[j] = time.perf_counter() - t0

    return results_sub, latencies_sub


def get_backoff_wait_ms(retry_count):
    """Calculate backoff wait time in milliseconds with exponential backoff and jitter"""
    wait_time_ceiling_ms = (1 << BACKOFF_CEILING) * BACKOFF_MILLIS
    wait_time_ms = BACKOFF_MILLIS

    for i in range(retry_count):
        wait_time_ms = min(wait_time_ms * 2, wait_time_ceiling_ms)

    return wait_time_ms + random.randint(0, 99)


def check_ydb_cli_available():
    """Check if ydb CLI binary is available"""
    if shutil.which("ydb") is None:
        raise RuntimeError("ydb CLI binary not found in PATH. Please install YDB CLI: https://ydb.tech/docs/en/reference/ydb-cli/install")

    # Verify it works by running a simple command
    try:
        result = subprocess.run(["ydb", "--help"], capture_output=True, text=True, timeout=10)
        if result.returncode != 0:
            raise RuntimeError(f"ydb CLI binary found but not working properly. Exit code: {result.returncode}")
    except subprocess.TimeoutExpired:
        raise RuntimeError("ydb CLI binary found but timed out during verification")
    except FileNotFoundError:
        raise RuntimeError("ydb CLI binary not found in PATH. Please install YDB CLI: https://ydb.tech/docs/en/reference/ydb-cli/install")


def get_min_max_partitions(num_dimensions, n):
    approximate_row_size_bytes = 32 + 4 * num_dimensions
    approximate_total_size = approximate_row_size_bytes * n

    # suppose 2 GB shards
    shard_count_by_size = (approximate_total_size >> 31)

    min_partitions = max(shard_count_by_size, MIN_SHARDS)
    max_partitions = min_partitions * 2

    return min_partitions, max_partitions


def drop_create_table(pool, table_name, num_dimensions, n):
    """Drop and create YDB table"""

    try:
        pool.execute_with_retries(f"DROP TABLE `{table_name}`")
    except:
        pass

    min_partitions, max_partitions = get_min_max_partitions(num_dimensions, n)

    rows_per_shard = n // min_partitions
    cur_row = rows_per_shard
    split_keys = []
    while cur_row < n:
        split_keys.append(str(cur_row))
        cur_row += rows_per_shard

    if len(split_keys) == 0:
        split_keys_str = ""
    else:
        split_keys = [str(int(x)) for x in split_keys]
        split_keys_str = ",PARTITION_AT_KEYS = (" + ",".join(split_keys) + ")"

    print(f"Creating table for {n} vectors of {num_dimensions} dimensions with {min_partitions} shards")

    query = f"""
        CREATE TABLE `{table_name}` (
            `id` Uint64,
            `embedding` String,
            PRIMARY KEY (`id`)
        )
        WITH (
            AUTO_PARTITIONING_BY_LOAD = DISABLED,
            AUTO_PARTITIONING_MIN_PARTITIONS_COUNT = {min_partitions},
            AUTO_PARTITIONING_MAX_PARTITIONS_COUNT = {max_partitions}
            {split_keys_str}
        );
    """

    try:
        pool.execute_with_retries(query)
    except Exception as e:
        print(f"Failed to create table `{table_name}`: ", e)
        raise e

    print(f"Table '{table_name}' created")


def set_partionining_policy(pool, table_name, index_name, num_dimensions, n):
    """Enables split by load"""

    min_partitions, max_partitions = get_min_max_partitions(num_dimensions, n)

    try:
        pool.execute_with_retries(f"""
            ALTER TABLE `{table_name}` SET (
                AUTO_PARTITIONING_BY_LOAD = ENABLED,
                AUTO_PARTITIONING_MIN_PARTITIONS_COUNT = {min_partitions},
                AUTO_PARTITIONING_MAX_PARTITIONS_COUNT = {max_partitions}
            );
        """)
        print(f"Split by load enabled for table '{table_name}'")

        # TODO: move 30 to constants

        index_table1 = f"{table_name}/{index_name}/indexImplLevelTable"
        pool.execute_with_retries(f"""
            ALTER TABLE `{index_table1}` SET (
                AUTO_PARTITIONING_BY_LOAD = ENABLED,
                AUTO_PARTITIONING_BY_SIZE = ENABLED,
                AUTO_PARTITIONING_PARTITION_SIZE_MB = 10,
                AUTO_PARTITIONING_MIN_PARTITIONS_COUNT = {min_partitions},
                AUTO_PARTITIONING_MAX_PARTITIONS_COUNT = {max_partitions}
            );
        """)
        print(f"Split by load enabled for table '{index_table1}'")

        index_table2 = f"{table_name}/{index_name}/indexImplPostingTable"
        pool.execute_with_retries(f"""
            ALTER TABLE `{index_table2}` SET (
                AUTO_PARTITIONING_BY_SIZE = ENABLED,
                AUTO_PARTITIONING_BY_LOAD = ENABLED,
                AUTO_PARTITIONING_PARTITION_SIZE_MB = 256,
                AUTO_PARTITIONING_MIN_PARTITIONS_COUNT = {min_partitions},
                AUTO_PARTITIONING_MAX_PARTITIONS_COUNT = {max_partitions}
            );
        """)
        print(f"Split by load enabled for table '{table_name}'")
    except:
        pass


def build_index(pool, endpoint, database, table_name, index_name, metric, num_dimensions, levels, clusters):
    """Create and wait to be ready the vector index"""

    print(f"Create index '{index_name}' for table '{table_name}'")

    table_path = database + "/" + table_name

    if metric == "angular":
        distance = "cosine"
    elif metric == "euclidean":
        distance = "euclidean"
    else:
        print(f"Unsupported metric: {metric}", file=sys.stderr)
        sys.exit(1)

    query = f"""
        ALTER TABLE `{table_path}`
        ADD INDEX `{index_name}`
        GLOBAL USING vector_kmeans_tree
        ON (embedding) COVER (embedding)
        WITH (
            distance="{distance}",
            vector_type="float",
            vector_dimension={num_dimensions},
            levels={levels},
            clusters={clusters}
        );
    """

    index_future = pool.execute_with_retries_async(query)
    try:
        # we set timeout to 1 second, because we want to check here
        # that requests have started execution and then check state manually
        result = index_future.result(timeout=1)[0]
    except ydb.issues.DeadlineExceed:
        print("DeadlineExceed for {}, but will check state manually".format(index_name))
    except concurrent.futures.TimeoutError:
        pass
    except Exception as e:
        print("Failed to create index {}: {}".format(index_name, e), file=sys.stderr)
        sys.exit(1)

    wait_all_indices(endpoint, database, index_name)


def wait_all_indices(endpoint, database, index_name):
    print(f"Waiting for {index_name} to be ready...")

    # TODO: use SDK? I don't see that it currently supports this
    # TODO: since we use CLI, we have a strong issue with setting auth properly

    command = [
        "ydb",
        "--endpoint",
        endpoint,
        "--database",
        database,
        "operation",
        "list",
        "buildindex",
        "--format",
        "proto-json-base64",
    ]

    while True:
        for i in range(10):
            result = subprocess.run(' '.join(command), capture_output=True, text=True, shell=True, executable='/bin/bash')
            if result.returncode != 0:
                time.sleep(10)

        if result.returncode != 0:
            print("Error getting index status: {}".format(result.stderr), file=sys.stderr)
            sys.exit(1)

        output = result.stdout
        if output == "":
            print("Error getting index status: empty output", file=sys.stderr)
            sys.exit(1)

        output = json.loads(output)
        operations = output["operations"]

        bad_states = (
            "STATE_UNSPECIFIED",
            "STATE_CANCELLATION",
            "STATE_CANCELLED",
            "STATE_REJECTION",
            "STATE_REJECTED",
        )

        in_progress_states = (
            "STATE_UNSPECIFIED",
            "STATE_PREPARING",
            "STATE_TRANSFERING_DATA",
            "STATE_APPLYING",
        )

        all_ready = False
        for op in operations:
            if op["metadata"]["state"] in bad_states:
                print(f"Error creating indices: {operations}", file=sys.stderr)
                sys.exit(1)

            if op["metadata"]["state"] in in_progress_states:
                break

            if "ready" in op:
                if not op["ready"]:
                    break

            if op["metadata"]["state"] == "STATE_DONE":
                if "status" in op:
                    if op["status"] != "SUCCESS":
                        print("Error creating indices: {}".format(op), file=sys.stderr)
                        sys.exit(1)
        else:
            all_ready = True

        if all_ready:
            time.sleep(2) # hack, because we have a small issue with reporting OK
            break


def initialize_ydb_from_env():
    """Initialize YDB driver from env"""
    driver_config = ydb.DriverConfig.default_from_connection_string(
        os.environ["YDB_CONNECTION_STRING"],
        credentials=ydb.credentials_from_env_variables(),
        use_all_nodes=True
    )

    driver = ydb.Driver(driver_config=driver_config)

    # Wait for the driver to become active
    driver.wait(timeout=5)
    return driver


def float_embedding_to_binary(arr):
    # arr is a 1D numpy float32 array
    # Serialize all floats, append a single byte 1 at the end
    return arr.tobytes() + b'\x01'


def send_batch_to_ydb(table_client, full_table_path, embeddings):
    """Insert vectors into YDB table using BulkUpsert (most efficient)"""

    # Prepare all data for bulk upsert
    rows = []
    for vector_id, embedding_binary in embeddings:
        rows.append({
            'id': vector_id,
            'embedding': float_embedding_to_binary(embedding_binary)
        })

    column_types = ydb.BulkUpsertColumns() \
        .add_column('id', ydb.PrimitiveType.Uint64) \
        .add_column('embedding', ydb.PrimitiveType.String)

    for attempt in range(MAX_RETRIES):
        try:
            table_client.bulk_upsert(full_table_path, rows, column_types)
            break
        except ydb.issues.Overloaded as e:
            if attempt == MAX_RETRIES - 1:
                raise e

            delay_ms = get_backoff_wait_ms(attempt)
            delay_s = delay_ms / 1000.0
            print(f"YDB overloaded (attempt {attempt + 1}/{MAX_RETRIES}), retrying in {delay_ms}ms: {e}")
            time.sleep(delay_s)
        except Exception as e:
            # For non-overload errors, don't retry
            raise e


class YDBVector(BaseANN):
    def __init__(self, metric, method_param):
        # Check if ydb CLI is available before proceeding
        check_ydb_cli_available()

        self._batch_threads = None

        self._metric = metric
        if method_param is None:
            method_param = {}
        self._method_param = method_param

        levels = self._method_param['levels']
        clusters = self._method_param['clusters']

        self._index_count = self._method_param.get('index_count', 1)
        self._index_name = INDEX_BASE_NAME + f"_{metric}_{clusters}x{levels}"

        try:
            self._driver = initialize_ydb_from_env()
            self._pool = ydb.QuerySessionPool(self._driver)
        except Exception as e:
            print("Unable to connect to YDB: ", e)
            raise e

        # <protocol>://<hostname>:<port>/?database=/path/to/the/database
        connection_string = os.environ["YDB_CONNECTION_STRING"]

        # Parse connection string to extract endpoint and database
        parsed_url = urlparse(connection_string)
        self._endpoint = f"{parsed_url.scheme}://{parsed_url.netloc}"

        self._use_stale_reads = False
        if "YDB_STALE_READS" in os.environ:
            self._use_stale_reads = os.environ["YDB_STALE_READS"] == "1"

        # Extract database from query parameters
        query_params = parse_qs(parsed_url.query)
        if 'database' not in query_params:
            raise ValueError(f"Database parameter not found in connection string: {connection_string}")
        self._database = query_params['database'][0]

        self._full_table_path = self._database + "/" + TABLE_NAME

        self._means_top_size = DEFAULT_MEANS_TOP_SIZE

    def fit(self, X):
        num_dimensions = X.shape[1]

        drop_create_table(self._pool, TABLE_NAME, num_dimensions, len(X))

        print("copying data...")
        sys.stdout.flush()
        num_rows = 0
        insert_start_time_sec = time.time()

        vectors_batch = []
        for i, embedding in enumerate(X):
            vectors_batch.append((i, embedding,))
            num_rows += 1
            if len(vectors_batch) == BATCH_SIZE:
                send_batch_to_ydb(self._driver.table_client, self._full_table_path, vectors_batch)
                vectors_batch = []

        if len(vectors_batch) != 0:
            send_batch_to_ydb(self._driver.table_client, self._full_table_path, vectors_batch)
            vectors_batch = []

        insert_elapsed_time_sec = time.time() - insert_start_time_sec
        print("inserted {} rows into table in {:.3f} seconds".format(num_rows, insert_elapsed_time_sec))

        index_start_time_sec = time.time()
        print("building index...")

        for i in range(1, self._index_count + 1):
            index_name = self._index_name
            if i > 1:
                index_name += f"_i{i}"
            build_index(
                self._pool,
                self._endpoint,
                self._database,
                TABLE_NAME,
                index_name,
                self._metric,
                num_dimensions,
                self._method_param['levels'],
                self._method_param['clusters'])

        # we have a race between reporting index ready and having it actually ready
        print("Indices are ready")
        time.sleep(10)

        for i in range(1, self._index_count + 1):
            index_name = self._index_name
            if i > 1:
                index_name += f"_i{i}"
            set_partionining_policy(self._pool, TABLE_NAME, index_name, num_dimensions, len(X))

        index_elapsed_time_sec = time.time() - index_start_time_sec
        print("built index in {:.3f} seconds".format(index_elapsed_time_sec))


    def query(self, v, n):
        index_name = self._index_name
        if self._index_count > 1:
            idx = random.randrange(self._index_count) + 1
            if idx > 1:
                index_name = self._index_name + f"_i{idx}"

        return query_impl(
            self._pool, self._database, self._use_stale_reads, index_name, self._metric, self._means_top_size, v, n)[0]

    def batch_query(self, X: np.ndarray, n: int) -> None:
        self._batch_threads = min(self._batch_threads, max(1, len(X)))
        print(f"Batching queries in {self._batch_threads} processes")

        try:
            mp.set_start_method("spawn", force=False)
        except RuntimeError:
            pass  # already set elsewhere

        total = len(X)
        results  = np.empty((total, n), dtype=int)
        latencies = np.empty(total, dtype=float)

        chunk = math.ceil(total / self._batch_threads)
        ranges = [(s, min(s + chunk, total)) for s in range(0, total, chunk)]

        ctx = mp.get_context("spawn")
        with concurrent.futures.ProcessPoolExecutor(max_workers=self._batch_threads, mp_context=ctx) as ex:
            future_to_range = {
                ex.submit(
                    proc_execute_sub_batch,
                    self._database,
                    self._metric,
                    self._use_stale_reads,
                    self._index_name,
                    self._index_count,
                    self._means_top_size,
                    X[s:e],        # <-- sliced copy to child
                    n,
                ): (s, e)
                for (s, e) in ranges
            }

            for future in concurrent.futures.as_completed(future_to_range):
                s, e = future_to_range[future]
                try:
                    res_sub, lat_sub = future.result()
                    results[s:e, :] = res_sub
                    latencies[s:e]  = lat_sub
                except Exception as exc:
                    print(f"exception in sub-batch ({s},{e}): {exc}")
                    # raise  # optionally fail-fast

        self.results = results
        self.latencies = latencies

    def get_batch_results(self) -> np.array:
        return self.results

    def get_batch_latencies(self) -> np.array:
        return self.latencies

    def set_query_arguments(self, means_top_size, opts=None, **kwargs):
        self._means_top_size = means_top_size

        options = {}
        if isinstance(opts, dict):
            options.update(opts)
        options.update(kwargs)

        if "threads" in options:
            self._batch_threads = options["threads"]

    def get_memory_usage(self):
        # TODO: Implement memory usage calculation
        return 0

    def get_additional(self) -> dict[str, Any]:
        d = {}
        if self._batch_threads:
            d["threads"] = self._batch_threads
        return d

    def __str__(self):
        result = "YDBVector("

        # Add metric if available
        if hasattr(self, '_metric') and self._metric:
            result += self._metric

        # Add method parameters
        param_parts = []
        if self._method_param:
            for k, v in self._method_param.items():
                param_parts.append(f"{k}={v}")

        # Add means_top_size if it's set to non-default value
        if hasattr(self, '_means_top_size') and self._means_top_size != DEFAULT_MEANS_TOP_SIZE:
            param_parts.append(f"means_top_size={self._means_top_size}")

        # Add parameters if any exist
        if param_parts:
            # Add comma if metric was already added
            if hasattr(self, '_metric') and self._metric:
                result += ", "

            result += ", ".join(param_parts)

        if self._batch_threads and 'threads' not in self._method_param:
            result += ", threads=" + str(self._batch_threads)

        result += ")"
        return result
