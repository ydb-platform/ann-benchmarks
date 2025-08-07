"""
This module supports connecting to a YDB instance and performing vector
indexing and search.

For authentication the module uses the standard YDB environment variables:
https://ydb.tech/docs/en/recipes/ydb-sdk/auth-env
"""

import concurrent.futures
import json
import numpy as np
import os
import random
import shutil
import struct
import subprocess
import sys
import threading
import time
import ydb

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


def enable_split_by_load(pool, table_name, num_dimensions, n):
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
    except:
        pass


def build_index(pool, endpoint, database, table_name, index_name, num_dimensions, levels, clusters):
    """Create and wait to be ready the vector index"""

    table_path = database + "/" + table_name

    query = f"""
        ALTER TABLE `{table_path}`
        ADD INDEX `{index_name}`
        GLOBAL USING vector_kmeans_tree
        ON (embedding) COVER (embedding)
        WITH (
            distance="cosine",
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

    print("Waiting for indices to be ready...")

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
            time.sleep(10) # hack, because we have a small issue with reporting OK
            print("Indices created")
            break
        time.sleep(10)

    print("Indices are ready")


def initialize_ydb_from_env():
    """Initialize YDB driver from env"""
    driver = ydb.Driver(
        connection_string=os.environ["YDB_CONNECTION_STRING"],
        credentials=ydb.credentials_from_env_variables(),
    )

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

        self.batch_threads = None

        self._metric = metric
        if method_param is None:
            method_param = {}
        self._method_param = method_param

        levels = self._method_param['levels'],
        clusters = self._method_param['clusters'])

        self._index_name = INDEX_BASE_NAME + f"_{metric}_{clusters}x{levels}"

        try:
            self.driver = initialize_ydb_from_env()
            self.pool = ydb.QuerySessionPool(self.driver)
        except Exception as e:
            print("Unable to connect to YDB: ", e)
            raise e

        # <protocol>://<hostname>:<port>/?database=/path/to/the/database
        connection_string = os.environ["YDB_CONNECTION_STRING"]

        # Parse connection string to extract endpoint and database
        parsed_url = urlparse(connection_string)
        self.endpoint = f"{parsed_url.scheme}://{parsed_url.netloc}"

        # Extract database from query parameters
        query_params = parse_qs(parsed_url.query)
        if 'database' not in query_params:
            raise ValueError(f"Database parameter not found in connection string: {connection_string}")
        self.database = query_params['database'][0]

        self.full_table_path = self.database + "/" + TABLE_NAME

        self.means_top_size = DEFAULT_MEANS_TOP_SIZE


    def fit(self, X):
        num_dimensions = X.shape[1]

        drop_create_table(self.pool, TABLE_NAME, num_dimensions, len(X))

        print("copying data...")
        sys.stdout.flush()
        num_rows = 0
        insert_start_time_sec = time.time()

        vectors_batch = []
        for i, embedding in enumerate(X):
            vectors_batch.append((i, embedding,))
            num_rows += 1
            if len(vectors_batch) == BATCH_SIZE:
                send_batch_to_ydb(self.driver.table_client, self.full_table_path, vectors_batch)
                vectors_batch = []

        if len(vectors_batch) != 0:
            send_batch_to_ydb(self.driver.table_client, self.full_table_path, vectors_batch)
            vectors_batch = []

        insert_elapsed_time_sec = time.time() - insert_start_time_sec
        print("inserted {} rows into table in {:.3f} seconds".format(num_rows, insert_elapsed_time_sec))

        enable_split_by_load(self.pool, TABLE_NAME, num_dimensions, len(X))

        index_start_time_sec = time.time()
        print("building index...")

        build_index(
            self.pool,
            self.endpoint,
            self.database,
            TABLE_NAME,
            self._index_name,
            num_dimensions,
            self._method_param['levels'],
            self._method_param['clusters'])

        index_elapsed_time_sec = time.time() - index_start_time_sec
        print("built index in {:.3f} seconds".format(index_elapsed_time_sec))


    def query(self, v, n):
        return self.query_impl(v, n)[0]

    def query_impl(self, v, n):
        start = time.perf_counter()
        binary_embedding = float_embedding_to_binary(v)
        query = f"""
            PRAGMA TablePathPrefix("{self.database}");

            pragma ydb.KMeansTreeSearchTopSize = "{self.means_top_size}";

            DECLARE $embedding_list as List<Float>;
            $TargetEmbedding = Knn::ToBinaryStringFloat($embedding_list);

            SELECT id, Knn::CosineDistance(embedding, $TargetEmbedding) as dist
            FROM `{TABLE_NAME}`
            VIEW `{self._index_name}`
            ORDER BY dist ASC
            LIMIT {n};
        """

        try:
            result_sets = self.pool.execute_with_retries(
                query,
                {
                    "$embedding_list": (v, ydb.ListType(ydb.PrimitiveType.Float)),
                },
            )

            rows = result_sets[0].rows
            ids = [row.id for row in rows]
            elapsed = time.perf_counter() - start
            return ids, elapsed
        except Exception as e:
            print("Query failed: ", e)
            raise e

    def batch_query(self, X: np.array, n: int) -> None:
        if 'threads' in self._method_param:
            self.batch_threads = self._method_param['threads']
        else:
            self.batch_threads = MAX_BATCH_QUERY_THREADS

        self.batch_threads = min(self.batch_threads, max(1, len(X)))

        results = np.empty((X.shape[0], n), dtype=int)
        latencies = np.empty(X.shape[0], dtype=float)
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.batch_threads) as executor:
            futures = {executor.submit(
                self.query_impl, q, n): i for i, q in enumerate(X)}
            for future in concurrent.futures.as_completed(futures):
                i = futures[future]
                try:
                    result, latency = future.result()
                    results[i] = result
                    latencies[i] = latency
                except Exception as x2:
                    print(f"exception getting batch results: {x2}")
        self.results = results
        self.latencies = latencies

    def get_batch_results(self) -> np.array:
        return self.results

    def get_batch_latencies(self) -> np.array:
        return self.latencies

    def set_query_arguments(self, means_top_size):
        self.means_top_size = means_top_size

    def get_memory_usage(self):
        # TODO: Implement memory usage calculation
        return 0

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
        if hasattr(self, 'means_top_size') and self.means_top_size != DEFAULT_MEANS_TOP_SIZE:
            param_parts.append(f"means_top_size={self.means_top_size}")

        # Add parameters if any exist
        if param_parts:
            # Add comma if metric was already added
            if hasattr(self, '_metric') and self._metric:
                result += ", "

            result += ", ".join(param_parts)

        if self.batch_threads and 'threads' not in self._method_param:
            result += ", threads=" + str(self.batch_threads)

        result += ")"
        return result
