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
INDEX_NAME = "idx_vector_items"

LEVELS = 3
CLUSTERS = 200

BATCH_SIZE = 1000

MAX_RETRIES = 100
BACKOFF_MILLIS = 10
BACKOFF_CEILING = 5


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


def drop_create_table(session, table_name, dimensions):
    """Drop and create YDB table"""

    try:
        session.execute_scheme(f"DROP TABLE `{table_name}`")
    except:
        pass

    query = f"""
        CREATE TABLE `{table_name}` (
            `id` Uint64,
            `embedding` String,
            PRIMARY KEY (`id`)
        )
    """

    try:
        session.execute_scheme(query)
    except Exception as e:
        print(f"Failed to create table `{table_name}`: ", e)
        raise e

    print(f"Table '{table_name}' created")


def build_index(session, endpoint, database, table_name, index_name, num_dimensions, levels, clusters):
    """Create and wait to be ready the vector index"""

    table_path = database + "/" + table_name

    query = f"""
        ALTER TABLE `{table_path}`
        ADD INDEX `{index_name}`
        GLOBAL USING vector_kmeans_tree
        ON (embedding)
        WITH (
            similarity=inner_product,
            vector_type="float",
            vector_dimension={num_dimensions},
            levels={levels},
            clusters={clusters}
        );
    """

    index_future = session.async_execute_scheme(query)
    try:
        # we set timeout to 1 second, because we want to check here
        # that requests have started execution and then check state manually
        result = index_future.result(timeout=1)
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
    def __init__(self, metric, method_param=None):
        # Check if ydb CLI is available before proceeding
        check_ydb_cli_available()

        self._metric = metric
        if method_param is None:
            method_param = {}
        self._method_param = method_param

        try:
            self.driver = initialize_ydb_from_env()
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


    def fit(self, X):
        num_dimensions = X.shape[1]

        drop_create_table(self.driver.table_client.session().create(), TABLE_NAME, num_dimensions)

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

        index_start_time_sec = time.time()
        print("building index...")

        build_index(
            self.driver.table_client.session().create(),
            self.endpoint,
            self.database,
            TABLE_NAME,
            INDEX_NAME,
            num_dimensions,
            LEVELS,
            CLUSTERS)

        index_elapsed_time_sec = time.time() - index_start_time_sec
        print("built index in {:.3f} seconds".format(index_elapsed_time_sec))

    def query(self, v, n):
        binary_embedding = float_embedding_to_binary(v)
        query = f"""
            PRAGMA TablePathPrefix("{self.database}");

            DECLARE $embedding_list as List<Float>;
            $TargetEmbedding = Knn::ToBinaryStringFloat($embedding_list);

            SELECT id, Knn::InnerProductSimilarity(embedding, $TargetEmbedding) as dist
            FROM `{TABLE_NAME}`
            VIEW `{INDEX_NAME}`
            ORDER BY dist DESC
            LIMIT {n};
        """

        try:
            session = self.driver.table_client.session().create()
            select_query = session.prepare(query)
            result_sets = session.transaction().execute(select_query, {"$embedding_list": v})
            rows = result_sets[0].rows
            ids = [row.id for row in rows]
            return ids
        except Exception as e:
            print("Query failed: ", e)
            raise e


    def set_query_arguments(self, *args):
        # Store query arguments for later use
        if args:
            self._query_args = args[0] if len(args) == 1 else args
        else:
            self._query_args = None

    def get_memory_usage(self):
        # TODO: Implement memory usage calculation
        return 0

    def __str__(self):
        return f"YDBVector(metric={self._metric}, method_param={self._method_param})"
