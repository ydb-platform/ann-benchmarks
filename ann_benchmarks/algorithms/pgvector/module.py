"""
This module supports connecting to a PostgreSQL instance and performing vector
indexing and search using the pgvector extension. The default behavior uses
the "ann" value of PostgreSQL user name, password, and database name, as well
as the default host and port values of the psycopg driver.

If PostgreSQL is managed externally, e.g. in a cloud DBaaS environment, the
environment variable overrides listed below are available for setting PostgreSQL
connection parameters:

ANN_BENCHMARKS_PG_USER
ANN_BENCHMARKS_PG_PASSWORD
ANN_BENCHMARKS_PG_DBNAME
ANN_BENCHMARKS_PG_HOST
ANN_BENCHMARKS_PG_PORT

This module starts the PostgreSQL service automatically using the "service"
command. The environment variable ANN_BENCHMARKS_PG_START_SERVICE could be set
to "false" (or e.g. "0" or "no") in order to disable this behavior.

This module will also attempt to create the pgvector extension inside the
target database, if it has not been already created.
"""

import concurrent.futures
import logging
import math
import multiprocessing as mp
import numpy as np
import os
import subprocess
import sys
import threading
import time

import pgvector.psycopg
import psycopg

from psycopg_pool import ConnectionPool

from typing import Dict, Any, Optional

from ..base.module import BaseANN
from ...util import get_bool_env_var

from time import perf_counter


METRIC_PROPERTIES = {
    "angular": {
        "distance_operator": "<=>",
        # A substring of e.g. vector_cosine_ops or halfvec_cosine_ops
        "ops_type": "cosine",
    },
    "euclidean": {
        "distance_operator": "<->",
        "ops_type": "l2",
    }
}

MAX_BATCH_QUERY_THREADS = 128

USE_SELECT1 = False
USE_MP = True


def proc_execute_sub_batch(connect_kwargs,
                           query_sql: str,
                           ef_search: int | None,
                           X_chunk: np.ndarray,  # THIS IS COPIED to the child
                           n: int):
    """
    Executes queries for the rows in X_chunk and returns (results_sub, latencies_sub).
    """
    conn = psycopg.connect(**connect_kwargs)

    if not USE_SELECT1:
        pgvector.psycopg.register_vector(conn)

    try:
        results_sub   = np.empty((len(X_chunk), n), dtype=int)
        latencies_sub = np.empty(len(X_chunk), dtype=float)

        with conn.cursor() as cursor:
            if ef_search is not None and not USE_SELECT1:
                cursor.execute(f"SET hnsw.ef_search = {ef_search}")

            if USE_SELECT1:
                # Fixed ids 0..n-1 for each query
                dummy_ids = np.arange(n, dtype=int)
                for j in range(len(X_chunk)):
                    t0 = perf_counter()
                    cursor.execute("SELECT 1")
                    cursor.fetchone()  # ensure round-trip completes
                    results_sub[j, :] = dummy_ids
                    latencies_sub[j]  = perf_counter() - t0
            else:
                for j, v in enumerate(X_chunk):
                    t0 = perf_counter()
                    cursor.execute(query_sql, (v, n), binary=True, prepare=True)
                    rows = cursor.fetchall()
                    # If fewer than n is acceptable, relax this check
                    if len(rows) != n:
                        raise RuntimeError(f"Expected {n} rows, got {len(rows)}")
                    results_sub[j, :] = np.fromiter((r[0] for r in rows), dtype=int, count=n)
                    latencies_sub[j]  = perf_counter() - t0

        return results_sub, latencies_sub
    finally:
        conn.close()


def get_pg_param_env_var_name(pg_param_name: str) -> str:
    return f'ANN_BENCHMARKS_PG_{pg_param_name.upper()}'


def get_pg_conn_param(
        pg_param_name: str,
        default_value: Optional[str] = None) -> Optional[str]:
    env_var_name = get_pg_param_env_var_name(pg_param_name)
    env_var_value = os.getenv(env_var_name, default_value)
    if env_var_value is None or len(env_var_value.strip()) == 0:
        return default_value
    return env_var_value


class IndexingProgressMonitor:
    """
    Continuously logs indexing progress, elapsed and estimated remaining
    indexing time.
    """

    MONITORING_DELAY_SEC = 0.5

    def __init__(self, psycopg_connect_kwargs: Dict[str, str]) -> None:
        self._psycopg_connect_kwargs = psycopg_connect_kwargs
        self.monitoring_condition = threading.Condition()
        self.stop_requested = False
        self.prev_phase = None
        self.prev_progress_pct = None
        self.prev_tuples_done = None
        self.prev_report_time_sec = None
        self.time_to_load_all_tuples_sec = None

    def report_progress(
            self,
            phase: str,
            progress_pct: Any,
            tuples_done: Any) -> None:
        if progress_pct is None:
            progress_pct = 0.0
        progress_pct = float(progress_pct)
        if tuples_done is None:
            tuples_done = 0
        tuples_done = int(tuples_done)
        if (phase == self.prev_phase and
                progress_pct == self.prev_progress_pct):
            return
        time_now_sec = time.time()

        elapsed_time_sec = time_now_sec - self.indexing_start_time_sec
        fields = [
            f"Phase: {phase}",
            f"progress: {progress_pct:.1f}%",
            f"elapsed time: {elapsed_time_sec:.3f} sec"
        ]
        if (self.prev_report_time_sec is not None and
            self.prev_tuples_done is not None and
            elapsed_time_sec):
            overall_tuples_per_sec = tuples_done / elapsed_time_sec
            fields.append(
                f"overall tuples/sec: {overall_tuples_per_sec:.2f}")

            time_since_last_report_sec = time_now_sec - self.prev_report_time_sec
            if time_since_last_report_sec > 0:
                cur_tuples_per_sec = ((tuples_done - self.prev_tuples_done) /
                                      time_since_last_report_sec)
                fields.append(
                    f"current tuples/sec: {cur_tuples_per_sec:.2f}")

        remaining_pct = 100 - progress_pct
        if progress_pct > 0 and remaining_pct > 0:
            estimated_remaining_time_sec = \
                elapsed_time_sec / progress_pct * remaining_pct
            estimated_total_time_sec = \
                elapsed_time_sec + estimated_remaining_time_sec
            fields.extend([
                "estimated remaining time: " \
                   f"{estimated_remaining_time_sec:.3f} sec" ,
                f"estimated total time: {estimated_total_time_sec:.3f} sec"
            ])
        print(", ".join(fields))
        sys.stdout.flush()

        self.prev_progress_pct = progress_pct
        self.prev_phase = phase
        self.prev_tuples_done = tuples_done
        self.prev_report_time_sec = time_now_sec

    def monitoring_loop_impl(self, monitoring_cur) -> None:
        while True:
            # Indexing progress query taken from
            # https://github.com/pgvector/pgvector/blob/master/README.md
            monitoring_cur.execute(
                "SELECT phase, " +
                "round(100.0 * blocks_done / nullif(blocks_total, 0), 1), " +
                "tuples_done " +
                "FROM pg_stat_progress_create_index");
            result_rows = monitoring_cur.fetchall()

            if len(result_rows) == 1:
                phase, progress_pct, tuples_done = result_rows[0]
                self.report_progress(phase, progress_pct, tuples_done)
                if (self.time_to_load_all_tuples_sec is None and
                    phase == 'building index: loading tuples' and
                    progress_pct is not None and
                    float(progress_pct) > 100.0 - 1e-7):
                    # Even after pgvector reports progress as 100%, it still spends
                    # some time postprocessing the index and writing it to disk.
                    # We keep track of the the time it takes to reach 100%
                    # separately.
                    self.time_to_load_all_tuples_sec = \
                        time.time() - self.indexing_start_time_sec
            elif len(result_rows) > 0:
                # This should not happen.
                print(f"Expected exactly one progress result row, got: {result_rows}")
            with self.monitoring_condition:
                if self.stop_requested:
                    return
                self.monitoring_condition.wait(
                    timeout=self.MONITORING_DELAY_SEC)
                if self.stop_requested:
                    return

    def monitor_progress(self) -> None:
        prev_phase = None
        prev_progress_pct = None
        with psycopg.connect(**self._psycopg_connect_kwargs) as monitoring_conn:
            with monitoring_conn.cursor() as monitoring_cur:
                self.monitoring_loop_impl(monitoring_cur)

    def start_monitoring_thread(self) -> None:
        self.indexing_start_time_sec = time.time()
        self.monitoring_thread = threading.Thread(target=self.monitor_progress)
        self.monitoring_thread.start()

    def stop_monitoring_thread(self) -> None:
        with self.monitoring_condition:
            self.stop_requested = True
            self.monitoring_condition.notify_all()
        self.monitoring_thread.join()
        self.indexing_time_sec = time.time() - self.indexing_start_time_sec

    def report_timings(self) -> None:
        print(f"pgvector total indexing time: {self.indexing_time_sec:3f} sec")
        if self.time_to_load_all_tuples_sec is not None:
            print("    Time to load all tuples into the index: {:.3f} sec".format(
                self.time_to_load_all_tuples_sec
            ))
            postprocessing_time_sec = \
                self.indexing_time_sec - self.time_to_load_all_tuples_sec
            print("    Index postprocessing time: {:.3f} sec".format(
                postprocessing_time_sec))
        else:
            print("    Detailed breakdown of indexing time not available.")

class PGVector(BaseANN):
    def __init__(self, metric, method_param):
        self._metric = metric
        self._m = method_param['M']
        self._ef_construction = method_param['efConstruction']

        self._ef_search = None

        self._method_param = method_param
        self._batch_threads = None
        self._pool = None

        self._psycopg_connect_kwargs: Dict[str, Any] = dict(
            autocommit=True,
        )
        for arg_name in ['user', 'password', 'dbname']:
            # The default value is "ann" for all of these parameters.
            self._psycopg_connect_kwargs[arg_name] = get_pg_conn_param(
                arg_name, 'ann')

        # If host/port are not specified, leave the default choice to the
        # psycopg driver.
        pg_host: Optional[str] = get_pg_conn_param('host')
        if pg_host is not None:
            self._psycopg_connect_kwargs['host'] = pg_host

        pg_port_str: Optional[str] = get_pg_conn_param('port')
        if pg_port_str is not None:
            self._psycopg_connect_kwargs['port'] = int(pg_port_str)

        self._psycopg_connect_kwargs["application_name"] = "ann-benchmarks/pgvector"

        should_start_service = get_bool_env_var(
            get_pg_param_env_var_name('start_service'),
            default_value=True)
        if should_start_service:
            subprocess.run(
                "service postgresql start",
                shell=True,
                check=True,
                stdout=sys.stdout,
                stderr=sys.stderr)
        else:
            print(
                "Assuming that PostgreSQL service is managed externally. "
                "Not attempting to start the service.")

        self._conn = psycopg.connect(**self._psycopg_connect_kwargs)
        self.ensure_pgvector_extension_created(self._conn)
        self.configure_connection(self._conn)

        if metric == "angular":
            self._query = "SELECT id FROM items ORDER BY embedding <=> %s LIMIT %s"
        elif metric == "euclidean":
            self._query = "SELECT id FROM items ORDER BY embedding <-> %s LIMIT %s"
        else:
            raise RuntimeError(f"unknown metric {metric}")

    def get_metric_properties(self) -> Dict[str, str]:
        """
        Get properties of the metric type associated with this index.

        Returns:
            A dictionary with keys distance_operator and ops_type.
        """
        if self._metric not in METRIC_PROPERTIES:
            raise ValueError(
                "Unknown metric: {}. Valid metrics: {}".format(
                    self._metric,
                    ', '.join(sorted(METRIC_PROPERTIES.keys()))
                ))
        return METRIC_PROPERTIES[self._metric]

    def ensure_pgvector_extension_created(self, conn: psycopg.Connection) -> None:
        """
        Ensure that `CREATE EXTENSION vector` has been executed.
        """
        with conn.cursor() as cur:
            # We have to use a separate cursor for this operation.
            # If we reuse the same cursor for later operations, we might get
            # the following error:
            # KeyError: "couldn't find the type 'vector' in the types registry"
            cur.execute(
                "SELECT EXISTS(SELECT 1 FROM pg_extension WHERE extname = 'vector')")
            pgvector_exists = cur.fetchone()[0]
            if pgvector_exists:
                print("vector extension already exists")
            else:
                print("vector extension does not exist, creating")
                cur.execute("CREATE EXTENSION vector")

    def fit(self, X):
        cur = self._conn.cursor()
        cur.execute("DROP TABLE IF EXISTS items")
        cur.execute("CREATE TABLE items (id int, embedding vector(%d))" % X.shape[1])
        cur.execute("ALTER TABLE items ALTER COLUMN embedding SET STORAGE PLAIN")
        print("copying data...")
        sys.stdout.flush()
        num_rows = 0
        insert_start_time_sec = time.time()
        with cur.copy("COPY items (id, embedding) FROM STDIN WITH (FORMAT BINARY)") as copy:
            copy.set_types(["int4", "vector"])
            for i, embedding in enumerate(X):
                copy.write_row((i, embedding))
                num_rows += 1
        insert_elapsed_time_sec = time.time() - insert_start_time_sec
        print("inserted {} rows into table in {:.3f} seconds".format(
            num_rows, insert_elapsed_time_sec))

        print("creating index...")
        sys.stdout.flush()
        create_index_str = \
            "CREATE INDEX ON items USING hnsw (embedding vector_%s_ops) " \
            "WITH (m = %d, ef_construction = %d)" % (
                self.get_metric_properties()["ops_type"],
                self._m,
                self._ef_construction
            )
        progress_monitor = IndexingProgressMonitor(self._psycopg_connect_kwargs)
        progress_monitor.start_monitoring_thread()

        try:
            cur.execute(create_index_str)
        finally:
            progress_monitor.stop_monitoring_thread()
        print("done!")
        progress_monitor.report_timings()

    def configure_connection(self, conn):
        if not USE_SELECT1:
            pgvector.psycopg.register_vector(conn)
            if self._ef_search is not None:
                conn.execute(f"SET hnsw.ef_search = {self._ef_search}")
                conn.commit()

    def start_pool(self):
        #logging.getLogger("psycopg.pool").setLevel(logging.DEBUG)
        if self._pool is not None:
            self._pool.close()

        max_size = int(self._batch_threads or MAX_BATCH_QUERY_THREADS)
        if max_size < 1:
            max_size = 1
        self._pool = ConnectionPool(
            kwargs=self._psycopg_connect_kwargs,
            min_size=max_size,
            max_size=max_size,
            configure=self.configure_connection,
            timeout=120.0,
            max_lifetime=7200,
            max_idle=7200,
        )
        self._pool.wait()

        print(f"Started pool with {max_size} connections")

    def batch_query(self, X: np.array, n: int) -> None:
        self._batch_threads = min(self._batch_threads, max(1, len(X)))

        if USE_MP:
            self.batch_query_mp(X, n)
        else:
            self.batch_query_thread_pool(X, n)

    def batch_query_thread_pool_naive(self, X: np.array, n: int) -> None:
        print(f"Batching queries in {self._batch_threads} threads, ef_search={self._ef_search}")
        self.start_pool()

        results = np.empty((X.shape[0], n), dtype=int)
        latencies = np.empty(X.shape[0], dtype=float)
        with concurrent.futures.ThreadPoolExecutor(max_workers=self._batch_threads) as executor:
            futures = {executor.submit(
                self.query_pooled, q, n): i for i, q in enumerate(X)}
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

    def batch_query_thread_pool(self, X: np.ndarray, n: int) -> None:
        print(f"Batching queries in {self._batch_threads} threads (via ThreadPool), ef_search={self._ef_search}, dummy={USE_SELECT1}")

        total = len(X)
        results  = np.empty((total, n), dtype=int)
        latencies = np.empty(total, dtype=float)

        chunk = math.ceil(total / self._batch_threads)
        ranges = [(s, min(s + chunk, total)) for s in range(0, total, chunk)]

        connect_kwargs = dict(self._psycopg_connect_kwargs)
        query_sql = self._query
        ef_search = self._ef_search

        with concurrent.futures.ThreadPoolExecutor(max_workers=self._batch_threads) as executor:
            future_to_range = {
                executor.submit(
                    proc_execute_sub_batch,
                    connect_kwargs,
                    query_sql,
                    ef_search,
                    X[s:e],      # copied slice to thread
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
                    print(f"exception in thread sub-batch ({s},{e}): {exc}")

        self.results = results
        self.latencies = latencies

    def batch_query_mp(self, X: np.ndarray, n: int) -> None:
        self._batch_threads = min(self._batch_threads, max(1, len(X)))
        print(f"Batching queries in {self._batch_threads} processes, ef_search={self._ef_search}, dummy={USE_SELECT1}")

        try:
            mp.set_start_method("spawn", force=False)
        except RuntimeError:
            pass  # already set elsewhere

        total = len(X)
        results  = np.empty((total, n), dtype=int)
        latencies = np.empty(total, dtype=float)

        chunk = math.ceil(total / self._batch_threads)
        ranges = [(s, min(s + chunk, total)) for s in range(0, total, chunk)]

        connect_kwargs = dict(self._psycopg_connect_kwargs)  # each process connects independently
        query_sql = self._query
        ef_search = self._ef_search

        ctx = mp.get_context("spawn")
        with concurrent.futures.ProcessPoolExecutor(max_workers=self._batch_threads, mp_context=ctx) as ex:
            future_to_range = {
                ex.submit(
                    proc_execute_sub_batch,
                    connect_kwargs,
                    query_sql,
                    ef_search,
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

    def set_query_arguments(self, ef_search, opts=None, **kwargs):
        # this will affect all new connections (i.e. from the pool)
        self._ef_search = ef_search
        if not USE_SELECT1:
            self._psycopg_connect_kwargs["options"] = f"-c hnsw.ef_search={self._ef_search}"

        # update existing "default" connection used in non-batch mode
        if not USE_SELECT1:
            with self._conn.cursor() as cur:
                cur.execute(f"SET hnsw.ef_search = {self._ef_search}")
            self._conn.commit()

        options = {}
        if isinstance(opts, dict):
            options.update(opts)
        options.update(kwargs)

        if "threads" in options:
            self._batch_threads = options["threads"]

    def query(self, v, n):
        connection = self._conn
        return self.query_impl(v, n, connection)[0]

    def query_pooled(self, v, n):
        with self._pool.connection() as connection:
            return self.query_impl(v, n, connection)

    def query_impl(self, v, n, connection):
        start = perf_counter()
        with connection.cursor() as cursor:
            if USE_SELECT1:
                cursor.execute("SELECT 1")
                cursor.fetchone()
                result = np.arange(n, dtype=int)
            else:
                cursor.execute(self._query, (v, n), binary=True, prepare=True)
                rows = cursor.fetchall()
                result = np.fromiter((row[0] for row in rows), dtype=int)

        elapsed = perf_counter() - start
        return result, elapsed

    def get_memory_usage(self):
        start = time.perf_counter()
        cur = self._conn.cursor()
        if cur is None:
            return 0
        cur.execute("SELECT COALESCE(pg_indexes_size(to_regclass('public.items')), 0)")
        return cur.fetchone()[0] / 1024

    def should_check_results(self):
        return not USE_SELECT1

    def get_additional(self) -> Dict[str, Any]:
        d = {}
        if self._batch_threads:
            d["threads"] = self._batch_threads
            if USE_MP:
                d["mp-mode"] = True
        return d

    def __str__(self):
        result = f"PGVector(m={self._m}, ef_construction={self._ef_construction}, ef_search={self._ef_search}"

        if self._batch_threads and self._batch_threads != 1:
            result += f", threads={self._batch_threads}"

        result += ")"
        return result
