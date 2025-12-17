This is a fork of [ann-benchmarks](https://github.com/erikbern/ann-benchmarks) with the following additions:
* YDB support
* Cohere/wikipedia-22-12-en-embeddings dataset from Timescale's [fork](https://github.com/timescale/ann-benchmarks.git), used in [this comparison](https://www.tigerdata.com/blog/pgvector-vs-qdrant)
* `table_results.py` utility

# Prerequisites

If you have Ubuntu 20.04:

```
#python3.10 из deadsnake
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt install python3.10 python3.10-distutils

# alt
sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 0
sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1
sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 2

sudo update-alternatives --config python3

# pip
curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10

pip3 install --upgrade pip setuptools wheel

# pgvector
pip3 install pgvector
pip3 install psycopg
pip3 install psycopg-pool
```

Ann's requirements
```
pip install -r requirements.txt
```

YDB:
```
pip3 install ydb
```

Make sure `ydb` CLI is in PATH.

# Configuration

Configuration is stored in `ann_benchmarks/algorithms/{ydb,pgvector}/{config.yml,config_batch.yml}`. You can configure:
* a number of configurations to run
* clusters, levels, overlap_clusters, kMeansTreeSearchTopSize, etc

If you run with `--batch` option, benchmark will use `config_batch.yml`, otherwise `config.yml`.

# Prebuild datasets

Building 100M wikipedia dataset requires 900 GiB of RAM. Thus, some prebuild datasets can be located [here](https://storage.yandexcloud.net/ann-data). Just download datasets to the data subfolder.

# Running pgvector

Setup environment:
```
export ANN_BENCHMARKS_PG_USER=vec
export ANN_BENCHMARKS_PG_PASSWORD=vec
export ANN_BENCHMARKS_PG_DBNAME=vec
export ANN_BENCHMARKS_PG_HOST=localhost
export ANN_BENCHMARKS_PG_PORT=5432
export ANN_BENCHMARKS_PG_START_SERVICE=no
```

Run

```
python3 -u run.py [--skip-dataload] --algorithm pgvector --dataset cohere-wikipedia-22-12-10M-angular --local [--batch]
```

Run IVFFLAT:
```
python3 -u run.py [--skip-dataload] --algorithm pgvector-flat --dataset cohere-wikipedia-22-12-10M-angular --local --batch --runs 5
```

# Running YDB

```
export YDB_ANONYMOUS_CREDENTIALS=1
export YDB_CONNECTION_STRING="grpc://<HOST>:2135/?database=/Root/db1"

python3 -u run.py [--skip-dataload] --algorithm ydb --dataset cohere-wikipedia-22-12-10M-angular --local [--batch]
```

To allow stale reads (followers / read replicas), please
```
export YDB_STALE_READS=1
```

# Getting results

Tabular:
```
./table_results.py --dataset cohere-wikipedia-22-12-10M-angular --count 10 [--batch]
```

Plot:
```
python3 -u plot.py --x-scale logit --dataset cohere-wikipedia-22-12-10M-angular --count 10 [--batch]
```
