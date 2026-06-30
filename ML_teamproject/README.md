# Local DNS Cache Poisoning ML Pipeline

This project builds RandomForest run bundles locally from resolver captures. The
pipeline is independent of the reference `sniffer/` and `random forest/`
directories and continues to work if those directories are removed.

## Label Rule

Each resolver-bound DNS response in a pcap becomes one CSV row. A row is
labeled as an attack (`label=1`) only when both conditions are true:

- A DNS question name or any answer/authority/additional record name or value
  contains `bank.test`.
- Any answer/authority/additional DNS record value contains `192.168.219.104`.

Every other accepted response is normal (`label=0`). The marker values are
fixed in `pipeline/dataset.py`.

## Local Run

Create a local virtual environment and install dependencies:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

Use the interactive entry point to fetch or select a pcap, label it, execute
the training notebook, create a run bundle, and optionally deploy after an
explicit confirmation:

```bash
.venv/bin/python main.py
```

For a local pcap, the same workflow can be started directly:

```bash
.venv/bin/python main.py run \
  --pcap data/raw/dataset_v000001.pcap \
  --resolver-ip 192.168.219.1 \
  --dataset-version v000001
```

The `resolver-ip` value must match the destination resolver IP captured in the
pcap. When `--run-version` is omitted, the next version after existing
`runs/run_vNNNNNN` directories is assigned automatically. The command writes:

```text
data/processed/features_v000001.csv
data/manifests/dataset_v000001.yaml
runs/run_v000010/run_manifest_v000010.yaml
runs/run_v000010/detection_run_v000010.executed.ipynb
runs/run_v000010/randomforest_v000010.joblib
runs/run_v000010/metrics_run_v000010.json
runs/run_v000010/confusion_matrix_run_v000010.csv
runs/run_v000010/feature_importance_run_v000010.csv
metrics/metrics_summary.csv
```

## Commands

`main.py` also exposes reproducible individual stages:

```bash
.venv/bin/python main.py fetch-data --dataset-version v000001 --host HOST \
  --username USER --remote-capture-dir /remote/data/captures

.venv/bin/python main.py build-dataset --pcap data/raw/dataset_v000001.pcap \
  --resolver-ip 192.168.219.1 --dataset-version v000001

.venv/bin/python main.py build-dataset \
  --pcap data/raw/external_dns_capture.pcap \
  --capture-scope all-dns-responses --dataset-version v000004

.venv/bin/python main.py train \
  --train-dataset-version v000001 --test-dataset-version v000002

.venv/bin/python main.py train \
  --train-dataset-versions "v000001 v000002" --test-dataset-version v000003

.venv/bin/python main.py prepare-run \
  --train-dataset-version v000001 --test-dataset-version v000002

.venv/bin/python main.py deploy-model --run-version v000010 --host HOST \
  --username USER --remote-model-dir /remote/models
```

Downloading and deployment do not overwrite same-named files unless
`--overwrite` is explicitly passed. Deployment still prompts for confirmation,
and the remote detector must be restarted before a newly uploaded highest
version model is selected.

Pass `--run-version vNNNNNN` when reproducing or overwriting a specific run.
DVC continues to use the explicit `training.run_version` in `params.yaml` so
its output path and lineage stay reproducible.

The default dataset capture scope remains `resolver-bound`, which requires the
internal resolver destination and monitored ports. Use
`--capture-scope all-dns-responses` only for external DNS pcaps; it removes
those capture-address constraints but keeps response-only processing and the
same fixed labeling rule. An external dataset with only normal labels cannot
be used alone as a training dataset.

## Mac Mini M4 Settings

Training uses the local `notebook/detection.ipynb` and
`config/randomforest.yaml`. The default randomized search runs twelve candidate
configurations with four concurrent candidate jobs and a single worker per
forest to keep memory pressure controlled on a 16 GB machine. Conversion
streams packets from the pcap, and training reads only label/model columns and
stores features as `float32` instead of retaining string provenance in memory.

Model fitting uses only `pipeline.schema.FEATURE_COLUMNS`. Label evidence,
packet identity, domains and IP metadata remain outside the predictor input.

## Manual Scratch Notebook

Open `notebook/manual_training.ipynb` for cell-by-cell experimentation. In its
configuration cell you can specify an official `OFFICIAL_RUN_VERSION` to load
an existing model read-only, set `SCENARIO_SPLIT = True`, choose separate
`TRAIN_DATASET_VERSIONS` and `TEST_DATASET_VERSIONS`, and temporarily override
selected features without modifying the official config. Scenario split trains
on whole train datasets and evaluates on whole test datasets without a random
row split.

Manual training writes only under `scratch/<experiment-name>/`; it does not
write official run bundles or `metrics/metrics_summary.csv`. A DVC-managed
dataset or model version can be selected in the cell as long as its file has
been checked out into the current workspace.

## Class Ratio

Train and test may each combine multiple dataset versions. Configure the
desired post-combination normal-to-attack ratio in `config/randomforest.yaml`:

```yaml
class_ratio:
  train: "8:2"
  test: "8:2"
```

Ratio handling deterministically downsamples only excess rows; it does not
modify processed CSVs or duplicate the minority class. Set a split to `null`
to retain all rows. Metrics and run manifests record configured ratios and
class counts before and after sampling.

## Threshold And Weight Comparison

After fixing a dataset/run, compare confusion matrices without modifying the
official run bundle:

```bash
.venv/bin/python main.py compare-thresholds --run-version v000014 \
  --analysis-name threshold_weight_v000014
```

Candidates are configured in `config/randomforest.yaml` with
`decision_thresholds` and `class_weight_grid`. Results are written under
`scratch/<analysis-name>/`.

## DVC Definition

`dvc.yaml` and `params.yaml` define the conversion and training lineage without
initializing Git, DVC, or a DVC remote. Training tracks one explicit train
scenario CSV and one explicit test scenario CSV through
`training.train_dataset_versions` and `training.test_dataset_versions`; each
parameter may contain multiple space-separated versions. The run input
manifest records the exact CSV list and checksums. Prepare the required
processed CSVs before reproducing the training stages. Set
`dataset.capture_scope` to `all-dns-responses` only when converting an external
DNS pcap.
