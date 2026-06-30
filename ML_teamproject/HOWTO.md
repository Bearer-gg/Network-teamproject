# DNS 캐시 포이즈닝 ML 파이프라인 사용 방법

이 문서는 이 저장소의 로컬 ML 파이프라인을 처음 실행하는 사람을 위한 안내서입니다.  
기준은 다음과 같습니다.

- 데이터셋 버전과 학습 실행 버전은 분리됩니다.
- `dataset_version`은 raw pcap과 processed CSV를 식별합니다.
- `run_version`은 한 번의 학습 실행과 그 산출물을 식별합니다.
- 노트북은 소스 템플릿이고, 실제 중심은 `run_version`입니다.
- 최종 산출물은 `runs/run_vXXXXXX/` 아래에 묶입니다.

## 1. 준비물

필요한 것은 다음입니다.

- macOS가 설치된 로컬 PC
- Python 3.14 이상
- 학습용 pcap 파일
- resolver IP
- Ubuntu 서버 접속 정보

이 프로젝트는 로컬에서 실행합니다. Colab은 사용하지 않습니다.

## 2. 가상환경 만들기

아래 명령으로 프로젝트 전용 가상환경을 만듭니다.

```bash
python3 -m venv .venv
```

의존성을 설치합니다.

```bash
.venv/bin/python -m pip install -r requirements.txt
```

## 3. 기본 구조

주요 디렉터리는 다음과 같습니다.

```text
data/raw/           # 가져온 원본 pcap
data/processed/     # feature CSV
data/manifests/     # dataset manifest
runs/run_vXXXXXX/   # run bundle
metrics/            # run summary CSV
```

`runs/run_vXXXXXX/` 안에는 아래가 들어갑니다.

- `randomforest_vXXXXXX.joblib`
- `metrics_run_vXXXXXX.json`
- `detection_run_vXXXXXX.executed.ipynb`
- `feature_importance_run_vXXXXXX.csv`
- `confusion_matrix_run_vXXXXXX.csv`
- `run_manifest_vXXXXXX.yaml`

## 4. 라벨 규칙

각 DNS 응답 행은 아래 조건으로 라벨링합니다.

`label=1` 공격 조건:

- DNS 질문명 또는 answer/authority/additional 레코드의 이름이나 값 어디든 `bank.test`가 포함되고
- answer/authority/additional 레코드 값 어디든 `192.168.219.104`가 포함되는 경우

그 외의 처리 대상 DNS 응답은 모두 `label=0`입니다.

이 규칙은 pcap 파일 이름이 아니라 DNS 내용 자체를 기준으로 적용됩니다.

## 5. pcap을 가져오기

Ubuntu 서버에 있는 `dataset_vXXXXXX.pcap` 파일을 Mac으로 복사할 수 있습니다.

```bash
.venv/bin/python main.py fetch-data \
  --dataset-version v000001 \
  --host <UBUNTU_HOST> \
  --username <SSH_USER> \
  --remote-capture-dir /remote/data/captures
```

옵션 설명:

- `--dataset-version`: 가져올 원본 pcap 버전
- `--host`: Ubuntu 호스트나 IP
- `--username`: SSH 사용자
- `--identity-file`: SSH 개인키 경로
- `--port`: SSH 포트가 기본값과 다를 때 사용
- `--remote-capture-dir`: Ubuntu 쪽 pcap 디렉터리

기본 동작은 덮어쓰지 않기입니다. 같은 파일이 이미 있으면 실패합니다.  
덮어쓰려면 `--overwrite`를 명시해야 합니다.

## 6. dataset 만들기

pcap을 processed CSV로 변환합니다.

```bash
.venv/bin/python main.py build-dataset \
  --pcap data/raw/dataset_v000001.pcap \
  --resolver-ip 192.168.219.1 \
  --dataset-version v000001
```

이 단계에서 다음이 만들어집니다.

- `data/processed/features_v000001.csv`
- `data/manifests/dataset_v000001.yaml`

변환은 스트리밍 방식으로 처리합니다.  
즉, pcap 전체를 한 번에 메모리에 올리지 않고 packet 단위로 읽습니다.

외부에서 가져온 일반 DNS pcap처럼 내부 resolver IP와 실험용 포트
`10053`, `20053`, `30053`, `1025`를 사용하지 않는 파일은 아래처럼
DNS 응답만 대상으로 변환합니다.

```bash
.venv/bin/python main.py build-dataset \
  --pcap data/raw/2015-03-24_capture1-only-dns.pcap \
  --capture-scope all-dns-responses \
  --dataset-version v000004
```

`all-dns-responses`는 resolver IP와 monitored port 조건만 해제합니다.
`DNS response` 조건과 `bank.test` 및 `192.168.219.104` 기반 라벨 규칙은
변경되지 않습니다. 따라서 일반 외부 pcap은 정상 라벨만 포함할 수 있으며,
그 경우 단독 train dataset으로는 학습할 수 없습니다.

## 7. run 실행하기

학습 실행은 `run_version`으로 관리합니다.
일반적인 새 학습에서는 `--run-version`을 생략할 수 있습니다. 프로그램이
`runs/` 아래의 기존 run 번호를 확인하고 다음 번호를 자동으로 선택합니다.
예를 들어 `run_v000001`, `run_v000002`가 있으면 새 학습은 `v000003`입니다.

여러 dataset을 하나의 run에 묶거나, DVC로 추적 가능한 입력 증거를 먼저 남기고 싶다면
먼저 run 입력 manifest를 만들 수 있습니다.

```bash
.venv/bin/python main.py prepare-run \
  --train-dataset-versions "v000001 v000003" \
  --test-dataset-versions "v000002 v000004"
```

이 명령은 자동 배정된 run의 `run_input_manifest_vXXXXXX.yaml`을 만듭니다.
이 파일에 어떤 dataset들이 들어가는지, 어떤 config를 썼는지, 어떤 checksum을 기준으로 했는지가 기록됩니다.

단일 dataset으로 하나의 run을 만드는 예시는 다음과 같습니다.

```bash
.venv/bin/python main.py train \
  --dataset-version v000001
```

위 명령은 과거 실험 재현을 위한 랜덤 row split 방식입니다. scenario가 준비된
평가에서는 사용하지 말고, train scenario와 test scenario를 분리하여 실행합니다.

```bash
.venv/bin/python main.py train \
  --train-dataset-version v000001 \
  --test-dataset-version v000002
```

이 경우 `v000001` 전체로 학습하고 `v000002` 전체로만 평가하며, 행 단위
랜덤 split은 수행하지 않습니다. train과 test 양쪽 모두 여러 dataset을 합칠
수 있습니다.

```bash
.venv/bin/python main.py train \
  --train-dataset-versions "v000001 v000003" \
  --test-dataset-versions "v000002 v000004"
```

train/test 목록에 같은 dataset이 포함되면 학습이 중단됩니다.
복수 목록에는 `--train-dataset-versions`를 권장하지만,
`--train-dataset-version "v000001 v000003"`처럼 단수 옵션에 공백 목록을
입력해도 동일하게 처리됩니다.

이미 만든 run을 같은 번호로 다시 실행하거나 특정 번호로 재현하려면
`--run-version v000010`처럼 직접 지정합니다. 같은 run 결과물을 덮어쓸 때는
추가로 `--overwrite`가 필요합니다.

## 8. 한 번에 다 하기

pcap 변환부터 학습까지 한 번에 돌릴 수 있습니다.

```bash
.venv/bin/python main.py run \
  --pcap data/raw/dataset_v000001.pcap \
  --resolver-ip 192.168.219.1 \
  --dataset-version v000001
```

이 명령은 다음 순서로 동작합니다.

1. pcap 변환
2. 라벨링
3. notebook 실행
4. RandomForest 학습
5. metrics 저장
6. confusion matrix 저장
7. feature importance 저장
8. run manifest 저장

마지막에 사용자가 확인하면 모델 업로드 단계로 넘어갈 수 있습니다.

## 9. 결과물 확인

한 run이 끝나면 아래 파일들이 생깁니다.

```text
runs/run_v000010/run_manifest_v000010.yaml
runs/run_v000010/detection_run_v000010.executed.ipynb
runs/run_v000010/randomforest_v000010.joblib
runs/run_v000010/metrics_run_v000010.json
runs/run_v000010/confusion_matrix_run_v000010.csv
runs/run_v000010/feature_importance_run_v000010.csv
metrics/metrics_summary.csv
```

`metrics_run_vXXXXXX.json`에는 다음이 들어갑니다.

- 사용한 `dataset_versions`
- 선택된 하이퍼파라미터
- split 방식
- seed
- metrics
- execution timestamp

`metrics/metrics_summary.csv`의 마지막 `comment` 열은 직접 메모를 적기 위한 공란입니다.
같은 run을 `--overwrite`로 다시 학습해도 기존 `comment` 내용은 유지됩니다.

## 10. 모델 배포

학습된 모델을 Ubuntu로 올릴 수 있습니다.

```bash
.venv/bin/python main.py deploy-model \
  --run-version v000010 \
  --host <UBUNTU_HOST> \
  --username <SSH_USER> \
  --remote-model-dir /remote/models
```

주의할 점:

- 배포는 자동 실행되지 않습니다.
- 같은 이름의 파일이 있으면 기본적으로 덮어쓰지 않습니다.
- 업로드 후에는 sniffer를 재시작해야 새 모델이 선택됩니다.

## 11. feature를 고를 수 있나

가능합니다.

기본값은 `pipeline/schema.py`의 `FEATURE_COLUMNS` 전체를 사용하는 것입니다.  
하지만 `config/randomforest.yaml`의 `selected_feature_columns`에 원하는 열을 적으면 그 feature만 학습합니다.

예시:

```yaml
selected_feature_columns:
  - frame_len
  - dns_id
  - ttl_max
```

이렇게 적으면 학습은 선택한 열만 사용합니다.

주의할 점:

- 선택한 이름은 반드시 `FEATURE_COLUMNS` 안에 있어야 합니다.
- 선택하지 않으면 기본적으로 전체 feature를 사용합니다.
- 모델 artifact의 `feature_columns`도 선택한 목록으로 기록됩니다.

원문 도메인/IP 문자열이나 라벨 부여에 직접 쓰인 indicator는
`excluded_feature_columns`로 지정되어 있습니다. 이 목록은
`qname`, 레코드 name/value, `contains_bank_test`, `contains_fake_ip` 계열을
학습에서 배제하기 위한 정책이며, 실수로 선택하려 하면 학습이 중단됩니다.

### 정상/공격 비율 설정

`config/randomforest.yaml`에서 train과 test의 정상:공격 비율을 설정할 수
있습니다. 여러 dataset을 선택한 경우 먼저 split별로 모두 합친 뒤 이 비율을
적용합니다.

```yaml
class_ratio:
  train: "8:2"  # normal:attack
  test: "8:2"   # normal:attack
```

비율 적용은 원본 CSV를 수정하거나 적은 클래스를 복제하지 않습니다. 과다한
클래스에서만 고정 seed로 행을 선택하는 deterministic downsampling입니다.
해당 split의 행을 전부 유지하려면 값을 `null`로 둡니다.

```yaml
class_ratio:
  train: "8:2"
  test: null
```

적용 전/후 라벨 수와 적용한 비율은 `metrics_run_vXXXXXX.json`과
`run_manifest_vXXXXXX.yaml` 또는 scratch manifest에 기록됩니다.

### Threshold와 weight 비교

확정된 run의 모델과 dataset을 기준으로 decision threshold와 class weight를
바꿔가며 confusion matrix를 비교할 수 있습니다.

`config/randomforest.yaml`에서 후보를 조정합니다.

```yaml
decision_thresholds: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
class_weight_grid:
  - null
  - balanced
  - balanced_subsample
  - {0: 1, 1: 2}
  - {0: 1, 1: 4}
```

실행:

```bash
.venv/bin/python main.py compare-thresholds \
  --run-version v000022 \
  --analysis-name threshold_weight_v000022
```

결과는 공식 run을 수정하지 않고 아래에 저장됩니다.

```text
scratch/threshold_weight_v000022/threshold_weight_confusion_matrix.csv
scratch/threshold_weight_v000022/threshold_weight_manifest.json
```

CSV에는 각 조합의 `TN`, `FP`, `FN`, `TP`, `precision`, `recall`, `f1_score`,
`false_positive_rate`, `false_negative_rate`가 들어갑니다.

### Feature 조합 실험

확정된 run의 dataset split, class ratio, RandomForest 설정을 그대로 두고
feature 조합만 바꿔 비교할 수 있습니다. 현재 기준 run은 `v000022`입니다.

```bash
.venv/bin/python main.py feature-experiments \
  --run-version v000022
```

이 명령은 threshold를 `0.3`으로 고정하고, class weight는 `v000022`에서 선택된
베이스라인 값을 그대로 사용합니다. 공식 `runs/run_v000022/` 파일은 수정하지
않고 아래 분석 폴더만 새로 만듭니다.

```text
outputs/feature_experiments/run_v000022/
  config.yaml
  feature_experiment_report.ipynb
  metrics_summary.csv
  confusion_matrices.csv
  feature_importances.csv
  feature_statistics.csv
  probability_summary.csv
  plots/
```

이미 같은 분석 폴더가 있으면 기본적으로 중단합니다. 같은 기준으로 다시 만들려면
명시적으로 덮어씁니다.

```bash
.venv/bin/python main.py feature-experiments \
  --run-version v000022 \
  --overwrite
```

실험 대상은 다음 feature set입니다.

- `all_current`: `v000022` 모델이 실제 사용한 현재 feature set
- `dropped_shortcut_only`: 이전에 drop한 shortcut 의심 feature만 사용
- `top_importance_only`: 현재 중요도가 높았던 feature만 사용
- `dns_structure_only`: DNS flag/count/record/section 구조 기반 feature
- `shortcut_suspects_only`: shortcut 의심 feature 묶음
- `ablate_top__...`: 상위 feature를 하나씩 제거
- `add_back_dropped__...`: 이전에 drop한 feature를 하나씩 예외적으로 재추가

`metrics_summary.csv`에서는 feature set별 `TN`, `FP`, `FN`, `TP`, `precision`,
`recall`, `f1`, `f2`, `false_positive_rate`, `false_negative_rate`를 봅니다.
`feature_statistics.csv`에는 label 기준 및 TP/FN/FP/TN 기준 분포 비교가 들어갑니다.
Mann-Whitney U test p-value, Cohen's d, separation score도 함께 기록됩니다.

그래프는 모두 PNG로 저장됩니다. 특히 먼저 볼 파일은 다음입니다.

- `plots/metrics_comparison.png`
- `plots/probability_distribution.png`
- `plots/confusion_matrix_all_current.png`
- `plots/feature_importance_all_current.png`
- `plots/hist_all_current_*`
- `plots/boxplot_all_current_*`

이 실험에서 이전에 drop한 feature를 다시 써보는 것은 정식 학습 정책 변경이
아닙니다. 통계적으로 TP/FN 또는 TN/FP 분리에 도움이 되는지 확인하기 위한
예외적 분석입니다.

숫자와 그래프를 노트북에서 한 번에 보려면 아래 파일을 엽니다.

```text
outputs/feature_experiments/run_v000022/feature_experiment_report.ipynb
```

이미 생성된 CSV/PNG만 가지고 리포트 노트북을 다시 만들 수도 있습니다. 이 명령은
모델을 다시 학습하지 않습니다.

```bash
.venv/bin/python main.py feature-report \
  --run-version v000022
```

## 12. 수동 학습과 모델 검토

셀 단위로 직접 실험하려면 `notebook/detection.ipynb`가 아니라
`notebook/manual_training.ipynb`를 엽니다. `detection.ipynb`는 공식 run
생성용 템플릿입니다.

수동 노트북의 설정 셀에서는 다음 값을 직접 정할 수 있습니다.

- `OFFICIAL_RUN_VERSION = 'v000002'`: 기존 정식 모델을 읽기 전용으로 확인
- `SCENARIO_SPLIT = True`: 랜덤 split 없이 scenario 단위 평가 사용
- `TRAIN_DATASET_VERSIONS = ['v000001']`: 학습에만 사용할 scenario
- `TEST_DATASET_VERSIONS = ['v000002']`: 평가에만 사용할 scenario
- `SCRATCH_NAME = 'manual_try_001'`: 실험 결과 폴더명
- `FEATURE_COLUMNS_OVERRIDE = ['frame_len', 'dns_id']`: 이 실험만의 피처 선택

scratch 학습도 같은 `config/randomforest.yaml`의 `class_ratio` 설정을
읽습니다.

`OFFICIAL_RUN_VERSION`으로 모델을 확인하는 셀은 기존 `joblib`과 metrics를 읽기만
하며 파일을 변경하지 않습니다. scratch 학습 셀을 실행하면 아래 위치에만
산출물이 생깁니다.

```text
scratch/manual_try_001/randomforest_scratch.joblib
scratch/manual_try_001/metrics_scratch.json
scratch/manual_try_001/confusion_matrix_scratch.csv
scratch/manual_try_001/feature_importance_scratch.csv
scratch/manual_try_001/scratch_manifest.yaml
```

scratch 실행은 `runs/`와 `metrics/metrics_summary.csv`를 수정하지 않습니다.
좋은 실험을 정식 기록으로 확정할 때만 같은 설정을 config에 반영하고
`main.py train`을 실행합니다.

DVC로 관리하는 과거 dataset 또는 정식 모델을 노트북에서 지정하는 것도
가능하지만, 지정한 `features_vXXXXXX.csv` 또는 `runs/run_vXXXXXX` 파일이
현재 작업 폴더에 DVC checkout 되어 있어야 읽을 수 있습니다.

## 13. DVC

이 프로젝트는 `dvc.yaml`과 `params.yaml`만 제공합니다.  
Git/DVC 초기화와 remote 설정은 사용자가 별도로 하면 됩니다.

`params.yaml`에서 다음 값을 바꿔서 사용합니다.

- `dataset_version`
- `training.run_version`
- `training.train_dataset_versions`
- `training.test_dataset_versions`
- `dataset.pcap`
- `dataset.resolver_ip`
- `dataset.capture_scope`

DVC stage는 재현 가능한 출력 경로가 고정되어야 하므로 `training.run_version`을
자동 증가시키지 않고 `params.yaml`에 명시한 값을 사용합니다.
현재 DVC 설정은 `data/processed/` 아래에서 train/test 목록에 지정한 여러 CSV를
입력으로 사용하고, run input manifest에 실제 사용 파일과 checksum을 기록합니다.
필요한 scenario의 processed CSV를 먼저 만든 다음 `dvc repro` 학습 단계를
실행합니다.

## 14. 자주 쓰는 흐름

가장 흔한 흐름은 이렇습니다.

1. pcap 가져오기
2. dataset 만들기
3. 선택 feature 설정
4. run 실행
5. 결과 확인
6. 필요하면 배포

## 15. 추천 실행 예시

```bash
.venv/bin/python main.py fetch-data \
  --dataset-version v000001 \
  --host 192.168.219.10 \
  --username ubuntu \
  --remote-capture-dir /data/captures

.venv/bin/python main.py build-dataset \
  --pcap data/raw/dataset_v000001.pcap \
  --resolver-ip 192.168.219.1 \
  --dataset-version v000001

.venv/bin/python main.py train \
  --train-dataset-version v000001 \
  --test-dataset-version v000002
```

여기까지 하면 run bundle이 완성됩니다.
