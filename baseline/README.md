# N24News 베이스라인 폴더 안내

## 폴더 구조

- data/: train.json, val.json, test.json, data_report.json 및 전처리·표집·검증 기록
- model/: 실행 코드, requirements, models.joblib, 특징 캐시(features/), 모델 설정과 검증 결과
- model/README_original.md: 이전 사용 안내 원문. 경로는 이 문서를 우선 참조하세요.

## 결과 보기

현재 검증 결과는 model/results.md, 상세 지표는 model/validation_results.json,
기사별 예측은 model/validation_predictions.csv에 있습니다.
데이터 준비 요약은 data/데이터준비결과.md에 있습니다.
기존 모델과 캐시는 이동만 했으며 다시 학습하지 않았습니다. 최종 test 평가는 아직 수행하지 않았습니다.

## 실행 경로

이 README가 있는 baseline 폴더에서 실행합니다.

```powershell
python model/baseline.py --help
```

기존 모델을 최종 평가할 때의 명령(실험 설정 확정 후 사용):

```powershell
python model/baseline.py evaluate-test --run-dir model --data-dir data --device cpu --batch-size 8
```

다른 길이 조건을 새로 학습하는 예시:

```powershell
python model/baseline.py train --run-dir model/runs/body64 --data-dir data --tokens 64 --device cpu --batch-size 8
```

--data-dir는 준비된 JSON 파일이 있는 폴더, --run-dir는 모델·캐시·결과 저장 폴더입니다.
현재 model/에는 학습한 32토큰 모델이 있으므로 동일 run-dir로 train을 재실행하면
기존 모델 보호를 위해 오류가 납니다. 새 실험은 새 run-dir를 사용하세요.

기본 기존 모델(model/)에서는 --data-dir를 생략해도 형제 data/ 폴더를 찾습니다.
예전처럼 데이터와 모델이 같은 run-dir에 있는 구조도 계속 지원합니다.

## 패키지 설치

```powershell
python -m pip install -r model/requirements.txt
```

모든 기사 이미지는 원래 N24News/imgs 경로를 그대로 참조합니다.
이번 폴더 정리에서 데이터 내용과 모델 가중치·캐시는 변경하지 않았습니다.
