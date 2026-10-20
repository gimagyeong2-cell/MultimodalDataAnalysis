# MultimodalDataAnalysis

텍스트 길이에 따른 이미지–텍스트 멀티모달 뉴스 주제 분류와 이미지 기여도 분석.

## 폴더

- `기사 자르기/`: Action Item 1 코드와 실제 8·32·128토큰 예시(`preview.html`, `prefixes.txt`).
- `baseline/data/`: 정제·표집한 학습 2,000개, 검증 500개, 최종 평가용 500개 및 전처리 기록.
- `baseline/model/`: RoBERTa-base / ResNet-50 고정 특징 기반 선형 분류 코드, 학습된 분류기,
  특징 캐시 및 검증 결과.
- `baseline/README.md`: 새 data/model 구조의 실행 안내.

## 검증 세트 예비 결과 — 본문 32토큰

| 모델 | Macro-F1 | Accuracy |
|---|---:|---:|
| 텍스트 단독 | 0.5768 | 60.0% |
| 이미지 단독 | 0.3248 | 35.8% |
| 이미지+텍스트 | 0.5680 | 60.2% |

이미지 추가 기여도는 -0.0088(0~1 척도), 약 -0.88 F1점이다.
설정 선택에 사용한 검증 세트 결과이며, 최종 test 평가는 아직 수행하지 않았다.
원논문 전체 모델의 재현이 아니라 프로포절을 위한 단순한 frozen-feature baseline이다.

## 실행

Python과 PyTorch/torchvision을 지원하는 환경에서 저장소 루트를 기준으로:

```powershell
python -m pip install -r baseline/model/requirements.txt
python baseline/model/baseline.py train --data-dir baseline/data --run-dir baseline/model/runs/new32 --tokens 32 --device cpu --batch-size 8
```

데이터 manifest의 image 경로는 작업한 PC의 N24News/imgs를 참조한다.
다른 PC에서는 공식 데이터셋을 별도로 확보하고 해당 경로를 맞춰야 한다.
모델 가중치와 토크나이저 최초 다운로드에는 인터넷이 필요하다.

## 데이터 및 출처

공식 N24News 배포본에서 표본을 정제·선정했다. 저장소에는 표본 기사 본문,
기사 자르기 HTML에 내장된 예시 이미지, 분류기와 특징 캐시가 포함된다.
전체 N24News 이미지 모음과 사전학습 인코더 가중치는 포함하지 않는다.
데이터·이미지의 이용 및 재배포 조건은 원배포처와 권리자의 조건을 별도로 확인해야 한다.

- Wang et al. (2022), N24News: A New Dataset for Multimodal News Classification, LREC.
  https://aclanthology.org/2022.lrec-1.729/
- 공식 배포처: https://github.com/billywzh717/N24News

코드 작성·전처리·결과 정리에 Codex를 사용했다. 자세한 사용 내역은 프로젝트 보고서에 기록한다.
