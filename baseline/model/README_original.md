# N24News 프로포절용 베이스라인

**Action Item 1(기사 하나를 8·32·128토큰으로 자르기)**부터 진행하려면
`ACTION1.md`를 읽고 `preview_article.py`를 실행하세요. CPU에서 토크나이저만 사용합니다.

**실제 배포본에 news/nytimes_train.json, nytimes_dev.json, nytimes_test.json과
imgs/가 있는 경우:** 기존 분할을 유지하는 `prepare_official.py`를 사용하세요.
아래의 일반 `baseline.py prepare`는 새 분할이 필요한 경우를 위한 별도 경로입니다.

```powershell
python prepare_official.py --data-root "D:/N24News" --out-dir "runs/official_3000"
python baseline.py train --run-dir "runs/official_3000" --tokens 32 --device cpu
```

prepare_official.py는 학습 2,000·검증 500·평가 500개를 기본으로 선정합니다.
실제 article/section 필드를 사용하며, 본문 앞의 알려진 사진 설명·제목을 조건부 제거합니다.
분할을 넘나드는 중복 그룹은 전부 제외하고, 최종 이미지 디코딩 및 중복 검사를 수행합니다.
out-dir에는 데이터준비결과.md, data_report.json, cleaning_examples.json과 각 분할이 저장됩니다.
새 길이 조건도 이 분할을 복사해 사용하세요. 원본 기사와 이미지 파일은 수정하지 않습니다.

같은 기사 본문의 앞 **32개 토큰**을 사용하는 텍스트 단독 모델과,
이미지 특징을 추가한 모델의 검증 Macro-F1을 비교합니다.

## 모델 구성

| 조건 | 특징 | 분류기 |
|---|---|---|
| text_only | 고정 RoBERTa-base의 CLS 벡터 | 표준화 + 로지스틱 회귀 |
| image_only | 고정 ResNet-50의 global average pooling 벡터 | 표준화 + 로지스틱 회귀 |
| early_fusion | 위 두 특징 벡터를 연결 | 표준화 + 로지스틱 회귀 |
| majority | 학습 세트에서 가장 많은 클래스 | 다수 클래스 예측 |

인코더는 학습하지 않고 eval 모드로 특징만 추출합니다. 표준화 통계도 학습 세트에서만
계산합니다. 분류기 C는 같은 후보 [0.01, 0.1, 1]에서 검증 Macro-F1으로 선택합니다.
class_weight는 사용하지 않습니다. 세 분류 모델 모두 동일 기사·분할을 사용합니다.

**원논문의 재현 모델은 아닙니다.** 원논문은 ResNet-50+ViT 하이브리드 이미지 인코더,
RoBERTa, 세 MLP 및 세 분류 손실을 사용합니다. 이 코드는 프로포절 전에 첫 숫자를
얻기 위한 frozen-feature linear baseline입니다. 고정 CLS의 성능이 낮더라도
이미지가 원래 무용하다는 결론으로 일반화하지 마세요.

## 1. 설치

Python 3.10~3.12의 새 환경 또는 GPU를 사용하는 Google Colab을 권장합니다.
아래 명령은 이 README와 baseline.py가 있는 폴더에서 실행하세요.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

PyTorch와 torchvision은 호환되는 버전을 함께 설치해야 합니다. CUDA 설치가 필요하면
https://pytorch.org/get-started/locally/ 에서 환경에 맞는 설치 명령을 사용하세요.
CPU에서도 동작하지만 특징 추출 시간이 더 걸립니다.
최초 학습은 Hugging Face와 PyTorch에서 모델 가중치를 다운로드하므로 인터넷과
다운로드 저장 공간이 필요합니다. 데이터는 별도로 확보해야 합니다.

## 2. 실제 JSON 필드 확인

공식 배포 압축본의 **본문과 이미지가 포함된 버전**을 사용하세요.
이 코드는 뉴스 웹사이트를 크롤링하지 않습니다.

```powershell
python baseline.py inspect --json "D:/N24News/nytimes_dataset_full.json"
```

기본적으로 아래 형태의 JSON 리스트를 예상합니다. 실제 필드 이름은 inspect로 확인하세요.
아래는 형식 설명용 예시이며 실제 기사나 실험 결과가 아닙니다.

```json
[
  {"category": "Sports", "body": "Article body ...", "image_id": "123"}
]
```

위 레코드의 이미지는 `images/123.jpg` 등으로 찾습니다. jpg/jpeg/png/webp를 지원합니다.
본문이 없으면 제목이나 요약으로 자동 대체하지 않습니다.

## 3. 기사 검증·중복 제거·분할 고정

```powershell
python baseline.py prepare --json "D:/N24News/nytimes_dataset_full.json" --images "D:/N24News/images" --run-dir "runs/body32_seed42" --sample-size 3000 --seed 42
```

기본 필드와 다르면 다음 옵션을 덧붙이세요.

```text
--label-field section
--body-field article_body
--image-id-field image_id
--image-field image_path
--records-key data
```

- `--image-field`는 이미지 ID 대신 이미지 경로 필드를 사용할 때 지정합니다.
  상대 경로는 `--images`를 기준으로 해석합니다.
- `--records-key`는 최상위가 리스트가 아니라 `{"data": [...]}` 형식일 때 지정합니다.
  최상위가 split별 객체라면 임의로 합치지 말고 기존 분할과 데이터 구조부터 확인하세요.
- 이미지·본문 누락, 손상 이미지와 정확히 중복되는 기사를 제외합니다.
- 본문 공백·대소문자 정규화 해시, 디코딩한 이미지 픽셀 해시, 기사 URL/ID로 중복을
  확인합니다. 유사 기사·크롭·재인코딩 사진까지 탐지하는 근접 중복 검사는 아닙니다.
- 같은 이미지가 여러 기사에 쓰였으면 첫 유효 기사만 유지합니다. 데이터 구성이
  달라질 수 있으므로 제외 개수와 클래스별 개수를 보고하세요.
- 클래스 비율에 가깝게 3,000개를 표집하고, 클래스별 약 **60/20/20**으로 학습/검증/평가를
  나눕니다. 반올림 때문에 개수는 조금 달라질 수 있습니다.
- 이는 원논문의 80/10/10과 다른 예비 실험용 분할입니다.
- 전체 24개 클래스가 유효하게 존재하는지 확인하며, 각 클래스 최소 5개가 필요합니다.
  이 최소값은 실행 조건일 뿐 통계적으로 충분하다는 의미가 아닙니다.
- `--sample-size 0`은 유효한 전체 데이터를 사용합니다.
- 기존 분할 덮어쓰기를 방지하기 위해 새/빈 run-dir만 허용합니다.

출력: `train.json`, `val.json`, `test.json`, `data_report.json`.
`data_report.json`에 원본·유효·실제 사용 개수, 제외 사유, 클래스별/분할별 개수가 저장됩니다.
여기에 원문 전체가 담긴 manifest가 생기므로 데이터 이용 조건에 따라 관리하세요.
표본 실험 결과를 데이터셋 전체의 확정적인 결론으로 보고하지 마세요.

## 4. 32토큰 첫 실험

```powershell
python baseline.py train --run-dir "runs/body32_seed42" --tokens 32 --batch-size 16
```

GPU 메모리가 부족하면 `--batch-size 4`, CPU를 사용하려면 `--device cpu`를 추가하세요.
분류기는 CPU에서 학습합니다. 특징은 features 폴더에 캐시합니다.

토큰 길이는 **본문 토큰만** 셉니다. 기본 32토큰에는 특수 토큰 2개가 추가되어
모델 입력은 최대 34토큰입니다. 짧은 본문은 실제 길이만 사용하고 배치 내 패딩을 적용합니다.
기본 RoBERTa는 총 입력 한도가 512이므로 본문은 최대 510토큰까지 지정할 수 있습니다.
`--tokens 512`는 조용히 자르지 않고 오류를 냅니다. 학기 중 512 조건을 정할 때는
"특수 토큰 포함 512(본문 510)"으로 명시하거나 더 긴 입력을 지원하는 인코더로 모든
길이 조건을 일관되게 바꾸세요. 32토큰 실험에는 이 문제가 없습니다.

저장되는 결과:

- `results.md`: 발표에 옮기기 쉬운 성능표
- `validation_results.json`: Macro-F1, Accuracy, 클래스별 지표, C 후보별 점수, 이미지 기여도
- `validation_predictions.csv`: 기사별 정답과 네 모델의 예측
- `models.joblib`: 학습된 분류기와 표준화 파라미터
- `model_settings.json`: 정확한 Hugging Face 커밋, 모델·길이·특징 설정
- `environment.json`: 실행한 패키지 버전

Macro-F1/Accuracy는 **0~1 값**, `delta_pp`는 차이를 100배한 점수입니다.
예: 0.02 차이면 0~100 척도에서 +2.00점입니다. 실제 숫자는 실행 후 생성됩니다.
이 단계는 검증 세트를 C 선택에 썼으므로 **"검증 세트 예비 결과"**로 발표하세요.
test.json은 train 명령이 읽거나 평가하지 않습니다.
학습 시 ConvergenceWarning이 나오면 수렴한 결과로 단정하지 말고 max-iter를 늘려
새 run에서 다시 실행하세요. 기본 모델에는 원논문의 보조 손실과 미세조정이 없습니다.

## 5. 최종 평가 — 실험 설정을 확정한 뒤

```powershell
python baseline.py evaluate-test --run-dir "runs/body32_seed42"
```

기존 분류기를 그대로 사용하며 다시 학습하거나 test에서 C를 고르지 않습니다.
`test_results.json`, `test_predictions.csv`가 생성됩니다. 이미 test 결과가 있으면
재평가를 막습니다. 다른 run을 만들어 같은 test를 반복 관찰하는 것도 피하세요.
여러 길이를 비교하는 최종 실험에서는 **모든 설정을 검증 세트로 확정한 후** 같은 holdout에서
미리 정한 전체 실험을 평가해야 합니다.

## 여러 길이 비교

기사 목록과 분할을 같게 유지하려면 원래 run의 train.json/val.json/test.json과
data_report.json을 새 빈 run으로 복사한 뒤 --tokens만 바꿔 train을 실행하세요.
원본 run의 모델이나 결과를 덮어쓰지 마세요. 여러 시드에서는 별도로 prepare를 실행합니다.

## 다운로드 없는 검증

```powershell
python -m unittest -v test_baseline.py
```

합성 이미지/레코드로 중복 제거, 결측 제외, 클래스 유지, 분할 간 누수, 시드 재현성을
검증합니다. 이 검사는 실제 모델 성능 검증이나 실데이터 학습을 대체하지 않습니다.

## 참고 자료 및 사용 기록

- 원논문: https://aclanthology.org/2022.lrec-1.729/
- 데이터 공식 배포처: https://github.com/billywzh717/N24News
- RoBERTa 문서: https://huggingface.co/docs/transformers/model_doc/roberta
- ResNet-50 문서: https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet50.html

데이터/이미지/코드의 이용 조건은 별도로 확인하세요. 이 폴더에 데이터와 가중치는
포함되어 있지 않습니다. 공식 모델 코드를 복제한 것이 아니라 설명에 따라 작성한
독립적인 단순 베이스라인입니다. 본 코드는 Codex와 함께 작성했습니다.
팀은 실제 실행·수정 내역과 사용 프롬프트를 기록하고, 최종 보고서 AI Use 절에 기재하세요.
