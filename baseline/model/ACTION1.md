# Action Item 1: 기사 하나 확인하고 8·32·128토큰으로 자르기

`preview_article.py`와 `baseline.py`를 같은 폴더에 두세요.
이 작업은 GPU가 필요 없고 모델 학습도 하지 않습니다. **RoBERTa 토크나이저**만
사용하므로 모델 가중치를 다운로드하지 않습니다. 최초에는 토크나이저 다운로드용
인터넷이 필요하며, 이후 `--local-files-only`로 캐시를 사용할 수 있습니다.

## 1. 설치

기존 베이스라인 requirements를 설치했다면 추가 설치는 필요 없습니다.
기사 자르기만 먼저 실행하려면 아래의 가벼운 의존성만 설치하세요.

```powershell
python -m pip install -r requirements_action1.txt
```

## 2. 실제 JSON 필드 확인

```powershell
python baseline.py inspect --json "D:/N24News/nytimes_dataset_full.json"
```

본문과 이미지가 실제로 포함된 배포본이 필요합니다.
기본 필드명은 body / headline / category / image_id입니다.
배포본의 필드명이 다르면 아래 옵션으로 지정하세요.

```text
--body-field article_body
--headline-field title
--label-field section
--image-id-field image_id
```

## 3. 기사 하나 자르기

아래 D:/N24News 경로 두 곳을 실제 파일·폴더 위치로 바꿔서 실행하세요.

```powershell
python preview_article.py --json "D:/N24News/nytimes_dataset_full.json" --images "D:/N24News/images" --index 0 --out-dir "action1/article0"
```

- `--index 0`: JSON 리스트의 첫 기사. 두 번째는 1, 세 번째는 2입니다.
- 기본 길이는 8·32·128 **본문 토큰**이며 특수 토큰은 길이에서 제외합니다.
- 다른 길이: `--lengths 8 16 32 64 128`
- 이미지 경로 자체가 저장된 필드라면 `--image-field image_path`를 사용합니다.
  상대 경로는 --images 폴더 기준입니다.
- `{"data": [...]}` 구조는 `--records-key data`를 추가합니다.
- 본문·제목·주제·이미지가 누락된 기사는 오류를 내므로 다른 index를 선택하거나
  필드명을 확인하세요. 자동으로 제목이나 요약을 본문 대신 쓰지 않습니다.
- 결과 폴더가 이미 차 있으면 덮어쓰지 않습니다. 새 out-dir을 지정하세요.

## 4. 결과 확인

| 파일 | 용도 |
|---|---|
| preview.html | 이미지·제목·주제·8/32/128토큰 본문을 함께 확인. 브라우저로 열기 |
| prefixes.txt | 콘솔 출력과 같은 잘린 텍스트. 슬라이드에 복사하기 |
| article.json | 원문, 토큰 ID, 실제 길이, 특수 토큰 포함 길이, 토크나이저 정보 |
| tokenizer/ | 재현을 위한 토크나이저 파일 스냅샷 |

HTML에는 원본 이미지가 내장되어 있으므로 HTML 하나를 복사해도 이미지를 볼 수 있습니다.
화면의 세 카드와 이미지 영역을 캡처해 데이터 소개 슬라이드에 넣을 수 있습니다.
기사와 사진이 포함된 파일의 공유 범위는 데이터 이용 조건에 맞추세요.

본문이 128토큰보다 짧으면 늘리지 않고 실제 길이를 표시합니다. 단어·문장을 기준으로
자르지 않으므로 중간에서 단어가 끝날 수 있습니다. 특히 byte-level 토큰 경계에서는
디코딩 결과 끝에 깨진 문자가 보일 수 있고, 이는 토큰을 임의로 추가해 수정하지 않습니다.
**정확한 모델 입력은 JSON에 저장한 토큰 ID**입니다. 출력된 문장을 다시 토큰화해서
학습하기보다, 기존 baseline.py처럼 원문에서 직접 토큰 길이를 제한하세요.

본문의 바깥 공백만 제거하는 규칙은 기존 베이스라인과 같습니다. 같은 토크나이저
버전에서 잘린 ID에 특수 토큰을 추가한 결과가 베이스라인의 truncation과 같은지
실행 중 확인합니다. 기본 RoBERTa 32토큰 입력은 특수 토큰 포함 34토큰입니다.
512 본문 토큰 예시를 만들 수는 있지만 특수 토큰을 더하면 모델 한도를 초과하므로
JSON과 HTML에 별도 표시합니다. Action Item 1의 8·32·128에는 해당하지 않습니다.

이 스크립트는 기사 하나를 보여주는 작업이며 데이터셋 전체를 자르거나 학습하지 않습니다.
기존 baseline.py의 train 단계는 각 기사의 원문을 지정한 토큰 수로 자동 제한합니다.

## 점검

```powershell
python -m unittest -v test_preview_article.py
```

가짜 토크나이저를 사용하는 다운로드 없는 테스트입니다. 특수 토큰 계산·짧은 원문 처리·
안전한 HTML 출력·파일 생성 경로를 점검하며, 실제 RoBERTa 다운로드 실행을 대체하지 않습니다.

토크나이저 API 참고: https://huggingface.co/docs/transformers/main_classes/tokenizer
