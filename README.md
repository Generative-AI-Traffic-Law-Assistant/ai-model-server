# ai-model-server
### 라이브러리 설치
서버 실행 전에 필요한 라이브러리 설치를 위해 다음 명령을 실행합니다.
```bash
pip install -r requirements.txt
```
### API Key 설정
OpenAI의 API를 이용하여 챗봇이 구성되기 때문에 환경 변수를 설정해야 합니다.

`.env`에 다음과 같이 설정합니다.
```
OPENAI_API_KEY=발급받은 API KEY
```
`openai_api.py` 파일에 의해 API KEY를 외부에 노출하지 않고 안전하게 보관할 수 있습니다.
```python
load_dotenv()

def get_openai_api_key():
    return os.getenv("OPENAI_API_KEY")
```
### FastAPI 서버 실행
```bash
uvicorn app:app --reload
```
### 서버 엔드포인트
#### 이미지 설명 텍스트 생성
- HTTP method : `POST`
- URL : `/ask`
- Request
```json
{
  "question": "도로교통법에 대해 알려줘"
}
```
- Response
```json
{
    "answer": {
        "query": "도로교통법에 대해 설명해줘",
        "result": "도로교통법은 도로에서 발생하는 교통사고를 ~~~"
    }
}
```
#### 이미지 설명 텍스트 생성
- HTTP method : `POST`
- URL : `/generate_text`
- Request
```json
{
  "Image_path": "이미지 링크",
  "additional_info": "예시 정보"
}
```
- Response
```json
{
    "description": "킥보드 탑승자가 신호 위반"
}
```
