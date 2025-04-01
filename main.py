import os
import importlib
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware  # CORS 미들웨어 import

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:3000",
    # "https://example.com",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,        # 요청을 허용할 출처 목록
    allow_credentials=True,       # 쿠키나 인증 정보를 포함할지 여부
    allow_methods=["*"],          # 허용할 HTTP 메서드 (모든 메서드 허용)
    allow_headers=["*"],          # 허용할 HTTP 헤더 (모든 헤더 허용)
)

# routers 폴더의 절대 경로를 가져옵니다.
routers_dir = os.path.join(os.path.dirname(__file__), "routers")

# routers 폴더 내의 모든 .py 파일을 순회합니다.
for filename in os.listdir(routers_dir):
    if filename.endswith(".py") and filename != "__init__.py":
        module_name = filename[:-3]  # 확장자 제거
        module = importlib.import_module(f"routers.{module_name}")
        # 각 모듈에 router 객체가 있다면 include_router()로 추가합니다.
        if hasattr(module, "router"):
            app.include_router(module.router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)


# rm -rf .venv (when.error)
# python3 -m venv .venv
# source .venv/bin/activate
# pip install fastapi uvicorn python-dotenv
# pip install gradio-client
# pip install python-multipart
# pip install boto3
# uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4

