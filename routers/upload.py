import os
import uuid
import mimetypes
from datetime import datetime

from fastapi import FastAPI, APIRouter, UploadFile, File, HTTPException
from dotenv import load_dotenv
from datetime import datetime
import boto3

# .env 파일 로드
load_dotenv()

# 환경변수에서 S3 설정 값 불러오기
AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_KEY")
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")
S3_REGION = os.getenv("S3_REGION")

# boto3 S3 클라이언트 생성 (자격증명 및 리전 지정)
s3 = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY,
    region_name=S3_REGION
)

def upload_to_s3(local_file_path: str) -> str:
    """
    로컬 파일을 S3에 업로드하고 공개 접근 가능한 URL을 반환합니다.
    파일명은 uuid와 확장자만 사용합니다.
    """
    ext = os.path.splitext(local_file_path)[1]
    today = datetime.utcnow().strftime("%Y-%m-%d")
    unique_filename = f"{today}/{uuid.uuid4()}{ext}"
    try:
        content_type, _ = mimetypes.guess_type(local_file_path)
        if content_type is None:
            content_type = "application/octet-stream"
        s3.upload_file(
            local_file_path,
            S3_BUCKET_NAME,
            unique_filename,
            ExtraArgs={
                "ContentDisposition": "inline",
                "ContentType": content_type
            }
        )
    except Exception as e:
        raise Exception(f"S3 업로드 실패: {str(e)}")
    s3_url = f"https://{S3_BUCKET_NAME}.s3.{S3_REGION}.amazonaws.com/{unique_filename}"
    return s3_url

router = APIRouter()

@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    이미지 파일을 받아 임시로 저장한 후 S3에 업로드하고 S3 URL을 반환합니다.
    파일명은 uuid와 확장자만 사용하여 한글 파일명으로 인한 문제를 방지합니다.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="이미지 파일만 업로드 가능합니다.")
    
    ext = os.path.splitext(file.filename)[1]
    temp_filename = f"/tmp/{uuid.uuid4()}{ext}"
    
    try:
        with open(temp_filename, "wb") as f:
            f.write(await file.read())
        s3_url = upload_to_s3(temp_filename)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)
    
    return {"url": s3_url}
