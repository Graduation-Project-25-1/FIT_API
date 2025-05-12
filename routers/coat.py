import os
import cv2
import numpy as np
import torch
import mediapipe as mp
import asyncio
import cv2.ximgproc as xiproc
from typing import Optional
from fastapi import APIRouter, HTTPException, Body
from gradio_client import Client, handle_file
from segment_anything import sam_model_registry, SamPredictor
from dotenv import load_dotenv

from .upload import upload_to_s3

load_dotenv()

router = APIRouter()
client = Client("BoyuanJiang/FitDiT", hf_token=os.getenv("HF_TOKEN1"))

# SAM 모델 한 번만 로드
SAM_CHECKPOINT  = os.getenv("SAM_CHECKPOINT", "./checkpoints/sam_vit_h.pth")
DEVICE    = "cpu"
sam_model = sam_model_registry["vit_h"](checkpoint=SAM_CHECKPOINT)
sam_model.to(DEVICE)
predictor = SamPredictor(sam_model)

def process_virtual_fitting(outer_path, inner_path, output_path):
    # 이미지 로드 및 크기 맞춤
    outer_img = cv2.imread(outer_path)
    inner_img = cv2.imread(inner_path)
    if inner_img.shape[:2] != outer_img.shape[:2]:
        inner_img = cv2.resize(inner_img, (outer_img.shape[1], outer_img.shape[0]))
    h, w = outer_img.shape[:2]

    # 살색 마스크 생성
    ycrcb = cv2.cvtColor(outer_img, cv2.COLOR_BGR2YCrCb)
    hsv = cv2.cvtColor(outer_img, cv2.COLOR_BGR2HSV)
    mask_ycrcb = cv2.inRange(ycrcb, np.array([95, 140, 105]), np.array([210, 165, 125]))
    mask_hsv = cv2.inRange(hsv, np.array([3, 30, 110]), np.array([20, 120, 240]))
    skin_mask = cv2.bitwise_or(mask_ycrcb, mask_hsv)

    # 손 제거
    rgb_img = cv2.cvtColor(outer_img, cv2.COLOR_BGR2RGB)
    hands = mp.solutions.hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5)
    results = hands.process(rgb_img)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for lm in hand_landmarks.landmark:
                x, y = int(lm.x * w), int(lm.y * h)
                cv2.circle(skin_mask, (x, y), 30, 0, -1)

    # 중심부 마스크 추출
    contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    focused_mask = np.zeros_like(skin_mask)
    cv2.drawContours(focused_mask, [max(contours, key=cv2.contourArea)], -1, 255, thickness=cv2.FILLED)
    inverse_mask = cv2.bitwise_not(focused_mask)

    # 침식 → outer 영역 추출
    kernel = np.ones((15, 3), np.uint8)
    eroded_outer_mask = cv2.erode(inverse_mask, kernel, iterations=1)
    eroded_result = cv2.bitwise_and(outer_img, outer_img, mask=eroded_outer_mask)

    # inner 중심부 추출 및 합성
    inverse_eroded_mask = cv2.bitwise_not(eroded_outer_mask)
    inner_foreground = cv2.bitwise_and(inner_img, inner_img, mask=inverse_eroded_mask)
    composite_result = cv2.add(eroded_result, inner_foreground)

    # 경계 블렌딩
    edge = cv2.Canny(inverse_eroded_mask, 100, 200)
    edge = cv2.dilate(edge, np.ones((1, 1), np.uint8), iterations=1)
    soft_mask = cv2.GaussianBlur(edge.astype(np.float32), (17, 17), 0) / 255.0
    feather_mask = np.where(inverse_eroded_mask == 255, 1.0, soft_mask)
    smoothed_result = (feather_mask[..., None] * inner_img + (1 - feather_mask[..., None]) * eroded_result).astype(np.uint8)
    return smoothed_result
    
@router.post("/coat")
async def coat(
    model_url: str = Body(default="https://2dfittingroom.s3.ap-northeast-2.amazonaws.com/2025-04-01/2f6810e0-6307-49bf-a7b5-a47e828bf9d8.jpg"),
    coat_url: str = Body(default="https://2dfittingroom.s3.ap-northeast-2.amazonaws.com/2025-05-04/b6d6ef83-b04e-4b80-baee-a047f77e53fe.webp"),
    inner_url: Optional[str] = Body(default="https://2dfittingroom.s3.ap-northeast-2.amazonaws.com/2025-04-01/427ca040-f318-4a9b-9258-7a7820b32f3f.png")
):
    try:
        # ── 1) Coat & Inner 마스크 생성을 비동기로 병렬 실행 ──
        print("mask start")
        coat_mask_task = asyncio.to_thread(
            client.predict,
            vton_img=handle_file(model_url),
            category="Upper-body",
            offset_top=0,
            offset_bottom=200,
            offset_left=0,
            offset_right=0,
            api_name="/generate_mask"
        )
        inner_mask_task = asyncio.to_thread(
            client.predict,
            vton_img=handle_file(model_url),
            category="Upper-body",
            offset_top=0,
            offset_bottom=0,
            offset_left=0,
            offset_right=0,
            api_name="/generate_mask"
        )
        coat_mask_resp, inner_mask_resp = await asyncio.gather(coat_mask_task, inner_mask_task)
        print("mask end")

        # Coat 마스크 후처리
        coat_pre_mask, coat_pose_resp = coat_mask_resp
        coat_processed_pre_mask = {
            "background": handle_file(coat_pre_mask["background"]) if coat_pre_mask.get("background") else None,
            "layers": [handle_file(layer) for layer in coat_pre_mask.get("layers", [])],
            "composite": handle_file(coat_pre_mask["composite"]) if coat_pre_mask.get("composite") else None,
            "id": coat_pre_mask.get("id")
        }
        coat_processed_pose = (
            handle_file(coat_pose_resp["path"]) if isinstance(coat_pose_resp, dict) and "path" in coat_pose_resp
            else handle_file(coat_pose_resp)
        )

        # Inner 마스크 후처리
        inner_pre_mask, inner_pose_resp = inner_mask_resp
        inner_processed_pre_mask = {
            "background": handle_file(inner_pre_mask["background"]) if inner_pre_mask.get("background") else None,
            "layers": [handle_file(layer) for layer in inner_pre_mask.get("layers", [])],
            "composite": handle_file(inner_pre_mask["composite"]) if inner_pre_mask.get("composite") else None,
            "id": inner_pre_mask.get("id")
        }
        inner_processed_pose = (
            handle_file(inner_pose_resp["path"]) if isinstance(inner_pose_resp, dict) and "path" in inner_pose_resp
            else handle_file(inner_pose_resp)
        )

        # ── 2) Coat & Inner 프로세스도 비동기로 병렬 실행 ──
        print("process start")
        coat_process_task = asyncio.to_thread(
            client.predict,
            vton_img=handle_file(model_url),
            garm_img=handle_file(coat_url),
            pre_mask=coat_processed_pre_mask,
            pose_image=coat_processed_pose,
            n_steps=20,
            image_scale=2,
            seed=-1,
            num_images_per_prompt=1,
            resolution="768x1024",
            api_name="/process"
        )
        inner_process_task = asyncio.to_thread(
            client.predict,
            vton_img=handle_file(model_url),
            garm_img=handle_file(inner_url),
            pre_mask=inner_processed_pre_mask,
            pose_image=inner_processed_pose,
            n_steps=20,
            image_scale=2,
            seed=-1,
            num_images_per_prompt=1,
            resolution="768x1024",
            api_name="/process"
        )
        coat_result, inner_result = await asyncio.gather(coat_process_task, inner_process_task)
        print("process end")

        # ── 3) 결과를 로컬 경로로 추출 & S3 업로드 ──
        print("coat_s3")
        coat_local = None
        if coat_result and isinstance(coat_result, list) and "image" in coat_result[0]:
            img_obj = coat_result[0]["image"]
            coat_local = img_obj.get("path") if isinstance(img_obj, dict) else img_obj
            temp_url1 = upload_to_s3(coat_local)
            print("coat : " + temp_url1)

        print("inner_s3")
        inner_local = None
        if inner_result and isinstance(inner_result, list) and "image" in inner_result[0]:
            img_obj = inner_result[0]["image"]
            inner_local = img_obj.get("path") if isinstance(img_obj, dict) else img_obj
            temp_url1 = upload_to_s3(inner_local)
            print("inner : " + temp_url1)

        print("openCV")
        # ── 4) OpenCV + SAM 레이어드 블렌딩 (아래 코드는 제공하신 스크립트 그대로) ──
        final_local = "final_output_full_soft_mask.png"
        final_result = process_virtual_fitting(coat_local, inner_local, final_local)

        cv2.imwrite(final_local, final_result)

        # ── 5) 최종 이미지 S3 업로드 및 URL 반환 ──
        result_s3_url = upload_to_s3(final_local)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {"url": result_s3_url}
