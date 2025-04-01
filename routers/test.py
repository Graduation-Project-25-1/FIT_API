import os
from fastapi import APIRouter, HTTPException, Body
from gradio_client import Client, handle_file
from dotenv import load_dotenv

from .upload import upload_to_s3

load_dotenv()

router = APIRouter()
client = Client("BoyuanJiang/FitDiT")

@router.post("/combined_process")
async def combined_process(
    model_url: str = Body(...),
    cloth_url: str = Body(...),
    category: str = "Upper-body",
    offset_top: float = 0,
    offset_bottom: float = 0,
    offset_left: float = 0,
    offset_right: float = 0,
    n_steps: int = 20,
    image_scale: float = 2,
    seed: int = -1,
    num_images_per_prompt: int = 1,
    resolution: str = "768x1024"
):
    try:
        print("mask start")
        mask_resp = client.predict(
            vton_img=handle_file(model_url),
            category=category,
            offset_top=offset_top,
            offset_bottom=offset_bottom,
            offset_left=offset_left,
            offset_right=offset_right,
            api_name="/generate_mask"
        )
        print("mask end")

        pre_mask, pose_image_resp = mask_resp

        processed_pre_mask = {
            "background": handle_file(pre_mask["background"]) if pre_mask.get("background") else None,
            "layers": [handle_file(layer) for layer in pre_mask.get("layers", [])],
            "composite": handle_file(pre_mask["composite"]) if pre_mask.get("composite") else None,
            "id": pre_mask.get("id")
        }

        if isinstance(pose_image_resp, dict) and "path" in pose_image_resp:
            processed_pose = handle_file(pose_image_resp["path"])
        else:
            processed_pose = handle_file(pose_image_resp)

        print("process start")
        result = client.predict(
            vton_img=handle_file(model_url),
            garm_img=handle_file(cloth_url),
            pre_mask=processed_pre_mask,
            pose_image=processed_pose,
            n_steps=n_steps,
            image_scale=image_scale,
            seed=seed,
            num_images_per_prompt=num_images_per_prompt,
            resolution=resolution,
            api_name="/process"
        )
        print("process end")

        result_s3_url = None
        if result and isinstance(result, list) and "image" in result[0]:
            img_obj = result[0]["image"]
            local_img_path = img_obj.get("path") if isinstance(img_obj, dict) else img_obj

            if local_img_path and os.path.exists(local_img_path):
                result_s3_url = upload_to_s3(local_img_path)

        if not result_s3_url:
            raise Exception("결과 이미지 업로드 실패 또는 결과가 없습니다.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {"url": result_s3_url}
