import os
from typing import Optional
from fastapi import APIRouter, HTTPException, Body
from gradio_client import Client, handle_file
from dotenv import load_dotenv

from .upload import upload_to_s3

load_dotenv()

router = APIRouter()
client = Client("BoyuanJiang/FitDiT", hf_token=os.getenv("HF_TOKEN"))

@router.post("/sum")
async def sum(
    model_url: str = Body("https://boyuanjiang-fitdit.hf.space/gradio_api/file=/tmp/gradio/555ddf7f9160ff20b7429dbb65f9c2167a4798fa756a8c8e7cef87d8cf5abef0/0223.jpg"),
    upper_url: Optional[str] = Body(default=None),
    upper_offset_bottom: int = Body(0),
    lower_url: Optional[str] = Body(default=None),
    lower_offset_bottom: int = Body(0),
):
    if not upper_url and not lower_url:
        raise HTTPException(status_code=400, detail="upper_url 또는 lower_url 중 하나는 필수입니다.")

    offset_top = 0
    offset_left = 0
    offset_right = 0
    n_steps = 20
    image_scale = 2
    seed = -1
    num_images_per_prompt = 1
    resolution = "768x1024"

    result_s3_url = model_url

    # UPPER
    if upper_url:
        try:
            print("[upper]mask start")
            mask_resp = client.predict(
                vton_img=handle_file(result_s3_url),
                category="Upper-body",
                offset_top=offset_top,
                offset_bottom=upper_offset_bottom,
                offset_left=offset_left,
                offset_right=offset_right,
                api_name="/generate_mask"
            )
            print("[upper]mask end")

            pre_mask, pose_image_resp = mask_resp

            processed_pre_mask = {
                "background": handle_file(pre_mask["background"]) if pre_mask.get("background") else None,
                "layers": [handle_file(layer) for layer in pre_mask.get("layers", [])],
                "composite": handle_file(pre_mask["composite"]) if pre_mask.get("composite") else None,
                "id": pre_mask.get("id")
            }

            processed_pose = handle_file(pose_image_resp["path"]) if isinstance(pose_image_resp, dict) and "path" in pose_image_resp else handle_file(pose_image_resp)

            print("[upper]process start")
            result = client.predict(
                vton_img=handle_file(result_s3_url),
                garm_img=handle_file(upper_url),
                pre_mask=processed_pre_mask,
                pose_image=processed_pose,
                n_steps=n_steps,
                image_scale=image_scale,
                seed=seed,
                num_images_per_prompt=num_images_per_prompt,
                resolution=resolution,
                api_name="/process"
            )
            print("[upper]process end")

            if result and isinstance(result, list) and "image" in result[0]:
                img_obj = result[0]["image"]
                local_img_path = img_obj.get("path") if isinstance(img_obj, dict) else img_obj
                if local_img_path and os.path.exists(local_img_path):
                    result_s3_url = upload_to_s3(local_img_path)

            if result_s3_url == model_url:
                raise Exception("[upper] 결과 이미지 업로드 실패 또는 결과가 없습니다.")
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    # LOWER
    if lower_url:
        try:
            print("[lower]mask start")
            mask_resp = client.predict(
                vton_img=handle_file(result_s3_url),
                category="Lower-body",
                offset_top=offset_top,
                offset_bottom=lower_offset_bottom,
                offset_left=offset_left,
                offset_right=offset_right,
                api_name="/generate_mask"
            )
            print("[lower]mask end")

            pre_mask, pose_image_resp = mask_resp

            processed_pre_mask = {
                "background": handle_file(pre_mask["background"]) if pre_mask.get("background") else None,
                "layers": [handle_file(layer) for layer in pre_mask.get("layers", [])],
                "composite": handle_file(pre_mask["composite"]) if pre_mask.get("composite") else None,
                "id": pre_mask.get("id")
            }

            processed_pose = handle_file(pose_image_resp["path"]) if isinstance(pose_image_resp, dict) and "path" in pose_image_resp else handle_file(pose_image_resp)

            print("[lower]process start")
            result = client.predict(
                vton_img=handle_file(result_s3_url),
                garm_img=handle_file(lower_url),
                pre_mask=processed_pre_mask,
                pose_image=processed_pose,
                n_steps=n_steps,
                image_scale=image_scale,
                seed=seed,
                num_images_per_prompt=num_images_per_prompt,
                resolution=resolution,
                api_name="/process"
            )
            print("[lower]process end")

            if result and isinstance(result, list) and "image" in result[0]:
                img_obj = result[0]["image"]
                local_img_path = img_obj.get("path") if isinstance(img_obj, dict) else img_obj
                if local_img_path and os.path.exists(local_img_path):
                    result_s3_url = upload_to_s3(local_img_path)

            if result_s3_url == model_url:
                raise Exception("[lower] 결과 이미지 업로드 실패 또는 결과가 없습니다.")
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    return {"url": result_s3_url}
