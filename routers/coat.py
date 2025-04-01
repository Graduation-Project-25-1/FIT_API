import os
from fastapi import APIRouter, HTTPException, Body
from gradio_client import Client, handle_file
from dotenv import load_dotenv

from .upload import upload_to_s3

load_dotenv()

router = APIRouter()
client = Client("BoyuanJiang/FitDiT", hf_token=os.getenv("HF_TOKEN"))

@router.post("/coat")
async def coat(
    model_url: str = Body("https://boyuanjiang-fitdit.hf.space/gradio_api/file=/tmp/gradio/555ddf7f9160ff20b7429dbb65f9c2167a4798fa756a8c8e7cef87d8cf5abef0/0223.jpg"),
    coat_url: str = Body("https://boyuanjiang-fitdit.hf.space/gradio_api/file=/tmp/gradio/b07e27aebc6ef0984e68a91d8f0b54e415ccfda1ad81c3a3cc8bd7738fdd34b3/clipboard.png"),
):
    try:
        print("mask start")

        mask_resp = client.predict(
            vton_img=handle_file(model_url),
            category="Upper-body",
            offset_top=0,
            offset_bottom=200,
            offset_left=0,
            offset_right=0,
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

        processed_pose = handle_file(pose_image_resp["path"]) if isinstance(pose_image_resp, dict) and "path" in pose_image_resp else handle_file(pose_image_resp)

        print("process start")

        result = client.predict(
            vton_img=handle_file(model_url),
            garm_img=handle_file(coat_url),
            pre_mask=processed_pre_mask,
            pose_image=processed_pose,
            n_steps=20,
            image_scale=2,
            seed=-1,
            num_images_per_prompt=1,
            resolution="768x1024",
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
