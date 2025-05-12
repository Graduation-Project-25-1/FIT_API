import os
from fastapi import APIRouter, HTTPException, Body
from gradio_client import Client, handle_file
from dotenv import load_dotenv

from .upload import upload_to_s3

load_dotenv()

router = APIRouter()
client = Client("BoyuanJiang/FitDiT", hf_token=os.getenv("HF_TOKEN1"))

@router.post("/dress")
async def dress(
    model_url: str = Body("https://2dfittingroom.s3.ap-northeast-2.amazonaws.com/2025-04-01/2f6810e0-6307-49bf-a7b5-a47e828bf9d8.jpg"),
    dress_url: str = Body("https://2dfittingroom.s3.ap-northeast-2.amazonaws.com/2025-04-01/21452436-7aa7-4037-9ae5-ccc85c31ab78.png"),
    offset_bottom: int = Body(0),
):
    result_s3_url = model_url

    offset_top = 0
    offset_left = 0
    offset_right = 0
    n_steps = 20
    image_scale = 2
    seed = 0
    num_images_per_prompt = 1
    resolution = "768x1024"

    try:
        if offset_bottom < 0 :
            print("[lower]mask start")
            mask_resp = client.predict(
                vton_img=handle_file(result_s3_url),
                category="Lower-body",
                offset_top=offset_top,
                offset_bottom=0,
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
                garm_img=handle_file("https://2dfittingroom.s3.ap-northeast-2.amazonaws.com/2025-05-10/a291fc58-77ce-43f9-9d8b-6850e613453c.png"),
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
                    print("pre : " + result_s3_url)

        print("mask start")

        mask_resp = client.predict(
            vton_img=handle_file(result_s3_url),
            category="Dresses",
            offset_top=0,
            offset_bottom=offset_bottom,
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
            vton_img=handle_file(result_s3_url),
            garm_img=handle_file(dress_url),
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
