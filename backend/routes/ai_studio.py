import asyncio
import os
import base64
import urllib.parse
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter(prefix="/api", tags=["AI Studio"])


def _require_token() -> str:
    token = os.environ.get("REPLICATE_API_TOKEN", "")
    if not token or token.startswith("your_"):
        raise HTTPException(
            status_code=500,
            detail="REPLICATE_API_TOKEN not set. Add it to backend/.env and restart the server.",
        )
    return token


class ImageGenRequest(BaseModel):
    prompt: str
    num_outputs: int = 2
    aspect_ratio: str = "1:1"


class VideoGenRequest(BaseModel):
    prompt: str


class RemoveObjectRequest(BaseModel):
    image: str  # data URI of original image
    mask: str   # data URI of user-drawn mask (white = erase, black = keep)


# ──────────────────────────────────────────────────────────────────────────────
# POST /api/generate-image
# Uses Pollinations.ai (free, no key, FLUX model).
# Backend FETCHES the images so the browser gets base64 — displays instantly.
# ──────────────────────────────────────────────────────────────────────────────
@router.post("/generate-image")
async def generate_image(req: ImageGenRequest):
    import httpx

    dims = {
        "1:1":  (1024, 1024),
        "16:9": (1344, 768),
        "9:16": (768,  1344),
        "4:3":  (1152, 896),
        "3:4":  (896,  1152),
    }
    w, h = dims.get(req.aspect_ratio, (1024, 1024))
    encoded = urllib.parse.quote(req.prompt)
    count   = min(max(req.num_outputs, 1), 4)

    async def fetch_one(seed: int) -> str:
        url = (
            f"https://image.pollinations.ai/prompt/{encoded}"
            f"?width={w}&height={h}&seed={seed}&model=flux&nologo=true&enhance=false"
        )
        async with httpx.AsyncClient(timeout=120.0, follow_redirects=True) as client:
            resp = await client.get(url)
        if resp.status_code != 200:
            return ""
        ct = resp.headers.get("content-type", "image/jpeg").split(";")[0].strip()
        b64 = base64.b64encode(resp.content).decode()
        return f"data:{ct};base64,{b64}"

    results = await asyncio.gather(*[fetch_one(i * 7919) for i in range(count)])
    images  = [r for r in results if r]

    if not images:
        raise HTTPException(status_code=500, detail="Pollinations.ai returned no images — try again.")

    return {"success": True, "images": images}


# ──────────────────────────────────────────────────────────────────────────────
# POST /api/generate-video
# Uses minimax/video-01 via Replicate (~6 s clip, 720p, takes 2-5 min).
# ──────────────────────────────────────────────────────────────────────────────
@router.post("/generate-video")
async def generate_video(req: VideoGenRequest):
    token = _require_token()

    def _run() -> str:
        import replicate
        client = replicate.Client(api_token=token)
        output = client.run(
            "minimax/video-01",
            input={"prompt": req.prompt, "prompt_optimizer": True},
        )
        first = output[0] if isinstance(output, list) else output
        return str(first.url) if hasattr(first, "url") else str(first)

    try:
        video_url = await asyncio.to_thread(_run)
        return {"success": True, "video_url": video_url}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ──────────────────────────────────────────────────────────────────────────────
# POST /api/remove-object
# Accepts user-drawn mask (white = erase, black = keep).
# Calls SDXL Inpainting directly — SAM step removed for speed (~20-30 s total).
# ──────────────────────────────────────────────────────────────────────────────
@router.post("/remove-object")
async def remove_object(req: RemoveObjectRequest):
    token = _require_token()

    def _run() -> str:
        import replicate
        client = replicate.Client(api_token=token)
        output = client.run(
            "stability-ai/stable-diffusion-inpainting:95b7223104132402a9ae91cc677285bc5eb997834bd2349fa486f53910fd68b3",
            input={
                "prompt": "clean background, seamless fill, natural texture, photorealistic",
                "negative_prompt": "blurry, distorted, artifacts, unnatural, duplicate",
                "image": req.image,
                "mask":  req.mask,
                "num_outputs": 1,
                "guidance_scale": 8,
                "num_inference_steps": 30,
            },
        )
        items = list(output)
        if not items:
            raise ValueError("Inpainting returned no output")
        item = items[0]
        return str(item.url) if hasattr(item, "url") else str(item)

    try:
        result_url = await asyncio.to_thread(_run)
        return {"success": True, "result_url": result_url}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
