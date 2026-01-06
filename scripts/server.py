# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os, sys
import io
import time
import numpy as np
from contextlib import asynccontextmanager

code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f"{code_dir}/../")

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import Response
import cv2
import torch
from omegaconf import OmegaConf
from core.utils.utils import InputPadder
from Utils import *
from core.foundation_stereo import *

# Global model instance
model = None
model_args = None


def load_model_at_startup(ckpt_dir: str):
    """Load model once at server startup"""
    global model, model_args

    cfg = OmegaConf.load(f"{os.path.dirname(ckpt_dir)}/cfg.yaml")
    if "vit_size" not in cfg:
        cfg["vit_size"] = "vitl"

    model_args = OmegaConf.create(cfg)
    model = FoundationStereo(model_args)

    ckpt = torch.load(ckpt_dir)
    logging.info(f"Loading checkpoint: global_step={ckpt['global_step']}, epoch={ckpt['epoch']}")
    model.load_state_dict(ckpt["model"])
    model.cuda()
    model.eval()

    logging.info("Model loaded successfully")


def decode_image(image_bytes: bytes) -> np.ndarray:
    """Decode image from bytes to RGB numpy array"""
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Failed to decode image")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def run_inference(img0_rgb: np.ndarray, img1_rgb: np.ndarray,
                  K: np.ndarray, baseline: float,
                  scale: float = 1.0, hiera: bool = False,
                  valid_iters: int = 32) -> np.ndarray:
    """Run stereo inference and return depth map"""

    # Preprocess images
    if scale < 1:
        img0_rgb = cv2.resize(img0_rgb, fx=scale, fy=scale, dsize=None)
        img1_rgb = cv2.resize(img1_rgb, fx=scale, fy=scale, dsize=None)
        K = K.copy()
        K[:2] *= scale

    H, W = img0_rgb.shape[:2]

    # Convert to tensors
    img0_tensor = torch.as_tensor(img0_rgb).cuda().float()[None].permute(0, 3, 1, 2)
    img1_tensor = torch.as_tensor(img1_rgb).cuda().float()[None].permute(0, 3, 1, 2)

    padder = InputPadder(img0_tensor.shape, divis_by=32, force_square=False)
    img0_tensor, img1_tensor = padder.pad(img0_tensor, img1_tensor)

    # Run inference
    with torch.cuda.amp.autocast(True):
        if not hiera:
            disp = model.forward(img0_tensor, img1_tensor, iters=valid_iters, test_mode=True)
        else:
            disp = model.run_hierachical(
                img0_tensor, img1_tensor, iters=valid_iters,
                test_mode=True, small_ratio=0.5
            )

    torch.cuda.synchronize()

    disp = padder.unpad(disp.float())
    disp = disp.data.cpu().numpy().reshape(H, W)

    # Remove invisible points
    yy, xx = np.meshgrid(np.arange(disp.shape[0]), np.arange(disp.shape[1]), indexing="ij")
    us_right = xx - disp
    invalid = us_right < 0
    disp = disp.copy()
    disp[invalid] = np.inf

    # Convert to depth
    depth = K[0, 0] * baseline / disp

    return depth


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize model on startup"""
    ckpt_dir = f"{code_dir}/pretrained_models/23-51-11/model_best_bp2.pth"
    set_logging_format()
    set_seed(0)
    torch.autograd.set_grad_enabled(False)
    load_model_at_startup(ckpt_dir)
    yield
    # Cleanup code would go here if needed


app = FastAPI(title="FoundationStereo Server", lifespan=lifespan)


@app.post("/infer")
async def infer_depth(
    left_image: UploadFile = File(...),
    right_image: UploadFile = File(...),
    fx: float = Form(...),
    fy: float = Form(...),
    cx: float = Form(...),
    cy: float = Form(...),
    baseline: float = Form(...),
    scale: float = Form(1.0),
    hiera: int = Form(0),
    valid_iters: int = Form(32)
):
    """
    Infer depth from stereo pair.

    Returns depth map as compressed numpy array (npz format).
    """
    try:
        start_time = time.time()

        # Decode images
        left_bytes = await left_image.read()
        right_bytes = await right_image.read()

        img0 = decode_image(left_bytes)
        img1 = decode_image(right_bytes)

        # Build intrinsics matrix
        K = np.array([[fx, 0, cx],
                      [0, fy, cy],
                      [0, 0, 1]], dtype=np.float32)

        # Run inference
        depth = run_inference(
            img0, img1, K, baseline,
            scale=scale, hiera=bool(hiera), valid_iters=valid_iters
        )

        # Compress depth map to npz format
        buffer = io.BytesIO()
        np.savez_compressed(buffer, depth=depth.astype(np.float32))
        buffer.seek(0)

        inference_time = time.time() - start_time
        logging.info(f"Inference completed in {inference_time:.3f}s")

        return Response(
            content=buffer.getvalue(),
            media_type="application/octet-stream",
            headers={
                "X-Inference-Time": str(inference_time),
                "X-Depth-Shape": f"{depth.shape[0]},{depth.shape[1]}"
            }
        )

    except Exception as e:
        logging.error(f"Error during inference: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/infer_png")
async def infer_depth_png(
    left_image: UploadFile = File(...),
    right_image: UploadFile = File(...),
    fx: float = Form(...),
    fy: float = Form(...),
    cx: float = Form(...),
    cy: float = Form(...),
    baseline: float = Form(...),
    scale: float = Form(1.0),
    hiera: int = Form(0),
    valid_iters: int = Form(32),
    depth_scale: float = Form(1000.0)
):
    """
    Infer depth from stereo pair.

    Returns depth map as 16-bit PNG (depth values in millimeters).
    """
    try:
        start_time = time.time()

        # Decode images
        left_bytes = await left_image.read()
        right_bytes = await right_image.read()

        img0 = decode_image(left_bytes)
        img1 = decode_image(right_bytes)

        # Build intrinsics matrix
        K = np.array([[fx, 0, cx],
                      [0, fy, cy],
                      [0, 0, 1]], dtype=np.float32)

        # Run inference
        depth = run_inference(
            img0, img1, K, baseline,
            scale=scale, hiera=bool(hiera), valid_iters=valid_iters
        )

        # Convert to 16-bit PNG (millimeters)
        depth_mm = (depth * depth_scale).astype(np.uint16)
        depth_mm[~np.isfinite(depth)] = 0

        # Encode as PNG
        success, buffer = cv2.imencode('.png', depth_mm)
        if not success:
            raise ValueError("Failed to encode depth as PNG")

        inference_time = time.time() - start_time
        logging.info(f"Inference completed in {inference_time:.3f}s")

        return Response(
            content=buffer.tobytes(),
            media_type="image/png",
            headers={
                "X-Inference-Time": str(inference_time),
                "X-Depth-Shape": f"{depth.shape[0]},{depth.shape[1]}",
                "X-Depth-Scale": str(depth_scale)
            }
        )

    except Exception as e:
        logging.error(f"Error during inference: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }


def main():
    """Run the FastAPI server."""
    import argparse
    parser = argparse.ArgumentParser(
        description='FoundationStereo Server'
    )
    parser.add_argument(
        '--host',
        type=str,
        default='0.0.0.0',
        help='Host to bind the server to (default: 0.0.0.0)'
    )
    parser.add_argument(
        '--port',
        type=int,
        default=1234,
        help='Port to bind the server to (default: 1234)'
    )

    args = parser.parse_args()

    print("=" * 80)
    print("Starting FoundationStereo Server")
    print(f"  Host: {args.host}")
    print(f"  Port: {args.port}")
    print("=" * 80)

    # Run server
    import uvicorn
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="info"
    )


if __name__ == '__main__':
    main()
