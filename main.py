import io
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
from pydantic import BaseModel
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    HEALTH_STR: str = "OK"
    MODEL_DIR: Path = Path("/opt/models")
    MODEL_NAME: str = "MeNet.onnx"
    MODEL_INPUT_SHAPE: tuple[int, int] = (256, 256)
    ONNX_PROVIDERS: list[str] = ["TensorrtExecutionProvider", "CUDAExecutionProvider"]


settings = Settings()
onnx_session: ort.InferenceSession
onnx_input_name: str
onnx_output_name: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    global onnx_session, onnx_input_name, onnx_output_name

    # Load model and warmup onnx session with dummy inputs
    model_binary = settings.MODEL_DIR / settings.MODEL_NAME
    onnx_session = ort.InferenceSession(model_binary, providers=settings.ONNX_PROVIDERS)
    onnx_input_name = onnx_session.get_inputs()[0].name
    onnx_output_name = onnx_session.get_outputs()[0].name
    preload_sample = np.random.uniform(
        0, 255, (1, *settings.MODEL_INPUT_SHAPE, 3)
    ).astype(np.float32)
    onnx_session.run([onnx_output_name], {onnx_input_name: preload_sample})
    yield


app = FastAPI(lifespan=lifespan)


@app.get("/health")
async def healthcheck() -> str:
    return settings.HEALTH_STR


class PredictionResult(BaseModel):
    class_name: list[str]
    confidence: list[float]


class PredictionResponse(JSONResponse):
    content: list[PredictionResult]


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    global onnx_session, onnx_input_name, onnx_output_name
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        contents = await file.read()
        image = (
            Image.open(io.BytesIO(contents))
            .convert("RGB")
            .resize(settings.MODEL_INPUT_SHAPE)
        )

        # MeNet does not require scaled inputs
        image_data = np.expand_dims(np.array(image).astype(np.float32), axis=0)
        results = onnx_session.run([onnx_output_name], {onnx_input_name: image_data})
        predictions = results[0][0]

        # TODO: this should be in a constants file in deepweeds.datasets
        classes = [
            "rubber_vine",
            "negative",
            "parthenium",
            "chinee_apple",
            "prickly_acacia",
            "snake_weed",
            "parkinsonia",
            "siam_weed",
            "lantana",
        ]

        top_results = [
            {"class_name": classes[idx], "confidence": float(predictions[idx])}
            for idx in np.argsort(predictions)[::-1]
        ]

        return PredictionResponse(content=top_results)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error processing image: {str(e)}"
        ) from e
