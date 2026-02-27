# main.py

from fastapi import FastAPI, Request
from contextlib import asynccontextmanager
from pydantic import BaseModel

from services.classifier_service import ClassifierService


class PredictionRequest(BaseModel):
    text: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load models once per worker
    service = ClassifierService.load()
    app.state.classifier_service = service
    yield
    # optional cleanup here


app = FastAPI(lifespan=lifespan)


@app.post("/predict")
async def predict(request: Request, payload: PredictionRequest):
    service: ClassifierService = request.app.state.classifier_service
    return service.predict(payload.text)