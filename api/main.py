from fastapi import FastAPI
from api.routes import predict, health

app = FastAPI(title="Food Classifier API")

app.include_router(predict.router)
app.include_router(health.router)
