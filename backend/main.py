
from fastapi import FastAPI
from backend.routes import health, predict

app = FastAPI(
    title="Satellite Land-Type Clustering API",
    description="Backend API for satellite image clustering using ResNet18 and K-Means",
    version="1.0.0"
)

app.include_router(health.router)
app.include_router(predict.router)


@app.get("/")
def root():
    return {
        "message": "Satellite Land-Type Clustering API is running",
        "status": "success"
    }
