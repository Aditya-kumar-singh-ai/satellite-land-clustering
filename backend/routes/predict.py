
from fastapi import APIRouter, UploadFile, File, HTTPException
from PIL import UnidentifiedImageError

from backend.services.predictor import predict_image

router = APIRouter(tags=["Prediction"])


@router.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Validate file type
    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(
            status_code=400,
            detail="Only JPG and PNG images are supported."
        )

    try:
        image_bytes = await file.read()

        if not image_bytes:
            raise HTTPException(
                status_code=400,
                detail="Uploaded file is empty."
            )

        cluster_id = predict_image(image_bytes)

        return {
            "filename": file.filename,
            "cluster_id": cluster_id,
            "status": "success"
        }

    except UnidentifiedImageError:
        raise HTTPException(
            status_code=400,
            detail="Invalid image file."
        )

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )
