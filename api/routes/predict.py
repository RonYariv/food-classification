from fastapi import APIRouter, UploadFile, File
from fastapi.responses import JSONResponse
from api.services import predict_image
from api.schemas import PredictResponse

router = APIRouter()

@router.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...)):
    try:
        # Save file temporarily
        image_path = f"temp_{file.filename}"
        with open(image_path, "wb") as f:
            f.write(await file.read())

        class_id, class_name, probability = predict_image(image_path)

        return PredictResponse(
            class_id=class_id,
            class_name=class_name,
            probability=float(probability)
        )

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
