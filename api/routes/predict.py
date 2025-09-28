from fastapi import APIRouter, UploadFile, File
from fastapi.responses import JSONResponse
from pathlib import Path
from api.services import predict_image

router = APIRouter()

@router.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Predict class of uploaded image using the service layer.
    """
    try:
        # Save uploaded file temporarily
        temp_path = Path("temp") / file.filename
        temp_path.parent.mkdir(exist_ok=True)
        with open(temp_path, "wb") as f:
            f.write(await file.read())

        # Call the service function
        predicted_class = predict_image(temp_path)

        # Clean up
        temp_path.unlink(missing_ok=True)

        return JSONResponse({"predicted_class": predicted_class})

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
