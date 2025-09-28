from pydantic import BaseModel

class PredictResponse(BaseModel):
    class_id: int
    class_name: str
    probability: float
