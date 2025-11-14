from pydantic import BaseModel
from datetime import datetime


class EmotionBase(BaseModel):
    predicted_emotion:str
    score: float
    date_of_creation: datetime

class CreateEmotion(EmotionBase):
    pass 


class ResponseEmotion():
    id : int


    class Config():
        orm_mode =True