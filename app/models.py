from sqlalchemy import String, Integer,Float,Column,DateTime
from app import database
from datetime import datetime




class emotion(database.base):
    __tablename__= "emotions"

    id = Column(Integer,primary_key=True, index=True)
    predicted_emotion = Column(String)
    score = Column(Float)
    date_of_creation = Column(DateTime, default=datetime.utcnow)



