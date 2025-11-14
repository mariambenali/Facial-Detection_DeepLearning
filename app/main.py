from fastapi import FastAPI,Depends, UploadFile, HTTPException, File
from tensorflow.keras.models import load_model
from sqlalchemy.orm import Session
from app import schema , models 
from app.database import base, engine, get_db
from fastapi.responses import JSONResponse
from model_DL.detect_and_predict import detect_and_predict_emotion
from PIL import Image
import numpy as np
import io
import app.database  as db
from datetime import datetime



app = FastAPI()
#Crée automatiquement les tables dans la base
base.metadata.create_all(bind=engine)

#load model CNN
model= load_model("model_DL/facial_model.keras")



@app.get("/emotions")
def get_emotions(db: Session= Depends(get_db)):
    emotions= db.query(models.emotion).all()
    return emotions


@app.post("/predict", response_model=schema.CreateEmotion)
async def predict_emotion(file:UploadFile= File(...), db: Session= Depends(get_db)):
    try:
        #read image
        content = await file.read() #because the function is asynchronous
        image = Image.open(io.BytesIO(content)).convert("RGB")
        image_np= np.array(image)  #convert image to array, every pixel convert to num 0-255


        #use the function from detect_and_predict.py 
        results= detect_and_predict_emotion(image_np)

        if len(results) == 0:
            raise HTTPException(status_code=400,detail="there is no face on the picture!")
        

        predicted_emotion, score= detect_and_predict_emotion(image_np)

        new_prediction ={
            "predicted_emotion" : predicted_emotion,
            "score" : float(score),
            "date_of_creation" : datetime.utcnow()

        }

        Prid = models.emotion(**new_prediction)

        db.add(Prid)
        db.commit()
        db.refresh(Prid)

        return Prid
    
    
    except Exception as e:
        print("❌ ERROR:", str(e))
        raise HTTPException(status_code=500, detail= f"Error!: {str(e)}") 
    
    
