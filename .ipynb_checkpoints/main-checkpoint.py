import tensorflow as tf
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import uvicorn 
import numpy as np
from keras.preprocessing import image

class Image(BaseModel):
    imageName : str
        
#model loading
cnn=tf.keras.models.load_model("./myModel_v2")

app = FastAPI()


origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
@app.get("/")
async def read_root():
  return {"message":"welcome to first"}

@app.post('/predict')
def get_image_category(data: Image):
    received = data.dict()
    test_image = image.load_img('single_prediction/'+received['imageName'], target_size = (128, 128))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    result = cnn.predict(test_image)
    if result[0][0] == 0:
        prediction = 'non_stop'
    else:
        prediction = 'stop'
    return {'classificaton_result': prediction}