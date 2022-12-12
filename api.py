# !pip install fastapi uvicorn

from fastapi import FastAPI,Request
from fastapi.responses import JSONResponse
import uvicorn
import PIL.Image as Image
import io
import base64
import numpy as np
from skimage.transform import rescale,resize
import keras
from starlette.middleware.cors import CORSMiddleware

app = FastAPI()
origins=["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
@app.post("/dnn/get_prediction")
async def getInformation(info : Request):
    req_info = await info.json()
    # print(req_info)
    b=base64.b64decode(req_info['image'])
# print(b)
    img=Image.open(io.BytesIO(b))
    # simg.save("geeks1.png")
    Qry=img
    Qry=Qry.convert("L")
    Qry=np.array(Qry.resize((224,224)))
    Qry=Qry.reshape((1,224,224))
    testimg=np.array(Qry)
    xt=[]
    testimg=testimg.astype('float32')
    testimg=testimg/np.max(testimg)
    testimg=resize(testimg,(1,224,224),anti_aliasing=True)
    testimg=np.array(testimg.flatten())
    testimg=testimg.astype('float32')
    xt.append(testimg)
    xt=np.array(xt)
    dnnModel1=keras.models.load_model('models/dnn/modelTrained1')
    p=dnnModel1.predict(xt)
    CLASSES=['COVID','NORMAL','PNEUMONIA']
    pred=CLASSES[np.argmax(p)]
    print(type(p))
    predProb=str(np.max(p))
    print(predProb)
    
    # img
    response= {
        "status" : "SUCCESS",
        "status_code" : 200,
        "prediction":pred,
        "probability":predProb
    }
    return (response)
    

@app.post("/cnn/get_prediction")
async def getInformation(info : Request):
    req_info = await info.json()
    # print(req_info)
    b=base64.b64decode(req_info['image'])
# print(b)
    img=Image.open(io.BytesIO(b))
    # simg.save("geeks1.png")
    Qry=img
    Qry=Qry.convert("RGB")
    Qry=np.array(Qry.resize((224,224)))
    Qry=Qry.reshape((224,224,3))
    testimg=np.array(Qry)
    xt=[]
    testimg=testimg.astype('float32')
    testimg=testimg/np.max(testimg)
    testimg=resize(testimg,(224,224,3),anti_aliasing=True)
    # testimg=np.array(testimg.flatten())
    testimg=testimg.astype('float32')
    xt.append(testimg)
    xt=np.array(xt)
    dnnModel1=keras.models.load_model('models/cnn/my_trained_cnn_model.h5')
    p=dnnModel1.predict(xt)
    CLASSES=['COVID','NORMAL','PNEUMONIA']
    pred=CLASSES[np.argmax(p)]
    print(type(p))
    predProb=str(np.max(p))
    print(predProb)
    
    # img
    response= {
        "status" : "SUCCESS",
        "status_code" : 200,
        "prediction":pred,
        "probability":predProb
    }
    return (response)
    
if __name__ == "__main__":
  uvicorn.run(app, host="0.0.0.0", port=8000)