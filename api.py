# !pip install fastapi uvicorn

from fastapi import FastAPI,Request
import uvicorn
import PIL.Image as Image
import io
import base64
import numpy as np
from skimage.transform import rescale,resize
import keras

app = FastAPI()
@app.post("/get_prediction")
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
    dnnModel1=keras.models.load_model('modelTrained1')
    p=dnnModel1.predict(xt)
    CLASSES=['COVID','NORMAL','PNEUMONIA']
    pred=CLASSES[np.argmax(p)]
    # img
    return {
        "status" : "SUCCESS",
        "data" : req_info,
        "image":pred
    }
    
    
if __name__ == "__main__":
  uvicorn.run(app, host="0.0.0.0", port=8000)