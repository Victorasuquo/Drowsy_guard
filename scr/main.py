from fastapi import FastAPI, File
from segmentation import get_model, get_image_from_bytes
from starlette.responses import Response
import io
from PIL import Image
import json

model = get_model()

app = FastAPI()

@app.get('/drowsy')
def everything_ok():
    return dict(msg='OK')

@app.post("/detect_image")
async def detect_image_return_json_result(file: bytes = File(...)):
    input_image = get_image_from_bytes(file)
    results = model(input_image)
    return "done" + str(type(results))
    # detect_res = results.pandas().xyxy[0].to_json(orient="records")
    # detect_res = json.loads(detect_res)
    # return {"result": detect_res}

@app.post("/object-to-img")
async def detect_food_return_base64_img(file: bytes = File(...)):
    input_image = get_image_from_bytes(file)
    results = model(input_image)
    return results

    # results.render()  # updates results.imgs with boxes and labels
    # for img in results.imgs:
    #     bytes_io = io.BytesIO()
    #     img_base64 = Image.fromarray(img)
    #     img_base64.save(bytes_io, format="jpeg")
    # return Response(content=bytes_io.getvalue(),media_type="image/jpeg")