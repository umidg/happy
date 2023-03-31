from typing import Union
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware


from boto3 import session
from botocore.client import Config
from datetime import datetime



ACCESS_ID = 'DO00R3Q7K3DCCDH7C6GM'
SECRET_KEY = 'P86OmYiPCKBVtXv56tk2AYzk7+2LKfm4jmvxXa+M7uI'

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins="*",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
from fastapi import FastAPI, File, UploadFile

import cv2
import numpy as np
from fastapi.responses import StreamingResponse
import moviepy.editor as mpe
   

CANNY_THRESH_1 = 10
CANNY_THRESH_2 = 200
MASK_DILATE_ITER = 10
MASK_ERODE_ITER = 10
MASK_COLOR = (1.0, 1.0, 1.0)  # In BGR format

app = FastAPI()
db = []

# Upload a file to your Space


# Image processing
def image_process(name):
    #-- Read image -----------------------------------------------------------------------
    image_source = name
    img = cv2.imread(image_source)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # -- Edge detection -------------------------------------------------------------------
    edges = cv2.Canny(gray, CANNY_THRESH_1, CANNY_THRESH_2)
    edges = cv2.dilate(edges, None)
    edges = cv2.erode(edges, None)
    
    # -- Find contours in edges, sort by area ---------------------------------------------
    contour_info = []
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    for c in contours:
        contour_info.append(
            (
                c,
                cv2.isContourConvex(c),
                cv2.contourArea(c),
            )
        )
    contour_info = sorted(contour_info, key=lambda c: c[2], reverse=True)
    max_contour = contour_info[0]

    # -- Create empty mask, draw filled polygon on it corresponding to largest contour ----
    # Mask is black, polygon is white
    mask = np.zeros(edges.shape)
    cv2.fillConvexPoly(mask, max_contour[0], (255))


    # -- Smooth mask, then blur it --------------------------------------------------------
    mask = cv2.dilate(mask, None, iterations=MASK_DILATE_ITER)
    mask = cv2.erode(mask, None, iterations=MASK_ERODE_ITER)
    
    mask_stack = np.dstack([mask] * 3)  # Create 3-channel alpha mask

    # -- Blend masked img into MASK_COLOR background --------------------------------------
    mask_stack = (mask_stack.astype("float32") / 255.0)  # Use float matrices,
    img = img.astype("float32") / 255.0  #  for easy blending

    masked = (mask_stack * img) + ((1 - mask_stack) * MASK_COLOR)  # Blend
    masked = (masked * 255).astype("uint8")
    
    # Define name of final output image
    image_source_split = image_source.split(".")
    final_output = image_source_split[0] + "_final_output_." + image_source_split[1]

    # Convert back to 8-bit
    cv2.imwrite(final_output, masked)
    
    img = cv2.imread(final_output)
    

    # Define path for output animation
    output_path = image_source_split[0] + ".mp4"

    # Create a video writer object using OpenCV
    # 20.0 is the number of frames per second
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(output_path, fourcc, 20.0, (img.shape[1], img.shape[0]))

    height, width, c = img.shape

    # Reference: https://www.geeksforgeeks.org/animate-image-using-opencv-in-python/
    for i in range(100):
        l = img[:, : (i % width)]
        r = img[:, (i % width) :]
        img = np.hstack((r, l))
        out.write(img)

    out.release()
    
    
    
    
    audios = mpe.AudioFileClip("babyshark.mp3").subclip(0,6)
    video1 = mpe.VideoFileClip('image.mp4')
    final = video1.set_audio(audios)
    final.write_videofile("image1.mp4",codec='libx264', audio_codec='aac', temp_audiofile='temp-audio.m4a', remove_temp=True)
    
    
    sessions = session.Session()
    client = sessions.client('s3', region_name='nyc3', endpoint_url='https://nyc3.digitaloceanspaces.com', aws_access_key_id=ACCESS_ID,aws_secret_access_key=SECRET_KEY)

    # Upload a file to your Space
    client.upload_file('image1.mp4', 'happyspace', 'image1.mp4')
    
    resource = sessions.resource("s3", endpoint_url="https://nyc3.digitaloceanspaces.com", region_name="nyc3", aws_access_key_id=ACCESS_ID,aws_secret_access_key=SECRET_KEY)
    resource.Object('happyspace', 'image1.mp4').Acl().put(ACL='public-read')
    
    # def iterfile():  # 
    #     with open("image1.mp4", mode="rb") as file_like:  # 
    #         yield from file_like  # 
            

    # return StreamingResponse(iterfile(), media_type="video/mp4")
    
    return "https://happyspace.nyc3.digitaloceanspaces.com/image1.mp4"
    
    


@app.post("/images")
async def create_upload_file(uploaded_file: UploadFile = File(...)):
    print(uploaded_file,uploaded_file.filename,"file",datetime.today().isoformat())
    name = "image."+uploaded_file.filename.split(".")[1]
    file_location = f"{name}"
    with open(file_location, "wb+") as file_object:
        file_object.write(uploaded_file.file.read())
        return image_process(name)
    

@app.post("/image")
async def create_upload_file(uploaded_file: UploadFile = File(...)):
    print(uploaded_file,"u",datetime.today().isoformat())
    
@app.get("/get")
async def getall():
    return datetime.today().isoformat()