# Cat or Dog Image Classifier: Backend File

import os
import io #<---Handles raw byte streams in memory (turns uploaded network bytes into image data).
#io lets you work with data in memory as if it were a file.
import base64 #<---base64 encodes visual images into text strings for HTML display.
import cv2 #<---OpenCV for image colour conversion and encoding.
import numpy as np #<---Used for numerical array operations on image pixel matrices.
from PIL import Image #<---Python Imaging library for opening and converting image formats.

#PyTorch deep learning framework.
import torch #<---Core PyTorch machine learning library.
import torch.nn.functional as F #<---Contains mathematical functions (e.g. softmax activation)
from torchvision import models, transforms #<---Contains pre-trained AI models and image preprocessing tools.
import torch.nn as nn #<---Imports PyTorch's neural network module which provides building blocks-
# such as nn.Linear for fully connected layers.

#FastAPI server and CORS security permissions.
#FastAPI is a web framework thats allows Python to listen to incoming HTTP requests from your browser.
from fastapi import FastAPI, HTTPException, UploadFile, Response, File #<---FastAPI web server components for handling HTTP requests.
from fastapi.middleware.cors import CORSMiddleware #<---Security middleware to handle Cross-Origin requests,-
# later enabled app.add_middleware().

#Grad-Cam 
from pytorch_grad_cam import GradCAM #<---Grad-CAM algorithm for AI heatmaps.
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget #<---Specifies target class for Grad-CAM.
from pytorch_grad_cam.utils.image import show_cam_on_image #<---Overlays Grad-CAM heatmap onto the uploaded image.

from fpdf import FPDF #<---Imports the pdf generation library allowing to create 

app = FastAPI() #<---Creates the web server instance.

#Communication and Security:
#Enable CORS (Cross-Origin Resource Sharing) so the local HTML/JS frontend can talk to Python.
#CORS is a browser security feature that blocks a webpage from communicating to a backend-
# running on a different port. Without this web browsers would automatically block cross-origin-
# for security reasons.

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], #<---Allows requests from every origin (e.g. localhost)
    allow_credentials=True,
    allow_methods=["*"], #<---Allows all HTTP requests (GET, POST, etc.)
    allow_headers=["*"],
)

#Introduction of AI model and Transfer Learning:
#Loads ResNet18 which is an 18-layer convolutional neural network (CNN) that is pre-trained on 1.28 million images.

model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

#Iterates through every weight parameter in the network and turns off gradient calculations.
#Convolutional neural networks learn hierarchical feature representations (edges, textures,-
# spots, and object shapes in the lower/middle layers)
#By setting gradient calculations to false, you freeze the pre-trained feature extractor layers so-
# PyTorch won't alter or degrade those learned visual weights.
#With "layer4" change only the early layers are frozen but layer4 and fc are kept active for Grad-CAM computation.
for name, param in model.named_parameters():
    if "layer4" not in name and "fc" not in name:
        param.requires_grad = False
    else:
        param.requires_grad = True #<---Enable gradient computation for Grad-CAM.

#Reads the number of input nodes feeding into ResNet-18s original final linear layer.
#in_features is a PyTorch attribute that stores the number of input nodes/features entering that layer.
num_features = model.fc.in_features

#This replaces the 1000-class output layer of the CNN into a 2-class output layer for Cats and Dogs.
#.fc stands for Fully Connected layer. In PyTorch fc is the variable name asssigned to the final classification/output layer.
model.fc = nn.Linear(num_features, 3) #<---Index 0 = Cat, Index 1 = Dog, Index 2 = Other.

#Load fine-tuned weights file if it exist in project folder.
weights_path = "cat_dog_resnet18.pth" #<---Assigns the path of the weights to variable "weights_path".
if os.path.exists(weights_path): #<---If the path exists...
    model.load_state_dict(torch.load(weights_path, weights_only=True)) #<---Loads the saved weights data and loads it into the model.
    print("Success: Loaded fine tuned weights.")
else:
    print("Warning: 'cat_dog_resnet18.pth' not found. Please run train.py first.")

model.eval() #<---Switches the neural network into inference/ evaluation mode meaning BatchNorm uses its stored-
# running statistics and Dropout is disabled.


#Preprocessing:
#Neural networks cant process raw image files (.jpg) as they require standardized grid matrices of numbers-
# because neural networks are mathematical engines they cannot read files they can only do arithmetic.

transform = transforms.Compose([ 
    transforms.Resize((224, 224)), #<---Resizes any image uploaded to 224x224 (ResNet Dimensions)-
    # as this means it can understand and process the image.
    
    transforms.ToTensor(), #<---Converts pixel colour values into a mathematical tensor. 
    #A tensor is a mathematical object that generalises scalars, vectors and matrices into higher-
    # dimensions to describe multilinear relationships. For example a greyscale image is a 2D tensor-
    # (height, width), a coloured image is a 3D tensor(height, width, colour channels).
    
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) #<---Normalises RGB-
    # channels using ImageNet dataset mean and standard deviation (Z = (X - mean) / std).
    #Expects a 3-channel tensor with values scaled [0.0, 1.0] from ToTensor(). 
    #Required so input image statistics match the pre-trained ResNet weights distribution.
    #The mean numbers are the average colour of red(.485), green(.456), blue(.406) from the 1.4 million-
    # images that were used to train the ResNet18 CNN. The std (standard deviation) represent the spread-
    # of colour intensity from the ImageNet training data. Mean and Std are unitless(dimensionless).
])



#File Validation settings.
ALLOWED_FILE_TYPES = {"image/jpeg", "image/jpg", "image/pjpeg", "image/png", "image/webp",}
ALLOWED_EXTENSIONS = {".jpeg", ".jpg", ".png", ".webp",}
max_file_size = 10 * 1024 * 1024 #<---Sets the maximum file size as 10MB.

#Confidence threshold.
confidence_threshold = 0.88 #<---Sets the minimum confidence score for classification to 88%.

#Health check endpoint:

@app.get("/") #<---app.get() Tells Fast API to handle GET requests and "/" tells it to use the root URL path.
def home(): #<---Defines a regular Python function named home. When FastAPI recieves a GET request at / it-
# calls this function.
    return {"status": "AI Backend is running smoothly."} #<---Returns a python dictionary containing a-
# key-value pair (key = "status") and (value = "AI backend ....") FastAPI automatically converts this-
# dictionary into a JSON response (200 OK).


#Image Classsification

@app.post("/predict") #<---Client uploads an image. @app is the FastAPI application object, post() tells FastAPI-
# to handle HTTP POST requests because the client is sending data to the server for processing, /predict is the-
# URL for the endpoint.
async def predict(file: UploadFile = File(...)): #<---Defines an asynchronus(coroutine) function named predict. 
    # file is the parameter name and UploadFile is a FastAPI type used to represent an uploaded file.
    #= File(...) tells FastAPI that the file must be supplied in the requests form data. ... is validation-
    # meaning the parameter is required. If the request doesnt include a file, FastAPI returns a validation error.
    
    await file.seek(0)
    image_bytes = await file.read() #<---file.read() reads the uploaded file and returns its contents as bytes.
        #This is an asynchronus operation. Await waits for that operation to complete and without it you would have-
        # a coroutine object rather than actual bytes. The resulting image bytes are asssigned to image_bytes.

    #Check for empty payload 
    if not image_bytes: 
        raise HTTPException( 
            status_code=400, 
            detail="Uploaded image file is empty (0 bytes)."
        )
    
    #Safely extract and lower-case content type and filename.
    content_type = (file.content_type or "").lower()
    filename = (file.filename or "").lower()
        
    #Check if file type or file extension is valid.
    has_valid_mime = content_type in ALLOWED_FILE_TYPES
    has_valid_ext = any(filename.endswith(ext) for ext in ALLOWED_EXTENSIONS)

    #Check file type.
    if not has_valid_mime and not has_valid_ext:
        raise HTTPException(
            status_code = 400,
            detail = (f"Unsupported file type '{file.content_type}'. Please upload .jpg, .png or webp image files.")
        )
    
    #Read bytes and check file size.
    if len(image_bytes) > max_file_size:
        raise HTTPException(
            status_code = 400,
            detail=("File size exceeds the maximum file size of 10MB."),
        )
    
    #Check for empty payload 
    if not image_bytes: 
        raise HTTPException( 
            status_code=400, 
            detail="Uploaded image file is empty (0 bytes)."
        )

    
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB") #<---This line does two things: it decodes the file-
    # bytes into an image and converts the images colour mode. It uses the pillow imaging library "from PIL import Image".
    #contents is a sequence of bytes. BytesIO saves those bytes in a file-like in-memory object. This lets 

    rgb_img = np.float32(image.resize((224, 224))) / 255.0 #<---Resizes the image to 224x224 pixels (ResNet dimensions)-
    # and converts the image into a NumPy array of 32-bit floating-point numbers, and normalise pixel values-
    # from 0-255 to 0-1 for the background image for the heatmap. 

    input_tensor = transform(image).unsqueeze(0) #<---Preprocesses the image by applying the transform section rules-
    # and adds a batch dimension so that it can be inputted into the ResNet model. 
    # A batch dimension is an identifier that tells the model how many images it is being given. 
    # Without a batch dimension the image data would have this shape (3, 224, 224) and with a batch-
    # dimension it would have this shape (1, 3, 224, 224) with the 1 indicating that it was only one image-
    # being inputted.

    #Disable gradient computation for faster inference and memory efficiency.
    with torch.no_grad():
        outputs = model(input_tensor) #<---Input image into ResNet-18 to get raw logits.
        #Logits are the raw scores that ResNet-18 produces for each possible class before-
        # converting them into probabilities.
        probabilities = F.softmax(outputs, dim=1) #<---Converts the raw logits provided into probabilities-
        # from 0.0 to 1.0.
        top_prob, top_class = torch.max(probabilities, 1) #<---Extracts highest probability value and index.

    #Converts PyTorch scalar tensors to standard Python integers and floats to make them easy to use and view.
    class_id = int(top_class.item())
    confidence = float(top_prob.item())

    
    if confidence < confidence_threshold:
        if class_id in (0, 1):
            #Was classified as cat or dog but confidence was overided to Neither.
            label = f"Neither Cat or Dog"
        else:
            #Was classified as Neither but confidence is too low so it returns uncertain instead.
            label = f"Uncertain"
    else:
        #Map ImageNet class ID indices to Cat, Dog or Neither label.
        if class_id == 0:
            label = f"Cat" 
        elif class_id == 1: 
            label = f"Dog" 
        elif class_id == 2:
            label = f"Neither Cat or Dog"

    #Verify that the file can be decoded safely.
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(
            status_code=400,
            detail=("Corrupt or invalid image file. Please upload valid image file."),
        )
    

    #Define target layer for Grad-CAM (last convolutional layer of ResNet-18)
    target_layers = [model.layer4[-1]]

    #Initialize Grad-CAM object with model and target convolutional layer.
    cam = GradCAM(model=model, target_layers=target_layers)

    #Specify the target class prediction to visualise.
    targets = [ClassifierOutputTarget(class_id)]

    #Generate 2D activation heatmap array for input image tensor.
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
    grayscale_cam = grayscale_cam[0, :] #<---Extract single 2D heatmap mask array from batch dimension.
    
    #This overlays the heatmap onto the un-normalised background RGB image.
    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

    #Convert RGB colour matrix to OpenCV BGR and encode as JPEG image buffer.
    _, buffer = cv2.imencode(".jpg", cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))

    #Encode binary JPEG buffer into Base64 ASCII text string.
    base64_heatmap = base64.b64encode(buffer).decode("utf-8")

    #Format string as a complete Data URL ready for HTML <img> tag src attribute.
    heatmap_data_url = f"data:image/jpeg;base64,{base64_heatmap}"

    #Returns prediction metadata and heatmap text string as JSON payload.
    return {
        "class_id": class_id,
        "label": label,
        "confidence": confidence,
        "heatmap": heatmap_data_url
    }


@app.post("/download-report")
async def download_report(data: dict):
    try:
        label = data.get("label", "N/A")
        confidence = data.get("confidence", "N/A")
        class_id = data.get("class_id", "N/A")
        heatmap_base64 = data.get("heatmap", "")

        #Format numerical confidence.
        if isinstance(confidence, float):
            confidence_str = f"{confidence * 100:.2f}%"
        else:
            confidence_str = str(confidence)

        #Initialize pdf.
        pdf = FPDF()
        pdf.add_page()

        #Header and Title
        pdf.set_font("Helvetica", "B", 18)
        pdf.cell(
            0,
            12,
            "AI Image Classification Report",
            new_x="LMARGIN",
            new_y="NEXT",
            align="C",
        )
        pdf.ln(8)

        #Prediction Summary Section
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, "1. Classification Summary", new_x="LMARGIN", new_y="NEXT")

        pdf.set_font("Helvetica", "", 11)
        pdf.cell(0, 7, f"  - Predicted Class: {label}", new_x="LMARGIN", new_y="NEXT")
        pdf.cell(
            0,
            7,
            f"  - Confidence Score: {confidence_str}",
            new_x="LMARGIN",
            new_y="NEXT",
        )
        pdf.cell(
            0, 7, f"  - Class ID: {class_id}", new_x="LMARGIN", new_y="NEXT"
        )
        pdf.ln(8)

        #Heatmap Image Section
        if heatmap_base64:
            pdf.set_font("Helvetica", "B", 12)
            pdf.cell(
                0,
                8,
                "2. Grad-CAM Heatmap Image",
                new_x="LMARGIN",
                new_y="NEXT",
            )

        #Decode Base64 string to PIL Image
        img_str = (
            heatmap_base64.split(",")[1]
            if "," in heatmap_base64
            else heatmap_base64
        )
        img_bytes = base64.b64decode(img_str)
        img = Image.open(io.BytesIO(img_bytes))

        #Save to temporary memory buffer for FPDF insertion
        temp_img = io.BytesIO()
        img.save(temp_img, format="JPEG")
        temp_img.seek(0)

        #Insert image into pdf (centered, 140mm width)
        pdf.image(temp_img, x=35, w=140)

        # Convert output to bytes
        pdf_output = pdf.output() 
        pdf_bytes = ( 
            pdf_output.encode("latin1") 
            if isinstance(pdf_output, str) 
        else bytes(pdf_output) )
        

        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={
                "Content-Disposition": (
                    'attachment; filename="AI_Classification_Report.pdf"'
                )
            },
        )

    except Exception as e:
      raise HTTPException(
        status_code=500, detail=f"Failed to generate PDF report: {str(e)}"
    )
    