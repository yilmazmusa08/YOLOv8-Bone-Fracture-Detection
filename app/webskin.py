import os
import cv2
import numpy as np
import pydicom
from PIL import Image
import streamlit as st
import torch
from matplotlib.colors import TABLEAU_COLORS 
from pathlib import Path
from ultralytics import YOLO


ALLOWED_EXTENSIONS = {"txt", "pdf", "png", "jpg", "jpeg", ".dcm"}
parent_root = Path(__file__).parent.parent.absolute().__str__() # os.path.dirname(os.path.abspath(__file__))
h, w = 640, 640
model_path = os.path.join(parent_root, "skin.pt")
device = "cuda" if torch.cuda.is_available() else "cpu"

def color_list():
    # Return first 10 plt colors as (r,g,b) https://stackoverflow.com/questions/51350872/python-from-color-name-to-rgb
    def hex2rgb(h):
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))

    return [hex2rgb(h) for h in TABLEAU_COLORS.values()]

colors = color_list()

def xyxy2xywhn(bbox, H, W):
    x1, y1, x2, y2 = bbox
    return [0.5*(x1+x2)/W, 0.5*(y1+y2)/H, (x2-x1)/W, (y2-y1)/H]

def xywhn2xyxy(bbox, H, W):
    x, y, w, h = bbox
    return [(x-w/2)*W, (y-h/2)*H, (x+w/2)*W, (y+h/2)*H]

def load_img(uploaded_file):
    """ Load image from bytes to numpy
    """
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)
    return opencv_image[..., ::-1]

def model_inference(model_path, image_np):
    model = YOLO(model_path)
    output = model.predict(source=image_np, save=False)
    out_img = output[0].plot()
    return output, out_img

def post_process(output):
    boxes = []
    texts = []
    for result in output:
        # Detection
        boxes.append(result.boxes.xyxy)   # box with xyxy format, (N, 4)
        texts.append(result.boxes.cls)    # cls, (N, 1)
    print("boxes :", boxes)
    print("texts :", texts)
    return boxes, texts

def convert_dicom_to_jpg(dicom_path, output_path):
    ds = pydicom.dcmread(dicom_path)
    pixel_array = ds.pixel_array
    image = Image.fromarray(pixel_array)
    image.save(output_path)

if __name__ == "__main__":
    st.title("Bone Fracture Detection")
    uploaded_file = st.file_uploader("Choose an image file", type=["png", "jpg", "jpeg", ".dcm"])
    if uploaded_file is not None:
        file_name = uploaded_file.name
        if not file_name.endswith(".dcm"):
            # Load and display the original image
            img = load_img(uploaded_file)
            print("Image : ", img)

            # Inference
            out, out_img = model_inference(model_path, img)
            out_boxes, out_txt = post_process(out)

            # Save the predicted image
            save_path = os.path.splitext(uploaded_file.name)[0] + "_predicted.jpg"
            cv2.imwrite(save_path, out_img)

            # Display the first image using st.image()
            st.image(out_img, caption="Prediction", channels="RGB")

            # Download buttons
            col1, col2 = st.columns(2)

            # Download prediction image
            _, img_data = cv2.imencode(".png", out_img)
            col1.download_button(
                label="Download prediction",
                data=img_data.tobytes(),
                file_name=save_path,
                mime="image/png"
            )

            id2names = {
            0: "boneanomaly", 1: "bonelesion", 2: "foreignbody", 
            3: "fracture", 4: "metal", 5: "periostealreaction", 
            6: "pronatorsign", 7:"softtissue", 8:"text"
            }


            # Count the occurrences of each element
            element_counts = {}

            out_txt = out_txt[0].cpu().tolist()

            for txt in out_txt:
                key = int(str(txt)[0])
                element_counts[key] = out_txt.count(txt)
            
            elements = [id2names[int(str(txt)[0])].lower() for txt in out_txt]
            # Combine counts and elements into a single output text
            output_text = "Detections : "
            output_text += ', '.join([f"{v} {id2names[k]}" for k, v in element_counts.items()])

            print(output_text)


            output_text += "\nBoxes: " + str(out_boxes[0])

            col2.download_button(
                label="Download detections",
                data=output_text,
                file_name=uploaded_file.name[:-4] + ".txt",
                mime="text/plain"
            )

        else:
            # Load and display the original image
            dcm = pydicom.dcmread(uploaded_file)
            img = dcm.pixel_array

            # Inference
            out, out_img = model_inference(model_path, img)
            out_boxes, out_txt = post_process(out)

            # Save the predicted image
            save_path = os.path.splitext(uploaded_file.name)[0] + "_predicted.jpg"
            cv2.imwrite(save_path, out_img)

            # Display the first image using st.image()
            st.image(out_img, caption="Prediction", channels="RGB")

            # Download buttons
            col1, col2 = st.columns(2)

            # Download prediction image
            _, img_data = cv2.imencode(".png", out_img)
            col1.download_button(
                label="Download prediction",
                data=img_data.tobytes(),
                file_name=save_path,
                mime="image/png"
            )

            id2names = {
            0: "boneanomaly", 1: "bonelesion", 2: "foreignbody", 
            3: "fracture", 4: "metal", 5: "periostealreaction", 
            6: "pronatorsign", 7:"softtissue", 8:"text"
            }

            # Count the occurrences of each element
            element_counts = {}

            out_txt = out_txt[0].cpu().tolist()

            for txt in out_txt:
                key = int(str(txt)[0])
                element_counts[key] = out_txt.count(txt)
            
            elements = [id2names[int(str(txt)[0])].lower() for txt in out_txt]
            # Combine counts and elements into a single output text
            output_text = "Detections : "
            output_text += ', '.join([f"{v} {id2names[k]}" for k, v in element_counts.items()])

            print(output_text)


            output_text += "\nBoxes: " + str(out_boxes[0])

            col2.download_button(
                label="Download detections",
                data=output_text,
                file_name=uploaded_file.name[:-4] + ".txt",
                mime="text/plain"
            )


