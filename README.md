Model Card: YOLOv8-Bone-Fracture-Detection 

Model Description:  

Detect and recognize bone fractures, implants, and other abnormalities in X-ray images with bounding box localization and label output. 

Fine-tuned by: Musa Yilmaz / Open Institue of Technology 

Model type: Object Detection & Recognition 

Language(s): 

Python: 99.3%  

Other: 0.7% 

 

License: GNU AFFERO GENERAL PUBLIC LICENSE Version 3, 19 November 2007 

Finetuned from model: YOLOv8 

Repository: https://github.com/yilmazmusa08/YOLOv8-Bone-Fracture-Detection 

 

Uses 

Intended Use: 

Assist radiologists and orthopedic surgeons in the detection and diagnosis of bone fractures and abnormalities in X-ray images. 

Aid in treatment planning by providing accurate localization and identification of fractures, implants, and other pathologies. 

Educational purposes to train medical students and residents in the interpretation of X-ray images. 

 

Out-of-Scope Use: 

Unauthorized surveillance or invasion of privacy. 

Non-medical purposes. 

 

Ethical Considerations: 

Data Bias: Model performance might vary depending on the diversity and representativeness of the training data. 

Privacy Concerns: X-ray images might contain sensitive patient information. 

Misuse Potential: Risk of misuse if the model is used for unauthorized purposes. 

 

How to Get Started with the Model 

Go to /App directory, use command: 

python streamlit run app.py 

 

Training Details: 

The training data originates from the "Bone Fracture v2" dataset available on Roboflow, accessible through the link provided. 

Dataset link: https://universe.roboflow.com/capjamesg/bone-fracture-v2/dataset/3 

Train Set: 1630 Images (71%) 

Validation Set:  440 Images (19%) 

Test Set:  220 Images (10%) 

  

Evaluation 

Performance Metrics: 

Accuracy (Detection): 80% 

Accuracy (Recognition): 65% 

Citation: 

BibTeX: 

@software{jocher2023ultralytics, 

  author = {Jocher, Glenn and Chaurasia, Ayush and Qiu, Jing}, 

  title = {Ultralytics YOLO}, 

  version = {8.0.0}, 

  date = {2023-1-10}, 

  url = {https://github.com/ultralytics/ultralytics}} 

APA: 

Jocher, G., Chaurasia, A., & Qiu, J. (2023). Ultralytics YOLO (Version 8.0.0) [Software]. Retrieved from https://github.com/ultralytics/ultralytics 

 
