# Detection-of-icons-on-Webpage
Detection of icons on webpage using MultiClass Classification and Deployed using StreamLit

As the problem statement defines it is an problem where the user uploads an Screenshot of an Image and the system should be able to identify the icon on the webpage screenshot and provide an output as an image with the properly classified image name. The Problem is an MultiClass Classification problem where there are several different approach which can be used to solve the given problem. 
The first approach which was followed was using YOLO V3 where the outputs can be used to identify the given problem. As the YOLO V3 model has prebuild model and prebuild weights the model is able to identify the classes which are only in COCO Dataset as the weights belong to that Dataset. And to be able to use YOLO i needed to create my own weights and own class name files which is an different kindoff approach which can be used to solve the given problem.

The second approach which can be used create the model is using VGG16 Model which is an conventional CV model used to identify objects in an image. The model is created which is using the approach of Transfer Learning where use of Early Stop and Call Back is done to save the best best model in .H5 format and later the same is used to make predictions using StreamLit. You can upload an Image using the dataset given by me and make predictions based on that.
For this project i have created my own dataset and trained my model. There are total 43 classes which can be used identify. Use the files in the icons subfolders to identify.

Well this is the Version 1.0 of this project, the future scope includes Version 2.0 which will be used for Object Detection in an image where multiple objects within an image can be Identified using the same and that will also be updated soon by me.

Pre-requisites: Installed StreamLit Any Version to run .py file

Below is the link to Google Drive for the .H5 file which has the saved Model and Weights with 43 different Classes
Attachments: https://drive.google.com/file/d/16bkcVb8lSektGm0HKMTtjk_6wK9mcYSo/view?usp=share_link
