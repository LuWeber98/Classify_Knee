# Classify Knee
This is a project which will use the annotated Dataset for Knee Arthritis Detection found on Kaggle: https://www.kaggle.com/datasets/hafiznouman786/annotated-dataset-for-knee-arthritis-detection

The dataset with 1650 knee X-Rays holds 5 image folders corresponding the K&L number from 0-4 to classify
the severity of arthritis    

Prescreening of the data set:
![Verteilung der KL-Grade](<General_Info/Verteilung der KL-Grade.png>)

Workflow:

1. Split the dataset into 3 datasets: train (70%), validate (15%) and test (15%)
2. Preprocess the images   
  Might apply a crop for the region of interest and CLAHE 
3. Apply Transfer Learning  
ResNet50 as starting point and fine tune as it provided best performance in the classification of osteoarthritis in knee X-Rays: https://www.nature.com/articles/s41598-025-24338-4
4. Test and check the model performance against each other

  
Current results:  
Custom CNN:   
  
ResNet 50: best result so far: Epoch [29/30] Train Loss: 0.7002 Train Acc: 0.7584 Val Loss: 0.9298 Val Acc: 0.6356
