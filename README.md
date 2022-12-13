# Skin Cancer Detection Machine Learning Project

Abstract – Throughout the years, skin cancer has always been an issue within the health industry, as cancer itself historical is known to be deadly in the United States. Skin cancer is the abnormal growth of skin cells with several invasive treatments to reduce tumors within the body. The most common types of skin cancer are Melanoma, Squamous Cell Carcinoma, and Basal Cell.  Skin cancer with early detection may reduce risk of progression and fatality among patients. Machine Learning methods within the industry has been increasingly utilized within various clinical applications, providing powerful decision making and analysis capabilities. Machine learning (ML) in the scientific study of algorithms and statistical models that computer systems use to perform a specific task without being explicitly programmed. Learning algorithms in many applications that we make use of daily. With Machine Learning, we can use image object-oriented detection via a supervised classification model to detect if someone is likely to have skin cancer or not. In this review, we will be focusing on the results of machine learning capabilities, using an array of techniques, and how accurate the learning models are when classifying skin cancer. Seven classes of skin cancer lesions are included in the analysis within Python. 

Key Terms: Machine Learning, Convolutional Neural Network, Python, Deep Learning, Skin Cancer
I. INTRODUCTION & BACKGROUND
Skin cancer is the out-of-control growth of abnormal cells in the epidermis, the outermost skin layer, caused by unrepaired DNA damage that triggers mutations. These mutations lead the skin cells to multiply rapidly and form malignant tumors [1]. The main types of skin cancer are basal cell carcinoma, squamous cell carcinoma, melanoma and Merkel cell carcinoma. According to Skin Cancer Foundation, approximately 2 people die every hour within the United States alone from skin cancer [2]. Skin cancer contributes to a high death toll within the United States, however the current methods available are not effective in diagnosing different types of skin cancer. Additionally, there has been debates about methodologies being able to diagnose different types of skin cancers.

The methods used right now to diagnose different types of skin cancer include performing biopsies and medical imaging [3]. There are many ways that a biopsy can be performed, including core-needle, punch, and shave biopsies [4]. Imaging tests can used for most skin cancer detection methods, especially for the main types of skin cancer discussed, BCC, SCC, and MCC. Imaging modalities include CT scans, X-ray, and MRI [3]. Imaging procedures are non-invasive and painless. Dermatologists typically identify the type of skin cancer by using a combination of both biopsies and image analysis to use their best judgement. The degree of each method depends on case-by-case basis. However, these options do not account for accurate levels of detection, as sometimes these non-automated methods misdiagnose patients and are limited by the knowledge of the dermatologist working with the patient. Thus, new diagnosis techniques such as machine learning are warranted, as there is an increase data and information that can be used to diagnose patients.

Deep learning has been revolutionizing the way we diagnose patients for medical conditions. Convolution Neural Networks (CNN), Support Vector Machine (SVM, and K Nearest Neighbors (KNN) are three different methods that are used widely within Clinical applications. A CNN is a deep learning algorithm that can take an input image, in this case the images of the skin from patient obtained. The algorithm can assign importance to specific characteristics that they detect within each image. Based off the detection of certain characteristics, it can determine if there is skin cancer present, and identify the type of skin cancer, based on the training data already available to the algorithm [5]. KNN for image detection works by quantifying certain features of the skin images, such as pigmentation. Based on the values of the pigmentation, we can plot these and store these in a database from the training data. The training data points value stored in the database are then compared to the input data quantifying the feature.  Based on the number of points (n points) from the training data being compared which is determined by the K optimal value, there will be a determination on the type of skin cancer the picture exhibits or if it does not exhibit skin cancer. Once all the photos are processed, the KNN will make a final determination of the type of skin cancer or if no skin cancer is present [6]. SVM is another type of supervised machine learning algorithm, which can be used for image detection via plotting each data item as a point in n-dimensional space, n being the number of features present [7]. In general, the value of each feature within the data set is the value of a particular coordinate. Classification is then performed by finding the optimal plane that can differentiate the number of labels well [7].

KNN have been used to detect skin cancer lesions [8] and CNN have been used to detect skin cancer lesions as well [9]. Both of these experiments were successful, with CNN having a 93.1% success rate [9] and the KNN having a 98% accuracy with their implementation. In another study, for detecting melanoma skin cancer lesions, an SVM model was successful, obtaining a 92.1% success rate [10]. However, it is still currently unknown which machine learning algorithm would be better for a clinical diagnosis with different algorithms, that contain the same testing data set.

The image data are made up of 7 classes of skin cancer. Melanocytic nevi (nv), Melanoma (mel), Benign keratosis-like lesions (bkl), Basal cell carcinoma (bcc), Actinic keratoses (akiec), Vascular lesions (vas), and Dermatofibroma (df).

The research objective of this study was to compare the effectiveness of CNN and SVM for classifying different skin cancer types. To achieve this, we trained and tested two machine learning models with identical data image sets that can identify 7 skin cancer lesion types. Afterwards, we compare the classification accuracy of the algorithms. We hypothesize that the CNN model will more accurately classify skin cancer types than the SVM model. 
II. DATA & METHODS
The dataset that is used in the research is called the HAM10000, which is a large collection of multi-source dermatoscopic images made of common pigmented skin lesions [11]. This dataset was used to train neural networks for automated diagnosis of pigmented skin lesions and is considered optimal for its small main variety of 7 skin lesion types. In total, the dataset provides 10015 image data depicting the skin lesions. For 50% of the labels of the lesions, they have been confirmed through histopathology. The other 50% of the labels of the skin lesions have been confirmed through either follow-up examinations of patients, expert consensuses, confirmation via confocal microscopy.

As depicted in figure 1, the database contains different amounts of data for each specific lesion that we will be observing. The lesion types are disproportionate of each other, the df type having the least amount of data images, with 115 images, and nv type having the most amount of data images having 6705 images.
 
 
Fig.1 Bar Graph representation of total skin lesion types present within the dataset.

Additionally, as depicted in figure 2, the locations of where each image was taken is different. The least being acral location, where 7 image data were taken. The most being the back, where 2192 image data were taken.
 
Fig.2 Bar Graph representation of total skin lesion locations where the image data were taken.

As depicted in figure 3, the genders of the patients that the photos were taken from were slightly off balance, with roughly 53.9% being male, 45.4% being female, and less than 1% being anonymous.

 
Fig.3 Bar Graph representation of total patients and their respective genders.

As depicted in figure 4, the age of each patient varies from 1 year of age to 80 years of age. The bulk of the image data coming from ages 35 through 60. Age group 45-49 having the most patients with 1299, and Age group 0-4 having the least patients with 39.

 
Fig.4 Histogram representation of total amount of patients and their respective age groups.

Since we have 7 different data types, 7 different labels are assigned to each one, depicted from 0 through 6. This was accomplished through python, after loading in all the image data, 7 different data frames are made which is specific to one label type. More specifically, akiec refers to label 0, bcc refers to label 1, bkl refers to label 2, df refers to label 3, mel refers to label 4, nv refers to label 5, and vasc refers to label 6.

After we separate the images based on their labels into their own data frames, we need to make sure that each data frame has the equal amount of image data. We cannot have one dataset with only 115 images while another has 6705 images. Therefore, resampling must be done. By doing resampling, databases with less than 500, are upscaled in quantity to become 500. Likewise, databases with more than 500 are downscaled in quantity to become 500. The data frames that were downscaled randomly had data taken away from the data frame until it reached 500. Meanwhile, data frames with less than 500 had data randomly repeated the data until 500 data files were reached. An important distinction to note is that no data augmentation was done. No new data was randomly generated that was not part of the original set.

After resampling is done, all the 7 individual data frames were merged into one data frame, consisting of 3500 data images, 500 from each respective label type. Manipulation of the data frames were then conducted to convert the entire data frame into a NumPy array, for the python-based CNN and SVM model to run. Of the 3500 data images, 25% of the image data are used for testing, while 75% of the image data are used for training the model of the CNN and SVM. That means 875 image data is being used for testing data set, and 2625 image data is being used for training data set.

The CNN model uses the sequential perceptron method. There are a total of 3 hidden layers which uses the rectified linear function (RELU) as the activation method. Within this research study, the CNN model will 100 epochs model tests. For the SVM model, within Python a CNN model can be converted into a SVM model via adding a parameter known as kernel regularizing. Linear activation functions are added to help determine the final output layer in the model creation. The SVM model is linear SVM based model, using SoftMax as an activation function as the linear activation function [12].

Figure 5 shows a 3 skin lesions of each cancer tumor type being used in this study. These are few of the many image data that will be processed into the SVM and CNN models.

 

Fig.5 Visuals of the actual images taken from patients of each of the 7 lesion types.

After the tests for running how well the CNN model works, graphs depicting the training and validation accuracy and loss will be shown to see how well the predictions were, and if there is any overfitting or underfitting. A bar chart depicting the incorrect labels will be shown based off the test. Additionally, a confusion matrix will be compiled to show the true positives, false positives, true negatives, and false negatives within the model.

The KNN model will load the same data, which has already been resampled for each label type individually and combined back into a single data frame. Of the 3500 data images, 25% of the image data are used for testing, while 75% is used for training and validation.

All the data of the training and validation will be plotted for visualization. Next, a KNN model will be implemented, implementing multiple k values at multiple times. The optimal k value will then be compared to the CNN model with the optimal number of epochs being used.
III. RESULTS
As depicted in figure 6, the CNN model training data loss at 100 epochs results in roughly 0.5 loss, while the validation data at 100 epochs is 0.8 loss. Loss is the values are that represents the difference from the true target state.
 
Fig.6 A graphical representation of CNN training & validation data loss over a period of 100 Epochs

As depicted in figure 7, the CNN model training data accuracy at 100 epochs results in roughly 0.8 accuracy, while the validation data at 100 epochs is 0.75 accuracy. Accuracy refers to the percentage correctly classified in predicting labels, in comparison to the true label types.
 
Fig.7 A graphical representation of CNN training & validation accuracy over a period of 100 Epochs

As depicted in figure 8, the CNN model incorrect number of predictions for each respective label type. Displayed in a fractional form, of the total fraction of each label type present in the validation/testing set.
 
Fig.8 A bar graph representation of CNN training & validation fractional accuracy correct of each label type over a period of 100 Epochs

As depicted in figure 9, the CNN model confusion matrix, where the true label axis refers to the actual labels, and predicted label axis refers to the predicted label type within the model. From this one can see the false positive predictions, true positive predictions, false negative predictions, and true negative predictions.
 
Fig.9 A graphical representation of CNN confusion matrix, what are the true labels and predictions, compared to false positives.

As depicted in figure 10, the CNN ROC curve and AUC, which the ROC curve calculated and plotted the true positive rate (TPR) against the false positive rate (FPR). This was done for each label type. The AUC is the area under the ROC curve, the higher the AUC curve value, the better the overall model performance.
 
Fig.10 An ROC graph which depicts the TPR and FPR against each other respectively, with an AUC calculation, depicting the performance of the model.

As depicted in figure 11, The SVM model training data loss at 100 epochs results in roughly around 0.95 loss, while the validation data at 100 epochs is 1.05 loss. Loss is the values are that represents the difference from the true target state.
 
Fig.11 A graphical representation of SVM training & validation data loss over a period of 100 Epochs

As depicted in figure 12, the SVM model training data accuracy at 100 epochs results in roughly 0.9 accuracy, while the validation data at 100 epochs is 0.7 accuracy. Accuracy refers to the percentage correctly classified in predicting labels, in comparison to the true label types.
 
Fig.12 A graphical representation of SVM training & validation data loss over a period of 100 Epochs

As depicted in figure 13, the SVM model incorrect number of predictions for each respective label type. Displayed in a fractional form, of the total fraction of each label type present in the validation/testing set.

 
Fig.13 A bar graph representation of SVM training & validation fractional accuracy correct of each label type over a period of 100 Epochs

As depicted in figure 14, the SVM model confusion matrix, where the true label axis refers to the actual labels, and predicted label axis refers to the predicted label type within the model. From this one can see the false positive predictions, true positive predictions, false negative predictions, and true negative predictions.
 
Fig.14 A graphical representation of SVM confusion matrix, what are the true labels and predictions, compared to false positives.

As depicted in figure 15, the CNN ROC curve and AUC, which the ROC curve calculated and plotted the true positive rate (TPR) against the false positive rate (FPR). This was done for each label type. The AUC is the area under the ROC curve, the higher the AUC curve value, the better the overall model performance.
 
Fig.15 A graphical representation of SVM confusion matrix, what are the true labels and predictions, compared to false positives.
IV. DISCUSSION & SUMMARY
The results depict many figures for the testing data set. The CNN model figures give us a lot of information about the performance of the CNN model. 

Within the set-up, 100 epochs were chosen as the standard for this experiment. It was not known at which epoch would the data be optimal for overfitting and underfitting for both models. Therefore, to maintain the integrity of the data and to allow analysis of the two models on the image data used, the epochs was set as 100 for both models and remained as such. As discussed below, there are signs of overfitting present in both models, which is not at 100. The models regardless have better accuracy of the testing data overall from 0 to 100. It is also witnessed that the accuracy of the testing data is not improving after 100, rather it remained relatively same after 100 epochs as the value percentage accuracy was not going up by whole percentages, rather decimal percentage values. Additionally, 100 epochs were the computer that models were built on could do without risk of overheating the laptop. Therefore, due to physical constraints of the laptop, 100 epochs were also chosen.

Based on the figure 6, the loss of the training and validation sets with the CNN model are very good and align well. The loss implies the model behaves well after each epoch, as it decreases over epochs. There is no major overfitting, with overfitting starting to become visible after 70 epochs. The loss, for the training dataset ends around 0.5 at 100 epochs. Meanwhile, the loss for validation dataset ends around 0.8 at 100 epochs.

Based on figure 7, the accuracy of the training and validation sets within the CNN both behaved very well. The accuracy saturating around 80 epochs, with the training accuracy staying around 80% accuracy in predictions while the validation accuracy stays around 70% accuracy in predictions.

Based on figure 8, label 5 experienced the worst prediction accuracy, 40% being misclassified, label 4 being the second most misclassified with 34%, label 2 had 33% misclassification, label 1 experienced 26% misclassification, label 0 experienced 24% misclassification, label 3 experienced 3% misclassification, and label 6 experienced 2% misclassification. There seems to be correlation with the resampling, the least affected label types from the resampling had higher misclassification, while the most impacted label types from resampling exhibits low misclassification. The order of labels that were most impacted to least impacted from resampling is 3,6,0,1,2,4, and 5.

Based on figure 9, the confusion matrix elaborates the overall model was able to predict the labels true positives, with minimal false positive amounts. In total 677 predictions were accurate, thus true positives. In total 198 predictions were misclassified, thus false positives in the testing data set.

Based on figure 10, the ROC curves were very good for each label, the TPR for all labels proved to be good against the FPR. The AUC calculations were relatively high for all the labels too. AUC scores being for label 0 is 0.9572, label 1 is 0.9515, label 2 is 0.9209, label 3 is 0.9977, label 4 is 0.9152, label 5 is 0.9063, and label 6 is 0.9991. Comparing this to figure 8, the worst prediction accuracy label types also received lower AUC scores. However, the AUC scores are all about 0.9 which means the scores are all good. 

The SVM model figures also give us information about the performance of the SVM model.
Based on the figure 11, the loss of the training and validation sets with the SVM model are very good and align well too. The loss implies the model behaves well after each epoch, as it decreases over epochs. There is overfitting present, which starts early on, with slight overfitting starting to become visible around 60 epochs. The loss, for the training dataset ends around 0.93 at 100 epochs. Meanwhile, the loss for validation dataset ends around 1.04 at 100 epochs.

Based on figure 12, the accuracy of the training and validation sets within the SVM both behaved very well. The accuracy saturating around 80 epochs, with the training accuracy staying around 90% accuracy in predictions while the validation accuracy stays around 70% accuracy in predictions.

Based on figure 13, label 5 experienced the worst prediction accuracy, 40% being misclassified, label 4 being the second most misclassified with 34%, label 2 had 33% misclassification, label 1 experienced 32% misclassification, label 0 experienced 31% misclassification, label 3 experienced 11% misclassification, and label 6 experienced 2% misclassification. There seems to be correlation with the resampling, the least affected label types from the resampling had higher misclassification, while the most impacted label types from resampling exhibits low misclassification. As stated, the order of labels that were most impacted to least impacted from resampling is 3,6,0,1,2,4, and 5.

Based on figure 9, the confusion matrix elaborates the overall model was able to predict the labels true positives, with minimal false positive amounts. In total 638 predictions were accurate, thus true positives. In total 237 predictions were misclassified, thus false positives in the testing data set.

Based on figure 10, the ROC curves were very good for each label, the TPR for all labels proved to be good against the FPR. The AUC calculations were relatively okay for all the labels too. AUC scores being for label 0 is 0.9297, label 1 is 0.8910, label 2 is 0.8840, label 3 is 0.9646, label 4 is 0.8991, label 5 is 0.8581, and label 6 is 0.9977. Again, comparing this to figure 13, the worst prediction accuracy label types also received lower AUC scores. Some label AUC scores dipped below 0.9, which is the threshold for this project, which is not good compared to the CNN. 

Since both models had the same order of misclassification label rankings, and there is a relationship between resampled labels and the misclassification of labels, the resampling method potentially could have influenced the outcome of the two models’ ability to accurately classify the labels.

The CNN outperformed the SVM for every label AUC score as depicted below in table 1. 



Table 1: AUC Scores for Each Label & Model Type
	Label 0	Label 1	Label 2	Label 3	Label 4	Label 5	Label 6
CNN 	0.957	0.951	0.920	0.997	0.915	0.906	0.999
SVM	0.929	0.891	0.884	0.964	0.899	0.858	0.997

The CNN model outperformed the SVM model when classifying the testing data, with CNN doing 677 predictions accurately, while SVM did 638 predictions accurately. However, it does take 13 seconds per epoch for the model to run, while the SVM model takes 2 seconds per epoch. Despite the SVM having an advantage in run time, the CNN has the edge in performance, accurately predicting more labels than the SVM. Therefore, the CNN has more supporting information for the CNN to be considered the better model in skin cancer tumor detection.

Limitations of this project is the image data sets given. Each tumor type was unproportionally provided for image data, so resampling had to be done. Another limitation was the model implementations, given the memory and hardware of the laptop used to create the model, higher quality models that take longer to predict and stronger computing power was unrealistic to make. Especially the idea of going significantly past 100 epochs, as the laptop would overheat too much.

Future directions I would like to take for this project is to come up with how resampling methods impact the data classification, as the resampling method has responsibility for how the results came the way they did, as it impacted the training data set used to train the model, and thus impacting the testing data set for classification. This project was also done at specifically 100 epochs. Next time, there could be value in finding the optimal epoch where there is relatively no overfitting or underfitting for both models. 
