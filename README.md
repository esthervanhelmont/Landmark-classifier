# Landmark-classifier
## Landmark Image Classifier using Convolutional Neural Networks (CNN)

This project uses a Convolutional Neural Network (CNN) to classify images of world landmarks.  
By training a deep learning model on labelled landmark photos, we can automatically recognize places such as famous buildings, statues, or natural sites.

This type of model is useful for tourism apps, automated tagging, cultural heritage archives, and visual search systems.

---

## Summary

This project builds and trains a CNN that can classify landmark images into predefined categories.  
The workflow includes:

- Loading and preprocessing image datasets  
- Applying data augmentation for robustness  
- Building a custom CNN model  
- Training, evaluating, and improving model performance  
- Generating predictions on new images  

The goal is to create a model that can generalize well to new unseen pictures of the same landmarks.

---

## Hypothesis

We assume that:

1. Images of landmarks contain consistent visual patterns (shapes, textures, colors).  
2. A CNN can learn these patterns if enough labelled training images are provided.  
3. Data augmentation will help the model handle real-world variation (lighting, zoom, angles).  

If the CNN learns these features well, it should accurately classify landmarks even with noise or small variations.

---

## Dataset Info

- **Content:** Images of different landmarks, each stored in separate class folders  
- **Format:** RGB images loaded through an image generator  
- **Labels:** Folder names represent the target class  
- **Train / Validation Split:** Automatically handled by the ImageDataGenerator  

### Example transformations applied

- Rescaling (normalizing pixel values)  
- Random rotation  
- Horizontal flipping  
- Zoom augmentation  
- Width & height shifts  

These steps improve generalization and reduce overfitting.

---

## Notebook Index

### **1. Load & Prepare Dataset**
- Import and inspect image folders  
- Create train and validation generators  
- Apply augmentation (rotation, zoom, etc.)  
- Resize images to model input shape (e.g., 150×150 or 224×224)

---

### **2. Build CNN Model**
The architecture typically includes:

- **Convolution layers** to extract features  
- **MaxPooling layers** to reduce spatial size  
- **Flatten layer** to convert output to a vector  
- **Dense layers** for final classification  
- **Softmax output layer** for multi-class prediction  

The model uses:

- **ReLU** activation  
- **Categorical Cross-Entropy** loss  
- **Adam** optimizer  

---

### **3. Train Model**
- Fit the model using the training generator  
- Validate using the validation generator each epoch  
- Track loss and accuracy over time  
- (Optional) Use callbacks like EarlyStopping or ModelCheckpoint  

---

### **4. Evaluate Performance**
- Plot training vs validation accuracy  
- Plot training vs validation loss  
- Detect overfitting or underfitting  
- Generate predictions for test images  
- Build a confusion matrix (if implemented)

---

### **5. Make Predictions**
The notebook demonstrates how to:

- Load a new image  
- Preprocess it (resize, scale)  
- Run the model prediction  
- Display the predicted class and probabilities  

---

## Final Results

The CNN successfully learns visual patterns from the landmark dataset and can classify unseen images with good accuracy for images of 59%

**Expected outcomes:**

- Accurate recognition of the landmark classes  
- Improved robustness thanks to augmentation  
- Better performance after tuning parameters and architecture  

---

## Takeaway

This project shows how CNNs can automatically classify complex visual data such as landmarks.  
With proper preprocessing, augmentation, and model tuning, deep learning becomes a powerful tool for image classification tasks.

---
