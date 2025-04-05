# 🕰️ Renaissance-layout-organization-recognition

## **Abstract**

This project focuses on segmenting ***main text regions*** in digitized ***early modern printed documents*** by learning to distinguish them from ***marginalia***, ***decorative borders***, and ***non-textual elements***. Leveraging a ***U-Net architecture*** with a ***ResNet18 encoder***, the system is trained on ***synthetic segmentation masks*** generated using ***Tesseract OCR*** and custom pre-processing of ***Renaissance-era scans***. The aim is to simulate an ***intelligent layout understanding system*** capable of isolating ***core textual zones***, supporting downstream tasks in ***OCR***, ***document restoration***, and ***digital archiving***. This work contributes to the development of robust pipelines for ***historical document analysis*** and the ***digital preservation*** of ***Renaissance textual heritage***.

---

## 🔎 **Approach**

### **1. Data Preparation**
- **Historical Image Collection**: *Renaissance-era printed document scans* are collected and curated. These contain both *main body text* and accompanying noise such as *marginalia*, *woodcut borders*, and *decorative elements*.
- **Synthetic Ground Truth Generation**: Segmentation masks are automatically generated using **`Tesseract OCR`**, which identifies *main text lines*. *Custom preprocessing techniques* filter out noisy or irrelevant text to refine these masks for *supervised training*.
- **Dataset Structuring**: Images and their corresponding masks are stored in a structured format (`train/`, `val/`, `test/`), facilitating easy loading via **`PyTorch Dataset`** and **`DataLoader`** classes.

### **2. Preprocessing**
- All input images are resized to a fixed resolution of **256×256** pixels.
- *Normalization* is applied using *ImageNet mean and standard deviation*, aligning with the expectations of the **ResNet18 encoder**.
- *Ground truth masks* are converted to binary format, where *main text regions* are labeled as `1` and all other content as `0`.

### **3. Model Architecture**
- The model is based on the **U-Net** segmentation architecture with a pretrained **ResNet18 encoder** (backbone from **`segmentation_models_pytorch`**).
- The *encoder* captures high-level semantic features while the *decoder* upsamples and fuses them to produce *pixel-wise predictions*.
- The *output layer* produces a *single-channel binary mask* identifying the **main text regions**.

### **4. Training Strategy**

#### **Loss Function**
- Uses **Binary Cross Entropy (BCE) Loss**, which is well-suited for *binary segmentation tasks* like identifying *main text versus background/marginalia*.

#### **Optimizer & Scheduler**
- The model is trained using the **Adam optimizer** with default *betas* and a learning rate of `1e-3`.
- A **learning rate scheduler** (**`ReduceLROnPlateau`**) is employed to dynamically reduce the learning rate when *validation loss* plateaus.

#### **Training Parameters**
- Trained for **500 epochs** with a **batch size of 4**.
- At each step, the model’s performance is evaluated using the **Dice coefficient**, providing a balance between *precision* and *recall* for *imbalanced segmentation tasks*.

### **5. Evaluation & Visualization**
- The model is evaluated on a held-out *validation set* with metrics such as **validation loss** and **Dice score** logged per epoch.
- Sample predictions are *visualized and saved* using **`matplotlib`**, showcasing how well the model distinguishes *main text from background noise*.
- *Real-time progress tracking* is done via **`tqdm`**, with *best model weights* saved automatically.

### **6. Output & Utility**
- The trained model generates clean **binary masks** that isolate the **core textual regions** of early modern documents.
- These masks can be used as *preprocessing inputs* for downstream tasks like:
  - **OCR text extraction**
  - **Document structure modeling**
  - **Visual restoration pipelines**
- By filtering out *decorative and marginal components*, the system enhances focus on *semantically meaningful content* for *historical document analysis*.


---

## ✔ **Evaluation Metrics**

To evaluate the performance of the **layout segmentation model**, I used the **Structural Similarity Index Measure (SSIM)** as the primary metric. SSIM compares the *visual similarity* between the predicted segmentation mask and the ground truth mask, providing a perceptual measure of how closely the predicted layout matches the expected one.

### **1. Structural Similarity Index (SSIM)**
- SSIM evaluates *luminance*, *contrast*, and *structural similarity* between two images.
- In the context of segmentation masks, it helps assess how well the model captures the *layout structure* and *text region boundaries* of historical documents.
- A **higher SSIM score** (closer to `1.0`) indicates that the predicted mask closely resembles the ground truth in terms of *spatial accuracy* and *layout consistency*.
- This is particularly important for *digitized Renaissance scans*, where layout structure is often noisy or visually complex due to *degradation*, *ink bleed*, and *decorative elements*.

The use of **SSIM** ensures that the model's output is not just pixel-accurate but also *structurally faithful*, enabling better performance in *downstream tasks* such as **OCR**, **document restoration**, and **digital archiving**.


### **2. Dice Coefficient**
- The Dice Coefficient (also known as the Sørensen–Dice index) measures the *overlap* between the predicted and ground truth regions.
- It is particularly effective for *binary segmentation tasks* where the focus is on identifying *main text blocks* versus *background* or *decorative elements*.
- Defined as:  
  \[
  \text{Dice} = \frac{2 \cdot |A \cap B|}{|A| + |B|}
  \]
  where \( A \) is the predicted region and \( B \) is the ground truth.

---

## 👀 **Results Analysis**

| **Metric** | **Best Score** | **Average Score** | **Median Score** | **Maximum Score** | **Minimum Score** |
|-----------|----------------|-------------------|------------------|-------------------|-------------------|
| **SSIM**     | 0.988759       | 0.838045          | 0.857945         | 0.988759          | 0.677889          |

---

## **Evaluation Summary**

### **SSIM**
The highest SSIM score observed was **0.9888**, indicating that the model was able to produce segmentation outputs with *near-perfect structural similarity* to the ground truth in ideal scenarios. On average, the SSIM score across the test set was **0.8380**, with a median of **0.8579**, suggesting that the model consistently captured the *spatial layout and structure* of main text regions across diverse Renaissance-era scans. Even the lowest SSIM score, **0.6779**, demonstrates that the model retained a reasonable level of *layout fidelity* even under challenging conditions such as *severe noise*, *decorative borders*, or *marginalia interference*.


---

## 🖼 **Recognized Layout Organization**

<p align="center">
  <img src="GEN_IMAGES/8.jpg" width="200" />
  <img src="GEN_IMAGES/15.jpg" width="200" />
  <img src="GEN_IMAGES/291.jpg" width="200" />
</p>

<p align="center">
  <img src="GEN_IMAGES/609.jpg" width="200" />
  <img src="GEN_IMAGES/318.jpg" width="200" />
  <img src="GEN_IMAGES/339.jpg" width="200" />
</p>

<p align="center">
  <img src="GEN_IMAGES/364.jpg" width="200" />
  <img src="GEN_IMAGES/529.jpg" width="200" />
  <img src="GEN_IMAGES/530.jpg" width="200" />
</p>




---

## 🚀 **Future Improvements**

### **1. Post-Processing with Morphological Refinement**
Although the model performs well in segmenting core text regions, applying *morphological operations* (e.g., dilation, erosion, contour smoothing) post-inference could enhance *region boundary accuracy* and eliminate small false positives from decorative or marginal elements.

### **2. Integration of Attention Mechanisms**
Incorporating *self-attention* or *spatial attention* modules into the U-Net architecture could help the model better focus on *contextual layout cues*, especially in distinguishing densely packed marginalia from the main body text.

### **3. Multi-Class Layout Segmentation**
Currently, the model performs binary segmentation (text vs non-text). Extending it to a *multi-class layout model*—detecting headings, footnotes, images, and decorative borders—would support more advanced downstream applications like *semantic layout parsing* and *region-specific OCR*.

### **4. Real Historical Data Fine-Tuning**
While synthetic segmentation masks serve as a solid foundation, fine-tuning the model on a *small set of manually labeled real Renaissance scans* could significantly boost performance by learning authentic layout distortions and artifacts.

### **5. Resolution Scaling and Tiling**
Handling full-page high-resolution scans can be computationally expensive. Implementing a *tiling strategy with overlapping windows* and recombining predictions can enable *scalable processing* without compromising layout context or precision.

### **6. Ensemble Learning or Model Distillation**
Combining outputs from multiple lightweight U-Net variants or using *knowledge distillation* can result in faster inference while preserving *high segmentation accuracy*, especially beneficial for deployment in *digital archiving pipelines*.


---

## 🔗 **Download Trained Model**

You can download the trained Layout Organization Recognition model weights here:
- [**Layout Organization Recognition Weights** (`layout_organization_recognition_model.pth`)](https://drive.google.com/file/d/1DbjUIDHpj9yE-W0vBb7LW3snDo527nkR/view?usp=sharing)

---

## 💻 **Implementation Guide**

Refer to this [**notebook**](layout_organization_recognition.ipynb) for the general guideline.

### **1. Install Required Packages**

Ensure that you have the required packages installed by running:

```bash
pip install -r requirements.txt
```

###2. Run the Main Script:
   
Execute the main script:
```bash
python layout_organization_recognition.py 
```
