# Potato Leaf Disease Detection with Hadoop, Spark and Deep Learning

This project builds a distributed image-classification pipeline for potato leaf disease detection. Images are stored in **Hadoop HDFS**, loaded and labelled with **Apache Spark**, then passed into PyTorch deep-learning models for classification.

The project was originally completed as part of my MSc Data Analytics coursework, and is now structured as a portfolio project for roles involving data infrastructure, reliability, distributed systems and machine-learning pipelines.

## Why This Project Matters

The goal is not only to train a classifier, but to show how an image dataset can be handled through a scalable data pipeline:

- Store image classes in HDFS for distributed access.
- Load image binaries with Spark using `spark.read.format("image")`.
- Extract labels from HDFS paths using Spark transformations.
- Convert Spark-loaded image data into model-ready tensors.
- Compare multiple model architectures using the same evaluation approach.

This makes the project relevant to infrastructure-oriented roles because it combines **distributed storage**, **parallel data processing**, **Python automation**, **model evaluation** and **resource-aware execution**.

## Architecture

```text
Kaggle potato leaf dataset
        |
        v
Hadoop HDFS
  /user1/potato_disease/
    Potato___Early_blight/
    Potato___Late_blight/
    Potato___healthy/
        |
        v
Apache Spark image loader
  - executor memory: 8g
  - driver memory: 8g
  - image schema validation
  - label extraction from paths
        |
        v
PyTorch preprocessing
  - resize to 224 x 224
  - normalization
  - augmentation
  - train/test split
        |
        v
Model training and evaluation
  - Custom CNN
  - VGG16
  - MobileNetV2
```

## Dataset

- **Source:** Kaggle potato leaf disease dataset
- **Images used:** 2,000 images loaded from HDFS in the model notebooks
- **Classes:** `Potato___Early_blight`, `Potato___Late_blight`, `Potato___healthy`
- **Input size:** 224 x 224 RGB
- **Train/test split:** 80% / 20%
- **Test set size:** 400 images

The full image dataset is not included in this repository because of size. Download it from Kaggle and upload the class folders to HDFS before running the notebooks.

## Results

| Model | Train Accuracy | Test Accuracy | Test Loss | Weighted F1 |
| --- | ---: | ---: | ---: | ---: |
| Custom CNN | 0.927 | 0.927 | 0.1626 | 0.92 |
| VGG16 | 0.951 | 0.930 | 0.1813 | 0.92 |
| MobileNetV2 | 0.953 | 0.948 | 0.2361 | 0.94 |

**Best overall model:** MobileNetV2 achieved the highest test accuracy at **94.8%** and the strongest weighted F1-score at **0.94**.

The saved training curves and confusion matrices are included in the repository:

- `training_validation_plots_cnn.png`
- `training_validation_plots.png`
- `training_validation_plots_mobilenet.png`
- `confusion_matrix_cnn.png`
- `confusion_matrix.png`
- `confusion_matrix_mobelinet.png`

## Model Comparison

### Custom CNN

The custom CNN provides a lightweight baseline with two convolutional layers, pooling, dropout and fully connected layers. It reached **92.7% test accuracy**, which is strong for a compact model.

### VGG16

VGG16 used pretrained convolutional layers with a custom classifier head. It reached **93.0% test accuracy** and performed well on Early Blight and Late Blight, but had lower recall on the Healthy class.

### MobileNetV2

MobileNetV2 produced the best result with **94.8% test accuracy**. It is also the most deployment-friendly option because it is lighter than VGG16 while achieving better performance.

## Infrastructure and Reliability Notes

This project is especially relevant to cloud infrastructure and SRE-style roles because it includes:

- **Distributed storage:** image classes are stored in HDFS.
- **Distributed processing:** Spark reads and labels image data before model training.
- **Resource configuration:** Spark driver and executor memory are explicitly configured.
- **Pipeline validation:** notebook cells print Spark schemas, sample records and extracted labels.
- **Reproducible evaluation:** all three models report train loss, test loss, accuracy, confusion matrix and classification report.

Potential production improvements:

- Containerize Hadoop, Spark and the Python runtime with Docker Compose.
- Add automated smoke tests for HDFS availability and Spark image loading.
- Add structured logging around ingestion, preprocessing and training stages.
- Track model metrics with MLflow or a similar experiment tracker.
- Add a lightweight inference API with `/health` and `/predict` endpoints.

## Repository Structure

```text
.
├── CA1Sem2MadinaSagatova2021255.ipynb   # Initial Spark/HDFS and TensorFlow exploration
├── CNN_Model.ipynb                       # Custom CNN implementation and evaluation
├── VGG16_model.ipynb                     # VGG16 transfer-learning implementation
├── MobiletNetV2_model.ipynb              # MobileNetV2 transfer-learning implementation
├── requirements.txt                      # Python dependencies
├── confusion_matrix*.png                 # Saved confusion matrices
└── training_validation_plots*.png        # Saved training curves
```

## Setup

### 1. Clone the Repository

```bash
git clone https://github.com/madinasagatova/Potato-Leaf-Disease-Detection.git
cd Potato-Leaf-Disease-Detection
```

### 2. Create a Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

On Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install Python Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

PyTorch installation can vary by operating system and GPU/CPU setup. If needed, install the correct PyTorch build from the official selector: https://pytorch.org/get-started/locally/

### 4. Start Hadoop and Upload the Dataset

Create the HDFS project directory:

```bash
hdfs dfs -mkdir -p /user1/potato_disease
```

Upload the three class folders:

```bash
hdfs dfs -put Potato___Early_blight /user1/potato_disease
hdfs dfs -put Potato___Late_blight /user1/potato_disease
hdfs dfs -put Potato___healthy /user1/potato_disease
```

Verify the upload:

```bash
hdfs dfs -ls /user1/potato_disease
```

### 5. Run the Notebooks

Start Jupyter:

```bash
jupyter notebook
```

Run the model notebooks:

1. `CNN_Model.ipynb`
2. `VGG16_model.ipynb`
3. `MobiletNetV2_model.ipynb`

The notebooks expect HDFS to be reachable at:

```text
hdfs://localhost:9000/user1/potato_disease/*
```

## Skills Demonstrated

- Python data engineering
- Hadoop HDFS storage
- Apache Spark image ingestion
- PyTorch model training
- Transfer learning with VGG16 and MobileNetV2
- Classification metrics and confusion matrix analysis
- Resource-aware ML pipeline execution

## Resume Summary

**Distributed ML pipeline for potato leaf disease detection:** built a Hadoop HDFS and Apache Spark image-processing pipeline, trained CNN, VGG16 and MobileNetV2 classifiers in PyTorch, and achieved **94.8% test accuracy** with MobileNetV2 across three disease classes.

## License

This academic project uses a public Kaggle dataset under the license stated by the dataset provider. Code in this repository is provided for academic and portfolio use.

## Author

Madina Sagatova  
MSc Data Analytics, CCT College Dublin
