# Clothing Fit Predictor

## Overview
The **Clothing Fit Predictor** is a Streamlit-based web application that predicts the fit of clothing based on user inputs such as bust size, weight, height, age, and other relevant factors. It leverages a **Random Forest Classifier** trained on rental clothing data.

---

## Folder Structure
```
Clothing Fit Predictor/
│-- Data/
│   ├── cleaned_data.csv
│   ├── renttherunway_final_data.json
│
│-- EDA and Model Script/
│   ├── script.ipynb  # Exploratory Data Analysis & Model Training
│
│-- Model and Objects/
│   ├── random_forest_model.pkl
│   ├── scaler.pkl
│   ├── label_encoders.pkl
│   ├── target_encoder.pkl
│
│-- main.py  # Streamlit App Script
│-- requirements.txt  # Required Libraries
│-- README.md  # Project Documentation
```

---

## Installation
### **Step 1: Clone the Repository**
```sh
git clone https://github.com/mushaid01/Clothing-Fit-Predictor.git
cd Clothing Fit Predictor
```

### **Step 2: Install Dependencies**
```sh
pip install -r requirements.txt
```

---

## Running the Application
### **Option 1: Run Streamlit**
```sh
streamlit run main.py
```

### **Option 2: Use Python Module Execution**
```sh
python -m streamlit run main.py
```

Once running, open the **localhost URL** in your browser to interact with the app.

---

## Dataset
The project utilizes rental clothing data:
- **`cleaned_data.csv`**: Preprocessed dataset used for training the model.
- **`renttherunway_final_data.json`**: Raw dataset before cleaning.

---

## Model & Objects
The trained model and preprocessing objects are stored in the `Model and Objects` folder:
- `random_forest_model.pkl` - Trained model.
- `scaler.pkl` - Scaler used for feature normalization.
- `label_encoders.pkl` - Encoders for categorical variables.
- `target_encoder.pkl` - Encoder for target labels.

---

## Notebook (EDA & Model Training)
The **`script.ipynb`** file contains:
- **Exploratory Data Analysis (EDA)**
- **Feature Engineering & Selection**
- **Model Training & Evaluation**

---

## Results
### **Predicted Fit Results and Visualizations**
#### Image 1:
![Result Image 1](https://drive.google.com/uc?id=1wDfyb7ebcVSzGh299xDoGUsqGmlPAexw)

#### Image 2:
![Result Image 2](https://drive.google.com/uc?id=1G-Pf1Lwwa973q_hUSPKR_1OiyLCXLwkX)

#### Image 3:
![Result Image 3](https://drive.google.com/uc?id=1Q8G2ZwZ-sQESWBZXhpgpZldBJoRaa0qc)

#### Image 4:
![Result Image 4](https://drive.google.com/uc?id=12-h4BWOyKPBAKxwWFikDjjAirWY80dBq)

---

## Requirements
The necessary dependencies are listed in **requirements.txt**:
```txt
streamlit
joblib
numpy
pandas
plotly
wordcloud
```

To install them, run:
```sh
pip install -r requirements.txt
```

---

## Author
Developed by **Mir Mushaidul Islam**.

For any queries, feel free to reach out!
