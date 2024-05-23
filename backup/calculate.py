import pandas as pd
import pickle
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.impute import SimpleImputer
from main.models import Datasets, Models  # Sesuaikan dengan nama aplikasi Anda

def load_dataset_and_model():
    # Ambil dataset terbaru dari basis data
    latest_dataset = Datasets.objects.last()
    if latest_dataset:
        dataset_path = latest_dataset.dataset.path
    else:
        raise Exception("No dataset found in the database.")

    # Ambil model berdasarkan dataset
    related_model = Models.objects.filter(title=latest_dataset.title).first()
    if related_model:
        model_path = related_model.model.path
    else:
        raise Exception("No model found in the database for the latest dataset.")

    # Muat dataset dan model
    data = pd.read_csv(dataset_path)
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    return data, model

def train_model_and_predict(age, sex, chest_pain_type, resting_bp, cholesterol, fasting_blood_sugar,
                            resting_ecg, max_heart_rate, exercise_angina, oldpeak, st_slope):
    # Langkah 1: Muat dataset dan model
    data, model = load_dataset_and_model()

    # Langkah 2: Persiapan Data
    X = data.drop('target', axis=1)  # Fitur
    y = data['target']  # Target
    
    # Mengisi missing values
    imputer = SimpleImputer(strategy='mean')
    X = imputer.fit_transform(X)

    # Langkah 3: Membagi data menjadi train dan test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Langkah 4: Melatih Model Decision Tree (model sudah dilatih)
    
    # 5. Evaluasi model
    somerows = data.head(),
    info = data.info(),
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    matrix = confusion_matrix(y_test, y_pred)
    
    # Menyiapkan input untuk prediksi
    new_data = pd.DataFrame({
        'age': [age],
        'sex': [sex],
        'chest pain type': [chest_pain_type],
        'resting bp s': [resting_bp],
        'cholesterol': [cholesterol],
        'fasting blood sugar': [fasting_blood_sugar],
        'resting ecg': [resting_ecg],
        'max heart rate': [max_heart_rate],
        'exercise angina': [exercise_angina],
        'oldpeak': [oldpeak],
        'ST slope': [st_slope]
    })

    # Mengisi missing values pada data baru
    new_data = imputer.transform(new_data)

    # Melakukan prediksi menggunakan model Decision Tree yang sudah dilatih
    prediction = model.predict(new_data)

    return prediction, accuracy, report, matrix, model
