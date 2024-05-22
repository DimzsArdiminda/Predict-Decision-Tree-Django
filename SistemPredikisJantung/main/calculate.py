import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def train_model_and_predict(age, sex, chest_pain_type, resting_bp, cholesterol, fasting_blood_sugar,
                            resting_ecg, max_heart_rate, exercise_angina, oldpeak, st_slope):
    # Langkah 1: Muat data dari lokasi yang tepat
    data = pd.read_csv("../data/dataset.csv")

    # Langkah 2: Persiapan Data
    X = data.drop('target', axis=1)  # Fitur
    y = data['target']  # Target

    # Langkah 3: Membagi data menjadi train dan test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Langkah 4: Membuat dan Melatih Model Decision Tree
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)
    
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

    # Melakukan prediksi menggunakan model Decision Tree yang sudah dilatih
    prediction = model.predict(new_data)

    return prediction, accuracy, report, matrix, model