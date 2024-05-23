import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Langkah 1: Muat data dari lokasi yang tepat
data = pd.read_csv("data/dataset.csv")

# Langkah 2: Eksplorasi Data
print("Beberapa baris data pertama: ")
print(data.head())  # Melihat beberapa baris pertama data
print("\nInformasi dataset: ")
print(data.info())  # Informasi mengenai dataset

# Langkah 3: Persiapan Data
X = data.drop('target', axis=1)  # Fitur
y = data['target']  # Target

# Langkah 4: Membagi data menjadi train dan test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Langkah 5: Membuat dan Melatih Model Decision Tree
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Langkah 6: Evaluasi Model
y_pred = model.predict(X_test)

# Langkah 7: Menampilkan hasil evaluasi
accuracy = accuracy_score(y_test, y_pred)
print("\nAkurasi Model: {:.2f}%".format(accuracy * 100))

print("\nLaporan Klasifikasi:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)


import pandas as pd
from sklearn.tree import DecisionTreeClassifier

# Input pengguna untuk prediksi
age = float(input("Masukkan usia (dalam tahun): "))
sex = int(input("Masukkan jenis kelamin (1 = Pria, 0 = Wanita): "))
chest_pain_type = int(input("Masukkan tipe nyeri dada (1, 2, 3, atau 4): "))
resting_bp = float(input("Masukkan tekanan darah istirahat (mm Hg): "))
cholesterol = float(input("Masukkan kolesterol (mg/dl): "))
fasting_blood_sugar = int(input("Masukkan gula darah puasa (1 = >120 mg/dl, 0 = <=120 mg/dl): "))
resting_ecg = int(input("Masukkan hasil elektrokardiogram istirahat (0, 1, atau 2): "))
max_heart_rate = float(input("Masukkan detak jantung maksimum yang dicapai (71-202): "))
exercise_angina = int(input("Apakah Anda mengalami angina yang dipicu oleh latihan? (1 = Ya, 0 = Tidak): "))
oldpeak = float(input("Masukkan depresi ST oldpeak: "))
st_slope = int(input("Masukkan kemiringan segmen ST puncak latihan (0, 1, atau 2): "))

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

# Menampilkan hasil prediksi
if prediction[0] == 0:
    print("Hasil prediksi: Tidak terkena penyakit jantung")
else:
    print("Hasil prediksi: Terkena penyakit jantung")
