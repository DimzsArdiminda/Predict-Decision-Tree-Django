
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import model

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
