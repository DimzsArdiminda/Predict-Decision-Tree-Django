from django.http import HttpResponse
from django.shortcuts import render
from .calculate import train_model_and_predict
import pandas as pd

def test1(request):
    return HttpResponse("<h1>Test 1</h1>")

def index(request):
    return render(request, 'pages/index.html')

def about(request):
    return render(request, 'pages/about/index.html')

def predict_view(request):
    if request.method == 'POST':
        # Mengambil nilai dari form sebagai string
        age_str = request.POST.get('age')
        sex_str = request.POST.get('sex')
        chest_pain_type_str = request.POST.get('chest_pain_type')
        resting_bp_str = request.POST.get('resting_bp')
        cholesterol_str = request.POST.get('cholesterol')
        fasting_blood_sugar_str = request.POST.get('fasting_blood_sugar')
        rest_ecg_str = request.POST.get('rest_ecg')
        max_heart_str = request.POST.get('max_heart_rate')
        exercise_angina_str = request.POST.get('exercise_angina')
        oldpeak_str = request.POST.get('oldpeak')
        st_slope_str = request.POST.get('slope')

        # Validasi dan konversi ke integer jika nilai tidak kosong
        try:
            age = int(age_str) if age_str else None
            sex = int(sex_str) if sex_str else None
            chest_pain_type = int(chest_pain_type_str) if chest_pain_type_str else None
            resting_bp = int(resting_bp_str) if resting_bp_str else None
            cholesterol = int(cholesterol_str) if cholesterol_str else None
            fasting_blood_sugar = int(fasting_blood_sugar_str) if fasting_blood_sugar_str else None
            rest_ecg = int(rest_ecg_str) if rest_ecg_str else None
            max_heart = int(max_heart_str) if max_heart_str else None
            exercise_angina = int(exercise_angina_str) if exercise_angina_str else None
            oldpeak = float(oldpeak_str) if oldpeak_str else None
            st_slope = int(st_slope_str) if st_slope_str else None
        except ValueError:
            # Penanganan jika konversi gagal (misalnya karena input tidak valid)
            return HttpResponse("Invalid input. Please enter valid values.")

        # Gunakan nilai-nilai integer tersebut untuk prediksi
        # new_data = pd.DataFrame({
        #     'age': [age],
        #     'sex': [sex],
        #     'chest_pain_type': [chest_pain_type],
        #     'resting_bp': [resting_bp],
        #     'cholesterol': [cholesterol],
        #     'fasting_blood_sugar': [fasting_blood_sugar],
        #     'rest_ecg': [rest_ecg],
        #     'max_heart_rate': [max_heart],
        #     'exercise_angina': [exercise_angina],
        #     'oldpeak': [oldpeak],
        #     'st_slope': [st_slope]
        # })
        # print(new_data)

        # # Lakukan prediksi menggunakan model
        prediction = train_model_and_predict(age, sex, chest_pain_type, resting_bp, cholesterol, fasting_blood_sugar, rest_ecg, max_heart, exercise_angina, oldpeak, st_slope)
        result = "Terkena penyakit jantung" if prediction[0] == 1 else "Tidak terkena penyakit jantung"

        return render(request, 'pages/index.html', {'result': result})