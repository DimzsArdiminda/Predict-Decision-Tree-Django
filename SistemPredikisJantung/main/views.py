from django.http import HttpResponse
from django.shortcuts import render
from .calculate import train_model_and_predict
import pandas as pd
from main.forms import DatasetForm
from main.models import Datasets

def test1(request):
    return HttpResponse("<h1>Test 1</h1>")

def index(request):
    return render(request, 'pages/index.html')

def form(request):
    return render(request, 'pages/predict/form.html')

def about(request):
    return render(request, 'pages/about/index.html')

def upload_dataset(request):
    if request.method == 'POST':
        form = DatasetForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
        else:
            context = {'form': form}
            return render(request, 'pages/upload/index.html', context)
    context = {'form': DatasetForm()}
    return render(request, 'pages/upload/index.html', context)

def show_dataset(request):
    data = Datasets.objects.all()
    context = {'data': data}
    return render(request, 'pages/upload/show.html', context)

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

        # dd
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
        prediction, accuracy, report, matrix = train_model_and_predict(age, sex, chest_pain_type, resting_bp, cholesterol, fasting_blood_sugar, rest_ecg, max_heart, exercise_angina, oldpeak, st_slope)
        result = "Terkena penyakit jantung" if prediction[0] == 1 else "Tidak terkena penyakit jantung"
        accuracy_message = "Akurasi Model: {:.2f}%".format(accuracy * 100)
        report_message = "Laporan Klasifikasi:\n{}".format(report)
        matrix_message = "Matriks Konfusi:\n{}".format(matrix)
        kosong = 'belum ada hasil'
        # condition if result, accuracy repot_massage matirxc is not None
        if result is not None and accuracy_message is not None and report is not None and matrix is not None:
            return render(request, 'pages/predict/index.html', {
                'result': result,
                'accuracy_message': accuracy_message,
                'classification_report': report,
                'confusion_matrix': matrix.tolist(),  # Ubah confusion matrix ke list untuk template
            })
        else:
                        return render(request, 'pages/predict/index.html', {
                'result': kosong,
            })
