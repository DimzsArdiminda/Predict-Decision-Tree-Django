from django.http import HttpResponse
from django.shortcuts import render, get_object_or_404, redirect
from .calculate import train_model_and_predict
import pandas as pd
from main.forms import DatasetForm
from main.models import Datasets, Models
from django.db.models import Q
import pandas as pd
from sklearn.tree import DecisionTreeClassifier 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
from django.core.files import File


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

def list_dataset(request):
    data = Datasets.objects.all()
    context = {'data': data}
    return render(request, 'pages/upload/list.html', context)


def delete_dataset(request, pk):
    data = get_object_or_404(Datasets, pk = pk)
    model = Models.objects.filter(Q(title = data.title)).first()
    if model:
        model.delete()
    data.delete()
    return redirect('list-dataset')


def display_dataset(request, pk):
    file = Datasets.objects.get(pk = pk)
    model = Models.objects.filter(Q(title = file.title)).exists()
    data = pd.read_csv(file.dataset)
    headers = data.columns.to_list()
    data_preview = data.head(10).to_html()
    context = {'data_preview': data_preview, 'headers' : headers, 'pk' : pk, 'model' : model}
    return render(request, 'pages/upload/display.html', context)


def create_model(request, pk):
    file = Datasets.objects.get(pk = pk)
    data = pd.read_csv(file.dataset)
    if request.method == 'POST':
        label_class = request.POST.get('label_class')
        feature_cols = data.drop(label_class, axis=1)
    
        X = feature_cols
        y = data[label_class]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
        
        model = DecisionTreeClassifier()
        model = model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        matrix = confusion_matrix(y_test, y_pred)
        
        filename = file.title.replace(" ", "-") + '.pkl'
        with open(filename, 'wb') as f:
            pickle.dump(model, f)
        
        with open(filename, 'rb') as f:
            model_file = File(f)
            Models.objects.create(
                title = file.title,
                model = model_file,
                accuracy = accuracy, 
                report = report, 
                matrix = matrix,
            )
        
        # context = {'accuracy': accuracy, 'report': report, 'matrix': matrix, "data" : file}
        # return render(request, 'pages/result/index.html', context)
        return redirect('model', pk)


def model(request, pk):
    data = Datasets.objects.get(pk = pk)
    model = Models.objects.filter(Q(title = data.title)).first()
    
    accuracy = f"{model.accuracy * 100:.2f}"
    report = model.report
    matrix_str = model.matrix.replace('[', ' ').replace(']', ' ')
    matrix = [list(map(int, row.split())) for row in matrix_str.split('\n')]
    
    context = {'accuracy': accuracy, 'report': report, 'matrix': matrix, "data" : data}
    return render(request, 'pages/result/index.html', context)

def predict(request):
    pass


def predict_view(request,pk):
    data = get_object_or_404(Datasets, pk=pk)
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
        prediction, accuracy, report, matrix, status = train_model_and_predict(age, sex, chest_pain_type, resting_bp, cholesterol, fasting_blood_sugar, rest_ecg, max_heart, exercise_angina, oldpeak, st_slope)
        result = "Terkena penyakit jantung" if prediction[0] == 1 else "Tidak terkena penyakit jantung"
        accuracy_message = "Akurasi Model: {:.2f}%".format(accuracy * 100)
        report_message = "Laporan Klasifikasi:\n{}".format(report)
        matrix_message = "Matriks Konfusi:\n{}".format(matrix)
        status_meesage = "Status: {}".format(status)
        return render(request, 'pages/predict/index.html', {
            'result': result,
            'accuracy_message': accuracy_message,
            'classification_report': report,
            'confusion_matrix': matrix.tolist(),  # Ubah confusion matrix ke list untuk template
        })
    else:
        context = {
            'result': "Belum ada hasil",
            'accuracy_message': None,
            'classification_report': None,
            'confusion_matrix': None,
            "Status_Model" : None,
        }
        return render(request, 'pages/predict/index.html', context)