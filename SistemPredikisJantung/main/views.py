import os
import base64
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
from django.db.models import Q
from django.core.files import File
from django.http import HttpResponse, FileResponse
from django.shortcuts import render, get_object_or_404, redirect
from sklearn import tree
from sklearn.model_selection import train_test_split 
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from .calculate import train_model_and_predict, load_dataset_and_model
from main.forms import DatasetForm, PredictForm
from main.models import Datasets, Models


def index(request):
    return render(request, 'pages/index.html')


def form(request, pk):
    context = {'pk': pk}
    return render(request, 'pages/predict/form.html', context)


def about(request):
    return render(request, 'pages/about/index.html')


def upload_dataset(request):
    if request.method == 'POST':
        form = DatasetForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            return redirect('list-dataset')
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
    data = get_object_or_404(Datasets, pk=pk)
    model = Models.objects.filter(Q(title=data.title)).first()
    model_path = 'model/' + data.title.replace(" ", "-") + '.pkl'
    visualization_path = 'output/' + data.title.replace(" ", "-") + '.pdf'
    if model:
        model.delete()
        os.remove(model_path)
        os.remove(visualization_path)
    data.delete()
    return redirect('list-dataset')


def display_dataset(request, pk):
    file = Datasets.objects.get(pk=pk)
    model = Models.objects.filter(Q(title=file.title)).exists()
    data = pd.read_csv(file.dataset)
    headers = data.columns.to_list()
    data_preview = data.head(10).to_html()
    context = {'data_preview': data_preview, 'headers': headers, 'pk': pk, 'model': model}
    return render(request, 'pages/upload/display.html', context)


def create_model(request, pk):
    file = Datasets.objects.get(pk=pk)
    data = pd.read_csv(file.dataset)
    if request.method == 'POST':
        label_class = request.POST.get('label_class')
        feature_cols = data.drop(label_class, axis=1)
        headers = feature_cols.columns.to_list()
        
        X = feature_cols
        y = data[label_class]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

        clf = DecisionTreeClassifier()
        model = clf.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        matrix = confusion_matrix(y_test, y_pred)

        # save model .pkl
        model_path = 'model/' + file.title.replace(" ", "-") + '.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        # save visualization model .pdf
        fig = plt.figure(figsize=(25,20))
        dot_data = tree.plot_tree(clf, 
                                feature_names=headers,
                                filled=True,
                                rounded=True)
        visualization_path = 'output/' + file.title.replace(" ", "-") + '.pdf'
        fig.savefig(visualization_path)
        
        with open(visualization_path, 'rb') as f1, open(model_path, 'rb') as f2:
            visualization_file = File(f1)
            model_file = File(f2)
            Models.objects.create(
                title=file.title,
                model=model_file,
                visualization=visualization_file,
                accuracy=accuracy,
                report=report,
                matrix=matrix,
            )

        return redirect('model', pk)


def model(request, pk):
    data = Datasets.objects.get(pk=pk)
    model = Models.objects.filter(Q(title=data.title)).first()
    
    accuracy = f"{model.accuracy * 100:.2f}"
    report = model.report
    matrix_str = model.matrix.replace('[', ' ').replace(']', ' ')
    matrix = [list(map(int, row.split())) for row in matrix_str.split('\n')]

    context = {'accuracy': accuracy, 
                'report': report, 
                'matrix': matrix,
                'model': model, 
                'pk': pk}
    return render(request, 'pages/result/index.html', context)


def serve_pdf(request, pk):
    model = Models.objects.filter(pk=pk).first()
    pdf_file_path = model.visualization.path
    return FileResponse(open(pdf_file_path, 'rb'), content_type='application/pdf')


def predict_view(request, pk):
    data = get_object_or_404(Datasets, pk=pk)
    model_data = Models.objects.filter(Q(title=data.title)).first()
    model_path = model_data.model.path
    with open(model_path, 'rb') as f:
        loaded_classifier = pickle.load(f)
    
    if request.method == 'POST':
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
            return HttpResponse("Invalid input. Please enter valid values.")

        # Menyiapkan input untuk prediksi
        new_data = pd.DataFrame({
            'age': [age],
            'sex': [sex],
            'chest pain type': [chest_pain_type],
            'resting bp s': [resting_bp],
            'cholesterol': [cholesterol],
            'fasting blood sugar': [fasting_blood_sugar],
            'resting ecg': [rest_ecg],
            'max heart rate': [max_heart],
            'exercise angina': [exercise_angina],
            'oldpeak': [oldpeak],
            'ST slope': [st_slope]
        })

        prediction = loaded_classifier.predict(new_data)
        result = "Terkena penyakit jantung" if prediction[0] == 1 else "Tidak terkena penyakit jantung"
        accuracy = f"{model_data.accuracy * 100:.2f}"
        report = model_data.report
        matrix_str = model_data.matrix.replace('[', ' ').replace(']', ' ')
        matrix = [list(map(int, row.split())) for row in matrix_str.split('\n')]

        if result is not None and accuracy is not None and report is not None and matrix is not None:
            return render(request, 'pages/predict/index.html', {
                'result': result,
                'accuracy_message': accuracy,
                'classification_report': report,
                'confusion_matrix': matrix,
                'pk': model_data.pk,
            })
        else:
            return render(request, 'pages/predict/index.html', {
                'result': "belum ada hasil",
            })
    else:
        return render(request, 'pages/predict/index.html', {
            'result': ' belum ada hasil',
        })

