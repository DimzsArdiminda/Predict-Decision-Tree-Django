from django.db import models
from django.core.exceptions import ValidationError
from django.core.validators import FileExtensionValidator

def validate_file(value):
    ext = value.name.split('.')[-1]
    if not ext.lower() == 'csv':
        raise ValidationError('Unsupported file extension. Only CSV files are supported.')
    if value.size > 5000000:
        raise ValidationError('File size should not exceed 5MB.')


class Datasets(models.Model):
    title = models.CharField(max_length=255)
    dataset = models.FileField(upload_to='dataset/', validators=[validate_file])
    
    def delete(self):
        self.dataset.delete()
        super().delete()
        
class Models(models.Model):
    title = models.CharField(max_length=255)
    model = models.FileField(upload_to='model/', validators=[FileExtensionValidator(['pkl'])])
    accuracy = models.FloatField(default=None)
    report = models.TextField(default=None)
    matrix = models.TextField(default=None)
        
    def delete(self):
        self.model.delete()
        super().delete()