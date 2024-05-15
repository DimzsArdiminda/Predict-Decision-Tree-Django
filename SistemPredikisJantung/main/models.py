from django.db import models


class Datasets(models.Model):
    title = models.CharField(max_length=255)
    dataset = models.FileField(upload_to='dataset/')
    
