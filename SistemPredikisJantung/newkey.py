"""
    this script will generate a new secret key for the Django project
    and print it to the console
    
    to use this script, run the following command in the terminal:
    python newkey.py and then copy the secret key to the SECRET_KEY variable in the .env file
    
    or 
    
    you can run the following command in the terminal:
    python manage.py shell -c "from django.core.management.utils import get_random_secret_key; print(get_random_secret_key())"

"""

from django.core.management.utils import get_random_secret_key

secret_key = get_random_secret_key()
print('Copy code : ',secret_key)
