# from django.templatetags.static import static

# ML_MODEL_FILE = static('machine_learning/titanic_model.sav')

import os
from django.conf import settings

ML_MODEL_FILE = os.path.join(settings.STATIC_ROOT, 'machine_learning/titanic_model.sav')


def prediction_model(pclass, sex, age, sibsp, parch, fare, embarked, title):
  import pickle
  x = [[pclass, sex, age, sibsp, parch, fare, embarked, title]]
  random_forest = pickle.load(open(ML_MODEL_FILE, 'rb'))
  prediction = random_forest.predict(x)

  if prediction == 0:
    prediction = 'Not survived'
  elif prediction == 1:
    prediction = 'Survived'
  else:
    # It should be 0 or 1.
    prediction = 'Error'

  return prediction

# Test the prediction model.
#prediction_model(1, 1, 11, 1, 1, 19, 1, 1)
