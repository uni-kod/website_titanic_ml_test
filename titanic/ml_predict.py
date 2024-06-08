OUTPUT_MODEL_FILE='/home/lap_3/Documents/work/courses/full_stack/python_ai_django//my_code/section_11/lecture_287/titanic_model.sav'


def prediction_model(pclass, sex, age, sibsp, parch, fare, embarked, title):
  import pickle
  x = [[pclass, sex, age, sibsp, parch, fare, embarked, title]]
  random_forest = pickle.load(open(OUTPUT_MODEL_FILE, 'rb'))
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
