import pandas as pd

training_data = pd.read_csv('training_results.csv').sort_values(by='date')
dates = sorted(set(training_data['date']))

sorted_by_exercise = training_data.groupby(['date', 'exercise'])['reps'].sum().reset_index().groupby('exercise')

bicep_curls_results = sorted_by_exercise.get_group('Bicep-curls')
pull_ups_results = sorted_by_exercise.get_group('Pull-ups')
push_ups_results = sorted_by_exercise.get_group('Push-ups')
squats_results = sorted_by_exercise.get_group('Squats')

print(bicep_curls_results)
print(pull_ups_results)
print(push_ups_results)
print(squats_results)
