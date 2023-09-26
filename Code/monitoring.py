experiment_ids = ["532585148427554333", "142704143808532679", "654653382223571780", "892816848484979934"]
model_names = ['LR Model lbfgs', 'LR Model newton-cholesky', 'RF Model 20 trees', 'RF Model 40 trees']
# Dict der Modells, um die Experiment-ID zu den Modells mappen zu können
model_dict = {"532585148427554333": "LR Model lbfgs",
            "142704143808532679": "LR Model newton-cholesky",
            "654653382223571780": "RF Model 20 trees",
            "892816848484979934": "RF Model 40 trees"}

import mlflow
mlflow.set_tracking_uri("file:///c:/Users/Paul%20Strohmeier/Desktop/ma-accantec/Code/mlruns")
print(mlflow.get_tracking_uri())

def get_column_mapping(new_data):
    from evidently import ColumnMapping

    # Regulärer Ausdrucksmuster für den Spaltenfilter
    pattern = r'^(?!.*route).*$'

    # Filtern der Spalten basierend auf dem Ausdrucksmuster
    cat_columns = list(new_data.filter(regex=pattern, axis=1))
    cat_columns.remove("age")
    cat_columns.remove("serious")
    print("Liste mit kategorischen Spalten erstellt: {}".format(cat_columns))

    # Numerische Spalten
    num_columns = list(new_data.filter(regex='route').columns)
    num_columns.append('age')
    print("Liste mit numerischen Spalten erstellt: {}".format(num_columns))

    ## Column Mapping erstellen
    column_mapping = ColumnMapping()

    column_mapping.target = 'serious'

    column_mapping.numerical_features = num_columns #list of numerical features
    column_mapping.categorical_features = cat_columns #list of categorical features

    column_mapping.datetime = None
    print("Column Mapping erstellt.")

    return column_mapping

def create_data_reports(new_data, date_info):
    from evidently.report import Report
    from evidently.metric_preset import DataDriftPreset, DataQualityPreset
    import os
    import pandas as pd

    # Datum der Datei für Ordnererstellung nutzen
    folder_path = r'c:\Users\Paul Strohmeier\Desktop\ma-accantec\Reports\{}'.format(date_info)

    # Ordner erstellen
    os.makedirs(folder_path)
    print(folder_path + " erfolgreich erstellt.")

    ### Berichte
    # Daten von dem aktuellen Modell
    path = r"C:\Users\Paul Strohmeier\Desktop\ma-accantec\Daten\csv\reference"
    file_name = os.listdir(path)[0]
    ref_data = pd.read_csv(os.path.join(path, file_name))
    print("Referenzdaten geladen: " + file_name)

    ## Data Drift Report
    print("Data Drift Report begonnen.")
    drift_report = Report(metrics=[DataDriftPreset(drift_share=0.2)])
    drift_report.run(reference_data=ref_data, current_data=new_data, column_mapping=get_column_mapping(new_data))
    drift_report.save_html(os.path.join(folder_path, "Drift_Report.html"))
    print("Data Drift Report gespeichert.")

    
    ## Quality Report
    print("Quality Report begonnen.")
    quality_report = Report(metrics=[DataQualityPreset()])
    quality_report.run(reference_data=ref_data, current_data=new_data, column_mapping=get_column_mapping(new_data))
    quality_report.save_html(os.path.join(folder_path, "Quality_Report.html"))
    print("Quality Report gespeichert.")
    
    
def drift_test(new_data, date_info):
    from evidently.test_suite import TestSuite
    from evidently.tests import TestShareOfDriftedColumns
    import os
    import pandas as pd

    path = r"C:\Users\Paul Strohmeier\Desktop\ma-accantec\Daten\csv\reference"
    file_name = os.listdir(path)[0]
    ref_data = pd.read_csv(os.path.join(path, file_name))

    data_drift = TestSuite(tests=[
        TestShareOfDriftedColumns(lt=0.2)
    ])

    data_drift.run(reference_data=ref_data, current_data=new_data, column_mapping=get_column_mapping(new_data))
    drift_dict = data_drift.as_dict()
    print(drift_dict['tests'][0]['description'])
    test_passed = drift_dict['summary']['all_passed']
    
    if test_passed:
        print('Es wurde kein Drift erkannt. Somit muss das Modell vorerst nicht neutrainiert werden.')
        performance_monitoring(new_data, date_info)
    else:
        # Trigger Retraining
        print('Data Drift erkannt. Die Modelle werden mit den neuen Daten neutrainiert.')
        retrain_models(new_data, date_info)

    # Grafik aktualisieren, nachdem alles durchgealaufen ist
    renew_chart(date_info)


def performance_monitoring(data, date_info):
    import mlflow
    from sklearn import metrics
    import json

    # Labels 
    true_labels = data.serious
    X_test = data.drop('serious', axis=1)

    # besten F1-Score vom letzten Training ermitteln
    f1_file = r'c:\Users\Paul Strohmeier\Desktop\ma-accantec\Daten\best_f1.json'

    with open(f1_file, 'r') as file:
        d = json.load(file)
        best_exp_id = d['experiment_id']
        best_f1_score = d['f1-score']
    print("Modell von bestem F1-Wert: {}".format(model_dict[best_exp_id]))
    print("Bester F1-Wert vom letzten Training: {}".format(best_f1_score))

    # besten F1-Score der vier Modelle ermitteln
    new_best_f1_score = 0
    new_best_exp_id = ''

    for model, exp_id in zip(model_names, experiment_ids):
        cur_model = mlflow.pyfunc.load_model(model_uri=f"models:/{model}/latest")
        predictions = cur_model.predict(X_test)
        test_f1 = metrics.f1_score(true_labels, predictions)

        if test_f1 > new_best_f1_score:
            new_best_f1_score = test_f1
            new_best_exp_id = exp_id

    f1_score_dif = (best_f1_score - new_best_f1_score) / best_f1_score
    print("Der prozentuale Unterschied zwischen dem ursprünglichen und neuem F1-Score des jeweils besten Modells liegt bei " + str(f1_score_dif))

    if f1_score_dif >= 0.05:
        print("Neutraining erforderlich, da F1-Score mindestens 5% schlechter.")
        retrain_models(data, date_info)
    else:
        print("F1-Score ist {} und somit nicht um mindestens 5% schlechter als beim letzen Training. Es erfolgt kein Neutraining".format(new_best_f1_score))
        log_model_performance(data, date_info)
        # Wenn anderes Modell nun besser, müssen die Tags geändert werden
        if best_exp_id != new_best_exp_id:
            change_role_tag()
        else: 
            print("Das momentane Champion-Modell hat mit {} immer noch den besten F1-Score.".format(new_best_f1_score))


def log_model_performance(data, date_info):
    import mlflow
    from sklearn import metrics

    # für jedes Modell die Metriken berechnen und loggen
    run_names = ['LR_lbfgs_', 'LR_newton-cholesky_', 'RF_20trees_', 'RF_40trees_']

    # int64 muss zu int32 konvertiert werden, damit die Modelle die Daten lesen können
    data = data.astype({col: 'int32' for col in data.select_dtypes('int64').columns})

    # Labels 
    true_labels = data['serious']
    X_test = data.drop('serious', axis=1)

    for model, exp_id, run_name in zip(model_names, experiment_ids, run_names):
        cur_model = mlflow.pyfunc.load_model(model_uri=f"models:/{model}/latest")
        print("Modell geladen: " + model)
        cur_run_name = run_name + date_info
        print("Runname: " + cur_run_name)

        with mlflow.start_run(experiment_id=exp_id, run_name=cur_run_name) as run:
            # Use the model to make predictions on the test dataset.
            predictions = cur_model.predict(X_test)

            # Test Metrics
            #test_accuracy = metrics.accuracy_score(true_labels, predictions)
            test_f1 = metrics.f1_score(true_labels, predictions)
            mlflow.log_metric('f1_score_X_test', test_f1)
            #test_precision = metrics.precision_score(true_labels, predictions)
            #test_recall = metrics.recall_score(true_labels, predictions)

            print("Test-Metriken für {} berechnet und geloggt".format(model))
       

def change_role_tag():
    import mlflow

    new_runs = mlflow.search_runs(
                experiment_ids=experiment_ids,
                max_results = 4,
                order_by=["start_time DESC"],
            )
    new_runs = new_runs.sort_values(by=['metrics.f1_score_X_test'], ascending=False).reset_index(drop=True)

    # Rollen der Modelle zuordnen
    champ_model = model_dict[new_runs.experiment_id[0]]
    challenger_model = model_dict[new_runs.experiment_id[1]]
    other_models = []
    other_models.append(model_dict[new_runs.experiment_id[2]])
    other_models.append(model_dict[new_runs.experiment_id[3]])

    # Tag der Modelle setzen
    client = mlflow.MlflowClient()
    client.set_registered_model_tag(champ_model, "role", "champion")
    client.set_registered_model_tag(challenger_model, "role", "challenger")
    for model in other_models:
        client.set_registered_model_tag(model, "role", "none")

    # Ausgabe für Debugging und Info
    print("Das neue Champion-Modell ist: " + champ_model)
    print("Das neue Challenger-Modell ist: " + challenger_model)
    

def retrain_models(new_data, date_info):
    import mlflow
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn import metrics
    import os
    import data_acquisition
    import json

    print(mlflow.get_tracking_uri())
    
    lr_params = ['lbfgs', 'newton-cholesky']
    lr_exp_id = experiment_ids[:2]
    rf_params = [20, 40]
    rf_exp_id = experiment_ids[2:]

    y = new_data['serious']
    X = new_data.drop('serious', axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    for solver, exp_id in zip(lr_params, lr_exp_id):

        run_name = 'LR_' + solver + '_' + date_info
        model_name = 'LR Model ' + solver

        with mlflow.start_run(experiment_id=exp_id, run_name=run_name) as run:

            # Create and train models.
            lr = LogisticRegression(solver=solver)
            lr.fit(X_train, y_train)

            # Use the model to make predictions on the test dataset.
            predictions = lr.predict(X_test)

            # Test Metrics
            #test_accuracy = metrics.accuracy_score(true_labels, predictions)
            test_f1 = metrics.f1_score(y_test, predictions)
            mlflow.log_metric('f1_score_X_test', test_f1)
            #test_precision = metrics.precision_score(true_labels, predictions)
            #test_recall = metrics.recall_score(true_labels, predictions)

            print("Test-Metriken für {} berechnet und geloggt".format(model_name))

            # Register Model
            signature = mlflow.models.signature.infer_signature(X_test, predictions)
            mlflow.sklearn.log_model(lr, model_name, signature=signature, registered_model_name=model_name)


    for trees, exp_id in zip(rf_params, rf_exp_id):

        run_name = 'RF_' + str(trees) + 'trees_' + date_info
        model_name = 'RF Model ' + str(trees) + ' trees'

        with mlflow.start_run(experiment_id=exp_id, run_name=run_name) as run:

            # Create and train models.
            rf = RandomForestClassifier(n_estimators=trees)
            rf.fit(X_train, y_train)

            # Use the model to make predictions on the test dataset.
            predictions = rf.predict(X_test)

            # Test Metrics
            #test_accuracy = metrics.accuracy_score(true_labels, predictions)
            test_f1 = metrics.f1_score(y_test, predictions)
            mlflow.log_metric('f1_score_X_test', test_f1)
            #test_precision = metrics.precision_score(true_labels, predictions)
            #test_recall = metrics.recall_score(true_labels, predictions)

            print("Test-Metriken für {} berechnet und geloggt".format(model_name))

            # Register Model
            signature = mlflow.models.signature.infer_signature(X_test, predictions)
            mlflow.sklearn.log_model(rf, model_name, signature=signature, registered_model_name=model_name)

    # Bestes Modell wählen
    change_role_tag()

    # alte Referenzdaten löschen
    path = r"C:\Users\Paul Strohmeier\Desktop\ma-accantec\Daten\csv\reference"
    file_name = os.listdir(path)[0]
    os.remove(os.path.join(path, file_name))

    # und neuen Datensatz als neue Referenzdaten abspeichern  
    prefix = 'ml_'
    counter = str(data_acquisition.read_counter_val())
    time_label = '_' + date_info
    new_file_name = prefix + counter + time_label + '.csv'

    new_data.to_csv(os.path.join(path, new_file_name), index=False)

    # Neuen besten F1-Score abspeichern
    new_runs = mlflow.search_runs(
        experiment_ids=experiment_ids,
        max_results = 4,
        order_by=["start_time DESC"],
    )
    new_id = new_runs.sort_values(by=['metrics.f1_score_X_test'], ascending=False)['experiment_id'][0]
    new_f1 = new_runs.sort_values(by=['metrics.f1_score_X_test'], ascending=False)['metrics.f1_score_X_test'][0]

    f1_file = r'c:\Users\Paul Strohmeier\Desktop\ma-accantec\Daten\best_f1.json'
    with open(f1_file, 'w') as file:
        data = {'experiment_id': new_id,
                'f1-score': new_f1}
        json.dump(data, file)
    print("F1-Datei aktualisiert.")

    # Marker aktualsieren, wann immer neutrainiert wurde
    training_dates = r'c:\Users\Paul Strohmeier\Desktop\ma-accantec\Daten\training_dates.json'
    with open(training_dates, 'r+') as file:
        data = json.load(file)
        data['training-dates'].append(date_info)
        file.seek(0)
        json.dump(data, file)
    print("Training-Dates aktualisiert.")


def renew_chart(date_info):
    import mlflow
    import matplotlib.pyplot as plt
    import os
    import json

    run = mlflow.search_runs(
        experiment_ids=experiment_ids,
        order_by=["start_time"],
    )

    run['model'] = run['experiment_id'].replace(model_dict)
    run['date'] = run['tags.mlflow.runName'].apply(lambda x: x.split('_')[-1])
    data = run[['model','metrics.f1_score_X_test', 'date']]
    print(data)

    plt.figure(figsize=(10, 6))
    for model, subset in data.groupby('model'):
        plt.plot(subset['date'], subset['metrics.f1_score_X_test'], marker='o', label=model)

    # Füge Markierungen für Zeitpunkte des Neutrainings hinzu
    training_dates = r'c:\Users\Paul Strohmeier\Desktop\ma-accantec\Daten\training_dates.json'
    with open(training_dates, 'r') as file:
        neutraining_dates = json.load(file)
    added_legend_entry = False
    for year in neutraining_dates['training-dates']:
        if not added_legend_entry:
            plt.axvline(x=year, color='gray', linestyle=':', linewidth=1.5, label='Training der Modelle')
            added_legend_entry = True
        else: 
            plt.axvline(x=year, color='gray', linestyle=':', linewidth=1.5)

    plt.xlabel('Datum')
    plt.ylabel('F1-Score')
    plt.title('F1-Score pro Modell')
    plt.legend(loc='upper left')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    #plt.ylim(0,1)

    # Grafik abspeichern!
    path = r"C:\Users\Paul Strohmeier\Desktop\ma-accantec\Grafiken\f1_score"
    file_name = date_info + '.jpg'
    plt.savefig(os.path.join(path, file_name))
    print("Grafik vom F1-Score aktualisiert. Neuste Version: " + file_name)