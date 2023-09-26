def filter_data(df):

    # unnötige Spalten entfernen
    cols_to_drop = ['patient.patientagegroup', 'authoritynumb', 'occurcountry', 'primarysource', 'primarysourcecountry', 'transmissiondate', 'receiptdate', 'fulfillexpeditecriteria', 'companynumb', 'duplicate', 'primarysource.literaturereference', 'patient.summary.narrativeincludeclinical']
    cols_to_drop += df.filter(regex='safety|format|reportduplicate|sender|receiver').columns.tolist()
    df = df.drop(cols_to_drop, axis=1, errors='ignore')

    print("Daten erfolgreich gefiltert.")

    return df

    # Tests des Datenschemas hier oder mit Evidently

def preprocess_data(df, date_info):
    import pandas as pd
    import os
    import data_acquisition
    import data_visualization
    from sklearn.preprocessing import StandardScaler

    # Datum formatieren
    df.receivedate = pd.to_datetime(df.receivedate)
    print("Datum formartiert.")

    # Datum löschen
    df.drop('receivedate', axis=1, inplace=True)

    '''
    # Reaktionen als Liste
    def get_reactionmeddrapt_list(row):
        reactionmeddrapt_list = []
        for reaction in row['patient.reaction']:
            reactionmeddrapt_list.append(reaction['reactionmeddrapt'])
        return reactionmeddrapt_list

    df['reactions'] = df.apply(get_reactionmeddrapt_list, axis=1)
    df.drop(['patient.reaction'], axis=1, inplace=True)
    print("Reaktionen als Liste gespeichert.")

    # Name der Medikamente
    def get_medicinalproduct_list(row):
        medicinalproduct_list = []
        for drug in row['patient.drug']:
            medicinalproduct_list.append(drug['medicinalproduct'])
        return medicinalproduct_list

    df['drugs'] = df.apply(get_medicinalproduct_list, axis=1)
    print("Reaktionen als Liste gespeichert.")
    '''
    df.drop(['patient.reaction'], axis=1, inplace=True)

    # Einnahmerouten
    def get_route_list(row):
        route_list = []
        for route in row['patient.drug']:
            if 'openfda' in route and 'route' in route['openfda']:
                route_list.append(route['openfda']['route'][0])
            else:
                route_list.append('Unknown')
        return route_list

    df['routes'] = df.apply(get_route_list, axis=1)
    print("Routen als Liste gespeichert.")

    # Mindestens eine Prescription Drug?
    df['min_one_pres'] = df['patient.drug'].apply(lambda x: 1 if any('openfda' in drug and 'product_type' in drug['openfda'] and drug['openfda']['product_type'][0] == 'HUMAN PRESCRIPTION DRUG' for drug in x) else 0)
    df.drop(['patient.drug'], axis=1, inplace=True)
    print("min_one_pres erstellt.")

    # Alter formatieren
    df['age'] = df.apply(lambda x: int(x['patient.patientonsetage'])*10 if x['patient.patientonsetageunit'] == '800' else x['patient.patientonsetage'], axis=1)
    df['age'] = df.apply(lambda x: 0 if x['patient.patientonsetageunit'] in ['802', '803', '804', '805'] else x['patient.patientonsetage'], axis=1)
    df.drop(['patient.patientonsetageunit'], axis=1, inplace=True)
    df.drop(['patient.patientonsetage'], axis=1, inplace=True)
    # NaNs löschen
    df.dropna(subset=['age'], inplace=True)
    # Zu int konvertieren
    df.age = pd.to_numeric(df.age)
    print("Alter formatiert.")

    # One-Hot-Encoding der Spalte "reporttype"
    df['report_spontaneous'] = df.apply(lambda x: 1 if x['reporttype'] == '1' else 0, axis=1)
    df['report_from_study'] = df.apply(lambda x: 1 if x['reporttype'] == '2' else 0, axis=1)
    df['report_other'] = df.apply(lambda x: 1 if x['reporttype'] == '3' else 0, axis=1)
    df['report_unknown'] = df.apply(lambda x: 1 if x['reporttype'] not in ['1','2','3'] else 0, axis=1)
    df.drop(['reporttype'], axis=1, inplace=True)
    print("One-Hot-Encoding durchgeführt für reporttype.")

    # One-Hot-Encoding der Spalte qualification
    df['quali_physician'] = df.apply(lambda x: 1 if x['primarysource.qualification'] == '1' else 0, axis=1)
    df['quali_pharmacist'] = df.apply(lambda x: 1 if x['primarysource.qualification'] == '2' else 0, axis=1)
    df['quali_other_health_prof'] = df.apply(lambda x: 1 if x['primarysource.qualification'] == '3' else 0, axis=1)
    df['quali_lawyer'] = df.apply(lambda x: 1 if x['primarysource.qualification'] == '4' else 0, axis=1)
    df['quali_consumer_or_nonhealthprof'] = df.apply(lambda x: 1 if x['primarysource.qualification'] not in ['1','2','3','4'] else 0, axis=1)
    df.drop(['primarysource.qualification'], axis=1, inplace=True)
    print("One-Hot-Encoding durchgeführt für qualification.")

    # One Hot Encoding Geschlecht
    df['patient.patientsex'].fillna(0)
    df['gender_male'] = df.apply(lambda x: 1 if x['patient.patientsex'] == '1' else 0, axis=1)
    df['gender_female'] = df.apply(lambda x: 1 if x['patient.patientsex'] == '2' else 0, axis=1)
    df['gender_unknown'] = df.apply(lambda x: 1 if x['patient.patientsex'] not in ['1','2'] else 0, axis=1)
    df.drop(['patient.patientsex'], axis=1, inplace=True)
    print("One-Hot-Encoding durcheführt für gender.")

    # Gewicht löschen, da zu viele NaNs
    df.drop(['patient.patientweight'], axis=1, inplace=True)
    print("Gewicht entfernt, da zu viele NaNs.")  

    # Serious (Target)
    df['serious'] = df.apply(lambda x: 1 if x['serious'] == '1' else 0, axis=1)
    df.drop(['seriousnessdeath','seriousnesslifethreatening','seriousnesshospitalization','seriousnessdisabling', 'seriousnesscongenitalanomali', 'seriousnessother'], axis=1, inplace=True)
    print("Serious encoded.")

    # TODO: Tests 

    ### Country
    # Country als String
    df['primarysource.reportercountry'] = df['primarysource.reportercountry'].astype("string")

    # Gewünschte Spalten für das One-Hot-Encoding
    desired_columns = ['country_ca', 'country_cn', 'country_de', 'country_es', 'country_fr', 'country_gb', 'country_it',
                   'country_jp', 'country_nl', 'country_us', 'country_other_or_unknown']

    # One-Hot-Encoding für den neuen Datensatz
    df_encoded = pd.get_dummies(df['primarysource.reportercountry'], prefix='country', dtype='int')
    df_encoded.columns = df_encoded.columns.str.lower()

    # Spalten im neuen Datensatz entfernen, die nicht in den gewünschten Spalten enthalten sind
    extra_columns = set(df_encoded.columns) - set(desired_columns)

    # Spalten, die entfernt wurden, zum country_other_or_unknown hinzufügen
    df_encoded['country_other_or_unknown'] = 0
    df_encoded['country_other_or_unknown'] += df_encoded[list(extra_columns)].sum(axis=1)
    df_encoded.drop(extra_columns, axis=1, inplace=True)

    # Spalten im neuen Datensatz an die gewünschten Spalten anpassen, falls im neuen DF noch nicht vorhanden
    for column in desired_columns:
        if column not in df_encoded.columns:
            df_encoded[column] = 0

    # DataFrames zusammenführen
    df = pd.concat([df.drop(['primarysource.reportercountry'], axis=1), df_encoded], axis=1)
    print("One-Hot-Encoding durchgeführt: Country")

    ### Routes
    # Gewünschte Spalten für das One-Hot-Encoding
    desired_columns_routes = ['route_intra-arterial', 'route_intradermal', 'route_intramuscular',
                        'route_intrauterine', 'route_intravenous', 'route_intravitreal',
                        'route_nasal', 'route_ophthalmic', 'route_oral', 'route_rectal',
                        'route_respiratory (inhalation)', 'route_subcutaneous',
                        'route_sublingual', 'route_topical', 'route_transdermal',
                        'route_vaginal', 'route_other_or_unknown']

    # One-Hot-Encoding für den neuen Datensatz
    routes = pd.get_dummies(df.routes.explode(), prefix='route', dtype='int').groupby(level=0).sum()
    routes.columns = routes.columns.str.lower()

    # Spalten im neuen Datensatz, die nicht gewünscht sind
    extra_columns = set(routes.columns) - set(desired_columns_routes)

    # Spalten, die entfernt wurden, zum country_other_or_unknown hinzufügen
    routes['route_other_or_unknown'] = 0
    routes['route_other_or_unknown'] += routes[list(extra_columns)].sum(axis=1)
    routes.drop(extra_columns, axis=1, inplace=True)

    # Spalten im neuen Datensatz an die gewünschten Spalten anpassen
    for column in desired_columns_routes:
        if column not in routes.columns:
            routes[column] = 0

    # DataFrames zusammenführen
    df = pd.concat([df.drop(['routes'], axis=1), routes], axis=1)
    df.rename(columns={'route_intra-arterial': 'route_intraarterial', 'route_respiratory (inhalation)': 'route_respiratory_inhalation'}, inplace=True)
    print("One-Hot-Encoding durchgeführt: Routes")

    # Reaktionen und Drugs löschen
    # df.drop(['reactions', 'drugs'], axis=1, inplace=True)

    # Spalten alphabetisch sortieren
    df = df.reindex(sorted(df.columns), axis=1)
    print("Spalten alphabetisch sortiert.")

    # Grafik erstellen bevor Age skaliert wird
    data_visualization.plot_data(df, date_info)
        
    # Initialisiere den StandardScaler
    scaler = StandardScaler()

    # Wende den StandardScaler nur auf die 'age'-Spalte an
    df['age'] = scaler.fit_transform(df[['age']])
    print("Age skaliert.")
    
    # Datensatz als CSV für SAS abspeichern
    prefix = 'ml_'
    counter = str(data_acquisition.read_counter_val())
    time_label = '_' + date_info
    path = r'c:\Users\Paul Strohmeier\Desktop\ma-accantec\Daten\csv\ml'
    file_name = prefix + counter + time_label + '.csv'
    df.to_csv(os.path.join(path, file_name), index=False)
    print("CSV für ML in SAS Viya erstellt.")

    # Test des Schemas 


    print("Daten erfolgreich vorverarbeitet.")

    return df.reset_index(drop=True)

def preprocess_for_sas_viz(df, date_info):
    import pandas as pd
    import os
    import data_acquisition

    # Alter
    df['age'] = df.apply(lambda x: int(x['patient.patientonsetage'])*10 if x['patient.patientonsetageunit'] == '800' else x['patient.patientonsetage'], axis=1)
    df['age'] = df.apply(lambda x: 0 if x['patient.patientonsetageunit'] in ['802', '803', '804', '805'] else x['patient.patientonsetage'], axis=1)
    df.drop(['patient.patientonsetageunit'], axis=1, inplace=True)
    df.drop(['patient.patientonsetage'], axis=1, inplace=True)
    
    # Datum
    df.receivedate = pd.to_datetime(df.receivedate)
    
    # Reporttype
    df['reporttype'] = df['reporttype'].replace(['1', '2', '3', '4'], ['spontaneaous', 'from study', 'other', 'unknown'])
    df.reporttype = df.reporttype.astype('string')

    # Selbes Vorgehen für qualification
    df['primarysource.qualification'].fillna('0', inplace=True)
    df['primarysource.qualification'] = df['primarysource.qualification'].replace(['0', '1', '2', '3', '4', '5'], ['unknown', 'physician', 'pharmacist', 'other_health_prof', 'lawyer', 'consumer_or_nonhealth_prof'])
    df['primarysource.qualification'] = df['primarysource.qualification'].astype('string')

    # Selbes bei Geschlecht
    df['patient.patientsex'].fillna('0', inplace=True)
    df['patient.patientsex'] = df['patient.patientsex'].replace(['0', '1', '2'], ['unknown', 'male', 'female'])
    df['patient.patientsex'] = df['patient.patientsex'].astype('string')

    # Liste der top 10 Länder
    countries = ['CA', 'CN', 'DE', 'ES', 'FR', 'GB', 'IT', 'JP', 'NL', 'US']
    # Werte ersetzen
    df['primarysource.reportercountry'] = df['primarysource.reportercountry'].apply(lambda x: 'other_or_unknown' if x not in countries else x)
    df['primarysource.reportercountry'] = df['primarysource.reportercountry'].astype('string')
    
    # Serious
    df['serious'] = df.apply(lambda x: 1 if x['serious'] == '1' else 0, axis=1)
    df.drop(['seriousnessdeath','seriousnesslifethreatening','seriousnesshospitalization','seriousnessdisabling', 'seriousnesscongenitalanomali', 'seriousnessother'], axis=1, inplace=True)
    
    # Löschen der Listen
    df.drop(['patient.drug'], axis=1, inplace=True)
    df.drop(['patient.reaction'], axis=1, inplace=True)

    # Datensatz als CSV für SAS abspeichern
    prefix = 'viz_'
    counter = str(data_acquisition.read_counter_val())
    time_label = '_' + date_info
    path = r'c:\Users\Paul Strohmeier\Desktop\ma-accantec\Daten\csv\viz'
    file_name = prefix + counter + time_label + '.csv'
    df.to_csv(os.path.join(path, file_name), index=False)
    print("Daten für Visualisierungen in SAS Viya vorbereitet und CSV abgespeichert.")