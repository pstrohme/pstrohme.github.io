def get_new_files_info():
    import pandas as pd
    import requests
    import json

    ## Infos über vorhandene Daten holen
    try:
        r = requests.get("https://api.fda.gov/download.json")
        print("API-Anfrage erfolgreich")
        data = r.json()
    except: 
        r = open("../Daten/all_dl_links.json")
        print("API-Url falsch oder nicht verfügbar. Es wird eine Datei-Liste genutzt, welche möglicherweise veraltet ist.")
        data = json.load(r)
    finally: 
        files = pd.DataFrame(data['results']['drug']['event']['partitions'])
        # Nur Daten ab 2020
        files = files[files[files['display_name'].str.contains('2020 Q1')].index[0]:].reset_index(drop=True)
        # Für jedes Quartal nur die erste Datei
        files = files[~files['display_name'].str.split('(').str[0].duplicated()]
        # Name umbennen (bsp.: '2020 Q1 (part1of33)' zu '2020 Q1)
        files['display_name'] = files['display_name'].str.replace(r'\s*\(.*\)', '', regex=True)
        files.reset_index(drop=True, inplace=True)
        print("Liste mit allen verfügbaren Daten erstellt.")
    
    return files

def read_counter_val():
    import json
    ## Daten mit Link aus Files herunterladen
    # Lesen des aktuellen Counter-Werts aus der Datei
    counter_file = r'c:\Users\Paul Strohmeier\Desktop\ma-accantec\Daten\counter.json'

    with open(counter_file, 'r') as file:
        data = json.load(file)
        counter = data['counter']
    print("Counter-Wert ist: {}".format(counter))
    return counter

def get_new_data(files):
    import pandas as pd
    import requests
    import json
    from zipfile import ZipFile
    from io import BytesIO
    import sys

    ## Daten mit Link aus Files herunterladen
    # Lesen des aktuellen Counter-Werts aus der Datei
    counter = read_counter_val()

    if counter > len(files) - 1:
        # Weitere Bearbeitung beenden, da keine neuen Daten
        print("Keine neuen Daten vorhanden. Counter bleibt bei {}. Es erfolgt keine weitere Verarbeitung und das Programm wird beendet.".format(counter))
        sys.exit()

    else:
        print("Neue Daten vorhanden.")
        zipurl = files.file[counter]
        print("Download von Datei {} gestartet.".format(zipurl))
        req = requests.get(zipurl)
        print("Download abgeschlossen.")
        with ZipFile(BytesIO(req.content)) as zip_file:
            with zip_file.open(zip_file.namelist()[0]) as json_file:
                json_data = json.load(json_file)
                data = json_data['results']

        # DataFrame erstellen
        df = pd.json_normalize(data)

        # Infos über Dateiname für spätere Visualisierungen (Bsp.: '2020 Q1' zu '2020Q1')
        date = files.display_name[counter].replace(" ","")
        print(date)

        counter += 1
        print("Counter hochgezählt. Dieser ist nun {}.".format(counter))

        # Speichern des aktualisierten Counter-Werts in der Datei
        counter_file = r'c:\Users\Paul Strohmeier\Desktop\ma-accantec\Daten\counter.json'
        
        with open(counter_file, 'w') as file:
            data = {'counter': counter}
            json.dump(data, file)
        print("Counter-Datei aktualisiert.")
        
        return df, date