import data_acquisition
import data_preprocessing
import data_visualization
import monitoring
import warnings

def main():
    # Infos über neue Daten
    files = data_acquisition.get_new_files_info()

    # Datenbeschaffung
    data, date_info = data_acquisition.get_new_data(files)
    
    # Datenfilterung
    filtered_data = data_preprocessing.filter_data(data)

    # Für Visualisierung aufbereiten und CSV für ML erstellen. Triggert Visualisierungen vor Skalierung von Age
    viz_data = data_preprocessing.preprocess_data(filtered_data.copy(), date_info)

    # CSV für SAS Visualisierungen erstellen
    data_preprocessing.preprocess_for_sas_viz(filtered_data, date_info)

    # Reports erstellen
    monitoring.create_data_reports(viz_data, date_info)

    # Drift Test (triggert anschließend Performance Monitoring)
    monitoring.drift_test(viz_data, date_info)

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
    main()

