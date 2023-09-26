def plot_data(df, date_info): 
    import matplotlib.pyplot as plt
    import numpy as np
    import os

    # Datum der Datei für Ordnererstellung nutzen
    folder_path = r'c:\Users\Paul Strohmeier\Desktop\ma-accantec\Grafiken\generated\{}'.format(date_info)

    # Ordner erstellen
    os.makedirs(folder_path)
    print(folder_path + " erfolgreich erstellt.")

    # Nach serious splitten
    df_serious = df[df.serious == 1]
    df_ns = df[df.serious == 0]

    ## Histogramm nach IBCS
    # Definiere die Bins-Intervalle
    bins = [i * 10 for i in range(12)]  # [0, 10, 20, ..., 100, 110]

    # Histogramm erstellen mit den definierten Bins
    plt.hist([df_serious.age, df_ns.age], bins=bins, density=True, color=['red', 'green'], label=['Serious', 'Not Serious'], align='left', rwidth=0.8)

    # Setze die Achsentitel
    plt.xlabel('Alter')
    plt.ylabel('Dichte')

    # Setze den Titel des Histogramms
    plt.title('Verteilung von Serious und Not Serious nach Alter')

    # Entferne die obere und rechte Achsenlinie
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    # Setze die Achsentick-Positionierung
    plt.gca().tick_params(axis='both', which='both', direction='out')
    plt.gca().xaxis.set_tick_params(width=0.5)
    plt.gca().yaxis.set_tick_params(width=0.5)

    # Setze die Achsentick-Labels
    plt.xticks(range(0, 101, 10), fontsize=8)
    plt.yticks(fontsize=8)

    # Setze die Achsentick-Linien
    plt.gca().yaxis.grid(color='gray', linestyle='--', linewidth=0.5)
    plt.legend()

    # Füge den statischen Text als Tooltip hinzu (auf Website)
    tooltip_text = 'Dieses Histogramm zeigt die Verteilung des Alters für den "Serious" und den "Not Serious" Fall. Da die Anzahl der Ereignisse in beiden Kategorien unterschiedlich ist, wird ein Dichte-Histogramm verwendet, um die Vergleichbarkeit der Verteilungen zu gewährleisten.'

    # Grafik im richtigen Ordner abspeichern
    plt.savefig(os.path.join(folder_path, "Verteilung_Alter_Serious"), bbox_inches='tight')
    plt.clf()

    ## Boxplot-Diagramm erstellen
    plt.boxplot([df_serious.age, df_ns.age], labels=['Serious', 'Not Serious'])
    plt.title('Boxplot des Alters, um Ausreißer zu erkennen')
    plt.ylabel('Alter')

    # y-Achse an der linken Seite positionieren
    ax = plt.gca()
    ax.yaxis.tick_left()
    ax.yaxis.set_label_position('left')

    # Achsentick-Positionierung
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)

    # Entferne überflüssige Rahmenlinien
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    # Hintergrundfarbe anpassen
    plt.gca().set_facecolor('white')

    # Anpassungen der Linienstärke
    plt.gca().xaxis.set_tick_params(width=0.5)
    plt.gca().yaxis.set_tick_params(width=0.5)

    # Grafik im richtigen Ordner abspeichern
    plt.savefig(os.path.join(folder_path, "Boxplot_Alter"), bbox_inches='tight')
    plt.clf()

    ## Histogramm nach IBCS
    # Definiere die Bins-Intervalle
    bins = [i * 10 for i in range(12)]  # [0, 10, 20, ..., 100, 110]

    # Histogramm erstellen mit den definierten Bins
    plt.hist(df.age, bins=bins, density=False, align='left', edgecolor='white', linewidth=0.5)

    # Setze die Achsentitel
    plt.xlabel('Alter')
    plt.ylabel('Anzahl')

    # Setze den Titel des Histogramms
    plt.title('Verteilung des Alters')

    # Entferne die obere und rechte Achsenlinie
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    # Setze die Achsentick-Positionierung
    plt.gca().tick_params(axis='both', which='both', direction='out')
    plt.gca().xaxis.set_tick_params(width=0.5)
    plt.gca().yaxis.set_tick_params(width=0.5)

    # Setze die Achsentick-Labels
    plt.xticks(range(0, 101, 10), fontsize=8)
    plt.yticks(fontsize=8)

    # Setze die Achsentick-Linien
    plt.gca().yaxis.grid(color='gray', linestyle='--', linewidth=0.5)

    # Grafik im richtigen Ordner abspeichern
    plt.savefig(os.path.join(folder_path, "Verteilung_Alter"), bbox_inches='tight')
    plt.clf()

    ## Serious nach Länder
    country_counts_serious = []
    country_counts_ns = []

    # Liste der Länder-Spalten
    country_columns = df.filter(regex='country').columns

    for country in country_columns:
        country_counts_serious.append(df_serious[country].sum())
        country_counts_ns.append(df_ns[country].sum())

    # Anzahl der Länder
    num_countries = len(country_columns)

    # Positionen der Balken
    bar_positions = np.arange(num_countries)

    # Umwandeln der Werte und Indizes in NumPy-Arrays
    country_counts_serious = np.array(country_counts_serious)
    country_counts_ns = np.array(country_counts_ns)

    # Sortiere die Balken nach der Anzahl der "Serious" Fälle
    sorted_indices = np.argsort(country_counts_serious)[::-1]
    sorted_country_counts_serious = country_counts_serious[sorted_indices]
    sorted_country_counts_ns = country_counts_ns[sorted_indices]
    sorted_country_columns = country_columns[sorted_indices]

    # Erstelle das Balkendiagramm für die "Serious" Fälle
    plt.barh(bar_positions, sorted_country_counts_serious, label='Serious', color='red')

    # Erstelle das Balkendiagramm für die "Not Serious" Fälle auf den bereits vorhandenen Balken
    plt.barh(bar_positions, sorted_country_counts_ns, label='Not Serious', color='green', left=sorted_country_counts_serious)

    # Achsentitel
    plt.title("Anzahl von Serious und Not Serious nach Ländern")
    plt.xlabel('Anzahl')
    plt.ylabel('Länder')

    # Achsentick-Positionen
    y_tick_labels = [country[8:].upper() for country in sorted_country_columns]
    plt.yticks(bar_positions, y_tick_labels)

    # Legende
    plt.legend()
    plt.gca().invert_yaxis()

    # Entferne die obere und rechte Achsenlinie
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    # Setze die Achsentick-Positionierung
    plt.gca().tick_params(axis='both', which='both', direction='out')
    plt.gca().xaxis.set_tick_params(width=0.5)
    plt.gca().yaxis.set_tick_params(width=0.5)
    # Setze die Achsentick-Linien
    plt.gca().xaxis.grid(color='gray', linestyle='--', linewidth=0.5)

    # Grafik im richtigen Ordner abspeichern
    plt.savefig(os.path.join(folder_path, "Anzahl_Länder_Serious"), bbox_inches='tight')
    plt.clf()

    ## Anteil Serious nach Einnahmeroute
    routes_columns = df.filter(regex='route').columns

    # Anzahl der Einnahmerouten
    num_routes = len(routes_columns)

    # Anteil der "Serious" Fälle pro Einnahmeroute
    serious_percentages = []

    # Anteil der "Not Serious" Fälle pro Einnahmeroute
    ns_percentages = []

    for route in routes_columns:
        total_count = df[route].sum()
        serious_count = df_serious[route].sum()
        ns_count = df_ns[route].sum()
        
        serious_percentages.append(serious_count / total_count)
        ns_percentages.append(ns_count / total_count)

    # Sortiere die Einnahmerouten nach dem Anteil von "Serious"
    sorted_indices = np.argsort(serious_percentages)
    sorted_routes_columns = np.array(routes_columns)[sorted_indices]
    sorted_serious_percentages = np.array(serious_percentages)[sorted_indices]
    sorted_ns_percentages = np.array(ns_percentages)[sorted_indices]

    # Positionen der Säulen
    bar_positions = np.arange(num_routes)

    # Erstelle das gestapelte horizontale Säulendiagramm
    plt.barh(bar_positions, sorted_serious_percentages, label='Serious', color='red')
    plt.barh(bar_positions, sorted_ns_percentages, left=sorted_serious_percentages, label='Not Serious', color='green')

    # Achsentitel
    plt.title("Anteil von Serious und Not Serious nach Einnahmeroute")
    plt.xlabel('Anteil')
    plt.ylabel('Einnahmeroute')

    # Achsentick-Positionen
    y_tick_labels = [route[6:] for route in sorted_routes_columns]
    plt.yticks(bar_positions, y_tick_labels)

    # Legende
    plt.legend()

    # Entferne die obere und rechte Achsenlinie
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    # Setze die Achsentick-Positionierung
    plt.gca().tick_params(axis='both', which='both', direction='out')
    plt.gca().xaxis.set_tick_params(width=0.5)
    plt.gca().yaxis.set_tick_params(width=0.5)

    # Setze die Achsentick-Linien
    plt.gca().xaxis.grid(color='gray', linestyle='--', linewidth=0.5)

    # Annotations
    for i, (serious, ns) in enumerate(zip(sorted_serious_percentages, sorted_ns_percentages)):
        plt.annotate(f'{serious:.2f}', xy=(serious/2, i), ha='center', va='center', color='black')
        plt.annotate(f'{ns:.2f}', xy=(serious+ns/2, i), ha='center', va='center', color='black')

    # Grafik im richtigen Ordner abspeichern
    plt.savefig(os.path.join(folder_path, "Anteil_Routes_Serious"), bbox_inches='tight')
    plt.clf()

    print("5 Grafiken erfolgreich erstellt und abgespeichert.")