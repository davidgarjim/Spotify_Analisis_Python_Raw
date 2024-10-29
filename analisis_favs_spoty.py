# -*- coding: utf-8 -*-
"""#ANÁLISIS FAVORITAS DE SPOTIFY"""

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from sklearn.model_selection import train_test_split
import shap
import xgboost
import spotipy
from spotipy.oauth2 import SpotifyOAuth


df = pd.read_csv('favoritas_hasta_septiembre24.csv')


def clean_df(data):

    required_columns = ['\t\t\t\tDanceability', 'Artist Name(s)', 'Album Name', 'Track ID', 'Duration (ms)',
                        'Track Name']

    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        raise ValueError(f"Faltan las siguientes columnas en el DataFrame: {missing_columns}")

    # Renombrar columnas
    data.rename(columns={'\t\t\t\tDanceability': 'Danceability'}, inplace=True)
    data.rename(columns={'Artist Name(s)': 'Artist'}, inplace=True)
    data.rename(columns={'Album Name': 'Album'}, inplace=True)
    data.rename(columns={'Track ID': 'ID'}, inplace=True)

    # Convertir la duración de milisegundos a segundos
    data['Duration'] = data['Duration (ms)'] / 1000
    data.drop(columns=['Duration (ms)'], inplace=True)

    data.drop_duplicates(subset=['Track Name', 'Artist'], keep='first', inplace=True)

    return data


def conocer_variables():
    explicacion = ("""Vamos a ver qué significa cada columna para poder comenzar a entender el Dataframe:
    
    1. Popularity (Popularidad)

        Descripción: Es una medida que indica qué tan popular es una canción en la plataforma de Spotify. Este valor es un número entre 0 y 100, donde 100 representa la canción más popular. La popularidad se basa en el número de reproducciones recientes, cuántas veces ha sido compartida, añadida a playlists y otros factores que Spotify considera.
        Rango: 0 a 100.

    2. Duration (ms)

        Descripción: Es la duración total de la canción medida en milisegundos (ms). Puedes dividir este valor por 60,000 para convertirlo en minutos y obtener una métrica más comprensible.
        Ejemplo: Una duración de 240,000 ms equivaldría a una canción de 4 minutos.

    3. Danceability (Bailabilidad)

        Descripción: Esta métrica mide lo adecuado que es una canción para bailar, basada en varios elementos musicales como el tempo, la estabilidad rítmica, la fuerza del beat, y la regularidad. Un valor más alto indica que la canción es más fácil de bailar.
        Rango: 0 a 1. Un valor de 1 significa que la canción es extremadamente bailable.

    4. Energy (Energía)

        Descripción: Es una medida de la intensidad y actividad percibida de una canción. Las canciones con alta energía tienden a sentirse rápidas, fuertes y ruidosas (por ejemplo, música de rock o metal), mientras que las canciones con baja energía son más suaves y relajadas.
        Rango: 0 a 1, donde un valor cercano a 1 indica alta energía.

    5. Key (Tonalidad)

        Descripción: Representa la clave musical de la canción, utilizando notación musical estándar donde:
            0 es Do (C)
            1 es Do# (C#) o Re♭ (D♭)
            Y así sucesivamente hasta 11 (Si, o B).
        Rango: 0 a 11, representando cada una de las 12 notas musicales de la escala cromática.

    6. Loudness (Volumen)

        Descripción: Es una medida del volumen promedio de una canción, en decibelios (dB). Las canciones modernas tienden a ser más fuertes debido a la "guerra del volumen", pero el rango generalmente está entre -60 dB y 0 dB. Un valor más bajo indica que la canción es más silenciosa.
        Rango: -60 dB a 0 dB.

    7. Mode (Modo)

        Descripción: Indica si la canción está en un modo mayor o menor:
            1 = Modo mayor
            0 = Modo menor
        Las canciones en modo mayor suelen percibirse como más alegres o felices, mientras que las canciones en modo menor tienden a sonar más tristes o melancólicas.

    8. Speechiness (Locuacidad)

        Descripción: Evalúa la presencia de palabras habladas en una pista. Las canciones con altos valores de locuacidad suelen tener mucho contenido hablado, como podcasts o pistas de rap.
        Rango: 0 a 1. Valores cercanos a 1 indican que la pista es principalmente hablada (como un audiolibro o un discurso).

    9. Acousticness (Acústica)

        Descripción: Mide qué tan acústica es una canción, es decir, qué tan probable es que la pista haya sido creada con instrumentos acústicos, en lugar de electrónicos o amplificados.
        Rango: 0 a 1. Un valor de 1 indica una alta probabilidad de que la pista sea completamente acústica.

    10. Instrumentalness (Instrumentalidad)

        Descripción: Mide qué tan instrumental es una canción. Valores altos indican que la pista probablemente no contiene letras o voces cantadas. Las canciones con un valor cercano a 1 son casi completamente instrumentales.
        Rango: 0 a 1.

    11. Liveness (Vivosidad)

        Descripción: Mide la probabilidad de que una pista se haya grabado en vivo, es decir, si contiene elementos que denotan la presencia de un público o un entorno en vivo.
        Rango: 0 a 1. Un valor de 1 indica una alta presencia de componentes en vivo.

    12. Valence (Valencia)

        Descripción: Indica el nivel de positividad o negatividad emocional transmitida por una canción. Canciones con altos valores de valencia tienden a ser más alegres y optimistas, mientras que los valores bajos están asociados con emociones más tristes o sombrías.
        Rango: 0 a 1, donde 1 es extremadamente positivo y 0 es extremadamente negativo.

    13. Tempo

        Descripción: Mide el tempo de la canción, es decir, la velocidad o el ritmo al que se reproduce. Se mide en pulsos por minuto (BPM). Las canciones con un tempo más alto tienden a sentirse más rápidas y enérgicas.
        Rango: Generalmente entre 0 BPM y 250 BPM.

    14. Time Signature (Compás)

        Descripción: Representa la cantidad de tiempos por compás en una canción. Es un valor entero, y las firmas de tiempo comunes son 4/4, 3/4, o 5/4. En la mayoría de la música popular, el compás de 4 tiempos es el más común.
        Rango: Valores enteros como 3, 4 o 5, donde el valor más común es 4 (4 tiempos por compás).
    """)
    print(explicacion)
    return (explicacion)


columns_var = ['Popularity', 'Duration', 'Danceability', 'Energy', 'Key',
               'Loudness', 'Mode', 'Speechiness', 'Acousticness', 'Instrumentalness',
               'Liveness', 'Valence', 'Tempo', 'Time Signature']


def eda(data):   #EDA

    print("\nEn primer lugar le echamos lo ojeamos para ver cómo viene en bruto:")
    print(data.head())

    print("\n\nColumnas:")
    print(data.columns)

    print("\n\nInformación del DataFrame:")
    print(data.info())

    # Verificar valores duplicados
    print("\nNúmero de ID de pistas duplicados:")
    print(data['ID'].duplicated().sum())

    print("\nEstadísticas descriptivas:")
    print(data.describe())


    # Comprobar valores nulos
    print("\n\nConteo de valores nulos:")
    null_counts = data.isnull().sum()
    null_counts.to_csv('df_nulos.csv')
    print(null_counts)



def var_num(data):
    df_var = data[columns_var].copy()
    df_var.index = data['ID']

    # .Describe de los datos
    print(df_var.describe())
    print(df_var.isnull().sum())
    print(df_var.corr())

    # Histogramas
    n_cols = 3
    n_rows = (len(columns_var) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    axes = axes.flatten()

    for i, column in enumerate(columns_var):
        ax = axes[i]
        df_var[column].plot(kind='hist', bins=20, ax=ax, title=column)
        ax.spines[['top', 'right']].set_visible(False)

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.savefig('histograms_variables.png')
    plt.show()

    # Normalizamos
    df_norm = (df_var - df_var.mean()) / df_var.std()

    # Heatmaps de correlación
    plt.figure(figsize=(14, 12))
    sns.heatmap(df_var.corr(), annot=True, cmap='coolwarm')
    plt.savefig('correlacion_heatmap.png')
    plt.show()

    # Scatterplots
    scatter_features = [('Popularity', 'Danceability'), ('Energy', 'Loudness'), ('Energy', 'Danceability')]
    for x, y in scatter_features:
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=x, y=y, data=df_var)
        plt.title(f'{x} vs {y}')
        plt.gca().spines[['top', 'right']].set_visible(False)
        plt.savefig(f'scatterplot_{x.lower()}_{y.lower()}.png')
        plt.show()




def genre(data):   #GÉNEROS
    # Recuento de géneros en bruto
    print('\n\nHacemos un recuento de los géneros en bruto:')
    # Crear un DataFrame alternativo para trabajar sobre los géneros sin perder la canción
    df_genres = data[['ID', 'Track Name', 'Genres']].set_index('ID')
    print('\n\n', df_genres.head(2))
    df_dummies_genres = df_genres.copy()  # Hacer una copia para trabajar
    print('\n\n',df_dummies_genres.head(2))

    # Lista de estilos musicales básicos
    basic_music_styles = [
        'rock', 'pop', 'hip hop', 'electronic', 'jazz', 'blues',
        'metal', 'folk', 'classical', 'latin', 'reggae', 'funk',
        'soul', 'country', 'punk', 'indie', 'alternative', 'rumba',
        'flamenco', 'new wave', 'psychedelic', 'dance', 'rap',
        'trap', 'world', 'ska'
    ]

    print("\n\nCogeremos estos estilos de música para servirnos en el recuento:", basic_music_styles)

    # Función para crear dummies en función de si se encuentra una palabra en la columna 'Genres'
    def create_dummies(df_dummies_genres, genre_column, styles):
        for style in styles:
            df_dummies_genres[style] = df_dummies_genres[genre_column].apply(lambda x: int(style in str(x).lower()))
        return df_dummies_genres

    # Crear dummies
    df_dummies_genres = create_dummies(df_genres, 'Genres', basic_music_styles)

    # Eliminar la columna original 'Genres'
    df_dummies_genres.drop('Genres', axis=1, inplace=True)


    # Crear columnas unificando géneros
    df_dummies_genres['hip_hop'] = df_dummies_genres[['hip hop', 'rap']].max(axis=1)
    df_dummies_genres['rumba_flamenco'] = df_dummies_genres[['rumba', 'flamenco']].max(axis=1)
    df_dummies_genres['punk_rock'] = df_dummies_genres[['punk', 'new wave']].max(axis=1)
    df_dummies_genres.drop(columns=['hip hop', 'rap', 'rumba', 'flamenco', 'punk', 'new wave'], inplace=True)

    # Eliminar géneros 'alternative' y 'dance', ya que son etiquetas que participan en varios géneros más grandes.
    for genre in ['alternative', 'dance']:
        if df_dummies_genres[genre].any():
            df_dummies_genres.drop(genre, axis=1, inplace=True)

    # Limpiar columnas 'rock' y 'pop'... Asignar 0 a 'rock' si alguna de las otras columnas tiene un 1

    columns_to_check_rock = [
        'pop', 'electronic', 'jazz', 'blues', 'metal', 'folk',
        'classical', 'latin', 'reggae', 'funk', 'soul', 'country',
        'indie', 'psychedelic', 'trap', 'world', 'ska',
        'hip_hop', 'rumba_flamenco', 'punk_rock'
    ]

    df_dummies_genres['rock'] = df_dummies_genres.apply(
        lambda row: 0 if row[columns_to_check_rock].max() == 1 else row['rock'], axis=1
    )

    # Repetir para 'pop'

    columns_to_check_pop = [
        'electronic', 'jazz', 'blues', 'metal', 'folk',
        'classical', 'latin', 'reggae', 'funk', 'soul', 'country',
        'indie', 'psychedelic', 'trap', 'world', 'ska',
        'hip_hop', 'rumba_flamenco', 'punk_rock'
    ]

    df_dummies_genres['pop'] = df_dummies_genres.apply(
        lambda row: 0 if row[columns_to_check_pop].max() == 1 else row['pop'], axis=1
    )

    # Hacer recuento final de los géneros, lo mostramos y lo guardamos
    df_dummies_NoNames = df_dummies_genres.drop('Track Name', axis=1)
    df_recuento_generos = df_dummies_NoNames.sum().sort_values(ascending=False)
    print('\n\n', df_recuento_generos)
    df_recuento_generos.to_csv('df_recuento_generos.csv')

    return df_dummies_genres  # Retornar el DataFrame final





def favs_vs_2023_vs_history(data):     #COMPARACIÓN CON TOP ESPAÑA 2023 Y MOST STREAMED SONGS IN HISTORY

    df_2023 = pd.read_csv('top_canciones_2023_espaa.csv')
    df_history = pd.read_csv('top_100_most_streamed_songs_on_spotify_updated.csv')

    clean_df(df_2023)
    clean_df(df_history)

    df_2023 = df_2023[columns_var]
    df_history = df_history[columns_var]
    df_favs = data[columns_var]

    df_combined = pd.concat([df_favs.assign(Category='Favoritas'),
                             df_history.assign(Category='Más Streameadas Historia'),
                             df_2023.assign(Category='Más Escuchadas 2023')],
                            ignore_index=True)

    df_combined.to_csv('df_combined.csv', index=False)

    vars = ['Danceability', 'Energy', 'Loudness', 'Valence', 'Tempo']

    # Reemplazar valores de duración superiores a 7 minutos por 7 minutos
    df_combined.loc[df_combined['Duration'] > 420, 'Duration'] = 420

    for var in ['Popularity', 'Duration'] + vars:
        for plot_type in ['violin', 'box']:
            plt.figure(figsize=(12, 6))
            if plot_type == 'violin':
                # Asignar hue a 'Category' en el violinplot
                sns.violinplot(x='Category', y=var, hue='Category', data=df_combined, palette="muted", legend=False)
            else:
                # Para el boxplot, asignamos hue y evitamos la leyenda
                sns.boxplot(x='Category', y=var, hue='Category', data=df_combined,
                            palette="Set2" if var == 'Popularity' else "Set1", legend=False)

            plt.title(f'Distribución de {var} ({plot_type.capitalize()} Plot)')
            plt.ylabel(var)
            plt.savefig(f'distribucion_{var.lower()}_{plot_type}.png')
            plt.close()


    # Calcular y graficar medias y medianas
    vars = ['Popularity', 'Danceability', 'Acousticness']

    for var in vars:

        means = [df[var].mean() for df in [df_2023, df_history, df_favs]]
        medians = [df[var].median() for df in [df_2023, df_history, df_favs]]


        for values, metric_name, plot_title in zip([means, medians], ['Media', 'Mediana'],
                                                   [f'Medias de {var}', f'Medianas de {var}']):
            plt.figure(figsize=(8, 6))
            plt.bar(['df_2023', 'df_history', 'df_favs'], values, color=['blue', 'green', 'red'])
            plt.title(plot_title)
            plt.ylabel(metric_name)

            if var == 'Acousticness':
                plt.ylim(0, 0.5)
            elif var == 'Danceability':
                plt.ylim(0, 1)
            elif var == 'Popularity':
                plt.ylim(0, 100)

            plt.savefig(f'{metric_name.lower()}_{var.lower()}.png')
            plt.close()


def artistas_albums(data):  #ARTISTAS Y ÁLBUMS
    data['Artist'].value_counts().sort_values(ascending=False).head(30)

    data['Artist'].value_counts()

    # Paso 1: Dividir los artistas por comas
    df_artistas = data['Artist'].str.split(',')

    # Paso 2: Expandir la columna en filas separadas, una por cada artista
    df_artistas_expandido = df_artistas.explode('Artist')

    df_artistas_expandido.head(3)

    conteo_artistas = df_artistas_expandido.value_counts().sort_values(ascending=False)

    print('\n\nEsta es la lista con mis 50 artistas con más canciones entre mis favoritas:\n\n', conteo_artistas.head(50))

    print('\n\nHay', conteo_artistas.count(), 'artistas en mi lista de favoritos')

    df_recuento_albums = data['Album'].value_counts().sort_values(ascending=False).head(40)
    print('\n\nEstos son los 5 álbums con más canciones entre mis favoritas:\n', df_recuento_albums.head(5))

    df_recuento_albums.to_csv('df_recuento_artistas.csv')







def playlist(data): #MIS PLAYLIST POR VARIABLES: MODELO DE CLUSTERIZACIÓN
    variables_clusters = ['Danceability', 'Energy', 'Tempo', 'Liveness', 'Acousticness']

    # Clustering con KMeans
    df_clusters = data[variables_clusters].copy()
    df_clusters['ID'] = data['ID']  # Asegúrate de que el ID esté incluido
    kmeans = KMeans(n_clusters=8, random_state=42)
    df_clusters['label_kmeans'] = kmeans.fit_predict(df_clusters[variables_clusters])

    # Visualizar los resultados del clustering
    print("\n\nConteo de clusters de KMeans:")
    print(df_clusters['label_kmeans'].value_counts())

    # Dividir los datos en entrenamiento y prueba
    X = df_clusters[variables_clusters]
    y = df_clusters['label_kmeans']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Entrenar un modelo XGBoost para predecir los clusters
    model = xgboost.XGBClassifier(seed=42)
    model.fit(X_train, y_train)

    # Aplicar SHAP para interpretar el modelo
    explainer = shap.Explainer(model, X_train)
    shap_values = explainer(X_test)

    # Imprimir dimensiones para verificar
    #print("Dimensiones de X_test:", X_test.shape)
    #print("Dimensiones de shap_values:", shap_values.shape)

    # Visualizar los valores SHAP solo para la primera clase (0 en este caso)
    shap.summary_plot(shap_values[:, :, 0], X_test)

    # Scatterplot de 'Energy' vs 'Acousticness' según los clusters de KMeans
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='Energy', y='Tempo', hue='label_kmeans', data=df_clusters, palette='Set1')
    plt.title('Clusters por Energy vs Tempo(kmeans)')
    plt.gca().spines[['top', 'right']].set_visible(False)
    plt.savefig('scatterplot_energy_acousticness_kmeans.png')
    plt.show()

    # Clustering con DBSCAN
    dbscan = DBSCAN()
    df_clusters['label_dbscan'] = dbscan.fit_predict(df_clusters[variables_clusters])

    # Visualizar los resultados del clustering DBSCAN
    print('\n\nAsí se agrupan por número de canciones cada cluster (DBSCAN):\n', df_clusters['label_dbscan'].value_counts())

    # Opcional: Visualizar DBSCAN
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='Energy', y='Tempo', hue='label_dbscan', data=df_clusters, palette='Set1')
    plt.title('Clusters por Energy vs Tempo (DBSCAN)')
    plt.gca().spines[['top', 'right']].set_visible(False)
    plt.savefig('scatterplot_energy_acousticness_dbscan.png')
    plt.show()

    # Características de una de cada cluster
    df_clusters[['Track Name', 'Artist', 'Popularity','Duration']] = data[['Track Name', 'Artist', 'Popularity','Duration']]
    canciones_por_cluster = df_clusters.groupby('label_kmeans').first().reset_index()

    print("\n\nValores y ejemplos de los clústeres del kmeans:")
    for index, row in canciones_por_cluster.iterrows():
        print(f"\n\nCluster {row['label_kmeans']}:")
        print(f"  Canción: {row.get('Track Name', 'No disponible')}")
        print(f"  Artista: {row.get('Artist', 'No disponible')}")
        print(f"  Popularidad: {row.get('Popularity', 'No disponible')}")
        print(f"  Duración: {row.get('Duration', 'No disponible')}")
        print(f"  Danceabilidad: {row['Danceability']}")
        print(f"  Energía: {row['Energy']}")
        print(f"  Tempo: {row['Tempo']}")

    return df_clusters  # Devuelve el DataFrame con las etiquetas de cluster




def llevarlo_a_spotify(data): ##TENGO QUE LLAMAR AL ARCHIVO CON CLUSTERS

    # Función para solicitar datos al usuario
    def solicitar_datos():
        print("Por favor, introduce la siguiente información:")

        client_id = input("Client ID de tu aplicación: ")
        client_secret = input("Client Secret de tu aplicación: ")
        redirect_uri = input("Redirect URI (ejemplo: http://localhost:8888/callback): ")
        nombre_playlist = input("Nombre de la playlist: ")

        return client_id, client_secret, redirect_uri, nombre_playlist

    # Función para crear una nueva playlist y añadir canciones
    def crear_playlist(data,sp, user_id, nombre_playlist, lista_ids):
        # Crear una nueva playlist
        playlist = sp.user_playlist_create(user=user_id, name=nombre_playlist, public=False)

        # Añadir canciones a la playlist
        sp.user_playlist_add_tracks(user=user_id, playlist_id=playlist['id'], tracks=lista_ids)
        print(f'Playlist "{nombre_playlist}" creada y canciones añadidas con éxito.')

    # Solicitar datos del usuario
    client_id, client_secret, redirect_uri, nombre_playlist = solicitar_datos()

    # Configuración de la autenticación
    sp = spotipy.Spotify(auth_manager=SpotifyOAuth(client_id=client_id,
                                                   client_secret=client_secret,
                                                   redirect_uri=redirect_uri,
                                                   scope='playlist-modify-private'))


    # Obtener los IDs de las canciones del índice del DataFrame
    lista_ids = data.index.tolist()  # Usa el índice como lista de IDs

    # Obtener el ID del usuario de Spotify
    user_id = sp.current_user()['id']

    # Crear la playlist
    crear_playlist(sp, user_id, nombre_playlist, lista_ids)


def conclusiones():
    # --- Conclusiones en primera persona ---
    texto_conclusiones = f"""
Mis conclusiones:
    
    ***EDA Y VARIABLES NUMÉRICAS:***
    
    1. Tan solo hay valores vacíos en las columnas 'Record label', la cual no ha sido utilizada en este análisis,
        por lo que no se han tenido que tomar decisiones al respecto; y en 'Genres', donde solo son 172 y se ha decidido
        no tener estas en cuenta.
    
    2.  Las medias y las medianas no varían mucho una respecto a la otra, pero si encontramos features con desviaciones con cola
        de caballo a izquierdas (Mode, acousticness, tempo) y a derechas (Liveness, time signature, valence).
    
    3.  La media y la mediana de la duración distan 8 segundos por lo que vemos que hay algún outlayer de más duración.
    
    4.  No se encuentra un número alto de canciones grabadas en directo (liveness).
    
    5.  Algunas features tienen correlacción (Loudness y energy, acousticness y energy, valence y daceability, acousticness y 
        loudness) pero conociendo cómo se produce la música y a que hace referencia cada feature, tiene sentido. Por ejemplo,
        las canciones acústicas suelen ser menos enérgicas.
    
    
    ***COMPARACIÓN CON LAS MÁS POPULARES DE LA HISTORIA Y DE 2023***
    
    1. **Popularidad**: Parece que mis canciones favoritas tienen una popularidad promedio bastante más baja
       que las canciones más streameadas de la historia y ligeramente inferior a las más escuchadas en 2023.
       Esto me hace pensar que mis gustos tienden a ser algo diferentes de lo que es popular a nivel global.
       
    2. **Duración**: La duración promedio de mis canciones favoritas es menor que las de las canciones más
       populares de la historia y de 2023. Me doy cuenta de que suelo preferir canciones más breves.

    3. **Danceability**: Me parece interesante que mis canciones tienen un nivel de "bailabilidad" que se asemeja
       al de las canciones más populares de 2023, lo que sugiere que me gusta el ritmo, aunque no siempre esté
       entre lo más escuchado.

    4. **Energy y Loudness**: Aparentemente, disfruto de música menos enérgica y menos "ruidosa" en comparación
       con las más populares de 2023. Esto podría sugerir que prefiero música más tranquila o menos intensa.

    5. **Valence**: Me he dado cuenta de que mis canciones favoritas son generalmente más alegres y positivas
       que las canciones más escuchadas globalmente, lo que refleja una inclinación hacia un tono emocional más optimista.
    
    6.  En general, este análisis me ha ayudado a ver cómo mis gustos musicales se alinean o se desvían de las tendencias globales,
        y me confirma que mi estilo es único en muchos sentidos.
    
    
    ***GÉNEROS, ARTISTAS Y ÁLBUMS***
    
    1.  En cuanto a géneros, he hecho una limpieza basándome en los géneros más habituales y eliminando los secundarios (Ejemplo: indie rock
        por lo general es una canción indie y no rock). He unificado géneros con similitudes como Rumba y flamenco, hip-hop y rap, etc. Además,
        por conocimiento de la música a aparecer he dejado géneros minoritarios como pueden ser reggae o ska.
        
    2.  Es llamativo ver como géneros como la música clásica o la electrónica están tan bajas, es posible que alguna de las canciones que no
        tenían género asignado pertenecía a uno de estos dos grupos.
        
    3.  El artista que más canciones tiene guardadas es Leiva y completando los 6 primeros puestos Melendi, Pereza (Grupo anterior de Leiva),
        Extremoduro y Fito y Fitipaldis. Para encontrar al primer artista de habla no hispana hay que irse hasta Arctic Monkeys (21 canciones,
        puesto 27). Esto hace presagiar que para que una canción de habla no hispana me guste debe tener algo más, además probablemente esto haya
        ido unido a mi desarrollo del lenguaje anglosajón o descubrimiento de música en habla italiana, turca o germana.
        
    4.  En la lista de artistas aparecen un total de 1546 artistas diferentes.
    
    5.  Los dos álbums más escuchados son álbums de versiones recopilatorios de grupos (Fitografía y All Star Smash Hits).
    
    
    ***PLAYLIST AUTOMÁTICAS***
    
    1.  Utilicemos el número de clústeres que utilicemos y el modelo de clusterización siempre suele tener más peso la feature
        Tempo para hacer la separación.


    ***CONCLUSIONES FINALES***
    
    1.  Se ha conseguido elaborar una herramienta capaz de realizar un análisis más completo que el reporte anual de Spotify,
        probablemente con la cantidad de datos que maneja Spotify entre usuarios podría aportar un análisis de mayor calidad
        con unos costes muy reducidos.
        
    2.  Se podría haber hecho la misma comparación con los géneros que con los valores numéricos respecto a las playlist más populares.

    3.  Es bonito ver cómo una playlist creada a lo largo de más de 12 años va evolucionando y hacen referencias a artistas retirados,
        o géneros de música que en este momento no escucho tanto.
    """
    print(texto_conclusiones)



def reporte_final(data):
    conocer_variables()
    clean_df(data)
    eda(data)
    var_num(data)
    favs_vs_2023_vs_history(data)
    genre(data)
    artistas_albums(data)
    playlist(data)
    #llevarlo_a_spotify(data)
    conclusiones()

reporte_final(df)


