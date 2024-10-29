**Prerequisites**

    Python 3.8+: This project requires Python 3.8 or higher.
    pip: Python package installer.

**Installation**

    Clone this repository:

    bash

git clone https://github.com/yourusername/spotify-data-analysis.git

  Navigate to the project directory:

  bash

  cd spotify-data-analysis

Install the required dependencies from requirements.txt:

   bash

    pip install -r requirements.txt

**Usage**

To start the analysis, run main.py. This script will load the datasets, perform data cleaning and analysis, and output the results.

  bash
  
  python main.py

  In main.py, you can follow the entire process of data loading, processing, and analysis. Be sure to have all datasets in the same directory as main.py (or adjust file paths as needed).

**Data**

The following datasets are required for main.py to execute the analysis:

    favoritas_hasta_septiembre24.csv: Contains the list of favorite songs up to September 24.
    top_100_most_streamed_songs_on_spotify_updated.csv: Holds the updated list of Spotify's top 100 most-streamed songs globally.
    top_canciones_2023_espaa.csv: Features the top songs in Spain for 2023.

Each dataset should be in CSV format and placed in the project directory for main.py to access.
Output Directory

The directory output is used to store all generated output files, such as visualizations, summaries, and analysis reports created by main.py. Ensure the output folder exists, or the script will create it during execution.

**License**

This project is licensed under the MIT License.
