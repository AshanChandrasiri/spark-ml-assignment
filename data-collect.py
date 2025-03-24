# import requests
# import os
# import csv
#
# # Set your Spotify API credentials
# SPOTIFY_AUTH_URL = "https://accounts.spotify.com/api/token"
# SPOTIFY_SEARCH_URL = "https://api.spotify.com/v1/search"
# CLIENT_ID = "your_spotify_client_id"
# CLIENT_SECRET = "your_spotify_client_secret"
#
#
# def get_spotify_token():
#     response = requests.post(SPOTIFY_AUTH_URL, {
#         "grant_type": "client_credentials",
#         "client_id": CLIENT_ID,
#         "client_secret": CLIENT_SECRET
#     })
#     token_info = response.json()
#     return token_info.get("access_token")
#
#
# def fetch_classical_tracks():
#     token = 'BQA2RX8A9HgPsy_dN2-qwlAouhFhlEs7yl_S3E9xeBc0sNAVnLo0e0PD56Xx1OdOQ9F0cu4LgcXWaVFAhp6JE1uzcGRiyJC3GUGNSX5rrj0nwlT0oppkNYWw1A7SV3OjS1wtN-ncg_m0vGSvPzfRlFl05yWj8jw_7O29kVFNjlnA3IAVORhMRIoHRCWMdaPN3bfq2Yo8MKuamW_v0QJJdZcbmyQUv0qS69iVKz_AZVMqtZ7YzkE8BJuUv2V14S1HMYm5hIJLg1SBOYWpnAcXB2GGZaCAuxsPhuoZSr5ey8gJ4Fe3oQ9VCmpn'
#     headers = {"Authorization": f"Bearer {token}"}
#     tracks = []
#
#     for offset in [0, 50]:  # Fetch two pages (50 + 50 = 100 tracks)
#         params = {
#             "q": "genre:classical",
#             "type": "track",
#             "limit": 50,
#             "offset": offset
#         }
#         response = requests.get(SPOTIFY_SEARCH_URL, headers=headers, params=params)
#         data = response.json()
#
#         for item in data.get("tracks", {}).get("items", []):
#             artist_name = item["artists"][0]["name"]  # First artist
#             track_name = item["name"]
#             release_date = item["album"]["release_date"][:4]  # Extract year
#             genre = "Classical"
#             lyrics = fetch_lyrics(artist_name, track_name)
#
#             tracks.append({
#                 "artist_name": artist_name,
#                 "track_name": track_name,
#                 "release_date": release_date,
#                 "genre": genre,
#                 "lyrics": lyrics
#             })
#
#     return tracks
#
#
# # Set your Genius API credentials
# GENIUS_API_URL = "https://api.genius.com"
# GENIUS_TOKEN = "o-CQR2aP7A5Hm3iXURT6PjeAhwoGR0FOzonZ55sqYt55C-LrYuihAytLhWXnZmsu"
#
#
# def fetch_lyrics(artist, track):
#     headers = {"Authorization": f"Bearer {GENIUS_TOKEN}"}
#     search_url = f"{GENIUS_API_URL}/search"
#     params = {"q": f"{artist} {track}"}
#     response = requests.get(search_url, headers=headers, params=params)
#     data = response.json()
#
#     if "response" in data and "hits" in data["response"]:
#         for hit in data["response"]["hits"]:
#             if hit["result"]["primary_artist"]["name"].lower() == artist.lower():
#                 return hit["result"]["url"]  # Returning Genius lyrics page URL
#     return "Lyrics not found"
#
#
# def save_to_csv(tracks, filename="classical_tracks.csv"):
#     with open(filename, mode='w', newline='', encoding='utf-8') as file:
#         writer = csv.DictWriter(file, fieldnames=["artist_name", "track_name", "release_date", "genre", "lyrics"])
#         writer.writeheader()
#         writer.writerows(tracks)
#
#
# if __name__ == "__main__":
#     classical_tracks = fetch_classical_tracks()
#     save_to_csv(classical_tracks)
#     print(f"Saved {len(classical_tracks)} tracks to classical_tracks.csv")


import pandas as pd
import random


def shuffle_and_select_rows(input_file, output_file, num_rows=110):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(input_file)

    # Shuffle the DataFrame rows
    df = df.sample(frac=1, random_state=84).reset_index(drop=True)

    # Select 110 random rows
    selected_rows = df.sample(n=num_rows, random_state=84)

    # Write the selected rows to a new CSV file
    selected_rows.to_csv(output_file, index=False)


# Example usage
input_file = './Student_dataset.csv'  # Replace with your input file path
output_file = './selected_rows_output.csv'  # Replace with your desired output file path

shuffle_and_select_rows(input_file, output_file)
