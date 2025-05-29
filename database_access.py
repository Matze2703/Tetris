import requests

# URL deiner API
url = "http://tetrispi.duckdns.org/score?sortiert=true"


def get_scores():
    try:
        response = requests.get(url)
        response.raise_for_status()

        scores = response.json()

        if not scores:
            print("Keine Scores gefunden.")
            return

        print("Scores:")
        for entry in scores:
            print(f"Nr: {entry.get('nr', '-')}, Name: {entry.get('name')}, Score: {entry.get('score')}")
    except requests.RequestException as e:
        print(f"Fehler beim Abrufen der Scores: {e}")


def add_score(name, score):
    try:
        payload = {"name": name, "score": score}
        response = requests.post(url, json=payload)
        response.raise_for_status()

        print("Score erfolgreich hochgeladen:", response.json())
    except requests.RequestException as e:
        print(f"Fehler beim Hochladen des Scores: {e}")


if __name__ == "__main__":
    
    # Beispiel zum Hochladen eines Scores
    add_score("Test2", 2345)

    get_scores()
