import requests

# URL der API
url = "http://tetrispi.duckdns.org/score?sortiert=true"


def get_scores():
    try:
        response = requests.get(url)
        response.raise_for_status()

        scores = response.json()

        if not scores:
            print("Keine Scores gefunden.")
            return 0

        print("Scores:")
        with open("Scores.txt","w") as datei:
            for entry in scores:
                datei.write(f"{entry.get('name')}: {entry.get('score')}\n")
                print(f"Nr: {entry.get('nr', '-')}, Name: {entry.get('name')}, Score: {entry.get('score')}")
        
        return 1

    except requests.RequestException as e:
        print(f"Fehler beim Abrufen der Scores: {e}")
        return 0    


def add_score(name, score):
    try:
        payload = {"name": name, "score": score}
        response = requests.post(url, json=payload)
        response.raise_for_status()

        print("Score erfolgreich hochgeladen:", response.json())
        return 1
    
    except requests.RequestException as e:
        print(f"Fehler beim Hochladen des Scores: {e}")
        return 0


if __name__ == "__main__":
    
    # Beispiel zum Hochladen eines Scores
    #add_score("Test4", 444)

    get_scores()
