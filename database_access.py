import requests, os

# URL der API
url = "http://tetrispi.duckdns.org/score?sortiert=true"


def get_scores():
    try:
        response = requests.get(url)
        response.raise_for_status()

        scores = response.json()

        if not scores:
            print("Keine Scores gefunden.")
            return 1

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

def delete_score(name):
    try:
        params = {"name": name}
        response = requests.delete(url, params=params)
        response.raise_for_status()

        print("Score erfolgreich gelöscht:", response.json())
        return 1

    except requests.RequestException as e:
        print(f"Fehler beim Löschen des Scores: {e}")
        return 0

# Dateien einlesen
def read_scores(file_path):
    scores = {}
    with open(file_path, 'r') as file:
        for line in file:
            if ':' in line:
                name, score = line.strip().split(':')
                scores[name.strip()] = int(score.strip())
    return scores

def update_scores():
    
    # Scores von Server lesen
    scores = read_scores("Scores.txt")
    # Lokale Scores lesen
    new_scores = read_scores("not_uploaded.txt")

    # Aktualisieren, wenn lokaler Score besser ist
    is_updated = True
    
    print(scores)
    print(new_scores)

    for name, new_score in new_scores.items():
        print(f"Synchronising {name}: {new_scores[name]}")
        new_scores[name] = new_score
        if name not in scores or new_score > scores[name]:
            print(f"Detected new score {name}: {new_scores[name]}")
            # Server aktualisieren
            delete_score(name.upper())
            if not add_score(name.upper(), new_score):
                is_updated = False
            else:
                print(f"{name}: {new_scores[name]} erfolgreich hochgeladen")
    
    # Wenn alle neuen Scores erfolgreich hochgeladen wurden, lokale entschlüsselte Sicherung löschen
    if is_updated == True:
        print("Erfolgreich synchronisiert")
        os.remove("not_uploaded.txt")
        # Wenn noch verschlüsselte Sicherung vorhanden ebenfalls löschen
        if os.path.isfile("not_uploaded.txt.enc"):
            os.remove("not_uploaded.txt.enc")

    # geladene Datei aktualisieren
    with open("Scores.txt", 'w') as file:
            for name, score in scores.items():
                file.write(f"{name}: {score}\n")
    return is_updated

if __name__ == "__main__":
    
    # Beispiel zum Hochladen eines Scores
    #add_score("Test4", 444)
    
    # Löschen eines Scores
    #delete_score("MATZE")
    
    #Auslesen der Scores
    get_scores()
