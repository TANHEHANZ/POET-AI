import csv
import requests

url = "https://poetrydb.org/author,title/Shakespeare;Sonnet"

poems_list = []

try:
    response = requests.get(url)
    
    if response.status_code == 200:
        poems_data = response.json()
        
        for poem in poems_data:
            title = poem["title"]
            author = poem["author"]
            lines = poem["lines"]
            
            poems_list.append({"title": title, "author": author, "lines": lines})
            
        with open("poems.csv", "w", newline="", encoding="utf-8") as csvfile:
            fieldnames = ["title", "author", "lines"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for poem in poems_list:
                writer.writerow(poem)
            
    else:
        print("Error al obtener los datos:", response.status_code)

except requests.exceptions.RequestException as e:
    print("Error de conexión:", e)
