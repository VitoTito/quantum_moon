import os

# Fonction pour détecter et supprimer les images sans fichier .txt correspondant
def detect_and_delete_images_without_txt():
    # Liste pour stocker les noms des images sans fichier .txt
    images_without_txt = []
    
    # Récupérer tous les fichiers du dossier actuel
    files = os.listdir('.')
    
    # Filtrer les fichiers .jpg
    jpg_files = [f for f in files if f.lower().endswith('.jpg')]
    
    # Pour chaque fichier .jpg, vérifier s'il y a un fichier .txt correspondant
    for jpg in jpg_files:
        txt_file = os.path.splitext(jpg)[0] + '.txt'
        if txt_file not in files:
            images_without_txt.append(jpg)
    
    # Afficher et supprimer les images sans fichier .txt
    if images_without_txt:
        print("Images sans fichier .txt correspondant (suppression en cours) :")
        for img in images_without_txt:
            print(f"Suppression de {img}")
            os.remove(img)
    else:
        print("Toutes les images ont un fichier .txt correspondant.")
    
    # Retourner la liste des images supprimées
    return images_without_txt

# Appel de la fonction si le script est exécuté directement
if __name__ == "__main__":
    images_without_txt = detect_and_delete_images_without_txt()
    print("\nListe des images supprimées :", images_without_txt)
