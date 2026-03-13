import pandas as pd
import os
from sklearn.metrics import accuracy_score, f1_score

# Chemins
SUBMISSIONS_DIR = 'submissions/'
TRUTH_PATH = 'truth.csv'
LEADERBOARD_PATH = 'leaderboard/leaderboard.csv'

def update_leaderboard():
    # 1. Charger la vérité terrain
    df_truth = pd.read_csv(TRUTH_PATH).sort_values('id').reset_index(drop=True)
    
    # 2. Liste pour stocker les nouveaux résultats
    all_results = []

    # 3. Parcourir tous les fichiers de soumission
    for filename in os.listdir(SUBMISSIONS_DIR):
        if filename.endswith(".csv"):
            team_name = filename.replace(".csv", "")
            sub_path = os.path.join(SUBMISSIONS_DIR, filename)
            
            try:
                df_sub = pd.read_csv(sub_path).sort_values('id').reset_index(drop=True)
                
                if len(df_sub) == len(df_truth):
                    acc = accuracy_score(df_truth['target'], df_sub['target'])
                    f1 = f1_score(df_truth['target'], df_sub['target'], average='weighted')
                    
                    all_results.append({
                        'Team': team_name,
                        'accuracy': round(acc, 4),
                        'f1_score': round(f1, 4)
                    })
            except Exception as e:
                print(f"⚠️ Erreur avec {team_name}: {e}")

    # 4. Créer le DataFrame et trier par la meilleure accuracy
    new_leaderboard = pd.DataFrame(all_results)
    new_leaderboard = new_leaderboard.sort_values(by='accuracy', ascending=False).reset_index(drop=True)
    
    # Ajouter le rang
    new_leaderboard.index += 1
    new_leaderboard.index.name = 'Rank'

    # 5. Sauvegarder (écrase l'ancien avec les données fraîches)
    os.makedirs(os.path.dirname(LEADERBOARD_PATH), exist_ok=True)
    new_leaderboard.to_csv(LEADERBOARD_PATH)
    print(f"✅ Leaderboard mis à jour avec {len(new_leaderboard)} participants.")

if __name__ == "__main__":
    update_leaderboard()