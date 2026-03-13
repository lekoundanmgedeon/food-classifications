import pandas as pd
import os
from sklearn.metrics import accuracy_score # Ou toute autre métrique (MAE, F1, etc.)

# Configuration
SUBMISSIONS_DIR = 'submissions/'  # Dossier où sont les fichiers des participants
GROUND_TRUTH_PATH = 'truth.csv'   # Votre fichier de réponse
OUTPUT_PATH = 'leaderboard.csv'

def evaluate_submissions():
    # 1. Charger la vérité terrain
    df_truth = pd.read_csv(GROUND_TRUTH_PATH).sort_values(by='id')
    
    results = []

    # 2. Parcourir les soumissions des participants
    for filename in os.listdir(SUBMISSIONS_DIR):
        if filename.endswith(".csv"):
            team_name = filename.replace(".csv", "")
            filepath = os.path.join(SUBMISSIONS_DIR, filename)
            
            try:
                df_sub = pd.read_csv(filepath).sort_values(by='id')
                
                # Vérification rapide (même nombre de lignes)
                if len(df_sub) == len(df_truth):
                    # 3. Calcul de la métrique
                    score = accuracy_score(df_truth['target'], df_sub['target'])
                    
                    results.append({
                        'Team': team_name,
                        'Score': round(score, 4)
                    })
            except Exception as e:
                print(f"Erreur avec le fichier {filename}: {e}")

    # 4. Créer le DataFrame final et trier
    leaderboard_df = pd.DataFrame(results)
    leaderboard_df = leaderboard_df.sort_values(by='Score', ascending=False).reset_index(drop=True)
    leaderboard_df.index += 1 # Commencer le classement à 1
    
    # 5. Sauvegarder pour Streamlit
    leaderboard_df.to_csv(OUTPUT_PATH, index_label='Rank')
    print("Leaderboard mis à jour !")

evaluate_submissions()