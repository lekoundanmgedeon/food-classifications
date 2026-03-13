import streamlit as st
import pandas as pd

# Configuration de la page
st.set_page_config(page_title="Food AI Challenge", layout="wide")

st.title("🏆 Food Classification Leaderboard")
st.markdown("Classement en temps réel des meilleurs modèles de reconnaissance culinaire.")

# Chargement des données
try:
    df = pd.read_csv("leaderboard/leaderboard.csv")

    # Nettoyage et tri (par défaut sur l'Accuracy)
    df = df.sort_values("accuracy", ascending=False).reset_index(drop=True)
    df.index += 1  # Pour afficher le rang (1, 2, 3...)

    # --- Zone de statistiques ---
    col1, col2, col3 = st.columns(3)
    col1.metric("Top Accuracy", f"{df['accuracy'].max()*100:.2f}%")
    col2.metric("Meilleur F1-Score", f"{df['f1_score'].max():.3f}")
    col3.metric("Participants", len(df))

    # --- Affichage du Tableau ---
    # On utilise st.dataframe avec un style pour surligner le gagnant
    st.subheader("Classement Officiel")
    
    styled_df = df.style.highlight_max(axis=0, subset=['accuracy', 'f1_score'], color='#2E7D32') \
                       .format({'accuracy': '{:.2%}', 'f1_score': '{:.3f}'})

    st.dataframe(
        styled_df,
        use_container_width=True,
        height=400
    )

    # Petit graphique pour visualiser l'écart entre les modèles
    st.divider()
    st.subheader("Visualisation des performances")
    st.bar_chart(df.set_index('Team')[['accuracy', 'f1_score']])

except FileNotFoundError:
    st.error("Le fichier leaderboard.csv est introuvable. Vérifiez le chemin.")
except Exception as e:
    st.exception(e)