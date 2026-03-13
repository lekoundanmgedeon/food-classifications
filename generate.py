import os
import pandas as pd
import json

test_dir = 'data/test'
output_file = 'truth.csv'

# 1. Récupérer les classes et les trier (Logique ImageFolder)
classes = sorted([d for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, d))])
class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}

# Optionnel : Sauvegarder le mapping pour vérification
with open('class_mapping.json', 'w') as f:
    json.dump(class_to_idx, f, indent=4)

data = []

# 2. Parcourir les dossiers pour créer la liste de vérité
for food_category in classes:
    category_path = os.path.join(test_dir, food_category)
    label_idx = class_to_idx[food_category] # L'index numérique (0, 1, 2...)
    
    for filename in os.listdir(category_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            data.append({
                'id': filename, 
                'target': label_idx  # On stocke l'ID numérique ici
            })

# 3. Sauvegarder
df_truth = pd.DataFrame(data)
df_truth.to_csv(output_file, index=False)

print(f"✅ Truth généré !")
print(f"Classes détectées ({len(classes)}) : {classes}")