import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Configurer Seaborn pour avoir des visualisations 
sns.set(style="whitegrid")

# Charger le jeu de données Titanic depuis l' URL 
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

# Afficher les premières lignes du DataFrame pour vérifier le chargement des données
df.head()

# Afficher les informations sur le DataFrame
df.info()

# Résumé statistique des colonnes numériques
df.describe()

# Vérifier s'il y a des valeurs manquantes
df.isnull().sum()

# Remplacer les valeurs manquantes dans la colonne 'Age' par la médiane des âges
df['Age'].fillna(df['Age'].median(), inplace=True)

# Remplacer les valeurs manquantes dans la colonne 'Embarked' par le mode
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# Optionnel : Supprimer la colonne 'Cabin' car elle contient trop de valeurs manquantes
df.drop(columns=['Cabin'], inplace=True)

# Visualisation de la survie par sexe
sns.countplot(x='Survived', hue='Sex', data=df)
plt.title('Survival by Gender')
plt.show()

# Visualisation de la survie par classe
sns.countplot(x='Survived', hue='Pclass', data=df)
plt.title('Survival by Passenger Class')
plt.show()

# Histogramme de la distribution des âges par survie
sns.histplot(data=df, x='Age', hue='Survived', kde=True)
plt.title('Age Distribution by Survival')
plt.show()

# Corrélation entre les variables
correlation_matrix = df.corr()

# Visualisation de la matrice de corrélation
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix')
plt.show()

