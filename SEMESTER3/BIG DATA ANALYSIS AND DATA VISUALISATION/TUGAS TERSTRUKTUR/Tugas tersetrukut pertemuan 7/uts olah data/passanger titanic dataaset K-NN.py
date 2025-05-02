from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Load dataset from local folder in Colab
df = pd.read_csv('passenger titanic dataset.csv') # Ubah 'filename.csv' sesuai dengan nama file Anda

# hilangkan data yang tidak dipakai
df = df.drop(['Name','Ticket','PassengerId','Cabin'],axis=1)

# Mengganti nilai NaN dalam kolom yg terdapat Nan dengan nilai tertentu untuk setiap kolom
replacement_values = {    'Fare': df['Fare'].median(),}
df.fillna(replacement_values, inplace=True)

# fungsi replace null pada kolom age
def impute_train_age(cols):
    Age = cols.iloc[0]
    Pclass = cols.iloc[1]
    
    if pd.isnull(Age):

        if Pclass == 1:
            return 37

        elif Pclass == 2:
            return 29

        else:
            return 24

    else:
        return Age
    
df['Age'] = df[['Age','Pclass']].apply(impute_train_age,axis=1)

# print(df.head())
# Mengecek keberadaan nilai NaN dalam DataFrame
# missing_values = df.isna().sum()
# print("Jumlah nilai NaN dalam setiap kolom:")
# print(missing_values)
# print(df)

label = LabelEncoder()
data_column = ['Sex','Embarked']
for column in data_column:
    df[column] = label.fit_transform(df[column])

# tentukan target
X = df.drop(['Survived'],axis = 1)
y = df['Survived']

# buat pembagian dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=42)
# # Encoding categorical variables
# label_encoders = {}
# for column in ['cut', 'color', 'clarity']:
# label_encoders[column] = LabelEncoder()
# df[column] = label_encoders[column].fit_transform(df[column])

# # Splitting dataset into features and target variable
# X = df.drop(['price'], axis=1)
# y = df['price']

# # Feature scaling
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)



# Training the K-NN model
k = 5 # Number of neighbors
knn_classifier = KNeighborsClassifier(n_neighbors=k)
knn_classifier.fit(X_train, y_train)

# Making predictions on the test set
y_pred = knn_classifier.predict(X_test)

# Confusion Matrix
# cm = confusion_matrix(y_test, y_pred)
# plt.figure(figsize=(10,7))
# sns.heatmap(cm, annot=True, fmt='d')
# plt.xlabel('Predicted')
# plt.ylabel('Actual')
# plt.title('Confusion Matrix')
# plt.show()

# Classification Report
report = classification_report(y_test, y_pred)
print("\nClassification Report:\n", report)

# Save the classification report as a text file
with open('classification_report.txt', 'w') as file:
    file.write(report)

# # Scatter Plot
# plt.figure(figsize=(10,7))
# plt.scatter(y_test, y_pred)
# plt.xlabel('Actual Prices')
# plt.ylabel('Predicted Prices')
# plt.title('Actual Prices vs Predicted Prices')
# plt.show()

# # Prediction
sample_data = X_test[:5] # Taking first 5 samples from the test set
sample_predictions = knn_classifier.predict(sample_data)
print("\nPrediksi untuk 5 yang pertama:")
for i in range(len(sample_data)):
    print("Sample", i+1, ":", sample_predictions[i])


# # Test and Score
score = knn_classifier.score(X_test, y_test)
print("\nAccuracy:", score)
