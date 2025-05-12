import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# Dataset ficticio
data = {
    'HorasEstudio': [2, 10, 4, 8, 1, 6, 7, 5, 3, 9],
    'ConocimientoPrevio': [3, 9, 5, 8, 2, 6, 7, 5, 4, 10],
    'Asistencia': [60, 90, 70, 95, 50, 80, 85, 75, 65, 100],
    'PromedioTareas': [5.5, 9.0, 6.5, 8.5, 4.0, 7.0, 8.0, 7.5, 5.0, 9.5],
    'TipoEstudiante': [1, 0, 1, 0, 1, 0, 0, 1, 1, 0],
    'Resultado': [0, 1, 0, 1, 0, 1, 1, 1, 0, 1]
}

# Crear DataFrame
df = pd.DataFrame(data)

# Separar características y etiquetas
X = df.drop('Resultado', axis=1)
y = df['Resultado']

# Dividir en datos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Crear y entrenar el modelo
modelo = DecisionTreeClassifier()
modelo.fit(X_train, y_train)

# Realizar predicciones y evaluar
y_pred = modelo.predict(X_test)
print("Precisión del modelo:", accuracy_score(y_test, y_pred))
print("\nReporte de Clasificación:\n", classification_report(y_test, y_pred))

# Probar con un nuevo estudiante
nuevo_estudiante = [[7, 8, 90, 8.0, 1]]  # puedes cambiar los valores
resultado = modelo.predict(nuevo_estudiante)
print("Resultado del nuevo estudiante:", "Aprobado" if resultado[0] == 1 else "No Aprobado")
