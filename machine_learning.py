import streamlit as st
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris


st.set_page_config(page_title="Machine Learning - FACV")

st.markdown("""
<style>
    .main { 
        background-color: #F7F9FC; 
    }
    .title { 
        text-align: center; font-size: 32px; font-weight: 700; color: #1F4172; 
    }
    .sub { 
        font-size: 18px; color: #3C4A60; text-align:center; margin-top: -10px; 
    }
    .section-title { 
        font-size: 22px; font-weight: 700; margin-top: 25px; color: #1F6E8C; 
    }
    .box { 
        padding: 1px; border-radius: 8px; background-color: #ffffff; border: 1px solid #DDE4EB; margin-top: 10px; 
    }
</style>
""", unsafe_allow_html=True)


st.markdown('<h1 class="title">Procesamiento de Datasets en Machine Learning</h1>', unsafe_allow_html=True)
opcion = st.radio("Seleccione una vista:", ["Ejercicio 1", "Ejercicio 2", "Ejercicio 3"], horizontal=True)

st.markdown('<h2 class="sub">Alumno: Chávez Vidal, Felix Andreé</h2>', unsafe_allow_html=True)

if opcion == "Ejercicio 1":
    st.markdown('<h3 class="section-title">1️⃣ Ejercicio 1: Análisis del Dataset “Titanic”</h3>', unsafe_allow_html=True)

    file = st.file_uploader("Subir un archivo CSV", type=["csv"])

    if file is not None:
        if file.name.lower() != "titanic.csv":
            st.warning("⚠️ Debes subir el archivo **titanic.csv**")
        else:
            dataset = pd.read_csv(file)
    
            with st.expander("Pasos del procesamiento"):
                st.markdown(
                    """
                    <h4 style='color: #FFF200;'>Exploración Inicial</h4>
                    <ul>
                        <li>Carga del dataset</li>
                        <li>Observación de los tipos de datos y los nulos</li>
                    </ul>
                    <br>
                    <h4 style='color: #FFF200;'>Limpieza de datos</h4>
                    <ul>
                        <li>
                            Eliminación de columnas irrelevantes: Name, Ticket y Cabin.
                        </li>
                        <li>
                            Reemplazo de valores nulos en Age (media) y Embarked (moda).
                        </li>
                    </ul>
                    <br>
                    <h4 style='color: #FFF200;'>Codificación de Variables Categóricas</h4>
                    <ul>
                        <li>
                            Transformó los valores categoricos en número y luego en columnas binarias para columnas Sex y Embarked.
                        </li>
                    </ul>
                    <br>
                    <h4 style='color: #FFF200;'>Normalización o estandarización</h4>
                    <ul>
                        <li>
                            Estandarización de variables Age y Fare.
                        </li>
                    </ul>
                    <br>
                    <h4 style='color: #FFF200;'>División en conjuntos de entrenamiento y prueba</h4>
                    <ul>
                        <li>
                            Observación de resultados con entrenamiento (70%) y prueba (30%)
                        </li>
                    </ul>
                    """
                    , unsafe_allow_html=True
                )

            st.markdown('<br><div class="box"></div>', unsafe_allow_html=True)



            # Eliminar columnas Name, Ticket o Cabin
            dataset = dataset.drop(columns=["Name", "Ticket", "Cabin"])

            x = dataset.iloc[:, [0,2,3,4,5,6,7,8]]   # Variables independientes
            y = dataset.iloc[:, 1]                   # Variable dependiente


            # Reemplazo de Age con la media y Embarked con la moda
            imputer_mean = SimpleImputer(strategy="mean")
            dataset["Age"] = imputer_mean.fit_transform(dataset["Age"].values.reshape(-1,1))

            imputer_mode = SimpleImputer(strategy="most_frequent")
            dataset["Embarked"] = imputer_mode.fit_transform(dataset[["Embarked"]])[:, 0]


            # Transformamos los valores categoricos en números
            le = LabelEncoder()
            dataset["Sex"] = le.fit_transform(dataset["Sex"])
            dataset["Embarked"] = le.fit_transform(dataset["Embarked"])

            # Transformamos los números en columnas binarias
            ct = ColumnTransformer(transformers=[('one_hot_encoder', OneHotEncoder(), ["Sex", "Embarked"])],remainder='passthrough')
            x = np.array(ct.fit_transform(dataset), dtype=np.float64)


            # Estandarización de las variables numéricas (Age, Fare).
            scaler = StandardScaler()
            dataset[["Age", "Fare"]] = scaler.fit_transform(dataset[["Age", "Fare"]])


            # Dividición de datos en entrenamiento (70%) y prueba (30%)
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=0)

            st.markdown('<h3 class="section-title">✏️ Tabla con los primeros 5 registros procesados</h3>', unsafe_allow_html=True)
            st.dataframe(dataset.head())


            st.markdown('<div style="color: #FFF200;"><b>Dimensiones Resultantes</b></div>', unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Entrenamiento (70%)**: {len(x_train)}")
            with col2:
                st.write(f"**Prueba (30%)**: {len(x_test)}")




elif opcion == "Ejercicio 2":
    st.markdown('<h3 class="section-title">2️⃣ Ejercicio 2: Procesamiento de Dataset “Student Performance”</h3>', unsafe_allow_html=True)

    file = st.file_uploader("Subir un archivo CSV", type=["csv"])

    if file is not None:
        if file.name.lower() != "student-mat.csv":
            st.warning("⚠️ Debes subir el archivo **student-mat.csv**")
        else:
            dataset = pd.read_csv(file)

            with st.expander("Pasos del procesamiento"):
                st.markdown(
                    """
                    <h4 style='color: #FFF200;'>Exploración Inicial</h4>
                    <ul>
                        <li>Carga del dataset</li>
                        <li>Análisis de las variables categóricas: Son school, sex, address, famsize, Pstatus, Fedu, Mjob, Fjob, reason, guardian, schoolsup, famsup, paid, activities, nursery, higher, internet y romantic</li>
                    </ul>
                    <br>
                    <h4 style='color: #FFF200;'>Limpieza de datos</h4>
                    <ul>
                        <li>
                            Eliminación de datos duplicados y valores inconsistentes
                        </li>
                    </ul>
                    <br>
                    <h4 style='color: #FFF200;'>Codificación de Variables Categóricas</h4>
                    <ul>
                        <li>
                            Aplicación del One Hot Encoding a las variables categóricas (school, sex, address,etc.).
                        </li>
                    </ul>
                    <br>
                    <h4 style='color: #FFF200;'>Normalización o estandarización</h4>
                    <ul>
                        <li>
                            Normalización de variables numéricas (age, absences, G1, G2).
                        </li>
                    </ul>
                    <br>
                    <h4 style='color: #FFF200;'>División en conjuntos de entrenamiento y prueba</h4>
                    <ul>
                        <li>
                            Separación de los datos en "X" y "Y" (características y variable objetivo).
                        </li>
                        <li>
                            División y observación de resultados con entrenamiento (80%) y prueba (20%).
                        </li>
                    </ul>
                    <br>
                    <h4 style='color: #FFF200;'>Análisis de correlación</h4>
                    <ul>
                        <li>
                            Análisis la correlación entre las notas G1, G2 y G3.
                        </li>
                    </ul>
                    """
                    , unsafe_allow_html=True
                )

            st.markdown('<br><div class="box"></div>', unsafe_allow_html=True)



            # Eliminar duplicados
            dataset = dataset.drop_duplicates()

            # Corrección de valores inconsistentes
            # Rango de edad de un estudiante (10–22)
            dataset = dataset[(dataset["age"] >= 10) & (dataset["age"] <= 22)]

            # Ausencias negativas deben ser 0
            dataset["absences"] = dataset["absences"].apply(lambda x: 0 if x < 0 else x)

            # Rango de notas (0-20)
            dataset = dataset[
                dataset["G1"].between(0, 20) &
                dataset["G2"].between(0, 20) &
                dataset["G3"].between(0, 20)
            ]


            # Aplique One Hot Encoding a las variables categóricas
            # Transformamos los valores categoricos en números
            le = LabelEncoder()
            dataset["school"] = le.fit_transform(dataset["school"])
            dataset["sex"] = le.fit_transform(dataset["sex"])
            dataset["address"] = le.fit_transform(dataset["address"])
            dataset["famsize"] = le.fit_transform(dataset["famsize"])
            dataset["Pstatus"] = le.fit_transform(dataset["Pstatus"])
            dataset["Fedu"] = le.fit_transform(dataset["Fedu"])
            dataset["Mjob"] = le.fit_transform(dataset["Mjob"])
            dataset["Fjob"] = le.fit_transform(dataset["Fjob"])
            dataset["reason"] = le.fit_transform(dataset["reason"])
            dataset["guardian"] = le.fit_transform(dataset["guardian"])
            dataset["schoolsup"] = le.fit_transform(dataset["schoolsup"])
            dataset["famsup"] = le.fit_transform(dataset["famsup"])
            dataset["paid"] = le.fit_transform(dataset["paid"])
            dataset["activities"] = le.fit_transform(dataset["activities"])
            dataset["nursery"] = le.fit_transform(dataset["nursery"])
            dataset["higher"] = le.fit_transform(dataset["higher"])
            dataset["internet"] = le.fit_transform(dataset["internet"])
            dataset["romantic"] = le.fit_transform(dataset["romantic"])

            # Transformamos los números en columnas binarias
            ct = ColumnTransformer(transformers=[('one_hot_encoder', OneHotEncoder(), ["school", "sex", "address", "famsize", "Pstatus", "Fedu", "Mjob", "Fjob", "reason", "guardian", "schoolsup", "famsup", "paid", "activities", "nursery", "higher", "internet", "romantic"])],remainder='passthrough')
            x = np.array(ct.fit_transform(dataset), dtype=np.float64)


            # Estandarización de las variables numéricas (age, absences, G1, G2).
            scaler = StandardScaler()
            dataset[["age", "absences", "G1", "G2"]] = scaler.fit_transform(dataset[["age", "absences", "G1", "G2"]])


            # Separe los datos en X y Y (características y variable objetivo).
            x = dataset.iloc[:,:-1].values          # Variables independientes
            y = dataset.iloc[:,32].values           # Variable dependiente (G3)

            # Dividición de datos en entrenamiento (80%) y prueba (20%)
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=0)


            st.markdown('<h3 class="section-title">✏️ Tabla con los primeros 5 registros procesados</h3>', unsafe_allow_html=True)
            st.dataframe(dataset.head())


            st.markdown('<div style="color: #FFF200;"><b>Dimensiones Resultantes</b></div>', unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Entrenamiento (80%)**: {len(x_train)}")
            with col2:
                st.write(f"**Prueba (20%)**: {len(x_test)}")


            st.markdown('<div style="color: #FFF200;"><br><b>Shape de entrenamiento y prueba.</b></div>', unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**x_train shape:** {x_train.shape}")
                st.write(f"**y_train shape:** {y_train.shape}")
            with col2:
                st.write(f"**x_test shape:** {x_test.shape}")
                st.write(f"**y_test shape:** {y_test.shape}")


            # Correlación entre G1, G2 y G3
            st.markdown('<h3 class="section-title"><br>📝 Analisis la correlación entre las notas G1, G2 y G3</h3>', unsafe_allow_html=True)

            correlacion = dataset[["G1", "G2", "G3"]].corr()
            matriz, ax = plt.subplots()
            sns.heatmap(correlacion, annot=True, ax=ax)
            st.pyplot(matriz)




else:
    st.markdown('<h3 class="section-title">3️⃣ Ejercicio 3: Dataset “Iris”</h3>', unsafe_allow_html=True)

    with st.expander("Pasos del procedimiento"):
        st.markdown(
            """
            <h4 style='color: #FFF200;'>Exploración Inicial</h4>
            <ul>
                <li>Carga del dataset desde sklearn.datasets</li>
                <li>Conversión a DataFrame y agregar nombres de las columnas</li>
            </ul>
            <br>
            <h4 style='color: #FFF200;'>Normalización o estandarización</h4>
            <ul>
                <li>
                    Estandarización de variables usando StandardScaler()
                </li>
            </ul>
            <br>
            <h4 style='color: #FFF200;'>División en conjuntos de entrenamiento y prueba</h4>
            <ul>
                <li>
                    Observación de resultados con entrenamiento (70%) y prueba (30%)
                </li>
            </ul>
            <br>
            <h4 style='color: #FFF200;'>Graficación y descripción</h4>
            <ul>
                <li>
                    Grafico de distribución de sepal length y petal length diferenciadas por clase (target)
                </li>
                <li>
                    Descripción de la dataset después de todo el procesamiento
                </li>
            </ul>
            """
            , unsafe_allow_html=True
        )

    st.markdown('<br><div class="box"></div>', unsafe_allow_html=True)

    # carga de dataset
    iris = load_iris()


    # conversión a un dataframe y agregado de nombres a las columnas.
    # iris.feature_names -> devuelve el nombre de las columnas
    # iris.data -> devuelve los registros del dataset
    dataset = pd.DataFrame(iris.data, columns=iris.feature_names)

    # la columna que tiene la variable objetivo le ponemos target, usada para la gráfica de dispersión
    dataset["target"] = iris.target


    # estandarización o normalización de variables
    scaler = StandardScaler()
    dataset[iris.feature_names] = scaler.fit_transform(dataset[iris.feature_names])


    # Separe los datos en X y Y (características y variable objetivo).
    x = dataset[iris.feature_names]         # Variables independientes
    y = dataset["target"]                   # Variable dependiente (target)

    # Dividición de datos en entrenamiento (70%) y prueba (30%)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=0)


    # Gráfico de dispersión con colores por clase
    st.markdown('<h3 class="section-title"><br>📊 Gráfico: Sepal Length vs Petal Length por Clase</h3>', unsafe_allow_html=True)

    fig, ax = plt.subplots()
    for clase in dataset["target"].unique():
        subset = dataset[dataset["target"] == clase]
        ax.scatter(subset["sepal length (cm)"], subset["petal length (cm)"], label=f"Clase {clase}")

    ax.set_xlabel("Sepal Length")
    ax.set_ylabel("Petal Length")
    ax.set_title("Distribución de características: Sepal Length vs Petal Length")
    ax.legend()
    st.pyplot(fig)


    # Estadísticas descriptivas del dataset estandarizado
    st.markdown('<h3 class="section-title"><br>📋 Estadísticas descriptivas del Dataset Estandarizado</h3>', unsafe_allow_html=True)
    st.write(dataset.describe())

