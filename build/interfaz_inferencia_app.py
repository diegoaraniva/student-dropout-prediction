import streamlit as st
import joblib
import pandas as pd

# Configuracion de la pagina
st.set_page_config(
    page_title="Prediccion de Desercion Estudiantil",
    page_icon="🎓",
    layout="wide"
)

# Cargar modelo y columnas (se cachea para no recargar en cada interaccion)
@st.cache_resource
def load_model():
    """
    Carga el modelo entrenado y las columnas de features.
    @st.cache_resource hace que esto se ejecute una sola vez.
    """
    model = joblib.load('best_model.pkl')
    feature_columns = joblib.load('feature_columns.pkl')
    model_name = joblib.load('best_model_name.pkl')
    return model, feature_columns, model_name

# Intentar cargar el modelo
try:
    model, feature_columns, model_name = load_model()
except FileNotFoundError:
    st.error("No se encontraron los archivos del modelo (best_model.pkl, feature_columns.pkl).")
    st.info("Primero debes entrenar el modelo ejecutando: **python build/train.py**")
    st.stop()

# Titulo principal
st.title("Prediccion de Desercion Estudiantil")
st.write(f"Modelo: **{model_name}** (entrenado con GridSearchCV)")

st.divider()

# Instrucciones
st.subheader("Ingrese los datos del estudiante")

# Catalogos del dataset
CARRERAS = [101, 102, 103, 104, 105, 106, 107, 108, 109, 202, 203, 206, 207, 208,
            209, 210, 211, 212, 218, 219, 301, 302, 303, 401, 402, 403, 404, 405,
            409, 413, 414, 415, 441, 444, 449, 701, 702, 703, 802]
PLANES = [2008, 2009, 2013, 2014, 2015, 2016, 2017, 2018, 2019]
ANIOS_INGRESO = [2015, 2016, 2017, 2018, 2019]
CICLOS_INGRESO = [1, 2]

# Dividir la interfaz en dos columnas
col1, col2 = st.columns(2)

# Columna izquierda: datos generales del estudiante
with col1:
    st.markdown("#### Datos Generales")

    carrera = st.selectbox(
        "Carrera",
        CARRERAS,
        index=CARRERAS.index(219),
        help="Codigo de la carrera del estudiante"
    )

    plan = st.selectbox(
        "Plan de Estudios",
        PLANES,
        index=PLANES.index(2013),
        help="Anio del plan de estudios"
    )

    sexo = st.selectbox(
        "Sexo",
        [102301, 102302],
        format_func=lambda x: "Masculino" if x == 102301 else "Femenino",
        help="Sexo del estudiante"
    )

    anio_ingreso = st.selectbox(
        "Anio de Ingreso",
        ANIOS_INGRESO,
        index=ANIOS_INGRESO.index(2019)
    )

    ciclo_ingreso = st.selectbox(
        "Ciclo de Ingreso",
        CICLOS_INGRESO,
        index=0,
        help="Ciclo academico en el que ingreso el estudiante"
    )

    # Cargar catalogo de campus desde el dataset
    @st.cache_data
    def get_campus():
        import pandas as pd
        df = pd.read_csv('Tbl_DesercionEstudiantil_PrimerAnio_2015_2019.csv')
        return sorted(df['IdCampus'].dropna().unique().astype(int).tolist())

    CAMPUS = get_campus()

    id_campus = st.selectbox(
        "Campus",
        CAMPUS,
        index=0,
        help="Codigo del campus"
    )

    # Cargar catalogo de instituciones desde el dataset
    @st.cache_data
    def get_instituciones():
        import pandas as pd
        df = pd.read_csv('Tbl_DesercionEstudiantil_PrimerAnio_2015_2019.csv')
        return sorted(df['InstitucionBach'].dropna().unique().astype(int).tolist())

    INSTITUCIONES = get_instituciones()

    institucion_bach = st.selectbox(
        "Institucion de Bachillerato",
        INSTITUCIONES,
        index=INSTITUCIONES.index(300) if 300 in INSTITUCIONES else 0,
        help="Codigo de la institucion de bachillerato del estudiante"
    )

    st.markdown("#### Beca")

    tiene_beca = st.selectbox(
        "Tiene Beca",
        [0, 1],
        format_func=lambda x: "Si" if x == 1 else "No"
    )

    porcentaje_beca = st.number_input(
        "Porcentaje de Beca (promedio)",
        min_value=0.0,
        max_value=100.0,
        value=0.0,
        step=5.0
    )

# Columna derecha: rendimiento academico
with col2:
    st.markdown("#### Rendimiento Ciclo 1")

    mat_inscritas_c1 = st.number_input("Materias Inscritas C1", min_value=1, max_value=8, value=5, step=1)
    mat_aprobadas_c1 = st.number_input("Materias Aprobadas C1", min_value=0, max_value=7, value=4, step=1)
    mat_reprobadas_c1 = st.number_input("Materias Reprobadas C1", min_value=0, max_value=6, value=1, step=1)
    tasa_aprobacion_c1 = st.number_input("Tasa de Aprobacion C1", min_value=0.0, max_value=1.0, value=0.8, step=0.05, help="Materias aprobadas / inscritas en C1")
    promedio_c1 = st.number_input("Promedio C1", min_value=0.0, max_value=10.0, value=7.0, step=0.1)

    st.markdown("#### Rendimiento Ciclo 2")

    mat_inscritas_c2 = st.number_input("Materias Inscritas C2", min_value=1, max_value=7, value=5, step=1)
    mat_aprobadas_c2 = st.number_input("Materias Aprobadas C2", min_value=0, max_value=7, value=3, step=1)
    mat_reprobadas_c2 = st.number_input("Materias Reprobadas C2", min_value=0, max_value=6, value=2, step=1)
    tasa_aprobacion_c2 = st.number_input("Tasa de Aprobacion C2", min_value=0.0, max_value=1.0, value=0.6, step=0.05, help="Materias aprobadas / inscritas en C2")
    promedio_c2 = st.number_input("Promedio C2", min_value=0.0, max_value=10.0, value=6.0, step=0.1)

    st.markdown("#### Consolidado Anio 1")

    total_mat_inscritas_anio1 = st.number_input("Total Materias Inscritas Anio 1", min_value=1, max_value=15, value=10, step=1)
    total_mat_aprobadas_anio1 = st.number_input("Total Materias Aprobadas Anio 1", min_value=0, max_value=15, value=7, step=1)
    total_mat_reprobadas_anio1 = st.number_input("Total Materias Reprobadas Anio 1", min_value=0, max_value=15, value=3, step=1)
    tasa_aprobacion_anio1 = st.number_input("Tasa de Aprobacion Anio 1", min_value=0.0, max_value=1.0, value=0.7, step=0.05, help="Materias aprobadas / inscritas en el anio")
    promedio_general_anio1 = st.number_input("Promedio General Anio 1", min_value=0.0, max_value=10.0, value=6.5, step=0.1)

st.divider()

# Boton para realizar la prediccion
if st.button("Predecir Desercion", type="primary"):

    # Construir diccionario con los datos ingresados
    student_data = {
        'Carrera': carrera,
        'Plan': plan,
        'IdCampus': id_campus,
        'Sexo': sexo,
        'AnioIngreso': anio_ingreso,
        'CicloIngreso': ciclo_ingreso,
        'InstitucionBach': institucion_bach,
        'TieneBeca': tiene_beca,
        'PorcentajeBeca_Promedio': porcentaje_beca,
        'MateriasInscritas_C1': mat_inscritas_c1,
        'MateriasAprobadas_C1': mat_aprobadas_c1,
        'MateriasReprobadas_C1': mat_reprobadas_c1,
        'TasaAprobacion_C1': tasa_aprobacion_c1,
        'PromedioCiclo_C1': promedio_c1,
        'MateriasInscritas_C2': mat_inscritas_c2,
        'MateriasAprobadas_C2': mat_aprobadas_c2,
        'MateriasReprobadas_C2': mat_reprobadas_c2,
        'TasaAprobacion_C2': tasa_aprobacion_c2,
        'PromedioCiclo_C2': promedio_c2,
        'TotalMateriasInscritas_Anio1': total_mat_inscritas_anio1,
        'TotalMateriasAprobadas_Anio1': total_mat_aprobadas_anio1,
        'TotalMateriasReprobadas_Anio1': total_mat_reprobadas_anio1,
        'TasaAprobacion_Anio1': tasa_aprobacion_anio1,
        'PromedioGeneral_Anio1': promedio_general_anio1
    }

    try:
        # Convertir a DataFrame
        input_df = pd.DataFrame([student_data])

        # Asegurar que las columnas esten en el orden correcto
        for col in feature_columns:
            if col not in input_df.columns:
                input_df[col] = 0
        input_df = input_df[feature_columns]

        # Realizar prediccion
        prediction = model.predict(input_df)[0]

        # Obtener probabilidades
        probability = model.predict_proba(input_df)[0]

        # Mostrar resultado
        st.success("Prediccion completada correctamente")

        # Mostrar resultados en tres columnas
        res_col1, res_col2, res_col3 = st.columns(3)

        with res_col1:
            st.metric(
                label="Resultado",
                value="DESERTA" if prediction == 1 else "NO DESERTA"
            )

        with res_col2:
            st.metric(
                label="Prob. Desercion",
                value=f"{probability[1]:.1%}"
            )

        with res_col3:
            st.metric(
                label="Prob. Permanencia",
                value=f"{probability[0]:.1%}"
            )

        # Alerta visual segun el resultado
        if prediction == 1:
            st.error("⚠️ ALERTA: Este estudiante tiene alta probabilidad de desertar. Se recomienda intervencion temprana.")
        else:
            st.info("Este estudiante tiene baja probabilidad de desertar.")

        st.divider()

        # Mostrar tabla con los datos ingresados
        st.subheader("Datos Ingresados")
        st.dataframe(pd.DataFrame([student_data]), use_container_width=True)

    except Exception as e:
        st.error(f"Error al realizar la prediccion: {str(e)}")
        st.info("Verifique que los archivos best_model.pkl y feature_columns.pkl esten en el directorio raiz del proyecto.")

# Footer
st.divider()
st.caption(f"Modelo entrenado: {model_name} | Dataset: Desercion Estudiantil Primer Anio (2015-2019)")