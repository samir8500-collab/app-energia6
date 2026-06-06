# Dashboard Energía EPM - Versión V4

Esta versión está organizada para Streamlit Cloud y reemplaza los gráficos extensos por gráficos más simples y entendibles.

## Archivo principal

En Streamlit Cloud usa este archivo como **Main file path**:

```text
energia_epm_modificada_v4.py
```

## Archivos que debes subir a GitHub

Sube estos archivos sueltos al repositorio:

```text
energia_epm_modificada_v4.py
requirements.txt
README_energia_v4.md
```

Importante: no subas archivos duplicados como `streamlit_app (1).py`. Si tienes archivos viejos, bórralos o no los uses.

## Gráficos incluidos

1. Barras: tarifa promedio anual.
2. Puntos: tendencia mensual general.
3. Barras: variación mensual porcentual.
4. Torta: participación visual por grupo.
5. Barras: top tarifas del último mes.
6. Dispersión + KMeans.
7. Predicción de la serie seleccionada.
8. Correlación simple en barras.

## Cambios frente a versiones anteriores

- Se eliminó la matriz de correlación extensa.
- La correlación ahora muestra solo las 8 series más relacionadas con la serie seleccionada.
- El KMeans agrupa tarifas según tarifa actual, crecimiento, volatilidad y tendencia mensual.
- Cada gráfico tiene una observación automática.
- El nombre del archivo principal cambió para evitar que Streamlit siga ejecutando archivos viejos.

## Requisitos

El archivo `requirements.txt` debe estar en la raíz del repositorio y contener:

```text
streamlit
pandas
numpy
openpyxl
plotly
scikit-learn
xlsxwriter
```

## Cómo desplegar en Streamlit

En Streamlit Cloud configura:

```text
Repository: tu repositorio
Branch: main
Main file path: energia_epm_modificada_v4.py
```

Luego haz clic en **Deploy** o **Reboot app** si ya existe una app previa.
