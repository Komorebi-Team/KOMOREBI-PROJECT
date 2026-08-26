# 📈 Estimación de Precio Objetivo para Clientes Recientes
> **Zrive Applied Data Science Project | 4Q25**

![Zrive](https://img.shields.io/badge/Program-Zrive%20Applied%20Data%20Science-blueviolet)
![Industry](https://img.shields.io/badge/Industry-Automotive-blue)
![Status](https://img.shields.io/badge/Status-Finished-green)

## 🎯 Contexto del Proyecto


Este proyecto se desarrolló para una empresa digital del sector de la automoción que conecta la demanda online con profesionales y concesionarios.

El reto consiste en optimizar la estrategia de precios una vez finalizado el periodo inicial de captación y adaptación del cliente, que suele durar aproximadamente tres meses. El objetivo es determinar una tarifa estable adecuada: un precio demasiado elevado puede incrementar el riesgo de abandono, mientras que un precio demasiado bajo reduce el margen de beneficio.

Por motivos de confidencialidad, la identidad de la empresa y determinadas características de los datos han sido anonimizadas.

## 🛠️ Objetivos del Proyecto
* **Definir la Variable Objetivo:** Construir una métrica que represente la facturación real consolidada (Precio Estable) tras el periodo de onboarding.
* **Algoritmo Predictivo:** Crear un modelo capaz de predecir dicho precio utilizando únicamente la información de los primeros meses de actividad.
* **Análisis de Correlación:** Determinar qué características (leads, volumen de anuncios, ubicación, etc.) tienen mayor impacto en la disposición a pagar.
* **Soporte Comercial:** Proponer rangos de precios y categorías de clientes para facilitar las negociaciones de renovación.

## 📊 Descripción de los Datos
El análisis se basa en tres tablas anonimizadas con datos desde enero de 2023 hasta mayo de 2025:
1. **Tabla de Anunciantes:** Información general, ubicación geográfica y estado de contratos.
2. **Snapshot Mensual:** Métricas de desempeño del funnel (shows, visitas, leads), anuncios publicados y facturación.
3. **Histórico de Bajas:** Registro detallado de los motivos y tipos de bajas para identificar churn real.

## 📂 Estructura del Repositorio
* `notebooks/`: Análisis exploratorio de datos (EDA) y feature engineering.
* `src/`: Código modular para procesamiento y entrenamiento de modelos.
* `reports/`: Documentación técnica, metodología y recomendaciones finales.

## 👥 Equipo
* Susana Pousada
* Javier Terry
* Manuel Morello
* Pablo Ruiz
* Jon Ugalde

---
> [!CAUTION]
> **DISCLAIMER:** Los datos proporcionados son privados y escalados; NO pueden ser compartidos bajo ninguna circunstancia fuera del entorno académico de Zrive.

