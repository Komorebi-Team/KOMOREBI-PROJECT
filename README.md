# 📈 Sumauto: Estimación de Precio Objetivo para Clientes Recientes
> **Zrive Applied Data Science Project | 4Q25**

![Zrive](https://img.shields.io/badge/Program-Zrive%20Applied%20Data%20Science-blueviolet)
![Client](https://img.shields.io/badge/Client-Sumauto-red)
![Status](https://img.shields.io/badge/Status-In%20Progress-yellow)

## 🎯 Contexto del Proyecto
Sumauto es un grupo de portales de anuncios clasificados de vehículos. Su modelo de negocio se basa en atraer tráfico y derivarlo a los anuncios de distintos profesionales y concesionarios. 

El reto principal de este proyecto es optimizar la estrategia de precios (pricing) tras el periodo inicial de captación (*onboarding*), que suele durar 3 meses. Es crítico determinar la **tarifa estable** de mercado: si el precio es muy alto, el cliente se marcha (*churn*); si es muy bajo, se pierde margen de beneficio.

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

