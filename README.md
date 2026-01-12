# 📈 Sumauto: Estimación de Precio Objetivo para Clientes Recientes
> **Zrive Applied Data Science Project | 4Q25**

![Zrive](https://img.shields.io/badge/Program-Zrive%20Applied%20Data%20Science-blueviolet)
![Client](https://img.shields.io/badge/Client-Sumauto-red)
![Status](https://img.shields.io/badge/Status-In%20Progress-yellow)

## 🎯 El Desafío de Negocio
[cite_start]Sumauto es un grupo de portales de anuncios clasificados de vehículos cuya captación de nuevos clientes se basa en descuentos agresivos iniciales (onboarding de ~3 meses)[cite: 4, 6, 7]. [cite_start]El reto crítico surge al finalizar este periodo: **¿Cuál es la tarifa estable que maximiza el beneficio sin provocar el abandono (churn) del cliente?**[cite: 7, 8].

[cite_start]Este proyecto busca desarrollar un modelo predictivo que estime el **"Precio Estable"** (facturación mensual consolidada) basándose en el comportamiento del anunciante durante sus primeros meses de vida en la plataforma[cite: 9, 11].

## 🛠️ Objetivos del Proyecto
1.  [cite_start]**Definición de Target:** Construir la métrica de "Precio Estable" tras el periodo de promoción[cite: 11].
2.  [cite_start]**Modelado Predictivo:** Crear un algoritmo que prediga dicho precio usando solo datos iniciales de actividad[cite: 12].
3.  [cite_start]**Análisis de Sensibilidad:** Identificar qué variables (leads, volumen de anuncios, ubicación, etc.) correlacionan con una mayor disposición a pagar[cite: 13].
4.  [cite_start]**Soporte Comercial:** Proponer rangos de precios y categorías de clientes para facilitar las negociaciones de renovación[cite: 14].

## 📊 Datos Utilizados
Se analizan tres fuentes de datos anonimizadas (2023-2025):
* [cite_start]**Información del Anunciante:** Ubicación, antigüedad y estado de contratos[cite: 17, 18].
* [cite_start]**Métricas de Desempeño:** Funnel de conversión (shows > visits > leads), anuncios contratados vs. publicados y facturación histórica[cite: 27, 28, 52].
* [cite_start]**Histórico de Bajas:** Análisis de churn para diferenciar bajas definitivas de cambios de contrato[cite: 55, 63].

## 📁 Estructura del Repositorio
* [cite_start]`notebooks/`: EDA, limpieza y feature engineering de las tablas de anunciantes y snapshots[cite: 70].
* [cite_start]`models/`: Desarrollo y validación del algoritmo de predicción de precio objetivo[cite: 71, 72].
* `src/`: Scripts modulares para el procesamiento de datos y cálculo de métricas de facturación.
* [cite_start]`reports/`: Informe final con metodología, resultados y recomendaciones de negocio[cite: 74].

## 👥 Equipo
* Susana Pousada
* Javier Terry
* Manuel Morello
* Pablo Ruiz
* Jon Ugalde

---
[cite_start]⚠️ **DISCLAIMER:** Los datos utilizados en este proyecto son privados y confidenciales. Este repositorio cumple con las políticas de privacidad y no contiene datasets originales crudos.

