```python
import pandas as pd
import os
from pathlib import Path
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns   

plt.style.use("ggplot")
```


```python
os.getcwd()
```




    '/home/jterryc/KOMOREBI-PROJECT/notebooks'




```python
os.listdir(".")
```




    ['EDA.md',
     'EDA_jtc.md',
     'EDA_jtc_old.ipynb',
     'komorebi_project_eda.ipynb',
     'eda_nueva_susana.ipynb',
     'EDA.ipynb',
     'EDA_jtc.ipynb',
     'notebooks.txt',
     'EDA_jon.ipynb',
     'EDA_jtc_files']




```python
wd_local_path = '../data/zrive_advertiser_withdrawals.parquet'
dim_local_path = '../data/zrive_dim_advertiser.parquet'
fct_local_path = '../data/zrive_fct_monthly_snapshot_advertiser.parquet'
```


```python
df_wd = pd.read_parquet(wd_local_path, engine='fastparquet')
df_dim = pd.read_parquet(dim_local_path, engine='fastparquet')
df_fct = pd.read_parquet(fct_local_path, engine='fastparquet')
```

<h4>1. Información General de los datos (DF_DIM)<h4>

><h4> 1.1 Análisis del df</h4>


```python
print(f"Forma: {df_dim.shape}")
print(f"\nColumnas:")
print(df_dim.columns.tolist())
print(f"\nTipos de datos:")
print(df_dim.dtypes)
```

    Forma: (7076, 8)
    
    Columnas:
    ['advertiser_zrive_id', 'province_id', 'updated_at', 'advertiser_province', 'advertiser_group_id', 'min_start_contrato_date', 'max_start_contrato_nuevo_date', 'contrato_churn_date']
    
    Tipos de datos:
    advertiser_zrive_id                       int64
    province_id                               int64
    updated_at                       datetime64[us]
    advertiser_province                      object
    advertiser_group_id                     float64
    min_start_contrato_date          datetime64[ns]
    max_start_contrato_nuevo_date    datetime64[ns]
    contrato_churn_date              datetime64[ns]
    dtype: object



```python
df_dim.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>advertiser_zrive_id</th>
      <th>province_id</th>
      <th>updated_at</th>
      <th>advertiser_province</th>
      <th>advertiser_group_id</th>
      <th>min_start_contrato_date</th>
      <th>max_start_contrato_nuevo_date</th>
      <th>contrato_churn_date</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>986</td>
      <td>1</td>
      <td>2025-05-01 02:46:04</td>
      <td>Álava</td>
      <td>194.0</td>
      <td>2025-01-07</td>
      <td>2025-01-07</td>
      <td>2025-05-17</td>
    </tr>
    <tr>
      <th>1</th>
      <td>6811</td>
      <td>1</td>
      <td>2025-02-05 01:02:08</td>
      <td>Álava</td>
      <td>NaN</td>
      <td>2025-01-24</td>
      <td>NaT</td>
      <td>2025-02-04</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4890</td>
      <td>2</td>
      <td>2024-08-09 13:38:43</td>
      <td>Albacete</td>
      <td>133.0</td>
      <td>2023-02-24</td>
      <td>NaT</td>
      <td>2023-06-03</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3428</td>
      <td>3</td>
      <td>2023-11-02 13:51:07</td>
      <td>Alicante</td>
      <td>NaN</td>
      <td>2023-03-17</td>
      <td>2023-03-17</td>
      <td>2023-11-01</td>
    </tr>
    <tr>
      <th>4</th>
      <td>547</td>
      <td>3</td>
      <td>2024-02-05 13:49:02</td>
      <td>Alicante</td>
      <td>NaN</td>
      <td>2023-05-03</td>
      <td>NaT</td>
      <td>2023-12-29</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_dim["updated_at"].max()
```




    Timestamp('2025-06-19 18:36:05')




```python
df_dim["advertiser_zrive_id"].is_unique
```




    True



En **df_dim**, tenemos 1 fila = 1 advertiser

><h4> 1.2 Análisis por provincia</h4>


```python
#Sumauto aplica descuentos por provincias porque no todas generan el mismo tráfico
advertiser_province_unique = df_dim["advertiser_province"].unique()
print(f"Provincias unicas:  {advertiser_province_unique}")

advertiser_province_nunique = df_dim["advertiser_province"].nunique()
print(f"Num Provincias unicas: {advertiser_province_nunique}")
```

    Provincias unicas:  ['Álava' 'Albacete' 'Alicante' 'Asturias' 'Badajoz' 'Barcelona' 'Burgos'
     'Cáceres' 'Cantabria' 'Castellón' 'Córdoba' 'La Coruña' 'Girona'
     'Granada' 'Guadalajara' 'Huesca' 'Islas Baleares' 'Jaén' 'Lugo' 'Madrid'
     'Málaga' 'Murcia' 'Navarra' 'Pontevedra' 'La Rioja' 'Segovia' 'Sevilla'
     'Teruel' 'Toledo' 'Valencia' 'Valladolid' 'Vizcaya' 'Zaragoza' 'Almería'
     'Ávila' 'Cádiz' 'Ciudad Real' 'Cuenca' 'Guipúzcoa' 'Huelva' 'León'
     'Lleida' 'Orense' 'Palencia' 'Las Palmas' 'Salamanca' 'Soria' 'Tarragona'
     'Tenerife' 'Zamora' 'Ceuta' 'Melilla']
    Num Provincias unicas: 52



```python
#top 15 provincias
top_provinces = df_dim['advertiser_province'].value_counts(normalize=True).head(15)
print(top_provinces)

#Porcentaje acumulado (Top5)
total_clients = len(df_dim)
top5= top_provinces.head(5).sum()*100
print(f"Top 5 provincias concentran {top5:.2f}% de los cientes")
```

    advertiser_province
    Madrid       0.169305
    Barcelona    0.111504
    Valencia     0.062041
    Málaga       0.052855
    Alicante     0.051018
    Sevilla      0.048332
    Murcia       0.036320
    Asturias     0.024449
    Toledo       0.022894
    Vizcaya      0.022753
    La Coruña    0.021057
    Tarragona    0.020774
    Granada      0.019927
    Córdoba      0.018796
    Zaragoza     0.016959
    Name: proportion, dtype: float64
    Top 5 provincias concentran 44.67% de los cientes


- Concentración geográfica muy alta. Top 5 provincias concentran 44.67% de los cientes (la mayoría del negocio está en grandes ciudades)
- No tendría sentido comparar directamente un cliente de Madrid o Barcelona con uno de Soria. 

><h4> 1.3 Análisis por grupos (concesionarios)</h4>


```python
#Grupo de concesionarios (adv_group_id)
clients_in_group = df_dim['advertiser_group_id'].notna().sum()
pct_in_group = (clients_in_group / len(df_dim)) * 100

print(f"Clientes que pertenecen a un grupo: {clients_in_group} ({pct_in_group:.2f}%)")
print(f"Clientes independientes: {df_dim['advertiser_group_id'].isna().sum()}")

n_groups = df_dim['advertiser_group_id'].nunique()
print(f"Grupos únicos: {n_groups}")

# Tamaño promedio de los grupos
group_sizes = df_dim.groupby('advertiser_group_id').size()
print("Distribución del tamaño de grupos:")
print(group_sizes.describe())
print()
print("Top 10 grupos más grandes:")
print(group_sizes.sort_values(ascending=False).head(10))

advertiser_groups = (
    df_dim
    .groupby("advertiser_group_id")["advertiser_zrive_id"]
    .nunique()
    .sort_values(ascending=False)
    .head(10)
)

sns.barplot(
    x=advertiser_groups.values,
    y=advertiser_groups.index,
    order=advertiser_groups.index,
    orient = "h"
)
plt.title("Top 10 grupos más grandes")
plt.show()
```

    Clientes que pertenecen a un grupo: 1323 (18.70%)
    Clientes independientes: 5753
    Grupos únicos: 112
    Distribución del tamaño de grupos:
    count    112.000000
    mean      11.812500
    std       31.438924
    min        1.000000
    25%        2.000000
    50%        4.000000
    75%       11.000000
    max      257.000000
    dtype: float64
    
    Top 10 grupos más grandes:
    advertiser_group_id
    125.0    257
    55.0     195
    48.0      78
    117.0     69
    100.0     49
    41.0      29
    68.0      22
    58.0      22
    132.0     22
    154.0     22
    dtype: int64



    
![png](EDA_files/EDA_17_1.png)
    



```python
# 1. Investigar grupo 125
top_group = df_dim[df_dim['advertiser_group_id'] == 125.0]
print(f"Provincias únicas del grupo 125: {top_group['advertiser_province'].nunique()}")
print("\nTop 10 provincias del grupo 125:")
print(top_group['advertiser_province'].value_counts().head(10))
```

    Provincias únicas del grupo 125: 43
    
    Top 10 provincias del grupo 125:
    advertiser_province
    Madrid       40
    Barcelona    37
    Alicante     13
    Málaga       12
    Vizcaya      11
    Sevilla      10
    Valencia      9
    Granada       7
    Toledo        6
    Murcia        6
    Name: count, dtype: int64



```python
top10 = (
    top_group["advertiser_province"]
    .value_counts()
    .head(10)
)

sns.barplot(
    x=top10.values,
    y=top10.index,
    orient = "h"
)
plt.show()

```


    
![png](EDA_files/EDA_19_0.png)
    


- Solo el 18,70% de los clientes pertenece a un grupo, mientras que el resto son independientes.
- La mayoria de los grupos son pequeños (P50 = 4 clientes)
- El grupo más grande (125) es nacional, opera en 43 provincias de 52 provincias totales y además fuerte concentración en ciudades grandes (coincide que es donde más trafico de clientes hay)

<h4>2. Información General de los datos (DF_WITHDRAWAL)<h4>


```python
df_wd.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 22668 entries, 0 to 22667
    Data columns (total 7 columns):
     #   Column                     Non-Null Count  Dtype         
    ---  ------                     --------------  -----         
     0   withdrawal_id              22668 non-null  int64         
     1   advertiser_zrive_id        22668 non-null  int64         
     2   withdrawal_status          22668 non-null  object        
     3   withdrawal_type            22668 non-null  object        
     4   withdrawal_creation_date   22668 non-null  datetime64[us]
     5   withdrawal_effective_date  14348 non-null  datetime64[ns]
     6   withdrawal_reason          22668 non-null  object        
    dtypes: datetime64[ns](1), datetime64[us](1), int64(2), object(3)
    memory usage: 1.2+ MB



```python
df_wd.isna().sum()
```




    withdrawal_id                   0
    advertiser_zrive_id             0
    withdrawal_status               0
    withdrawal_type                 0
    withdrawal_creation_date        0
    withdrawal_effective_date    8320
    withdrawal_reason               0
    dtype: int64




```python
print(f"Forma: {df_wd.shape}")
print(f"\nColumnas:")
print(df_wd.columns.tolist())
print(f"\nTipos de datos:")
print(df_wd.dtypes)
```

    Forma: (22668, 7)
    
    Columnas:
    ['withdrawal_id', 'advertiser_zrive_id', 'withdrawal_status', 'withdrawal_type', 'withdrawal_creation_date', 'withdrawal_effective_date', 'withdrawal_reason']
    
    Tipos de datos:
    withdrawal_id                         int64
    advertiser_zrive_id                   int64
    withdrawal_status                    object
    withdrawal_type                      object
    withdrawal_creation_date     datetime64[us]
    withdrawal_effective_date    datetime64[ns]
    withdrawal_reason                    object
    dtype: object



```python
df_wd['withdrawal_effective_date'] = pd.to_datetime(df_wd['withdrawal_effective_date'])
df_wd.head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>withdrawal_id</th>
      <th>advertiser_zrive_id</th>
      <th>withdrawal_status</th>
      <th>withdrawal_type</th>
      <th>withdrawal_creation_date</th>
      <th>withdrawal_effective_date</th>
      <th>withdrawal_reason</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>259</td>
      <td>Cerrada</td>
      <td>TOTAL</td>
      <td>2012-06-19 07:12:34</td>
      <td>2012-07-01</td>
      <td>RESULTADOS</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>221</td>
      <td>Cerrada</td>
      <td>TOTAL</td>
      <td>2012-06-19 07:16:34</td>
      <td>NaT</td>
      <td>RESULTADOS</td>
    </tr>
    <tr>
      <th>2</th>
      <td>7</td>
      <td>492</td>
      <td>Cerrada</td>
      <td>TOTAL</td>
      <td>2012-06-20 07:10:16</td>
      <td>2012-07-01</td>
      <td>RESULTADOS</td>
    </tr>
    <tr>
      <th>3</th>
      <td>12</td>
      <td>481</td>
      <td>Cerrada</td>
      <td>TOTAL</td>
      <td>2012-06-20 11:59:36</td>
      <td>2012-07-01</td>
      <td>RESULTADOS</td>
    </tr>
    <tr>
      <th>4</th>
      <td>16</td>
      <td>457</td>
      <td>Cerrada</td>
      <td>TOTAL</td>
      <td>2012-06-20 15:41:39</td>
      <td>2012-07-01</td>
      <td>FALTA DE USO/TIEMPO</td>
    </tr>
    <tr>
      <th>5</th>
      <td>36</td>
      <td>95</td>
      <td>Cerrada</td>
      <td>TOTAL</td>
      <td>2012-06-22 10:40:37</td>
      <td>2012-07-01</td>
      <td>RAZONES ECONOMICAS</td>
    </tr>
    <tr>
      <th>6</th>
      <td>49</td>
      <td>620</td>
      <td>Cerrada</td>
      <td>TOTAL</td>
      <td>2012-06-26 15:53:04</td>
      <td>NaT</td>
      <td>RAZONES ECONOMICAS</td>
    </tr>
    <tr>
      <th>7</th>
      <td>53</td>
      <td>443</td>
      <td>Cerrada</td>
      <td>TOTAL</td>
      <td>2012-06-27 08:00:09</td>
      <td>NaT</td>
      <td>RESULTADOS</td>
    </tr>
    <tr>
      <th>8</th>
      <td>57</td>
      <td>675</td>
      <td>Cerrada</td>
      <td>TOTAL</td>
      <td>2012-06-27 10:55:17</td>
      <td>2012-07-01</td>
      <td>RESULTADOS</td>
    </tr>
    <tr>
      <th>9</th>
      <td>58</td>
      <td>195</td>
      <td>Cerrada</td>
      <td>TOTAL</td>
      <td>2012-06-27 11:19:15</td>
      <td>2012-07-01</td>
      <td>RESULTADOS</td>
    </tr>
  </tbody>
</table>
</div>



><h4>2.1 Análisis de Razones de Baja<h4>

Se consideran bajas definitivas aquellas donde: 
1. withdrawal_type = 'TOTAL' 
2. withdrawal_status != 'Denegada' 
3. withdrawal_reason not in ('Upselling-cambio de contrato', 'Cambio a Bundle Online', 
'Cambio de Contrato/propuesta/producto')


```python
withdrawal_reason_not = ['Upselling-cambio de contrato', 'Cambio a Bundle Online','Cambio de Contrato/propuesta/producto']
withdrawal_reason_yes = set(df_wd.withdrawal_reason.values).difference(set(withdrawal_reason_not))
withdrawal_reason_yes
```




    {'Baja Bundle AS24',
     'CESE DE ACTIVIDAD',
     'CORONAVIRUS',
     'Desconocido',
     'FALTA DE PRODUCTO',
     'FALTA DE USO/TIEMPO',
     'FIN DE CONTRATO',
     'Incidencias de la web',
     'MOROSIDAD',
     'No acepta subida',
     'OTROS',
     'RATIO RESULTADO-INVERSION',
     'RAZONES ECONOMICAS',
     'RESULTADOS',
     'Reestructuración cuentas de grupo'}




```python
# Creamos nueva columna binaria para definir las filas con withdrawal definitivo
df_wd["is_definitive_withdrawal"] = (
    df_wd.withdrawal_reason.isin(withdrawal_reason_yes) &
    (df_wd.withdrawal_type == "TOTAL") &
    (df_wd.withdrawal_status != "Denegada")
).astype(int)
df_wd
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>withdrawal_id</th>
      <th>advertiser_zrive_id</th>
      <th>withdrawal_status</th>
      <th>withdrawal_type</th>
      <th>withdrawal_creation_date</th>
      <th>withdrawal_effective_date</th>
      <th>withdrawal_reason</th>
      <th>is_definitive_withdrawal</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>259</td>
      <td>Cerrada</td>
      <td>TOTAL</td>
      <td>2012-06-19 07:12:34</td>
      <td>2012-07-01</td>
      <td>RESULTADOS</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>221</td>
      <td>Cerrada</td>
      <td>TOTAL</td>
      <td>2012-06-19 07:16:34</td>
      <td>NaT</td>
      <td>RESULTADOS</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>7</td>
      <td>492</td>
      <td>Cerrada</td>
      <td>TOTAL</td>
      <td>2012-06-20 07:10:16</td>
      <td>2012-07-01</td>
      <td>RESULTADOS</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>12</td>
      <td>481</td>
      <td>Cerrada</td>
      <td>TOTAL</td>
      <td>2012-06-20 11:59:36</td>
      <td>2012-07-01</td>
      <td>RESULTADOS</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>16</td>
      <td>457</td>
      <td>Cerrada</td>
      <td>TOTAL</td>
      <td>2012-06-20 15:41:39</td>
      <td>2012-07-01</td>
      <td>FALTA DE USO/TIEMPO</td>
      <td>1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>22663</th>
      <td>55613</td>
      <td>684</td>
      <td>Cerrada</td>
      <td>TOTAL</td>
      <td>2025-05-30 10:57:08</td>
      <td>2025-08-01</td>
      <td>RATIO RESULTADO-INVERSION</td>
      <td>1</td>
    </tr>
    <tr>
      <th>22664</th>
      <td>55614</td>
      <td>251</td>
      <td>Cerrada</td>
      <td>TOTAL</td>
      <td>2025-05-30 10:58:12</td>
      <td>2025-08-01</td>
      <td>RATIO RESULTADO-INVERSION</td>
      <td>1</td>
    </tr>
    <tr>
      <th>22665</th>
      <td>55615</td>
      <td>80</td>
      <td>Cerrada</td>
      <td>TOTAL</td>
      <td>2025-05-30 10:59:03</td>
      <td>2025-08-01</td>
      <td>RATIO RESULTADO-INVERSION</td>
      <td>1</td>
    </tr>
    <tr>
      <th>22666</th>
      <td>55616</td>
      <td>5149</td>
      <td>Cerrada</td>
      <td>TOTAL</td>
      <td>2025-05-30 11:15:01</td>
      <td>2025-07-01</td>
      <td>MOROSIDAD</td>
      <td>1</td>
    </tr>
    <tr>
      <th>22667</th>
      <td>55617</td>
      <td>1634</td>
      <td>Cerrada</td>
      <td>TOTAL</td>
      <td>2025-05-30 11:29:06</td>
      <td>NaT</td>
      <td>MOROSIDAD</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>22668 rows × 8 columns</p>
</div>




```python
df_wd["advertiser_zrive_id"].value_counts()
```




    advertiser_zrive_id
    122     58
    2915    47
    1378    44
    953     36
    56      35
            ..
    6927     1
    6294     1
    6409     1
    6825     1
    6922     1
    Name: count, Length: 6021, dtype: int64




```python
#Top 15 razones de baja (%)
reason_dist = (
    df_wd
    .loc[df_wd.is_definitive_withdrawal == 1, "withdrawal_reason"]
    .value_counts(normalize=True)
    .head(15)
) *100
print(round(reason_dist, 2))

```

    withdrawal_reason
    RESULTADOS                           36.14
    MOROSIDAD                            19.94
    RAZONES ECONOMICAS                   14.06
    RATIO RESULTADO-INVERSION            10.16
    FALTA DE USO/TIEMPO                   5.34
    OTROS                                 4.50
    FALTA DE PRODUCTO                     3.59
    CESE DE ACTIVIDAD                     2.86
    Reestructuración cuentas de grupo     1.15
    FIN DE CONTRATO                       0.90
    CORONAVIRUS                           0.63
    No acepta subida                      0.32
    Desconocido                           0.23
    Incidencias de la web                 0.13
    Baja Bundle AS24                      0.04
    Name: proportion, dtype: float64


>La principal causa de baja está relacionada con los resultados, junto con razones economicas y morosidad suma mas del 60%, lo que indica que el churn está muy relacionado con la retorno del servicio

<h4>3. Información General de los datos (DF_FCT)<h4>


```python
df_fct.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 96829 entries, 0 to 96828
    Data columns (total 23 columns):
     #   Column                         Non-Null Count  Dtype  
    ---  ------                         --------------  -----  
     0   advertiser_zrive_id            96829 non-null  int64  
     1   period_int                     96829 non-null  int64  
     2   monthly_contracted_ads         96829 non-null  int64  
     3   monthly_published_ads          96829 non-null  int64  
     4   monthly_unique_published_ads   96829 non-null  int64  
     5   monthly_distinct_ads           86777 non-null  float64
     6   monthly_oro_ads                96829 non-null  int64  
     7   monthly_plata_ads              96829 non-null  int64  
     8   monthly_destacados_ads         96829 non-null  int64  
     9   monthly_pepitas_ads            96829 non-null  int64  
     10  monthly_shows                  96829 non-null  float64
     11  monthly_visits                 96829 non-null  float64
     12  monthly_leads                  96829 non-null  int64  
     13  monthly_total_phone_views      96829 non-null  int64  
     14  monthly_total_calls            96829 non-null  int64  
     15  monthly_total_emails           96829 non-null  int64  
     16  monthly_total_invoice          96829 non-null  float64
     17  monthly_total_reference_price  84499 non-null  float64
     18  monthly_unique_calls           96829 non-null  int64  
     19  monthly_unique_emails          96829 non-null  int64  
     20  monthly_unique_leads           96829 non-null  int64  
     21  monthly_avg_ad_price           88362 non-null  float64
     22  has_active_contract            96829 non-null  bool   
    dtypes: bool(1), float64(6), int64(16)
    memory usage: 16.3 MB



```python
df_fct[["advertiser_zrive_id", "period_int"]].duplicated().any()
```




    np.False_



Tabla **df_fct**: 1 fila = 1 mes


```python
print(f"Forma: {df_fct.shape}")
print(f"\nColumnas:")
print(df_fct.columns.tolist())

```

    Forma: (96829, 23)
    
    Columnas:
    ['advertiser_zrive_id', 'period_int', 'monthly_contracted_ads', 'monthly_published_ads', 'monthly_unique_published_ads', 'monthly_distinct_ads', 'monthly_oro_ads', 'monthly_plata_ads', 'monthly_destacados_ads', 'monthly_pepitas_ads', 'monthly_shows', 'monthly_visits', 'monthly_leads', 'monthly_total_phone_views', 'monthly_total_calls', 'monthly_total_emails', 'monthly_total_invoice', 'monthly_total_reference_price', 'monthly_unique_calls', 'monthly_unique_emails', 'monthly_unique_leads', 'monthly_avg_ad_price', 'has_active_contract']



```python
(df_fct < 0).any()
```




    advertiser_zrive_id              False
    period_int                       False
    monthly_contracted_ads           False
    monthly_published_ads            False
    monthly_unique_published_ads     False
    monthly_distinct_ads             False
    monthly_oro_ads                  False
    monthly_plata_ads                False
    monthly_destacados_ads           False
    monthly_pepitas_ads              False
    monthly_shows                    False
    monthly_visits                   False
    monthly_leads                    False
    monthly_total_phone_views        False
    monthly_total_calls              False
    monthly_total_emails             False
    monthly_total_invoice             True
    monthly_total_reference_price    False
    monthly_unique_calls             False
    monthly_unique_emails            False
    monthly_unique_leads             False
    monthly_avg_ad_price             False
    has_active_contract              False
    dtype: bool




```python
df_fct = pd.read_parquet(fct_local_path, engine='fastparquet')
df_fct[df_fct["monthly_total_invoice"] < 0]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>advertiser_zrive_id</th>
      <th>period_int</th>
      <th>monthly_contracted_ads</th>
      <th>monthly_published_ads</th>
      <th>monthly_unique_published_ads</th>
      <th>monthly_distinct_ads</th>
      <th>monthly_oro_ads</th>
      <th>monthly_plata_ads</th>
      <th>monthly_destacados_ads</th>
      <th>monthly_pepitas_ads</th>
      <th>...</th>
      <th>monthly_total_phone_views</th>
      <th>monthly_total_calls</th>
      <th>monthly_total_emails</th>
      <th>monthly_total_invoice</th>
      <th>monthly_total_reference_price</th>
      <th>monthly_unique_calls</th>
      <th>monthly_unique_emails</th>
      <th>monthly_unique_leads</th>
      <th>monthly_avg_ad_price</th>
      <th>has_active_contract</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>49074</th>
      <td>5242</td>
      <td>202403</td>
      <td>20</td>
      <td>19</td>
      <td>19</td>
      <td>17.0</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>-130.6</td>
      <td>693.3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>22405.88</td>
      <td>True</td>
    </tr>
    <tr>
      <th>84780</th>
      <td>2260</td>
      <td>202502</td>
      <td>1000</td>
      <td>996</td>
      <td>996</td>
      <td>1049.0</td>
      <td>6</td>
      <td>6</td>
      <td>6</td>
      <td>0</td>
      <td>...</td>
      <td>57</td>
      <td>18</td>
      <td>16</td>
      <td>-1833.3</td>
      <td>NaN</td>
      <td>15</td>
      <td>51</td>
      <td>66</td>
      <td>309080.72</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
<p>2 rows × 23 columns</p>
</div>



- Aparecen dos registros negativos en monthly_total_invoice. Al ser casos muy concretos y no representar el comportamiento normal de facturación, tiene sentido excluirlos del análisis para no distorsionar la distribución.


```python
df_fct = df_fct[df_fct["monthly_total_invoice"] >= 0].copy()
```

- Para el análisis de pricing solo se tienen en cuenta los meses con contrato activo (has_active_contract = True).
Los meses sin contrato tienen facturación 0 y precios de referencia nulos, por lo que no aportan información útil sobre el valor real del cliente y distorsionan las métricas.


```python
# Miramos si hay usuarios con has_active_contract = 0 en el periodo de estudio (2023 - 2025)
(
    df_fct
    .groupby("advertiser_zrive_id")["has_active_contract"]
    .sum()
    .loc[lambda x: x==0]
)
```




    advertiser_zrive_id
    7036    0
    7066    0
    7069    0
    Name: has_active_contract, dtype: int64



Los usuarios 7036, 7066 y 7069 no tuvieron ningún contrato activo en el periodo de estudio, por lo que los excluímos del análisis


```python
plt.figure()
plt.boxplot(df_fct["monthly_total_invoice"], vert = False, showfliers= False)
plt.xlabel("Monthly total invoice")
plt.ylabel("Distribution of monthly total invoice")
plt.show()
```


    
![png](EDA_files/EDA_45_0.png)
    


><h4> Permanencia Clientes<h4>

Analizamos los datos temporales en **df_fct**, que es la información sobre la actividad de los df_dim en el periodo de estudio (2023-2025)


```python
# Convertimos la columna "period_int" a YYYY-MM-DD
try:
    df_fct["period_int"] = pd.to_datetime(df_fct["period_int"].astype(str), format="%Y%m")
except ValueError:
    pass
df_fct.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>advertiser_zrive_id</th>
      <th>period_int</th>
      <th>monthly_contracted_ads</th>
      <th>monthly_published_ads</th>
      <th>monthly_unique_published_ads</th>
      <th>monthly_distinct_ads</th>
      <th>monthly_oro_ads</th>
      <th>monthly_plata_ads</th>
      <th>monthly_destacados_ads</th>
      <th>monthly_pepitas_ads</th>
      <th>...</th>
      <th>monthly_total_phone_views</th>
      <th>monthly_total_calls</th>
      <th>monthly_total_emails</th>
      <th>monthly_total_invoice</th>
      <th>monthly_total_reference_price</th>
      <th>monthly_unique_calls</th>
      <th>monthly_unique_emails</th>
      <th>monthly_unique_leads</th>
      <th>monthly_avg_ad_price</th>
      <th>has_active_contract</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>2023-01-01</td>
      <td>75</td>
      <td>47</td>
      <td>47</td>
      <td>56.0</td>
      <td>6</td>
      <td>6</td>
      <td>6</td>
      <td>0</td>
      <td>...</td>
      <td>14</td>
      <td>15</td>
      <td>0</td>
      <td>1714.2</td>
      <td>2416.7</td>
      <td>12</td>
      <td>3</td>
      <td>15</td>
      <td>42667.86</td>
      <td>True</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>2023-01-01</td>
      <td>150</td>
      <td>31</td>
      <td>31</td>
      <td>55.0</td>
      <td>10</td>
      <td>10</td>
      <td>4</td>
      <td>0</td>
      <td>...</td>
      <td>16</td>
      <td>2</td>
      <td>2</td>
      <td>293.3</td>
      <td>3643.3</td>
      <td>2</td>
      <td>2</td>
      <td>4</td>
      <td>22742.40</td>
      <td>True</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>2023-01-01</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>25420.83</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>2023-01-01</td>
      <td>85</td>
      <td>79</td>
      <td>79</td>
      <td>86.0</td>
      <td>3</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>10</td>
      <td>8</td>
      <td>2</td>
      <td>1165.0</td>
      <td>3023.3</td>
      <td>6</td>
      <td>6</td>
      <td>12</td>
      <td>23259.52</td>
      <td>True</td>
    </tr>
    <tr>
      <th>4</th>
      <td>6</td>
      <td>2023-01-01</td>
      <td>20</td>
      <td>20</td>
      <td>20</td>
      <td>25.0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>10</td>
      <td>4</td>
      <td>3</td>
      <td>336.3</td>
      <td>515.0</td>
      <td>4</td>
      <td>11</td>
      <td>15</td>
      <td>43439.00</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 23 columns</p>
</div>




```python
df_fct[df_fct["advertiser_zrive_id"]==16]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>advertiser_zrive_id</th>
      <th>period_int</th>
      <th>monthly_contracted_ads</th>
      <th>monthly_published_ads</th>
      <th>monthly_unique_published_ads</th>
      <th>monthly_distinct_ads</th>
      <th>monthly_oro_ads</th>
      <th>monthly_plata_ads</th>
      <th>monthly_destacados_ads</th>
      <th>monthly_pepitas_ads</th>
      <th>...</th>
      <th>monthly_total_phone_views</th>
      <th>monthly_total_calls</th>
      <th>monthly_total_emails</th>
      <th>monthly_total_invoice</th>
      <th>monthly_total_reference_price</th>
      <th>monthly_unique_calls</th>
      <th>monthly_unique_emails</th>
      <th>monthly_unique_leads</th>
      <th>monthly_avg_ad_price</th>
      <th>has_active_contract</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>10</th>
      <td>16</td>
      <td>2023-01-01</td>
      <td>75</td>
      <td>20</td>
      <td>20</td>
      <td>78.0</td>
      <td>6</td>
      <td>6</td>
      <td>6</td>
      <td>0</td>
      <td>...</td>
      <td>29</td>
      <td>18</td>
      <td>6</td>
      <td>1329.1</td>
      <td>3180.0</td>
      <td>16</td>
      <td>10</td>
      <td>26</td>
      <td>22120.16</td>
      <td>True</td>
    </tr>
    <tr>
      <th>3551</th>
      <td>16</td>
      <td>2023-02-01</td>
      <td>75</td>
      <td>30</td>
      <td>30</td>
      <td>94.0</td>
      <td>6</td>
      <td>6</td>
      <td>6</td>
      <td>0</td>
      <td>...</td>
      <td>30</td>
      <td>24</td>
      <td>9</td>
      <td>730.0</td>
      <td>2210.0</td>
      <td>21</td>
      <td>22</td>
      <td>43</td>
      <td>22509.08</td>
      <td>True</td>
    </tr>
    <tr>
      <th>6914</th>
      <td>16</td>
      <td>2023-03-01</td>
      <td>75</td>
      <td>59</td>
      <td>59</td>
      <td>152.0</td>
      <td>6</td>
      <td>6</td>
      <td>6</td>
      <td>0</td>
      <td>...</td>
      <td>23</td>
      <td>19</td>
      <td>6</td>
      <td>730.0</td>
      <td>2210.0</td>
      <td>16</td>
      <td>23</td>
      <td>39</td>
      <td>23040.26</td>
      <td>True</td>
    </tr>
    <tr>
      <th>10337</th>
      <td>16</td>
      <td>2023-04-01</td>
      <td>75</td>
      <td>37</td>
      <td>37</td>
      <td>110.0</td>
      <td>6</td>
      <td>6</td>
      <td>6</td>
      <td>0</td>
      <td>...</td>
      <td>29</td>
      <td>45</td>
      <td>9</td>
      <td>730.0</td>
      <td>2210.0</td>
      <td>21</td>
      <td>24</td>
      <td>45</td>
      <td>22711.89</td>
      <td>True</td>
    </tr>
    <tr>
      <th>13757</th>
      <td>16</td>
      <td>2023-05-01</td>
      <td>75</td>
      <td>58</td>
      <td>58</td>
      <td>134.0</td>
      <td>6</td>
      <td>6</td>
      <td>6</td>
      <td>0</td>
      <td>...</td>
      <td>22</td>
      <td>14</td>
      <td>5</td>
      <td>730.0</td>
      <td>2210.0</td>
      <td>11</td>
      <td>13</td>
      <td>24</td>
      <td>23576.57</td>
      <td>True</td>
    </tr>
    <tr>
      <th>17146</th>
      <td>16</td>
      <td>2023-06-01</td>
      <td>75</td>
      <td>81</td>
      <td>81</td>
      <td>170.0</td>
      <td>6</td>
      <td>6</td>
      <td>6</td>
      <td>0</td>
      <td>...</td>
      <td>19</td>
      <td>10</td>
      <td>6</td>
      <td>730.0</td>
      <td>2210.0</td>
      <td>9</td>
      <td>13</td>
      <td>22</td>
      <td>23734.82</td>
      <td>True</td>
    </tr>
    <tr>
      <th>20507</th>
      <td>16</td>
      <td>2023-07-01</td>
      <td>75</td>
      <td>72</td>
      <td>72</td>
      <td>182.0</td>
      <td>6</td>
      <td>6</td>
      <td>6</td>
      <td>0</td>
      <td>...</td>
      <td>18</td>
      <td>10</td>
      <td>8</td>
      <td>730.0</td>
      <td>2210.0</td>
      <td>9</td>
      <td>15</td>
      <td>24</td>
      <td>23482.54</td>
      <td>True</td>
    </tr>
    <tr>
      <th>23946</th>
      <td>16</td>
      <td>2023-08-01</td>
      <td>75</td>
      <td>65</td>
      <td>65</td>
      <td>139.0</td>
      <td>6</td>
      <td>6</td>
      <td>6</td>
      <td>0</td>
      <td>...</td>
      <td>35</td>
      <td>27</td>
      <td>13</td>
      <td>730.0</td>
      <td>2210.0</td>
      <td>21</td>
      <td>20</td>
      <td>41</td>
      <td>25109.20</td>
      <td>True</td>
    </tr>
    <tr>
      <th>27169</th>
      <td>16</td>
      <td>2023-09-01</td>
      <td>75</td>
      <td>50</td>
      <td>50</td>
      <td>111.0</td>
      <td>6</td>
      <td>6</td>
      <td>6</td>
      <td>0</td>
      <td>...</td>
      <td>8</td>
      <td>11</td>
      <td>6</td>
      <td>730.0</td>
      <td>2210.0</td>
      <td>10</td>
      <td>8</td>
      <td>18</td>
      <td>24902.29</td>
      <td>True</td>
    </tr>
    <tr>
      <th>30369</th>
      <td>16</td>
      <td>2023-10-01</td>
      <td>75</td>
      <td>62</td>
      <td>62</td>
      <td>113.0</td>
      <td>6</td>
      <td>6</td>
      <td>6</td>
      <td>0</td>
      <td>...</td>
      <td>22</td>
      <td>14</td>
      <td>3</td>
      <td>730.0</td>
      <td>2210.0</td>
      <td>11</td>
      <td>7</td>
      <td>18</td>
      <td>25592.64</td>
      <td>True</td>
    </tr>
    <tr>
      <th>33513</th>
      <td>16</td>
      <td>2023-11-01</td>
      <td>75</td>
      <td>66</td>
      <td>66</td>
      <td>123.0</td>
      <td>6</td>
      <td>6</td>
      <td>6</td>
      <td>0</td>
      <td>...</td>
      <td>51</td>
      <td>41</td>
      <td>19</td>
      <td>730.0</td>
      <td>2210.0</td>
      <td>28</td>
      <td>33</td>
      <td>61</td>
      <td>23384.88</td>
      <td>True</td>
    </tr>
    <tr>
      <th>36730</th>
      <td>16</td>
      <td>2023-12-01</td>
      <td>75</td>
      <td>29</td>
      <td>29</td>
      <td>130.0</td>
      <td>6</td>
      <td>6</td>
      <td>6</td>
      <td>0</td>
      <td>...</td>
      <td>22</td>
      <td>13</td>
      <td>23</td>
      <td>730.0</td>
      <td>2210.0</td>
      <td>11</td>
      <td>22</td>
      <td>33</td>
      <td>24592.14</td>
      <td>True</td>
    </tr>
    <tr>
      <th>39936</th>
      <td>16</td>
      <td>2024-01-01</td>
      <td>75</td>
      <td>39</td>
      <td>39</td>
      <td>81.0</td>
      <td>6</td>
      <td>6</td>
      <td>6</td>
      <td>0</td>
      <td>...</td>
      <td>11</td>
      <td>28</td>
      <td>8</td>
      <td>730.0</td>
      <td>2210.0</td>
      <td>14</td>
      <td>15</td>
      <td>29</td>
      <td>24449.31</td>
      <td>True</td>
    </tr>
    <tr>
      <th>43128</th>
      <td>16</td>
      <td>2024-02-01</td>
      <td>75</td>
      <td>50</td>
      <td>50</td>
      <td>108.0</td>
      <td>6</td>
      <td>6</td>
      <td>6</td>
      <td>0</td>
      <td>...</td>
      <td>27</td>
      <td>21</td>
      <td>8</td>
      <td>916.7</td>
      <td>2210.0</td>
      <td>13</td>
      <td>13</td>
      <td>26</td>
      <td>24401.25</td>
      <td>True</td>
    </tr>
    <tr>
      <th>46368</th>
      <td>16</td>
      <td>2024-03-01</td>
      <td>75</td>
      <td>75</td>
      <td>75</td>
      <td>129.0</td>
      <td>6</td>
      <td>6</td>
      <td>6</td>
      <td>0</td>
      <td>...</td>
      <td>12</td>
      <td>12</td>
      <td>7</td>
      <td>916.7</td>
      <td>2210.0</td>
      <td>9</td>
      <td>21</td>
      <td>30</td>
      <td>23140.10</td>
      <td>True</td>
    </tr>
    <tr>
      <th>49668</th>
      <td>16</td>
      <td>2024-04-01</td>
      <td>75</td>
      <td>62</td>
      <td>62</td>
      <td>134.0</td>
      <td>6</td>
      <td>6</td>
      <td>6</td>
      <td>0</td>
      <td>...</td>
      <td>12</td>
      <td>25</td>
      <td>11</td>
      <td>916.7</td>
      <td>2210.0</td>
      <td>14</td>
      <td>22</td>
      <td>36</td>
      <td>22845.80</td>
      <td>True</td>
    </tr>
    <tr>
      <th>53025</th>
      <td>16</td>
      <td>2024-05-01</td>
      <td>75</td>
      <td>67</td>
      <td>67</td>
      <td>126.0</td>
      <td>6</td>
      <td>6</td>
      <td>6</td>
      <td>0</td>
      <td>...</td>
      <td>24</td>
      <td>24</td>
      <td>13</td>
      <td>1300.0</td>
      <td>2210.0</td>
      <td>18</td>
      <td>26</td>
      <td>44</td>
      <td>24122.71</td>
      <td>True</td>
    </tr>
    <tr>
      <th>56465</th>
      <td>16</td>
      <td>2024-06-01</td>
      <td>75</td>
      <td>56</td>
      <td>56</td>
      <td>122.0</td>
      <td>6</td>
      <td>6</td>
      <td>6</td>
      <td>0</td>
      <td>...</td>
      <td>21</td>
      <td>27</td>
      <td>14</td>
      <td>1300.0</td>
      <td>2210.0</td>
      <td>18</td>
      <td>24</td>
      <td>42</td>
      <td>23996.58</td>
      <td>True</td>
    </tr>
    <tr>
      <th>59995</th>
      <td>16</td>
      <td>2024-07-01</td>
      <td>75</td>
      <td>67</td>
      <td>67</td>
      <td>135.0</td>
      <td>6</td>
      <td>6</td>
      <td>6</td>
      <td>0</td>
      <td>...</td>
      <td>25</td>
      <td>29</td>
      <td>16</td>
      <td>1300.0</td>
      <td>2210.0</td>
      <td>19</td>
      <td>38</td>
      <td>57</td>
      <td>22224.49</td>
      <td>True</td>
    </tr>
    <tr>
      <th>63382</th>
      <td>16</td>
      <td>2024-08-01</td>
      <td>75</td>
      <td>59</td>
      <td>59</td>
      <td>59.0</td>
      <td>6</td>
      <td>6</td>
      <td>6</td>
      <td>0</td>
      <td>...</td>
      <td>21</td>
      <td>22</td>
      <td>10</td>
      <td>2008.3</td>
      <td>2210.0</td>
      <td>15</td>
      <td>16</td>
      <td>31</td>
      <td>22658.31</td>
      <td>True</td>
    </tr>
    <tr>
      <th>66705</th>
      <td>16</td>
      <td>2024-09-01</td>
      <td>75</td>
      <td>53</td>
      <td>53</td>
      <td>86.0</td>
      <td>6</td>
      <td>6</td>
      <td>6</td>
      <td>0</td>
      <td>...</td>
      <td>22</td>
      <td>15</td>
      <td>9</td>
      <td>2008.3</td>
      <td>2210.0</td>
      <td>12</td>
      <td>14</td>
      <td>26</td>
      <td>22859.88</td>
      <td>True</td>
    </tr>
    <tr>
      <th>70051</th>
      <td>16</td>
      <td>2024-10-01</td>
      <td>75</td>
      <td>75</td>
      <td>75</td>
      <td>155.0</td>
      <td>6</td>
      <td>6</td>
      <td>6</td>
      <td>0</td>
      <td>...</td>
      <td>23</td>
      <td>19</td>
      <td>9</td>
      <td>2008.3</td>
      <td>2210.0</td>
      <td>9</td>
      <td>13</td>
      <td>22</td>
      <td>24597.54</td>
      <td>True</td>
    </tr>
    <tr>
      <th>73360</th>
      <td>16</td>
      <td>2024-11-01</td>
      <td>75</td>
      <td>61</td>
      <td>61</td>
      <td>118.0</td>
      <td>6</td>
      <td>6</td>
      <td>6</td>
      <td>0</td>
      <td>...</td>
      <td>27</td>
      <td>22</td>
      <td>9</td>
      <td>2008.3</td>
      <td>2210.0</td>
      <td>14</td>
      <td>14</td>
      <td>28</td>
      <td>25853.52</td>
      <td>True</td>
    </tr>
    <tr>
      <th>76688</th>
      <td>16</td>
      <td>2024-12-01</td>
      <td>75</td>
      <td>65</td>
      <td>65</td>
      <td>74.0</td>
      <td>6</td>
      <td>6</td>
      <td>6</td>
      <td>0</td>
      <td>...</td>
      <td>27</td>
      <td>17</td>
      <td>6</td>
      <td>2008.3</td>
      <td>2210.0</td>
      <td>12</td>
      <td>10</td>
      <td>22</td>
      <td>27278.64</td>
      <td>True</td>
    </tr>
    <tr>
      <th>80137</th>
      <td>16</td>
      <td>2025-01-01</td>
      <td>75</td>
      <td>59</td>
      <td>59</td>
      <td>129.0</td>
      <td>6</td>
      <td>6</td>
      <td>6</td>
      <td>0</td>
      <td>...</td>
      <td>33</td>
      <td>40</td>
      <td>4</td>
      <td>2210.0</td>
      <td>2210.0</td>
      <td>20</td>
      <td>11</td>
      <td>31</td>
      <td>28572.69</td>
      <td>True</td>
    </tr>
    <tr>
      <th>83537</th>
      <td>16</td>
      <td>2025-02-01</td>
      <td>75</td>
      <td>74</td>
      <td>74</td>
      <td>127.0</td>
      <td>6</td>
      <td>6</td>
      <td>6</td>
      <td>0</td>
      <td>...</td>
      <td>12</td>
      <td>16</td>
      <td>4</td>
      <td>2210.0</td>
      <td>2210.0</td>
      <td>13</td>
      <td>11</td>
      <td>24</td>
      <td>27475.60</td>
      <td>True</td>
    </tr>
    <tr>
      <th>86950</th>
      <td>16</td>
      <td>2025-03-01</td>
      <td>75</td>
      <td>75</td>
      <td>75</td>
      <td>85.0</td>
      <td>6</td>
      <td>6</td>
      <td>6</td>
      <td>0</td>
      <td>...</td>
      <td>16</td>
      <td>28</td>
      <td>3</td>
      <td>2210.0</td>
      <td>2210.0</td>
      <td>21</td>
      <td>8</td>
      <td>29</td>
      <td>28667.28</td>
      <td>True</td>
    </tr>
    <tr>
      <th>90293</th>
      <td>16</td>
      <td>2025-04-01</td>
      <td>75</td>
      <td>75</td>
      <td>75</td>
      <td>73.0</td>
      <td>6</td>
      <td>6</td>
      <td>6</td>
      <td>0</td>
      <td>...</td>
      <td>10</td>
      <td>17</td>
      <td>1</td>
      <td>2210.0</td>
      <td>2210.0</td>
      <td>10</td>
      <td>4</td>
      <td>14</td>
      <td>27241.08</td>
      <td>True</td>
    </tr>
    <tr>
      <th>93580</th>
      <td>16</td>
      <td>2025-05-01</td>
      <td>75</td>
      <td>75</td>
      <td>75</td>
      <td>73.0</td>
      <td>6</td>
      <td>6</td>
      <td>6</td>
      <td>0</td>
      <td>...</td>
      <td>15</td>
      <td>11</td>
      <td>2</td>
      <td>2210.0</td>
      <td>2210.0</td>
      <td>5</td>
      <td>7</td>
      <td>12</td>
      <td>27241.08</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
<p>29 rows × 23 columns</p>
</div>




```python
df_months = (
    df_fct
    .groupby("advertiser_zrive_id")["period_int"]
    .nunique()
    .reset_index()
    .rename(columns={"period_int":"n_months"})
)

print(df_months["n_months"].describe())
```

    count    6968.000000
    mean       13.895953
    std         9.970284
    min         1.000000
    25%         5.000000
    50%        11.000000
    75%        25.000000
    max        29.000000
    Name: n_months, dtype: float64



```python
sns.histplot(df_months, x="n_months", bins=30)

plt.title("Nº de meses por anunciante")
plt.xlabel("Nº meses")
plt.ylabel("Anunciantes")
```




    Text(0, 0.5, 'Anunciantes')




    
![png](EDA_files/EDA_50_1.png)
    


Distribución bimodal; podemos comprobar que los df_fct se concentran en dos grupos principales:
- Corto-medio plazo (5 meses aprox.)
- Largo plazo (29 meses aprox.)


```python
active_users_month = (
    df_fct
    .groupby("period_int")["advertiser_zrive_id"]
    .nunique()
    .reset_index()
    .rename(columns={"advertiser_zrive_id":"n_active_users"})
)

active_users_month.plot(
    x="period_int",
    y="n_active_users",
    kind="line",
    figsize=(15,4)
)
active_users_month["n_active_users"].describe()
```




    count      29.000000
    mean     3338.862069
    std       100.918186
    min      3144.000000
    25%      3258.000000
    50%      3346.000000
    75%      3413.000000
    max      3542.000000
    Name: n_active_users, dtype: float64




    
![png](EDA_files/EDA_52_1.png)
    



```python
cv = active_users_month["n_active_users"].std()/active_users_month["n_active_users"].mean()
print(f"Coeficiente de variación: {cv:.2f}")
```

    Coeficiente de variación: 0.03


- La media de advertisers activos por mes es 3339, con CV=3%, lo que representa poca variación en el periodo de estudio


```python
df_fct["price_bucket"] = pd.cut(
    df_fct["monthly_total_invoice"],
    bins=[0, 100, 250, 500, 1000, 5000, 40000],
    labels=["<100", "100-250", "250-500", "500-1k", "1k-5k", "5k+"]
)

df_fct["price_bucket"].value_counts(normalize=True).sort_index()
```




    price_bucket
    <100       0.231605
    100-250    0.241092
    250-500    0.233906
    500-1k     0.168397
    1k-5k      0.112941
    5k+        0.012059
    Name: proportion, dtype: float64




```python
df_fct["price_diff_pct"] = (
    df_fct["monthly_total_invoice"]
    / df_fct["monthly_total_reference_price"]
) - 1
df_fct["price_diff_pct"].describe()

```




    count    84498.000000
    mean        -0.483692
    std          0.677025
    min         -1.000000
    25%         -0.702794
    50%         -0.500000
    75%         -0.229515
    max        125.640769
    Name: price_diff_pct, dtype: float64



- La mayor parte de los clientes se concentra en un rango de precio mensual bajo: cerca del 70% paga menos de 500€, con un peso muy relevante entre 100–250€ y 250–500€. 
- Al comprarar el precio de tarifa frente a la facturación mensual, los clientes pagan significativamente menos del precio de tarifa (-50%), esto significa que el precio tarifa está mal planteado


```python
df_fct["monthly_contracted_ads"].quantile(0.95)
```




    np.float64(300.0)




```python
df_fct["monthly_contracted_ads"].describe(percentiles=[0.95])
```




    count    96827.000000
    mean       108.101191
    std        490.209838
    min          0.000000
    50%         20.000000
    95%        300.000000
    max       8200.000000
    Name: monthly_contracted_ads, dtype: float64




```python
p95 = df_fct["monthly_contracted_ads"].quantile(0.95)

sns.histplot(
    df_fct[df_fct['monthly_contracted_ads'] <= p95]['monthly_contracted_ads'],
    bins=30
)

plt.axvline(p95, color='red', linestyle='--', label='P95')
plt.legend()
plt.title("Distribución de anuncios contratados (hasta P95)")
plt.show()
```


    
![png](EDA_files/EDA_60_0.png)
    


- Distribución del tamaño de los clientes según nº de meses contratados. Para visualizarlo mejor, se muestra el percentil 95 ya que la distribución tiene cola larga hacia la derecha: muchos anunciantes con pocos anuncios y pocos anunciantes con muchísimos anuncios
- La mitad tiene contratados 20 anuncios o menos


```python
df_fct.groupby("period_int")["advertiser_zrive_id"].nunique().mean()
```




    np.float64(3338.862068965517)




```python
# Clientes únicos
print(f"Clientes únicos: {df_fct['advertiser_zrive_id'].nunique():}")
#media de clientes por meses
clientes_por_mes = (
    df_fct
        .groupby("period_int")["advertiser_zrive_id"]
        .nunique()
)
print(f"Análisis de clientes por mes")
clientes_por_mes.describe()

```

    Clientes únicos: 6968
    Análisis de clientes por mes





    count      29.000000
    mean     3338.862069
    std       100.918186
    min      3144.000000
    25%      3258.000000
    50%      3346.000000
    75%      3413.000000
    max      3542.000000
    Name: advertiser_zrive_id, dtype: float64



- La mayoría de clientes son pequeños o medianos: lo normal es contratar unos 20 anuncios y publicar alrededor de 12.  
- Los datos están muy descompensados, con unos pocos clientes muy grandes que suben la media, por lo que los percentiles representan mejor la realidad que la media.  
- Aunque hay casi 7.000 clientes distintos en total, de media hay unos 3.340 clientes únicos activos, lo que muestra bastante rotación de clientes en el tiempo.  
- El recorrido de los anuncios es claro (se muestran, se visitan y luego generan leads), pero los leads no son muchos, así que el valor del servicio está más en el rendimiento que en el volumen.

><h4>Creación columnas nuevas<h4>


```python
# Lead Conversion Rate (% de visitas se convierten en leads)
df_fct["conversion_rate"] = np.where(
    df_fct["monthly_visits"] > 0,
    df_fct["monthly_leads"] / df_fct["monthly_visits"],
    np.nan
)

# Cost Per Lead (cuanto paga el cliente por cada lead generado)
df_fct["cost_per_lead"] = np.where(
    df_fct["monthly_leads"] > 0,
    df_fct["monthly_total_invoice"] / df_fct["monthly_leads"],
    np.nan
)

# Ratio de Uso (anuncios publicados vs contratados)
df_fct["usage_ratio"] = (df_fct["monthly_published_ads"]/df_fct["monthly_contracted_ads"].replace(0, np.nan))
```


```python
df_fct["usage_ratio"].describe()
```




    count    96484.000000
    mean         0.720402
    std          0.361695
    min          0.000000
    25%          0.500000
    50%          0.800000
    75%          1.000000
    max          3.700000
    Name: usage_ratio, dtype: float64



Distribución del ratio de uso: anuncios publicados frente a los contratados. Se comprueba que el un 25% de los df_dim tienen un ratio > 1. Como está a nivel user - mes, hacemos un groupby para ver la info a nivel usuario y ver el total de anuncios contratados y publicados


```python
df_usage_ratio = (
    df_fct
    .groupby("advertiser_zrive_id")[["monthly_contracted_ads", "monthly_unique_published_ads"]]
    .sum()
    .assign(
    usage_ratio=lambda x:
        x["monthly_unique_published_ads"] / x["monthly_contracted_ads"].replace(0, np.nan) 
    )
)
df_usage_ratio
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>monthly_contracted_ads</th>
      <th>monthly_unique_published_ads</th>
      <th>usage_ratio</th>
    </tr>
    <tr>
      <th>advertiser_zrive_id</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>375</td>
      <td>169</td>
      <td>0.450667</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4350</td>
      <td>1728</td>
      <td>0.397241</td>
    </tr>
    <tr>
      <th>3</th>
      <td>105</td>
      <td>51</td>
      <td>0.485714</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2990</td>
      <td>3029</td>
      <td>1.013043</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1075</td>
      <td>779</td>
      <td>0.724651</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>7047</th>
      <td>3</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>7051</th>
      <td>20</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>7052</th>
      <td>3</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>7066</th>
      <td>90</td>
      <td>86</td>
      <td>0.955556</td>
    </tr>
    <tr>
      <th>7069</th>
      <td>20</td>
      <td>9</td>
      <td>0.450000</td>
    </tr>
  </tbody>
</table>
<p>6968 rows × 3 columns</p>
</div>




```python
df_usage_ratio["usage_ratio"].describe()
```




    count    6956.000000
    mean        0.642500
    std         0.325140
    min         0.000000
    25%         0.429151
    50%         0.709938
    75%         0.889720
    max         3.000000
    Name: usage_ratio, dtype: float64



Sigue habiendo usuarios que tienen más anuncios publicados que contratados, puede ser un error en los datos


```python
df_fct["conversion_rate"].describe(percentiles=[0.75])
```




    count    91430.000000
    mean         0.007997
    std          0.080387
    min          0.000000
    50%          0.003727
    75%          0.006829
    max         18.750000
    Name: conversion_rate, dtype: float64



Ratio de conversión: % de visitas que se convierten en leads
- Fuertemente sesgada hacia la derecha
- Algunos casos muestran ratio > 1, lo cual no tiene sentido


```python
sns.histplot(
    df_fct["cost_per_lead"], 
    bins = 100
)
plt.xscale("log")
plt.xlabel("Coste por lead (€)")
plt.title("Distribución del coste por lead")
plt.show()
```


    
![png](EDA_files/EDA_74_0.png)
    


>La distribución del coste por lead está muy sesgada a la derecha; existe una cola larga de pocos clientes con costes muy elevados


```python
len(set(df_fct.advertiser_zrive_id) - set(df_dim.advertiser_zrive_id))
```




    0



Todos los df_dim en **fct** están en **dim**


```python
df_master = df_fct.merge(
    df_dim,
    on="advertiser_zrive_id",
    how="left"
)
df_master.shape
```




    (96827, 35)




```python
df_master.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>advertiser_zrive_id</th>
      <th>period_int</th>
      <th>monthly_contracted_ads</th>
      <th>monthly_published_ads</th>
      <th>monthly_unique_published_ads</th>
      <th>monthly_distinct_ads</th>
      <th>monthly_oro_ads</th>
      <th>monthly_plata_ads</th>
      <th>monthly_destacados_ads</th>
      <th>monthly_pepitas_ads</th>
      <th>...</th>
      <th>conversion_rate</th>
      <th>cost_per_lead</th>
      <th>usage_ratio</th>
      <th>province_id</th>
      <th>updated_at</th>
      <th>advertiser_province</th>
      <th>advertiser_group_id</th>
      <th>min_start_contrato_date</th>
      <th>max_start_contrato_nuevo_date</th>
      <th>contrato_churn_date</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>2023-01-01</td>
      <td>75</td>
      <td>47</td>
      <td>47</td>
      <td>56.0</td>
      <td>6</td>
      <td>6</td>
      <td>6</td>
      <td>0</td>
      <td>...</td>
      <td>0.003962</td>
      <td>95.233333</td>
      <td>0.626667</td>
      <td>11</td>
      <td>2023-06-16 09:35:01</td>
      <td>Cádiz</td>
      <td>NaN</td>
      <td>2020-12-23</td>
      <td>NaT</td>
      <td>2023-05-31</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>2023-01-01</td>
      <td>150</td>
      <td>31</td>
      <td>31</td>
      <td>55.0</td>
      <td>10</td>
      <td>10</td>
      <td>4</td>
      <td>0</td>
      <td>...</td>
      <td>0.002003</td>
      <td>73.325000</td>
      <td>0.206667</td>
      <td>11</td>
      <td>2023-11-06 16:01:14</td>
      <td>Cádiz</td>
      <td>100.0</td>
      <td>2020-09-25</td>
      <td>2020-09-25</td>
      <td>NaT</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>2023-01-01</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>8</td>
      <td>2024-08-07 13:37:09</td>
      <td>Barcelona</td>
      <td>41.0</td>
      <td>2024-05-06</td>
      <td>NaT</td>
      <td>2024-07-15</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>2023-01-01</td>
      <td>85</td>
      <td>79</td>
      <td>79</td>
      <td>86.0</td>
      <td>3</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>0.004640</td>
      <td>83.214286</td>
      <td>0.929412</td>
      <td>48</td>
      <td>2023-05-03 16:37:02</td>
      <td>Vizcaya</td>
      <td>NaN</td>
      <td>2022-09-30</td>
      <td>NaT</td>
      <td>NaT</td>
    </tr>
    <tr>
      <th>4</th>
      <td>6</td>
      <td>2023-01-01</td>
      <td>20</td>
      <td>20</td>
      <td>20</td>
      <td>25.0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>0.001811</td>
      <td>21.018750</td>
      <td>1.000000</td>
      <td>48</td>
      <td>2025-03-12 13:48:02</td>
      <td>Vizcaya</td>
      <td>NaN</td>
      <td>2022-10-10</td>
      <td>NaT</td>
      <td>NaT</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 35 columns</p>
</div>




```python
df_master.columns
```




    Index(['advertiser_zrive_id', 'period_int', 'monthly_contracted_ads',
           'monthly_published_ads', 'monthly_unique_published_ads',
           'monthly_distinct_ads', 'monthly_oro_ads', 'monthly_plata_ads',
           'monthly_destacados_ads', 'monthly_pepitas_ads', 'monthly_shows',
           'monthly_visits', 'monthly_leads', 'monthly_total_phone_views',
           'monthly_total_calls', 'monthly_total_emails', 'monthly_total_invoice',
           'monthly_total_reference_price', 'monthly_unique_calls',
           'monthly_unique_emails', 'monthly_unique_leads', 'monthly_avg_ad_price',
           'has_active_contract', 'price_bucket', 'price_diff_pct',
           'conversion_rate', 'cost_per_lead', 'usage_ratio', 'province_id',
           'updated_at', 'advertiser_province', 'advertiser_group_id',
           'min_start_contrato_date', 'max_start_contrato_nuevo_date',
           'contrato_churn_date'],
          dtype='object')




```python
province_stats = (
    df_master
        .groupby('advertiser_province')['monthly_total_invoice']
        .median()
        .sort_values(ascending=False)
        .head(10)
)

plt.figure(figsize=(10, 5))
sns.barplot(
    x=province_stats.values,
    y=province_stats.index
)

plt.title('Facturación mediana por provincia (Top 10)')
plt.xlabel('Facturación mensual (€)')
plt.ylabel('Provincia')
plt.tight_layout()
plt.show()
```


    
![png](EDA_files/EDA_81_0.png)
    



```python
df_cost_plot = df_fct[
    (df_fct["cost_per_lead"] > 0) &
    (df_fct["cost_per_lead"] <= 600)
]

plt.figure(figsize=(8,5))

sns.histplot(
    df_cost_plot["cost_per_lead"],
    bins=40
)

plt.xlabel("Coste por lead (€)")
plt.ylabel("Número de clientes")
plt.title("Distribución del coste por lead")
plt.show()
```


    
![png](EDA_files/EDA_82_0.png)
    


<h4>CONCLUSIÓN</h4>

- El 55% de clientes (3,894 de 7,076) han solicitado baja al menos una vez
- La causa principal por la que chacer churn es "VALOR PERCIBIDO INSUFICIENTE", no ven suficiente retorno de lo que pagan
- El precio de tarifa no es realista: casi nadie lo paga y la mayoría de clientes está por debajo de 500€/mes.
- Hay diferencias claras por provincia, así que no tiene sentido un precio único para todos.
- Las métricas clave que explican el churn y deben guiar el pricing son:
     - Coste por lead

     - Ratio de conversión

     - Ratio de uso

     - Provincia (valor del tráfico)

     - Antigüedad del cliente

- Data quality: tanto en ratio de conversión como ratio de uso hay valores > 1, lo cual no tiene sentido

## Rango de fechas

Comprobamos en qué fechas tenemos datos en cada tabla


```python
print("\n--- RANGO DE FECHAS ---")
print(f"df_dim  min_start_contrato_date: "
      f"{df_dim['min_start_contrato_date'].min()} "
      f"→ {df_dim['min_start_contrato_date'].max()}")
print(f"df_fct     period_int:              "
      f"{df_fct['period_int'].min()} "
      f"→ {df_fct['period_int'].max()}")
print(f"df_wd  withdrawal_creation_date: "
      f"{df_wd['withdrawal_creation_date'].min()} "
      f"→ {df_wd['withdrawal_creation_date'].max()}")
```

    
    --- RANGO DE FECHAS ---
    df_dim  min_start_contrato_date: 2009-10-16 00:00:00 → 2025-06-20 00:00:00
    df_fct     period_int:              2023-01-01 00:00:00 → 2025-05-01 00:00:00
    df_wd  withdrawal_creation_date: 2012-06-19 07:12:34 → 2025-05-30 11:29:06


- En las tablas dim y wd hay datos mucho anteriores al periodo de estudio, que comienza en 2023. Los datos que nos sobran serían aquellos de df_wd donde la fecha de baja definitiva (withdrawal_effective_date) es menor a 2023

## Comprobación: ¿advertisers en df_fct están en df_dim?


```python
set(df_fct["advertiser_zrive_id"].unique()).difference(set(df_dim["advertiser_zrive_id"].unique()))
```




    set()



Sí, todos los advertisers de df_fct están en la tabla df_dim, con lo cual tenemos información sobre los contratos de todos

## Análisis comportamiento advertisers

Para tener una foto completa del comportamiento de los usuarios e información contractual de cada uno, hacemos un merge entre df_fct y df_dim


```python
df_dim.columns
```




    Index(['advertiser_zrive_id', 'province_id', 'updated_at',
           'advertiser_province', 'advertiser_group_id', 'min_start_contrato_date',
           'max_start_contrato_nuevo_date', 'contrato_churn_date'],
          dtype='object')




```python
df_fct_dim = df_fct.merge(
    df_dim[["advertiser_zrive_id", "province_id", "advertiser_province", "advertiser_group_id", "min_start_contrato_date", "max_start_contrato_nuevo_date", "contrato_churn_date"]],
    how="left",
    on="advertiser_zrive_id"
    )
df_fct_dim.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>advertiser_zrive_id</th>
      <th>period_int</th>
      <th>monthly_contracted_ads</th>
      <th>monthly_published_ads</th>
      <th>monthly_unique_published_ads</th>
      <th>monthly_distinct_ads</th>
      <th>monthly_oro_ads</th>
      <th>monthly_plata_ads</th>
      <th>monthly_destacados_ads</th>
      <th>monthly_pepitas_ads</th>
      <th>...</th>
      <th>price_diff_pct</th>
      <th>conversion_rate</th>
      <th>cost_per_lead</th>
      <th>usage_ratio</th>
      <th>province_id</th>
      <th>advertiser_province</th>
      <th>advertiser_group_id</th>
      <th>min_start_contrato_date</th>
      <th>max_start_contrato_nuevo_date</th>
      <th>contrato_churn_date</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>2023-01-01</td>
      <td>75</td>
      <td>47</td>
      <td>47</td>
      <td>56.0</td>
      <td>6</td>
      <td>6</td>
      <td>6</td>
      <td>0</td>
      <td>...</td>
      <td>-0.290686</td>
      <td>0.003962</td>
      <td>95.233333</td>
      <td>0.626667</td>
      <td>11</td>
      <td>Cádiz</td>
      <td>NaN</td>
      <td>2020-12-23</td>
      <td>NaT</td>
      <td>2023-05-31</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>2023-01-01</td>
      <td>150</td>
      <td>31</td>
      <td>31</td>
      <td>55.0</td>
      <td>10</td>
      <td>10</td>
      <td>4</td>
      <td>0</td>
      <td>...</td>
      <td>-0.919496</td>
      <td>0.002003</td>
      <td>73.325000</td>
      <td>0.206667</td>
      <td>11</td>
      <td>Cádiz</td>
      <td>100.0</td>
      <td>2020-09-25</td>
      <td>2020-09-25</td>
      <td>NaT</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>2023-01-01</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>8</td>
      <td>Barcelona</td>
      <td>41.0</td>
      <td>2024-05-06</td>
      <td>NaT</td>
      <td>2024-07-15</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>2023-01-01</td>
      <td>85</td>
      <td>79</td>
      <td>79</td>
      <td>86.0</td>
      <td>3</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>-0.614659</td>
      <td>0.004640</td>
      <td>83.214286</td>
      <td>0.929412</td>
      <td>48</td>
      <td>Vizcaya</td>
      <td>NaN</td>
      <td>2022-09-30</td>
      <td>NaT</td>
      <td>NaT</td>
    </tr>
    <tr>
      <th>4</th>
      <td>6</td>
      <td>2023-01-01</td>
      <td>20</td>
      <td>20</td>
      <td>20</td>
      <td>25.0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>-0.346990</td>
      <td>0.001811</td>
      <td>21.018750</td>
      <td>1.000000</td>
      <td>48</td>
      <td>Vizcaya</td>
      <td>NaN</td>
      <td>2022-10-10</td>
      <td>NaT</td>
      <td>NaT</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 34 columns</p>
</div>



Buscamos y eliminamos los usuarios inactivos entre 2023 - 2025


```python
(
    df_fct
    .groupby("advertiser_zrive_id")["has_active_contract"]
    .sum()
    .loc[lambda x: x==0]
)
```




    advertiser_zrive_id
    7036    0
    7066    0
    7069    0
    Name: has_active_contract, dtype: int64




```python
advertisers_inactivos = [7036, 7066, 7069]
df_fct_dim = df_fct_dim[~df_fct_dim["advertiser_zrive_id"].isin(advertisers_inactivos)].copy()
```


```python
df_fct_dim["min_start_contrato_date"].isna().any()
```




    np.False_



## Tipos de clientes en el periodo de estudio (2023-2025)

Entre 2023 y 2025, podemos tener clientes con diferentes comportamientos:
1) Clientes que entraron con un contrato activo: filtramos y nos quedamos con los siguientes (punto 2.)
2) Clientes que se dieron de alta una vez empezado el periodo de estudio:
    - Se dieron de baja antes del final del periodo (2025) -> tienen una fecha contrato_churn_date != NaT
    - Siguen de alta -> contrato_churn_date == NaT


```python
df_fct_dim[(df_fct_dim["has_active_contract"]==True)]["period_int"].max()
```




    Timestamp('2025-05-01 00:00:00')



Último `period_int` del dataset en el que existe al menos un usuario activo. Todos los advertisers con un `contrato_churn_date` > 2025-05-01 se consideran censurados, es decir, activos

### Clientes que entraton con un contrato activo

Estos no nos sirven, ya que no tenemos información del comportamiento en sus primeros meses. En caso de que estos mismos se den de baja y vuelvan a darse de alta pasado el 2023-01-01, sí nos valdrían ya que ahí sí tenemos información de su comportamiento en sus primeros meses del nuevo contrato.

Por definición del negocio, tenemos:
- **min_start_contrato_date**: Fecha inicial del primer contrato (antigüedad). 
- **max_start_contrato_nuevo_date**: Fecha inicial del último contrato de tipo: nuevo, en  caso de existir. Indica si un anunciante se dio de baja y retornó a la plataforma. 

Por tanto, para este filtro usaremos como criterio si su **min_start_contrato_date** es >= 2023-01-01, así nos aseguramos que al menos el primer contrato empezó en el periodo de estudio



```python
df_fct_dim = df_fct_dim.sort_values(["advertiser_zrive_id", "period_int"])
df_fct_dim = df_fct_dim[df_fct_dim["min_start_contrato_date"] >= "2023-01-01"].copy()
df_fct_dim.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>advertiser_zrive_id</th>
      <th>period_int</th>
      <th>monthly_contracted_ads</th>
      <th>monthly_published_ads</th>
      <th>monthly_unique_published_ads</th>
      <th>monthly_distinct_ads</th>
      <th>monthly_oro_ads</th>
      <th>monthly_plata_ads</th>
      <th>monthly_destacados_ads</th>
      <th>monthly_pepitas_ads</th>
      <th>...</th>
      <th>price_diff_pct</th>
      <th>conversion_rate</th>
      <th>cost_per_lead</th>
      <th>usage_ratio</th>
      <th>province_id</th>
      <th>advertiser_province</th>
      <th>advertiser_group_id</th>
      <th>min_start_contrato_date</th>
      <th>max_start_contrato_nuevo_date</th>
      <th>contrato_churn_date</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>2023-01-01</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>8</td>
      <td>Barcelona</td>
      <td>41.0</td>
      <td>2024-05-06</td>
      <td>NaT</td>
      <td>2024-07-15</td>
    </tr>
    <tr>
      <th>53017</th>
      <td>3</td>
      <td>2024-05-01</td>
      <td>35</td>
      <td>17</td>
      <td>17</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>NaN</td>
      <td>0.024917</td>
      <td>0.0</td>
      <td>0.485714</td>
      <td>8</td>
      <td>Barcelona</td>
      <td>41.0</td>
      <td>2024-05-06</td>
      <td>NaT</td>
      <td>2024-07-15</td>
    </tr>
    <tr>
      <th>56457</th>
      <td>3</td>
      <td>2024-06-01</td>
      <td>35</td>
      <td>17</td>
      <td>17</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>NaN</td>
      <td>0.000000</td>
      <td>NaN</td>
      <td>0.485714</td>
      <td>8</td>
      <td>Barcelona</td>
      <td>41.0</td>
      <td>2024-05-06</td>
      <td>NaT</td>
      <td>2024-07-15</td>
    </tr>
    <tr>
      <th>59986</th>
      <td>3</td>
      <td>2024-07-01</td>
      <td>35</td>
      <td>17</td>
      <td>17</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>NaN</td>
      <td>0.000000</td>
      <td>NaN</td>
      <td>0.485714</td>
      <td>8</td>
      <td>Barcelona</td>
      <td>41.0</td>
      <td>2024-05-06</td>
      <td>NaT</td>
      <td>2024-07-15</td>
    </tr>
    <tr>
      <th>59988</th>
      <td>5</td>
      <td>2024-07-01</td>
      <td>100</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>30</td>
      <td>5</td>
      <td>2</td>
      <td>0</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.000000</td>
      <td>48</td>
      <td>Vizcaya</td>
      <td>193.0</td>
      <td>2024-07-30</td>
      <td>2024-09-30</td>
      <td>NaT</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 34 columns</p>
</div>




```python
df_fct_dim["advertiser_zrive_id"].nunique()
```




    3657



Tras aplicar este filtro, nos quedamos con 3657 advertisers cuyo primer contrato (min_start_contrato_date) empieza ≥ 2023-01-01. De esta forma, tenemos el comportamiento completo de todos los advertisers desde su entrada en la plataforma, evitando así sesgos derivados de actividad previa no registrada.

## Análisis a nivel **contrato**

El problema de negocio radica en encontrar un precio estable según el comportamiento del advertiser, el cual siempre está asociado a un contrato único. Por ello, la unidad de análisis debe ser el contrato y no el advertiser, ya que un mismo advertiser puede tener varios contratos y no quedarían correctamente representados al hacer la agregación de los datos a nivel advertiser. Por tanto, definimos una nueva columna `contract_id` para identificar los contratos. 

Antes de nada, creamos `df_contract`, una copia de `df_fct_dim` pero con las columnas relevantes para el análisis


```python
df_fct_dim.columns
```




    Index(['advertiser_zrive_id', 'period_int', 'monthly_contracted_ads',
           'monthly_published_ads', 'monthly_unique_published_ads',
           'monthly_distinct_ads', 'monthly_oro_ads', 'monthly_plata_ads',
           'monthly_destacados_ads', 'monthly_pepitas_ads', 'monthly_shows',
           'monthly_visits', 'monthly_leads', 'monthly_total_phone_views',
           'monthly_total_calls', 'monthly_total_emails', 'monthly_total_invoice',
           'monthly_total_reference_price', 'monthly_unique_calls',
           'monthly_unique_emails', 'monthly_unique_leads', 'monthly_avg_ad_price',
           'has_active_contract', 'price_bucket', 'price_diff_pct',
           'conversion_rate', 'cost_per_lead', 'usage_ratio', 'province_id',
           'advertiser_province', 'advertiser_group_id', 'min_start_contrato_date',
           'max_start_contrato_nuevo_date', 'contrato_churn_date'],
          dtype='object')




```python
contract_analysis_cols = ['advertiser_zrive_id',
       'province_id', 'advertiser_group_id', 'min_start_contrato_date',
       'max_start_contrato_nuevo_date', 'contrato_churn_date','has_active_contract', 'period_int']

df_contract = df_fct_dim[contract_analysis_cols].copy()
```

### Criterio contrato: ¿cómo identificamos los contratos?


Definimos un contrato como un tramo continuo de actividad de un mismo advertiser_zrive_id, donde se cumple:
- `has_active_contract == True`
- `period_int` son consecutivos a nivel mensual

Es decir:
- si dos filas activas pertenecen a meses consecutivos, forman parte del mismo contrato
- si entre dos meses activos existe un salto de más de 1 mes, se considera que comienza un nuevo contrato
- si un mes pasa de `has_active_contract == False` a `True`, se considera también el inicio de un nuevo contrato

#### 1) Calculamos la diferencia mensual entre `period_int` y su valor previo


```python
df_contract = df_contract.sort_values(["advertiser_zrive_id", "period_int"]).copy()
try:
    df_contract["period_int"] = df_contract["period_int"].dt.to_period("M")
except Exception:
    pass
```


```python
df_contract["prev_period_int"] = df_contract.groupby("advertiser_zrive_id")["period_int"].shift(1)
df_contract["month_diff"] = np.where(
    df_contract["prev_period_int"].notna(),
    df_contract["period_int"].astype("int64") - df_contract["prev_period_int"].astype("int64"),
    np.nan
)
```

#### 2) Aplicamos la lógica:
Un contrato empieza si:
- `has_active_contract == True`
- y además se cumple alguna de las siguientes condiciones:
    - `prev_has_active_contract == False` (es decir, pasa de False a True, luego comienza un contrato) 
    - `month_diff > 1` (es decir, si la fila anterior pertenece a un contrato activo pero de un periodo anterior a 1 mes)

Donde:
- `prev_has_active_contract` representa la columna `has_active_contract` desplazada una posición hacia atrás a nivel de ``advertiser_zrive_id``,


```python
df_contract["prev_has_active_contract"] = (
    df_contract
    .groupby("advertiser_zrive_id")["has_active_contract"]
    .shift(1)
)

df_contract["new_contract_start"] = (
    df_contract["has_active_contract"] &
    (
        (df_contract["prev_has_active_contract"] == False) | 
        (df_contract["month_diff"] > 1)
    )
)
```

#### 3) Asignamos el ID de contrato

Finalmente, asignamos un ID de contrato en las filas donde `has_active_contract == True`, ya que solo los periodos activos forman parte de un contrato. Para ello, combinamos el ``advertiser_zrive_id`` con el acumulado de la variable ``new_contract_start``, de modo que cada nuevo inicio detectado genera un identificador distinto dentro de un mismo advertiser. Así, cada tramo continuo de actividad queda asociado a un contrato único.


```python
df_contract["contract_id"] = (
    df_contract["advertiser_zrive_id"].astype(str)
    + "_"
    + df_contract
    .groupby("advertiser_zrive_id")["new_contract_start"]
    .cumsum()
    .astype(str)
).where(df_contract["has_active_contract"], np.nan)

df_contract[df_contract["advertiser_zrive_id"]==1781]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>advertiser_zrive_id</th>
      <th>province_id</th>
      <th>advertiser_group_id</th>
      <th>min_start_contrato_date</th>
      <th>max_start_contrato_nuevo_date</th>
      <th>contrato_churn_date</th>
      <th>has_active_contract</th>
      <th>period_int</th>
      <th>prev_period_int</th>
      <th>month_diff</th>
      <th>prev_has_active_contract</th>
      <th>new_contract_start</th>
      <th>contract_id</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1272</th>
      <td>1781</td>
      <td>46</td>
      <td>154.0</td>
      <td>2023-04-19</td>
      <td>NaT</td>
      <td>NaT</td>
      <td>False</td>
      <td>2023-01</td>
      <td>NaT</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4726</th>
      <td>1781</td>
      <td>46</td>
      <td>154.0</td>
      <td>2023-04-19</td>
      <td>NaT</td>
      <td>NaT</td>
      <td>False</td>
      <td>2023-02</td>
      <td>2023-01</td>
      <td>1.0</td>
      <td>False</td>
      <td>False</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>8088</th>
      <td>1781</td>
      <td>46</td>
      <td>154.0</td>
      <td>2023-04-19</td>
      <td>NaT</td>
      <td>NaT</td>
      <td>False</td>
      <td>2023-03</td>
      <td>2023-02</td>
      <td>1.0</td>
      <td>False</td>
      <td>False</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>11485</th>
      <td>1781</td>
      <td>46</td>
      <td>154.0</td>
      <td>2023-04-19</td>
      <td>NaT</td>
      <td>NaT</td>
      <td>True</td>
      <td>2023-04</td>
      <td>2023-03</td>
      <td>1.0</td>
      <td>False</td>
      <td>True</td>
      <td>1781_1</td>
    </tr>
    <tr>
      <th>14897</th>
      <td>1781</td>
      <td>46</td>
      <td>154.0</td>
      <td>2023-04-19</td>
      <td>NaT</td>
      <td>NaT</td>
      <td>True</td>
      <td>2023-05</td>
      <td>2023-04</td>
      <td>1.0</td>
      <td>True</td>
      <td>False</td>
      <td>1781_1</td>
    </tr>
    <tr>
      <th>18277</th>
      <td>1781</td>
      <td>46</td>
      <td>154.0</td>
      <td>2023-04-19</td>
      <td>NaT</td>
      <td>NaT</td>
      <td>True</td>
      <td>2023-06</td>
      <td>2023-05</td>
      <td>1.0</td>
      <td>True</td>
      <td>False</td>
      <td>1781_1</td>
    </tr>
    <tr>
      <th>21654</th>
      <td>1781</td>
      <td>46</td>
      <td>154.0</td>
      <td>2023-04-19</td>
      <td>NaT</td>
      <td>NaT</td>
      <td>True</td>
      <td>2023-07</td>
      <td>2023-06</td>
      <td>1.0</td>
      <td>True</td>
      <td>False</td>
      <td>1781_1</td>
    </tr>
    <tr>
      <th>25036</th>
      <td>1781</td>
      <td>46</td>
      <td>154.0</td>
      <td>2023-04-19</td>
      <td>NaT</td>
      <td>NaT</td>
      <td>True</td>
      <td>2023-08</td>
      <td>2023-07</td>
      <td>1.0</td>
      <td>True</td>
      <td>False</td>
      <td>1781_1</td>
    </tr>
    <tr>
      <th>28245</th>
      <td>1781</td>
      <td>46</td>
      <td>154.0</td>
      <td>2023-04-19</td>
      <td>NaT</td>
      <td>NaT</td>
      <td>True</td>
      <td>2023-09</td>
      <td>2023-08</td>
      <td>1.0</td>
      <td>True</td>
      <td>False</td>
      <td>1781_1</td>
    </tr>
    <tr>
      <th>31429</th>
      <td>1781</td>
      <td>46</td>
      <td>154.0</td>
      <td>2023-04-19</td>
      <td>NaT</td>
      <td>NaT</td>
      <td>False</td>
      <td>2023-10</td>
      <td>2023-09</td>
      <td>1.0</td>
      <td>True</td>
      <td>False</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>34591</th>
      <td>1781</td>
      <td>46</td>
      <td>154.0</td>
      <td>2023-04-19</td>
      <td>NaT</td>
      <td>NaT</td>
      <td>True</td>
      <td>2023-11</td>
      <td>2023-10</td>
      <td>1.0</td>
      <td>False</td>
      <td>True</td>
      <td>1781_2</td>
    </tr>
    <tr>
      <th>37799</th>
      <td>1781</td>
      <td>46</td>
      <td>154.0</td>
      <td>2023-04-19</td>
      <td>NaT</td>
      <td>NaT</td>
      <td>True</td>
      <td>2023-12</td>
      <td>2023-11</td>
      <td>1.0</td>
      <td>True</td>
      <td>False</td>
      <td>1781_2</td>
    </tr>
    <tr>
      <th>40987</th>
      <td>1781</td>
      <td>46</td>
      <td>154.0</td>
      <td>2023-04-19</td>
      <td>NaT</td>
      <td>NaT</td>
      <td>True</td>
      <td>2024-01</td>
      <td>2023-12</td>
      <td>1.0</td>
      <td>True</td>
      <td>False</td>
      <td>1781_2</td>
    </tr>
    <tr>
      <th>44175</th>
      <td>1781</td>
      <td>46</td>
      <td>154.0</td>
      <td>2023-04-19</td>
      <td>NaT</td>
      <td>NaT</td>
      <td>True</td>
      <td>2024-02</td>
      <td>2024-01</td>
      <td>1.0</td>
      <td>True</td>
      <td>False</td>
      <td>1781_2</td>
    </tr>
    <tr>
      <th>47410</th>
      <td>1781</td>
      <td>46</td>
      <td>154.0</td>
      <td>2023-04-19</td>
      <td>NaT</td>
      <td>NaT</td>
      <td>True</td>
      <td>2024-03</td>
      <td>2024-02</td>
      <td>1.0</td>
      <td>True</td>
      <td>False</td>
      <td>1781_2</td>
    </tr>
    <tr>
      <th>50711</th>
      <td>1781</td>
      <td>46</td>
      <td>154.0</td>
      <td>2023-04-19</td>
      <td>NaT</td>
      <td>NaT</td>
      <td>True</td>
      <td>2024-04</td>
      <td>2024-03</td>
      <td>1.0</td>
      <td>True</td>
      <td>False</td>
      <td>1781_2</td>
    </tr>
    <tr>
      <th>54084</th>
      <td>1781</td>
      <td>46</td>
      <td>154.0</td>
      <td>2023-04-19</td>
      <td>NaT</td>
      <td>NaT</td>
      <td>True</td>
      <td>2024-05</td>
      <td>2024-04</td>
      <td>1.0</td>
      <td>True</td>
      <td>False</td>
      <td>1781_2</td>
    </tr>
    <tr>
      <th>57523</th>
      <td>1781</td>
      <td>46</td>
      <td>154.0</td>
      <td>2023-04-19</td>
      <td>NaT</td>
      <td>NaT</td>
      <td>True</td>
      <td>2024-06</td>
      <td>2024-05</td>
      <td>1.0</td>
      <td>True</td>
      <td>False</td>
      <td>1781_2</td>
    </tr>
    <tr>
      <th>61015</th>
      <td>1781</td>
      <td>46</td>
      <td>154.0</td>
      <td>2023-04-19</td>
      <td>NaT</td>
      <td>NaT</td>
      <td>True</td>
      <td>2024-07</td>
      <td>2024-06</td>
      <td>1.0</td>
      <td>True</td>
      <td>False</td>
      <td>1781_2</td>
    </tr>
    <tr>
      <th>64384</th>
      <td>1781</td>
      <td>46</td>
      <td>154.0</td>
      <td>2023-04-19</td>
      <td>NaT</td>
      <td>NaT</td>
      <td>True</td>
      <td>2024-08</td>
      <td>2024-07</td>
      <td>1.0</td>
      <td>True</td>
      <td>False</td>
      <td>1781_2</td>
    </tr>
    <tr>
      <th>67711</th>
      <td>1781</td>
      <td>46</td>
      <td>154.0</td>
      <td>2023-04-19</td>
      <td>NaT</td>
      <td>NaT</td>
      <td>True</td>
      <td>2024-09</td>
      <td>2024-08</td>
      <td>1.0</td>
      <td>True</td>
      <td>False</td>
      <td>1781_2</td>
    </tr>
    <tr>
      <th>71063</th>
      <td>1781</td>
      <td>46</td>
      <td>154.0</td>
      <td>2023-04-19</td>
      <td>NaT</td>
      <td>NaT</td>
      <td>True</td>
      <td>2024-10</td>
      <td>2024-09</td>
      <td>1.0</td>
      <td>True</td>
      <td>False</td>
      <td>1781_2</td>
    </tr>
    <tr>
      <th>74360</th>
      <td>1781</td>
      <td>46</td>
      <td>154.0</td>
      <td>2023-04-19</td>
      <td>NaT</td>
      <td>NaT</td>
      <td>True</td>
      <td>2024-11</td>
      <td>2024-10</td>
      <td>1.0</td>
      <td>True</td>
      <td>False</td>
      <td>1781_2</td>
    </tr>
    <tr>
      <th>77692</th>
      <td>1781</td>
      <td>46</td>
      <td>154.0</td>
      <td>2023-04-19</td>
      <td>NaT</td>
      <td>NaT</td>
      <td>True</td>
      <td>2024-12</td>
      <td>2024-11</td>
      <td>1.0</td>
      <td>True</td>
      <td>False</td>
      <td>1781_2</td>
    </tr>
    <tr>
      <th>81137</th>
      <td>1781</td>
      <td>46</td>
      <td>154.0</td>
      <td>2023-04-19</td>
      <td>NaT</td>
      <td>NaT</td>
      <td>True</td>
      <td>2025-01</td>
      <td>2024-12</td>
      <td>1.0</td>
      <td>True</td>
      <td>False</td>
      <td>1781_2</td>
    </tr>
    <tr>
      <th>84540</th>
      <td>1781</td>
      <td>46</td>
      <td>154.0</td>
      <td>2023-04-19</td>
      <td>NaT</td>
      <td>NaT</td>
      <td>True</td>
      <td>2025-02</td>
      <td>2025-01</td>
      <td>1.0</td>
      <td>True</td>
      <td>False</td>
      <td>1781_2</td>
    </tr>
    <tr>
      <th>87953</th>
      <td>1781</td>
      <td>46</td>
      <td>154.0</td>
      <td>2023-04-19</td>
      <td>NaT</td>
      <td>NaT</td>
      <td>True</td>
      <td>2025-03</td>
      <td>2025-02</td>
      <td>1.0</td>
      <td>True</td>
      <td>False</td>
      <td>1781_2</td>
    </tr>
    <tr>
      <th>91271</th>
      <td>1781</td>
      <td>46</td>
      <td>154.0</td>
      <td>2023-04-19</td>
      <td>NaT</td>
      <td>NaT</td>
      <td>True</td>
      <td>2025-04</td>
      <td>2025-03</td>
      <td>1.0</td>
      <td>True</td>
      <td>False</td>
      <td>1781_2</td>
    </tr>
    <tr>
      <th>94543</th>
      <td>1781</td>
      <td>46</td>
      <td>154.0</td>
      <td>2023-04-19</td>
      <td>NaT</td>
      <td>NaT</td>
      <td>True</td>
      <td>2025-05</td>
      <td>2025-04</td>
      <td>1.0</td>
      <td>True</td>
      <td>False</td>
      <td>1781_2</td>
    </tr>
  </tbody>
</table>
</div>




```python
n_contracts_per_advertiser = (
    df_contract
    .groupby("advertiser_zrive_id")["contract_id"]
    .nunique()
)
print(n_contracts_per_advertiser.value_counts())
n_contracts_per_advertiser.value_counts().plot(kind="bar")
print()
plt.xlabel("Nº contratos")
plt.ylabel("Nº advertisers")
plt.title("Distribución de advertisers según nº de contratos")
```

    contract_id
    1    3389
    2     259
    3       8
    5       1
    Name: count, dtype: int64
    





    Text(0.5, 1.0, 'Distribución de advertisers según nº de contratos')




    
![png](EDA_files/EDA_115_2.png)
    


La mayoría de advertisers tiene 1 único contrato en el periodo de estudio. Analizamos el resto de casos:


```python
for i in range(1,6):
    print(f'Advertisers con {i} contrato/s: {df_contract.groupby("advertiser_zrive_id")["contract_id"].nunique().loc[lambda x: x == i].index}')

```

    Advertisers con 1 contrato/s: Index([   3,    5,    8,    9,   13,   14,   17,   19,   20,   32,
           ...
           7038, 7039, 7040, 7041, 7042, 7045, 7046, 7047, 7051, 7052],
          dtype='int64', name='advertiser_zrive_id', length=3389)
    Advertisers con 2 contrato/s: Index([  28,   48,   49,   67,   78,  163,  169,  261,  365,  402,
           ...
           6042, 6078, 6118, 6210, 6223, 6243, 6244, 6353, 6379, 6396],
          dtype='int64', name='advertiser_zrive_id', length=259)
    Advertisers con 3 contrato/s: Index([567, 1554, 1691, 1752, 4360, 4385, 4852, 5173], dtype='int64', name='advertiser_zrive_id')
    Advertisers con 4 contrato/s: Index([], dtype='int64', name='advertiser_zrive_id')
    Advertisers con 5 contrato/s: Index([2087], dtype='int64', name='advertiser_zrive_id')



```python
df_contract[df_contract["advertiser_zrive_id"]==2087]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>advertiser_zrive_id</th>
      <th>province_id</th>
      <th>advertiser_group_id</th>
      <th>min_start_contrato_date</th>
      <th>max_start_contrato_nuevo_date</th>
      <th>contrato_churn_date</th>
      <th>has_active_contract</th>
      <th>period_int</th>
      <th>prev_period_int</th>
      <th>month_diff</th>
      <th>prev_has_active_contract</th>
      <th>new_contract_start</th>
      <th>contract_id</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>31593</th>
      <td>2087</td>
      <td>40</td>
      <td>NaN</td>
      <td>2023-10-26</td>
      <td>2025-03-06</td>
      <td>2025-04-07</td>
      <td>True</td>
      <td>2023-10</td>
      <td>NaT</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>2087_0</td>
    </tr>
    <tr>
      <th>34762</th>
      <td>2087</td>
      <td>40</td>
      <td>NaN</td>
      <td>2023-10-26</td>
      <td>2025-03-06</td>
      <td>2025-04-07</td>
      <td>True</td>
      <td>2023-11</td>
      <td>2023-10</td>
      <td>1.0</td>
      <td>True</td>
      <td>False</td>
      <td>2087_0</td>
    </tr>
    <tr>
      <th>47575</th>
      <td>2087</td>
      <td>40</td>
      <td>NaN</td>
      <td>2023-10-26</td>
      <td>2025-03-06</td>
      <td>2025-04-07</td>
      <td>True</td>
      <td>2024-03</td>
      <td>2023-11</td>
      <td>4.0</td>
      <td>True</td>
      <td>True</td>
      <td>2087_1</td>
    </tr>
    <tr>
      <th>50878</th>
      <td>2087</td>
      <td>40</td>
      <td>NaN</td>
      <td>2023-10-26</td>
      <td>2025-03-06</td>
      <td>2025-04-07</td>
      <td>True</td>
      <td>2024-04</td>
      <td>2024-03</td>
      <td>1.0</td>
      <td>True</td>
      <td>False</td>
      <td>2087_1</td>
    </tr>
    <tr>
      <th>64547</th>
      <td>2087</td>
      <td>40</td>
      <td>NaN</td>
      <td>2023-10-26</td>
      <td>2025-03-06</td>
      <td>2025-04-07</td>
      <td>True</td>
      <td>2024-08</td>
      <td>2024-04</td>
      <td>4.0</td>
      <td>True</td>
      <td>True</td>
      <td>2087_2</td>
    </tr>
    <tr>
      <th>67870</th>
      <td>2087</td>
      <td>40</td>
      <td>NaN</td>
      <td>2023-10-26</td>
      <td>2025-03-06</td>
      <td>2025-04-07</td>
      <td>True</td>
      <td>2024-09</td>
      <td>2024-08</td>
      <td>1.0</td>
      <td>True</td>
      <td>False</td>
      <td>2087_2</td>
    </tr>
    <tr>
      <th>74520</th>
      <td>2087</td>
      <td>40</td>
      <td>NaN</td>
      <td>2023-10-26</td>
      <td>2025-03-06</td>
      <td>2025-04-07</td>
      <td>True</td>
      <td>2024-11</td>
      <td>2024-09</td>
      <td>2.0</td>
      <td>True</td>
      <td>True</td>
      <td>2087_3</td>
    </tr>
    <tr>
      <th>77849</th>
      <td>2087</td>
      <td>40</td>
      <td>NaN</td>
      <td>2023-10-26</td>
      <td>2025-03-06</td>
      <td>2025-04-07</td>
      <td>False</td>
      <td>2024-12</td>
      <td>2024-11</td>
      <td>1.0</td>
      <td>True</td>
      <td>False</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>88111</th>
      <td>2087</td>
      <td>40</td>
      <td>NaN</td>
      <td>2023-10-26</td>
      <td>2025-03-06</td>
      <td>2025-04-07</td>
      <td>True</td>
      <td>2025-03</td>
      <td>2024-12</td>
      <td>3.0</td>
      <td>False</td>
      <td>True</td>
      <td>2087_4</td>
    </tr>
    <tr>
      <th>91426</th>
      <td>2087</td>
      <td>40</td>
      <td>NaN</td>
      <td>2023-10-26</td>
      <td>2025-03-06</td>
      <td>2025-04-07</td>
      <td>True</td>
      <td>2025-04</td>
      <td>2025-03</td>
      <td>1.0</td>
      <td>True</td>
      <td>False</td>
      <td>2087_4</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_wd[df_wd["advertiser_zrive_id"]==2087]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>withdrawal_id</th>
      <th>advertiser_zrive_id</th>
      <th>withdrawal_status</th>
      <th>withdrawal_type</th>
      <th>withdrawal_creation_date</th>
      <th>withdrawal_effective_date</th>
      <th>withdrawal_reason</th>
      <th>is_definitive_withdrawal</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2863</th>
      <td>21251</td>
      <td>2087</td>
      <td>Cerrada</td>
      <td>TOTAL</td>
      <td>2018-02-01 11:59:48</td>
      <td>2018-02-01</td>
      <td>FIN DE CONTRATO</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3007</th>
      <td>21743</td>
      <td>2087</td>
      <td>Cerrada</td>
      <td>PARCIAL</td>
      <td>2018-04-02 07:51:31</td>
      <td>2018-04-01</td>
      <td>FIN DE CONTRATO</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3400</th>
      <td>23140</td>
      <td>2087</td>
      <td>Cerrada</td>
      <td>TOTAL</td>
      <td>2018-10-22 11:18:34</td>
      <td>2018-11-01</td>
      <td>FIN DE CONTRATO</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



- Analizamos el advertiser ``2087``, que tiene 5 contratos diferentes. Observamos que ninguno de los contratos dura más de 3 meses, por tanto quedaría fuera del modelo, ya que necesitamos al menos 3 meses de actividad en la plataforma


```python
contract_duration = (
    df_contract
    .groupby("contract_id")
    .size()
)

contract_dist = (
    contract_duration
    .value_counts(normalize=True)
    .sort_index()
)

contract_dist.plot(kind="bar")

plt.title("Distribución de la duración de contratos (meses)")
plt.xlabel("Nº meses")
plt.ylabel("Proporción de contratos")
plt.show()
contract_duration.describe()
```


    
![png](EDA_files/EDA_121_0.png)
    





    count    3936.000000
    mean        8.567327
    std         6.302138
    min         1.000000
    25%         4.000000
    50%         6.000000
    75%        12.000000
    max        29.000000
    dtype: float64




```python
pct_ge_3_months = len(contract_duration.loc[lambda x: x>=3])/len(contract_duration)*100
print(f"{pct_ge_3_months:.2f}% de los contratos duran 3 o más meses")
```

    91.46% de los contratos duran 3 o más meses


## Definición del **churn**

En la siguiente sección, vamos a definir el churn de los usuarios de forma binaria:
- Churn 1: se dió de baja al mes 3
- Churn 0: permaneció más de 3 meses


```python

```


```python

```


```python

```


```python

```


```python

```
