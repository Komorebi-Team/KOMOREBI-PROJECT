```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from typing import Dict, Any, Optional, List

plt.style.use("ggplot")
```


```python
os.getcwd()
```




    '/home/jterryc/KOMOREBI-PROJECT/notebooks'




```python
os.listdir(".")
```




    ['EDA_jtc.ipynb', 'notebooks.txt', 'EDA_jon.ipynb']




```python
wd_local_path = '../data/zrive_advertiser_withdrawals.parquet'
dim_local_path = '../data/zrive_dim_advertiser.parquet'
fct_local_path = '../data/zrive_fct_monthly_snapshot_advertiser.parquet'
```


```python
df_wd = pd.read_parquet(wd_local_path, engine='fastparquet')
df_dim = pd.read_parquet(dim_local_path, engine='fastparquet')

```

### Witdrawals


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
float(df_wd.withdrawal_effective_date.isnull().sum()/df_wd.shape[0])
```




    0.36703723310393505



- 36.7% de valores faltantes en la columna "withdrawal_effective_date" (no viene explicada en la memoria)
- Tiene un gran porcentaje de valores faltantes, con lo cual no parece que sean MAR (Missing At Random)


```python
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
      <th>...</th>
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
    </tr>
  </tbody>
</table>
<p>22668 rows × 7 columns</p>
</div>




```python
df_wd.shape
```




    (22668, 7)




```python
def print_unique(df: pd.DataFrame, column: str = None):
    if column is None:
        for col in df.columns:
            if df[col].dtype == object:
                print(f"- {col} ({df[col].nunique()}): {df[col].unique()}")
    else:
        print(f"- {column} ({df[column].nunique()}): {df[column].unique()}")

print_unique(df_wd)
```

    - withdrawal_status (3): ['Cerrada' 'Denegada' 'Pendiente Administración']
    - withdrawal_type (2): ['TOTAL' 'PARCIAL']
    - withdrawal_reason (18): ['RESULTADOS' 'FALTA DE USO/TIEMPO' 'RAZONES ECONOMICAS' 'MOROSIDAD'
     'RATIO RESULTADO-INVERSION' 'OTROS' 'FALTA DE PRODUCTO'
     'CESE DE ACTIVIDAD' 'FIN DE CONTRATO' 'Cambio a Bundle Online'
     'Baja Bundle AS24' 'Upselling-cambio de contrato' 'No acepta subida'
     'CORONAVIRUS' 'Cambio de Contrato/propuesta/producto'
     'Incidencias de la web' 'Reestructuración cuentas de grupo' 'Desconocido']



```python
withdrawal_reason_not = ['Upselling-cambio de contrato', 'Cambio a Bundle Online','Cambio de Contrato/propuesta/producto']
withdrawal_reason_yes = [reason for reason in df_wd.withdrawal_reason.unique() if reason not in withdrawal_reason_not]
withdrawal_reason_yes
```




    ['RESULTADOS',
     'FALTA DE USO/TIEMPO',
     'RAZONES ECONOMICAS',
     'MOROSIDAD',
     'RATIO RESULTADO-INVERSION',
     'OTROS',
     'FALTA DE PRODUCTO',
     'CESE DE ACTIVIDAD',
     'FIN DE CONTRATO',
     'Baja Bundle AS24',
     'No acepta subida',
     'CORONAVIRUS',
     'Incidencias de la web',
     'Reestructuración cuentas de grupo',
     'Desconocido']




```python
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
df_wd[df_wd.is_definitive_withdrawal == 1].withdrawal_reason.unique()
```




    array(['RESULTADOS', 'FALTA DE USO/TIEMPO', 'RAZONES ECONOMICAS',
           'MOROSIDAD', 'RATIO RESULTADO-INVERSION', 'OTROS',
           'FALTA DE PRODUCTO', 'CESE DE ACTIVIDAD', 'FIN DE CONTRATO',
           'Baja Bundle AS24', 'No acepta subida', 'CORONAVIRUS',
           'Incidencias de la web', 'Reestructuración cuentas de grupo',
           'Desconocido'], dtype=object)




```python
df_wd[df_wd.is_definitive_withdrawal == 0]
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
      <th>63</th>
      <td>352</td>
      <td>883</td>
      <td>Denegada</td>
      <td>TOTAL</td>
      <td>2012-09-26 08:27:14</td>
      <td>NaT</td>
      <td>MOROSIDAD</td>
      <td>0</td>
    </tr>
    <tr>
      <th>437</th>
      <td>2309</td>
      <td>314</td>
      <td>Cerrada</td>
      <td>PARCIAL</td>
      <td>2013-11-19 14:25:49</td>
      <td>NaT</td>
      <td>MOROSIDAD</td>
      <td>0</td>
    </tr>
    <tr>
      <th>439</th>
      <td>2329</td>
      <td>851</td>
      <td>Cerrada</td>
      <td>PARCIAL</td>
      <td>2013-11-19 16:23:46</td>
      <td>2013-12-01</td>
      <td>FIN DE CONTRATO</td>
      <td>0</td>
    </tr>
    <tr>
      <th>441</th>
      <td>2336</td>
      <td>267</td>
      <td>Cerrada</td>
      <td>PARCIAL</td>
      <td>2013-11-25 08:14:37</td>
      <td>2013-12-01</td>
      <td>FIN DE CONTRATO</td>
      <td>0</td>
    </tr>
    <tr>
      <th>444</th>
      <td>2357</td>
      <td>355</td>
      <td>Cerrada</td>
      <td>PARCIAL</td>
      <td>2013-11-28 12:02:49</td>
      <td>NaT</td>
      <td>RAZONES ECONOMICAS</td>
      <td>0</td>
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
      <th>22644</th>
      <td>55594</td>
      <td>6999</td>
      <td>Cerrada</td>
      <td>TOTAL</td>
      <td>2025-05-29 15:06:44</td>
      <td>2025-05-29</td>
      <td>Upselling-cambio de contrato</td>
      <td>0</td>
    </tr>
    <tr>
      <th>22646</th>
      <td>55596</td>
      <td>1405</td>
      <td>Cerrada</td>
      <td>PARCIAL</td>
      <td>2025-05-30 06:53:33</td>
      <td>2025-05-30</td>
      <td>Cambio de Contrato/propuesta/producto</td>
      <td>0</td>
    </tr>
    <tr>
      <th>22652</th>
      <td>55602</td>
      <td>6857</td>
      <td>Cerrada</td>
      <td>TOTAL</td>
      <td>2025-05-30 07:52:10</td>
      <td>2025-05-31</td>
      <td>Cambio de Contrato/propuesta/producto</td>
      <td>0</td>
    </tr>
    <tr>
      <th>22658</th>
      <td>55608</td>
      <td>3866</td>
      <td>Cerrada</td>
      <td>TOTAL</td>
      <td>2025-05-30 09:36:59</td>
      <td>2025-05-30</td>
      <td>Cambio de Contrato/propuesta/producto</td>
      <td>0</td>
    </tr>
    <tr>
      <th>22659</th>
      <td>55609</td>
      <td>1001</td>
      <td>Cerrada</td>
      <td>TOTAL</td>
      <td>2025-05-30 09:56:22</td>
      <td>2025-05-30</td>
      <td>Upselling-cambio de contrato</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>6727 rows × 8 columns</p>
</div>




```python
df_wd.advertiser_zrive_id.value_counts()
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
real_withdrawal_pct = df_wd[df_wd["is_definitive_withdrawal"] == 1].shape[0]/df_wd.shape[0]
real_withdrawal_pct
```




    0.7032380448208929



Si se consideran bajas definitivas aquellas donde se cumplen estas 3 condiciones simultáneamente:
1. withdrawal_type = 'TOTAL'
2. withdrawal_status != 'Denegada'
3. withdrawal_reason not in ('Upselling-cambio de contrato', 'Cambio a Bundle Online',
'Cambio de Contrato/propuesta/producto')

Obtenemos que el 70.71% de registros de posibles abandonos fueron abandonos definitivos

Pregunta: a partir de ahora, ¿nos importa el df sólo cuando is_permanent_withdrawal = 1, o con ambos casos posibles?


```python
df_wd.isnull().sum()
```




    withdrawal_id                   0
    advertiser_zrive_id             0
    withdrawal_status               0
    withdrawal_type                 0
    withdrawal_creation_date        0
    withdrawal_effective_date    8320
    withdrawal_reason               0
    is_definitive_withdrawal        0
    dtype: int64




```python
df_wd.withdrawal_status.value_counts()
```




    withdrawal_status
    Cerrada                     22662
    Denegada                        4
    Pendiente Administración        2
    Name: count, dtype: int64




```python
df_wd.withdrawal_status.value_counts(normalize=True).plot(kind='bar')
print(df_wd.withdrawal_status.value_counts(normalize=True))
```

    withdrawal_status
    Cerrada                     0.999735
    Denegada                    0.000176
    Pendiente Administración    0.000088
    Name: proportion, dtype: float64



    
![png](EDA_jtc_files/EDA_jtc_22_1.png)
    


Pregunta: predomina la categoría "Cerrada" con casi un 99.99%. ¿Habría que eliminarla? Entiendo que no da mucha información ya que el dataset de por sí son los casos en los que la salida del cliente fue definitiva, con lo cual el estado de de "withdrawal" no tiene mucha relevancia aquí


```python
df_wd.withdrawal_type.value_counts(normalize=True)
```




    withdrawal_type
    TOTAL      0.92721
    PARCIAL    0.07279
    Name: proportion, dtype: float64




```python
df_wd.withdrawal_creation_date = df_wd.withdrawal_creation_date.dt.normalize()
```


```python
df_wd.is_definitive_withdrawal.value_counts(normalize=True).plot(kind="bar")
```




    <Axes: xlabel='is_definitive_withdrawal'>




    
![png](EDA_jtc_files/EDA_jtc_26_1.png)
    



```python
df_wd[df_wd.is_definitive_withdrawal == 1]
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
      <td>2012-06-19</td>
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
      <td>2012-06-19</td>
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
      <td>2012-06-20</td>
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
      <td>2012-06-20</td>
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
      <td>2012-06-20</td>
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
      <td>2025-05-30</td>
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
      <td>2025-05-30</td>
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
      <td>2025-05-30</td>
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
      <td>2025-05-30</td>
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
      <td>2025-05-30</td>
      <td>NaT</td>
      <td>MOROSIDAD</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>15941 rows × 8 columns</p>
</div>




```python
df_wd.describe()
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
      <th>withdrawal_creation_date</th>
      <th>withdrawal_effective_date</th>
      <th>is_definitive_withdrawal</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>22668.000000</td>
      <td>22668.000000</td>
      <td>22668</td>
      <td>14348</td>
      <td>22668.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>37768.396374</td>
      <td>2329.098774</td>
      <td>2021-11-25 13:00:55.267337</td>
      <td>2022-02-09 22:13:12.863117056</td>
      <td>0.703238</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>2012-06-19 00:00:00</td>
      <td>2012-06-28 00:00:00</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>29216.750000</td>
      <td>881.000000</td>
      <td>2020-03-24 00:00:00</td>
      <td>2020-06-01 00:00:00</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>41392.500000</td>
      <td>1902.000000</td>
      <td>2022-12-12 00:00:00</td>
      <td>2023-02-10 00:00:00</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>48760.250000</td>
      <td>3515.000000</td>
      <td>2024-03-19 00:00:00</td>
      <td>2024-05-16 00:00:00</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>55617.000000</td>
      <td>7026.000000</td>
      <td>2025-05-30 00:00:00</td>
      <td>2025-12-01 00:00:00</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>std</th>
      <td>13864.117631</td>
      <td>1752.783975</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.456841</td>
    </tr>
  </tbody>
</table>
</div>



En este dataset, creo que no tiene mucho sentido hacer un .describe()

### Dim Advertiser

Contiene información general y descriptiva de los anunciantes de la plataforma.

- 🔑 **advertiser_zive_id**: Identificador único del anunciante.
- **province_id**: Identificador de la provincia del anunciante.
- **advertiser_province**: Nombre de la provincia del anunciante.
- **updated_at**: Fecha y hora de la última actualización de los datos del anunciante.
- **advertiser_group_id**: Identificador del grupo al que pertenece el anunciante, en caso de formar parte de uno.
- **min_start_contrato_date**: Fecha de inicio del primer contrato del anunciante (antigüedad en la plataforma).
- **max_start_contrato_nuevo_date**: Fecha de inicio del último contrato de tipo *nuevo*, si existe. Indica si el anunciante se dio de baja y posteriormente regresó a la plataforma.
- **contrato_churn_date**: Fecha de finalización del último contrato.  
  - Si es **nula**, el anunciante tiene un contrato activo sin fecha de finalización especificada.  
  - Si es **futura**, no implica necesariamente una baja, ya que el contrato puede renovarse automáticamente.

Pregunta: ¿**contrato_churn_date** es futura con respecto a qué? Creo que respecto al period_int (mes de la tabla fct, que es la foto "actual") 




```python
df_dim.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 7076 entries, 0 to 7075
    Data columns (total 8 columns):
     #   Column                         Non-Null Count  Dtype         
    ---  ------                         --------------  -----         
     0   advertiser_zrive_id            7076 non-null   int64         
     1   province_id                    7076 non-null   int64         
     2   updated_at                     7076 non-null   datetime64[us]
     3   advertiser_province            7076 non-null   object        
     4   advertiser_group_id            1323 non-null   float64       
     5   min_start_contrato_date        7076 non-null   datetime64[ns]
     6   max_start_contrato_nuevo_date  5208 non-null   datetime64[ns]
     7   contrato_churn_date            3894 non-null   datetime64[ns]
    dtypes: datetime64[ns](3), datetime64[us](1), float64(1), int64(2), object(1)
    memory usage: 442.4+ KB



```python
print_unique(df_dim)
```

    - advertiser_province (52): ['Álava' 'Albacete' 'Alicante' 'Asturias' 'Badajoz' 'Barcelona' 'Burgos'
     'Cáceres' 'Cantabria' 'Castellón' 'Córdoba' 'La Coruña' 'Girona'
     'Granada' 'Guadalajara' 'Huesca' 'Islas Baleares' 'Jaén' 'Lugo' 'Madrid'
     'Málaga' 'Murcia' 'Navarra' 'Pontevedra' 'La Rioja' 'Segovia' 'Sevilla'
     'Teruel' 'Toledo' 'Valencia' 'Valladolid' 'Vizcaya' 'Zaragoza' 'Almería'
     'Ávila' 'Cádiz' 'Ciudad Real' 'Cuenca' 'Guipúzcoa' 'Huelva' 'León'
     'Lleida' 'Orense' 'Palencia' 'Las Palmas' 'Salamanca' 'Soria' 'Tarragona'
     'Tenerife' 'Zamora' 'Ceuta' 'Melilla']



```python
df_dim.isnull().sum()
```




    advertiser_zrive_id                 0
    province_id                         0
    updated_at                          0
    advertiser_province                 0
    advertiser_group_id              5753
    min_start_contrato_date             0
    max_start_contrato_nuevo_date    1868
    contrato_churn_date              3182
    dtype: int64



Missing values:
- **advertiser_group_id** indica el grupo al que pertenece el anunciante, (Pregunta) ¿esta información es relevante? Hay anunciantes que no pertenecen a ningun grupo
- **max_start_contrato_nuevo_date** indica la fecha en la que un anunciante que se había dado de baja, se volivó a dar de alta. Tiene sentido que haya valores nulos por que habrá casos en los que el anunciante se dió de baja y no volvió. Una cuestión interesante es ver con qué precio se dió de alta tanto la primera como la segunda vez
- **contrato_churn_date** indica la fecha final del último contrato. 

Pregunta: ¿Los valores nulos indican que el contrato está activo?


```python
float(df_dim.max_start_contrato_nuevo_date.isna().sum())/df_dim.shape[0]
```




    0.2639909553420011




```python
df_dim[df_dim["max_start_contrato_nuevo_date"].isna() & df_dim["contrato_churn_date"].isna()]
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
      <th>98</th>
      <td>1405</td>
      <td>1</td>
      <td>2025-06-06 09:54:13</td>
      <td>Álava</td>
      <td>191.0</td>
      <td>2020-04-07</td>
      <td>NaT</td>
      <td>NaT</td>
    </tr>
    <tr>
      <th>104</th>
      <td>7026</td>
      <td>1</td>
      <td>2025-05-14 12:14:57</td>
      <td>Álava</td>
      <td>NaN</td>
      <td>2025-05-14</td>
      <td>NaT</td>
      <td>NaT</td>
    </tr>
    <tr>
      <th>105</th>
      <td>2080</td>
      <td>1</td>
      <td>2024-01-12 15:45:13</td>
      <td>Álava</td>
      <td>48.0</td>
      <td>2020-03-16</td>
      <td>NaT</td>
      <td>NaT</td>
    </tr>
    <tr>
      <th>110</th>
      <td>1384</td>
      <td>1</td>
      <td>2025-03-03 12:31:07</td>
      <td>Álava</td>
      <td>NaN</td>
      <td>2021-04-13</td>
      <td>NaT</td>
      <td>NaT</td>
    </tr>
    <tr>
      <th>111</th>
      <td>993</td>
      <td>1</td>
      <td>2024-01-12 15:00:49</td>
      <td>Álava</td>
      <td>48.0</td>
      <td>2020-03-10</td>
      <td>NaT</td>
      <td>NaT</td>
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
      <th>3269</th>
      <td>2708</td>
      <td>50</td>
      <td>2024-08-31 11:01:19</td>
      <td>Zaragoza</td>
      <td>NaN</td>
      <td>2021-09-23</td>
      <td>NaT</td>
      <td>NaT</td>
    </tr>
    <tr>
      <th>3271</th>
      <td>2310</td>
      <td>50</td>
      <td>2024-02-28 13:43:30</td>
      <td>Zaragoza</td>
      <td>NaN</td>
      <td>2019-04-01</td>
      <td>NaT</td>
      <td>NaT</td>
    </tr>
    <tr>
      <th>3273</th>
      <td>3257</td>
      <td>50</td>
      <td>2022-11-25 23:18:06</td>
      <td>Zaragoza</td>
      <td>162.0</td>
      <td>2022-05-18</td>
      <td>NaT</td>
      <td>NaT</td>
    </tr>
    <tr>
      <th>3276</th>
      <td>2665</td>
      <td>50</td>
      <td>2024-01-12 15:31:17</td>
      <td>Zaragoza</td>
      <td>48.0</td>
      <td>2020-03-16</td>
      <td>NaT</td>
      <td>NaT</td>
    </tr>
    <tr>
      <th>3278</th>
      <td>492</td>
      <td>50</td>
      <td>2025-01-14 12:33:40</td>
      <td>Zaragoza</td>
      <td>163.0</td>
      <td>2022-04-08</td>
      <td>NaT</td>
      <td>NaT</td>
    </tr>
  </tbody>
</table>
<p>939 rows × 8 columns</p>
</div>




```python
df_dim[df_dim["max_start_contrato_nuevo_date"].isna() & df_dim["contrato_churn_date"].notna()]
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
    <tr>
      <th>5</th>
      <td>1789</td>
      <td>3</td>
      <td>2024-11-07 13:26:50</td>
      <td>Alicante</td>
      <td>185.0</td>
      <td>2021-11-15</td>
      <td>NaT</td>
      <td>2024-11-06</td>
    </tr>
    <tr>
      <th>6</th>
      <td>5023</td>
      <td>3</td>
      <td>2024-12-04 11:55:31</td>
      <td>Alicante</td>
      <td>NaN</td>
      <td>2023-04-03</td>
      <td>NaT</td>
      <td>2023-07-02</td>
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
      <th>7055</th>
      <td>5738</td>
      <td>29</td>
      <td>2024-01-24 13:50:41</td>
      <td>Madrid</td>
      <td>NaN</td>
      <td>2024-01-17</td>
      <td>NaT</td>
      <td>2025-06-30</td>
    </tr>
    <tr>
      <th>7057</th>
      <td>1424</td>
      <td>30</td>
      <td>2024-11-11 12:13:45</td>
      <td>Málaga</td>
      <td>NaN</td>
      <td>2022-09-06</td>
      <td>NaT</td>
      <td>2025-06-30</td>
    </tr>
    <tr>
      <th>7066</th>
      <td>2718</td>
      <td>42</td>
      <td>2024-09-04 16:57:46</td>
      <td>Tarragona</td>
      <td>NaN</td>
      <td>2021-02-01</td>
      <td>NaT</td>
      <td>2025-06-30</td>
    </tr>
    <tr>
      <th>7067</th>
      <td>3396</td>
      <td>42</td>
      <td>2024-12-06 07:25:38</td>
      <td>Tarragona</td>
      <td>NaN</td>
      <td>2022-10-10</td>
      <td>NaT</td>
      <td>2025-06-30</td>
    </tr>
    <tr>
      <th>7075</th>
      <td>4194</td>
      <td>50</td>
      <td>2023-10-10 11:21:22</td>
      <td>Zaragoza</td>
      <td>NaN</td>
      <td>2022-07-21</td>
      <td>NaT</td>
      <td>2025-06-30</td>
    </tr>
  </tbody>
</table>
<p>929 rows × 8 columns</p>
</div>




```python
n_active_contracts = df_dim[df_dim["contrato_churn_date"].isna()]
n_active_contracts
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
      <th>98</th>
      <td>1405</td>
      <td>1</td>
      <td>2025-06-06 09:54:13</td>
      <td>Álava</td>
      <td>191.0</td>
      <td>2020-04-07</td>
      <td>NaT</td>
      <td>NaT</td>
    </tr>
    <tr>
      <th>99</th>
      <td>614</td>
      <td>1</td>
      <td>2025-06-17 13:13:47</td>
      <td>Álava</td>
      <td>193.0</td>
      <td>2022-10-20</td>
      <td>2024-02-21</td>
      <td>NaT</td>
    </tr>
    <tr>
      <th>100</th>
      <td>487</td>
      <td>1</td>
      <td>2024-11-11 13:56:58</td>
      <td>Álava</td>
      <td>NaN</td>
      <td>2023-11-21</td>
      <td>2023-11-21</td>
      <td>NaT</td>
    </tr>
    <tr>
      <th>101</th>
      <td>2861</td>
      <td>1</td>
      <td>2024-08-08 12:14:49</td>
      <td>Álava</td>
      <td>190.0</td>
      <td>2020-04-28</td>
      <td>2024-02-28</td>
      <td>NaT</td>
    </tr>
    <tr>
      <th>102</th>
      <td>6703</td>
      <td>1</td>
      <td>2025-05-19 08:42:47</td>
      <td>Álava</td>
      <td>125.0</td>
      <td>2024-12-20</td>
      <td>2024-12-20</td>
      <td>NaT</td>
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
      <th>3275</th>
      <td>5499</td>
      <td>50</td>
      <td>2025-02-21 09:12:19</td>
      <td>Zaragoza</td>
      <td>NaN</td>
      <td>2023-10-10</td>
      <td>2023-10-10</td>
      <td>NaT</td>
    </tr>
    <tr>
      <th>3276</th>
      <td>2665</td>
      <td>50</td>
      <td>2024-01-12 15:31:17</td>
      <td>Zaragoza</td>
      <td>48.0</td>
      <td>2020-03-16</td>
      <td>NaT</td>
      <td>NaT</td>
    </tr>
    <tr>
      <th>3277</th>
      <td>7073</td>
      <td>50</td>
      <td>2025-06-17 17:03:47</td>
      <td>Zaragoza</td>
      <td>NaN</td>
      <td>2025-06-17</td>
      <td>2025-06-17</td>
      <td>NaT</td>
    </tr>
    <tr>
      <th>3278</th>
      <td>492</td>
      <td>50</td>
      <td>2025-01-14 12:33:40</td>
      <td>Zaragoza</td>
      <td>163.0</td>
      <td>2022-04-08</td>
      <td>NaT</td>
      <td>NaT</td>
    </tr>
    <tr>
      <th>3279</th>
      <td>7053</td>
      <td>50</td>
      <td>2025-06-05 12:24:32</td>
      <td>Zaragoza</td>
      <td>NaN</td>
      <td>2025-06-04</td>
      <td>2025-06-04</td>
      <td>NaT</td>
    </tr>
  </tbody>
</table>
<p>3182 rows × 8 columns</p>
</div>




```python
n_active_contracts.shape[0]/df_dim.shape[0]
```




    0.44968908988128886



Casi el 45% de los contratos están activos a día de hoy


```python
(
    df_dim
    .groupby('advertiser_province')["advertiser_zrive_id"]
    .count()
    .sort_values(ascending=False)
    .head(30)
).plot(kind='bar')
```




    <Axes: xlabel='advertiser_province'>




    
![png](EDA_jtc_files/EDA_jtc_41_1.png)
    


**Outliers**:
- Pocas provincias concentran muchísimos anunciantes 
- Gran mayoría de provincias tienen pocos anunciantes

Pregunta: ¿Debemos transformar esta variable para tratar los outliers?


```python
df_dim['contrato_churn_date'] - df_dim['min_start_contrato_date']
```




    0       130 days
    1        11 days
    2        99 days
    3       229 days
    4       240 days
              ...   
    7071    280 days
    7072   1003 days
    7073    482 days
    7074    994 days
    7075   1075 days
    Length: 7076, dtype: timedelta64[ns]




```python
df_dim['contrato_churn_date'] - df_dim['min_start_contrato_date']
```




    0       130 days
    1        11 days
    2        99 days
    3       229 days
    4       240 days
              ...   
    7071    280 days
    7072   1003 days
    7073    482 days
    7074    994 days
    7075   1075 days
    Length: 7076, dtype: timedelta64[ns]



Número de días que el anunciante permaneció en la plataforma


```python
df_dim['max_start_contrato_nuevo_date'] - df_dim['min_start_contrato_date']
```




    0        0 days
    1           NaT
    2           NaT
    3        0 days
    4           NaT
             ...   
    7071     0 days
    7072   852 days
    7073   323 days
    7074     0 days
    7075        NaT
    Length: 7076, dtype: timedelta64[ns]




```python
df_dim.iloc[7072]
```




    advertiser_zrive_id                              789
    province_id                                       48
    updated_at                       2025-04-30 13:40:11
    advertiser_province                          Vizcaya
    advertiser_group_id                              NaN
    min_start_contrato_date          2022-10-01 00:00:00
    max_start_contrato_nuevo_date    2025-01-30 00:00:00
    contrato_churn_date              2025-06-30 00:00:00
    Name: 7072, dtype: object



Por ejemplo este anunciante:
- Se dió de alta por primera vez el 2022-10-01
- Se volvió a dar de alta el 2025-01-30
- Se dió de baja el 2025-06-30 (Pregunta: ¿sabemos si se fué antes de que se le acabara el contrato? ¿hay que distinguir entre si se le acabó el contrato y no renovó y si abandonó antes de lo previsto? ¿O ambas casuísticas se tratan igual?)


```python
df_dim.iloc[7075]
```




    advertiser_zrive_id                             4194
    province_id                                       50
    updated_at                       2023-10-10 11:21:22
    advertiser_province                         Zaragoza
    advertiser_group_id                              NaN
    min_start_contrato_date          2022-07-21 00:00:00
    max_start_contrato_nuevo_date                    NaT
    contrato_churn_date              2025-06-30 00:00:00
    Name: 7075, dtype: object




```python
df_dim.dropna()
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
      <th>11</th>
      <td>5071</td>
      <td>5</td>
      <td>2024-05-28 12:02:26</td>
      <td>Asturias</td>
      <td>156.0</td>
      <td>2023-05-16</td>
      <td>2023-05-16</td>
      <td>2023-12-19</td>
    </tr>
    <tr>
      <th>18</th>
      <td>2422</td>
      <td>8</td>
      <td>2024-08-09 09:48:54</td>
      <td>Barcelona</td>
      <td>41.0</td>
      <td>2022-07-19</td>
      <td>2022-07-19</td>
      <td>2024-06-28</td>
    </tr>
    <tr>
      <th>27</th>
      <td>1178</td>
      <td>9</td>
      <td>2019-12-28 20:14:49</td>
      <td>Burgos</td>
      <td>120.0</td>
      <td>2020-02-05</td>
      <td>2020-02-05</td>
      <td>2024-12-26</td>
    </tr>
    <tr>
      <th>31</th>
      <td>2210</td>
      <td>12</td>
      <td>2024-11-14 16:06:14</td>
      <td>Cantabria</td>
      <td>130.0</td>
      <td>2023-06-02</td>
      <td>2023-06-02</td>
      <td>2024-11-12</td>
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
      <th>6896</th>
      <td>5479</td>
      <td>46</td>
      <td>2025-05-28 11:37:29</td>
      <td>Valencia</td>
      <td>154.0</td>
      <td>2023-11-13</td>
      <td>2023-11-13</td>
      <td>2025-04-30</td>
    </tr>
    <tr>
      <th>6936</th>
      <td>6100</td>
      <td>11</td>
      <td>2025-06-04 14:05:01</td>
      <td>Cádiz</td>
      <td>184.0</td>
      <td>2024-05-27</td>
      <td>2024-05-27</td>
      <td>2025-05-31</td>
    </tr>
    <tr>
      <th>6974</th>
      <td>6104</td>
      <td>29</td>
      <td>2025-06-04 14:11:10</td>
      <td>Madrid</td>
      <td>184.0</td>
      <td>2024-05-28</td>
      <td>2024-05-28</td>
      <td>2025-05-31</td>
    </tr>
    <tr>
      <th>6991</th>
      <td>5555</td>
      <td>30</td>
      <td>2025-06-19 13:22:46</td>
      <td>Málaga</td>
      <td>159.0</td>
      <td>2023-11-07</td>
      <td>2023-11-07</td>
      <td>2025-05-31</td>
    </tr>
    <tr>
      <th>7037</th>
      <td>310</td>
      <td>47</td>
      <td>2025-06-01 02:04:07</td>
      <td>Valladolid</td>
      <td>112.0</td>
      <td>2022-07-19</td>
      <td>2025-03-04</td>
      <td>2025-05-31</td>
    </tr>
  </tbody>
</table>
<p>289 rows × 8 columns</p>
</div>




```python

```

### FCT

Métricas mensuales relacionadas con el desempeño de los anunciantes.

#### Identificadores
- 🔑 **advertiser_zive_id**: Identificador único del anunciante.
- 🔑 **period_int**: Identificador del mes en formato `YYYYMM`.

#### Métricas de anuncios
- **monthly_contracted_ads**: Número total de anuncios contratados.
- **monthly_published_ads**: Número total de anuncios publicados.
- **monthly_unique_published_ads**: Número total de anuncios únicos publicados, en caso de publicar anuncios en diferentes provincias.
- **monthly_distinct_ads**: Número total de anuncios diferentes publicados en un mes (rotación total mensual).

#### Anuncios premium
- **monthly_oro_ads**: Número total de anuncios Oro contratados.
- **monthly_plata_ads**: Número total de anuncios Plata contratados.
- **monthly_destacados_ads**: Número total de anuncios Destacados contratados.
- **monthly_pepitas_ads**: Número total de anuncios Pepitas contratados.

#### Métricas de interacción
- **monthly_shows_ads**: Número total de búsquedas de anuncios publicados.
- **monthly_visits_ads**: Número total de visitas de anuncios publicados.
- **monthly_leads**: Número total de leads de anuncios publicados.
- **monthly_total_phone_views**: Número total de vistas del teléfono del anunciante de anuncios publicados.
- **monthly_total_calls**: Número total de llamadas al anunciante de anuncios publicados.
- **monthly_total_emails**: Número total de emails al anunciante de anuncios publicados.

#### Métricas económicas
- **monthly_total_invoice**: Facturación mensual.
- **monthly_total_reference_price**: Precio tarifa mensual.
- **monthly_avg_ad_price**: Precio medio de los anuncios publicados.

#### Métricas únicas
- **monthly_unique_calls**: Número de llamadas al anunciante de anuncios únicos publicados.
- **monthly_unique_emails**: Número de emails al anunciante de anuncios únicos publicados.
- **monthly_unique_leads**: Número de leads de anuncios únicos publicados.

#### Estado del contrato
- **has_active_contract**: Indica si el anunciante tiene una propuesta activa en el mes.




```python
df_fct = pd.read_parquet(fct_local_path, engine='fastparquet')
```


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
      <td>202301</td>
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
      <td>202301</td>
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
      <td>202301</td>
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
      <td>202301</td>
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
      <td>202301</td>
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
try:
    df_fct["period_int"] = pd.to_datetime(df_fct['period_int'].astype(str), format="%Y%m")
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
df_fct.isnull().sum()
```




    advertiser_zrive_id                  0
    period_int                           0
    monthly_contracted_ads               0
    monthly_published_ads                0
    monthly_unique_published_ads         0
    monthly_distinct_ads             10052
    monthly_oro_ads                      0
    monthly_plata_ads                    0
    monthly_destacados_ads               0
    monthly_pepitas_ads                  0
    monthly_shows                        0
    monthly_visits                       0
    monthly_leads                        0
    monthly_total_phone_views            0
    monthly_total_calls                  0
    monthly_total_emails                 0
    monthly_total_invoice                0
    monthly_total_reference_price    12330
    monthly_unique_calls                 0
    monthly_unique_emails                0
    monthly_unique_leads                 0
    monthly_avg_ad_price              8467
    has_active_contract                  0
    dtype: int64




```python
print_unique(df_fct)
```


```python
df_fct.head(10)
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
    <tr>
      <th>5</th>
      <td>7</td>
      <td>2023-01-01</td>
      <td>50</td>
      <td>42</td>
      <td>42</td>
      <td>49.0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>18</td>
      <td>5</td>
      <td>3</td>
      <td>763.3</td>
      <td>1006.7</td>
      <td>4</td>
      <td>7</td>
      <td>11</td>
      <td>19495.74</td>
      <td>True</td>
    </tr>
    <tr>
      <th>6</th>
      <td>10</td>
      <td>2023-01-01</td>
      <td>49</td>
      <td>37</td>
      <td>37</td>
      <td>56.0</td>
      <td>11</td>
      <td>5</td>
      <td>8</td>
      <td>0</td>
      <td>...</td>
      <td>8</td>
      <td>3</td>
      <td>0</td>
      <td>75.0</td>
      <td>1645.0</td>
      <td>3</td>
      <td>3</td>
      <td>6</td>
      <td>37954.00</td>
      <td>True</td>
    </tr>
    <tr>
      <th>7</th>
      <td>11</td>
      <td>2023-01-01</td>
      <td>85</td>
      <td>6</td>
      <td>6</td>
      <td>7.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>150.2</td>
      <td>NaN</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>45955.71</td>
      <td>True</td>
    </tr>
    <tr>
      <th>8</th>
      <td>12</td>
      <td>2023-01-01</td>
      <td>125</td>
      <td>102</td>
      <td>102</td>
      <td>148.0</td>
      <td>21</td>
      <td>21</td>
      <td>8</td>
      <td>0</td>
      <td>...</td>
      <td>17</td>
      <td>7</td>
      <td>1</td>
      <td>374.9</td>
      <td>6855.0</td>
      <td>5</td>
      <td>3</td>
      <td>8</td>
      <td>43456.25</td>
      <td>True</td>
    </tr>
    <tr>
      <th>9</th>
      <td>15</td>
      <td>2023-01-01</td>
      <td>150</td>
      <td>112</td>
      <td>112</td>
      <td>328.0</td>
      <td>10</td>
      <td>10</td>
      <td>4</td>
      <td>0</td>
      <td>...</td>
      <td>24</td>
      <td>13</td>
      <td>5</td>
      <td>1101.7</td>
      <td>NaN</td>
      <td>10</td>
      <td>9</td>
      <td>19</td>
      <td>16965.04</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
<p>10 rows × 23 columns</p>
</div>




```python
df_fct["advertiser_zrive_id"].nunique()
```




    6968



En la base de datos tenemos información de **6968** anunciantes distintos


```python
df_dim["advertiser_zrive_id"].nunique()
```




    7076




```python
ids_df1 = set(df_fct['advertiser_zrive_id'])
ids_df2 = set(df_dim['advertiser_zrive_id'])

solo_en_df1 = ids_df1 - ids_df2
solo_en_df2 = ids_df2 - ids_df1

df_dim[df_dim['advertiser_zrive_id'].isin(solo_en_df2)]

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
      <th>12</th>
      <td>3388</td>
      <td>5</td>
      <td>2022-10-28 13:28:56</td>
      <td>Asturias</td>
      <td>NaN</td>
      <td>2024-03-11</td>
      <td>NaT</td>
      <td>2024-03-15</td>
    </tr>
    <tr>
      <th>16</th>
      <td>4095</td>
      <td>8</td>
      <td>2022-10-07 09:43:18</td>
      <td>Barcelona</td>
      <td>NaN</td>
      <td>2022-07-01</td>
      <td>NaT</td>
      <td>2024-06-27</td>
    </tr>
    <tr>
      <th>27</th>
      <td>1178</td>
      <td>9</td>
      <td>2019-12-28 20:14:49</td>
      <td>Burgos</td>
      <td>120.0</td>
      <td>2020-02-05</td>
      <td>2020-02-05</td>
      <td>2024-12-26</td>
    </tr>
    <tr>
      <th>201</th>
      <td>7070</td>
      <td>3</td>
      <td>2025-06-11 15:42:23</td>
      <td>Alicante</td>
      <td>NaN</td>
      <td>2025-06-11</td>
      <td>2025-06-11</td>
      <td>NaT</td>
    </tr>
    <tr>
      <th>240</th>
      <td>6918</td>
      <td>3</td>
      <td>2025-06-18 13:01:44</td>
      <td>Alicante</td>
      <td>NaN</td>
      <td>2025-06-11</td>
      <td>2025-06-11</td>
      <td>NaT</td>
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
      <th>6633</th>
      <td>1800</td>
      <td>9</td>
      <td>2019-12-28 19:59:58</td>
      <td>Burgos</td>
      <td>120.0</td>
      <td>2024-07-29</td>
      <td>2024-07-29</td>
      <td>2025-03-31</td>
    </tr>
    <tr>
      <th>6642</th>
      <td>2662</td>
      <td>12</td>
      <td>2021-01-21 12:17:37</td>
      <td>Cantabria</td>
      <td>NaN</td>
      <td>2024-06-07</td>
      <td>2024-06-07</td>
      <td>2025-03-31</td>
    </tr>
    <tr>
      <th>6708</th>
      <td>545</td>
      <td>30</td>
      <td>2021-01-21 12:21:40</td>
      <td>Málaga</td>
      <td>NaN</td>
      <td>2024-06-03</td>
      <td>2024-06-03</td>
      <td>2025-03-31</td>
    </tr>
    <tr>
      <th>7009</th>
      <td>2351</td>
      <td>40</td>
      <td>2022-10-28 10:07:04</td>
      <td>Sevilla</td>
      <td>NaN</td>
      <td>2020-11-11</td>
      <td>2020-11-11</td>
      <td>2025-05-31</td>
    </tr>
    <tr>
      <th>7016</th>
      <td>1626</td>
      <td>40</td>
      <td>2024-08-09 13:54:39</td>
      <td>Sevilla</td>
      <td>100.0</td>
      <td>2021-05-01</td>
      <td>NaT</td>
      <td>2025-05-31</td>
    </tr>
  </tbody>
</table>
<p>108 rows × 8 columns</p>
</div>




```python
df_dim.advertiser_group_id.nunique()
```




    112



Hay 112 grupos de anunciantes


```python
advertiser_groups = (
    df_dim
    .groupby("advertiser_group_id")["advertiser_zrive_id"]
    .nunique()
    .sort_values(ascending=False)
)
advertiser_groups.head(10).plot(kind='bar', title="Top 10 grupos con más anunciantes")
```




    <Axes: title={'center': 'Top 10 grupos con más anunciantes'}, xlabel='advertiser_group_id'>




    
![png](EDA_jtc_files/EDA_jtc_66_1.png)
    



```python
advertiser_groups.describe()
```




    count    112.000000
    mean      11.812500
    std       31.438924
    min        1.000000
    25%        2.000000
    50%        4.000000
    75%       11.000000
    max      257.000000
    Name: advertiser_zrive_id, dtype: float64




```python
advertiser_groups.plot(kind='box', vert=False)
```




    <Axes: >




    
![png](EDA_jtc_files/EDA_jtc_68_1.png)
    


#### Merge 

Hacemos un left join en **Zrive_fct_monthly_snapshot_advertiser (df_fct)** para tener la foto completa con la información de **Zrive_dim_advertiser (df_dim)**


```python
df_fct_merged = df_fct.merge(right=df_dim, how='left', on='advertiser_zrive_id')
```


```python
# Proporción de anunciantes pertenecientes a un grupo y anunciantes individuales

advertiser_group_proportion = (
    df_fct_merged
    .groupby("advertiser_zrive_id")["advertiser_group_id"]
    .nunique()
)

advertiser_group_proportion.value_counts(normalize=True).plot(kind='bar', title="Usuarios con grupo vs. individuales")
```




    <Axes: title={'center': 'Usuarios con grupo vs. individuales'}, xlabel='advertiser_group_id'>




    
![png](EDA_jtc_files/EDA_jtc_71_1.png)
    



```python
print(f"% advertisers pertenecientes a grupos: {advertiser_group_proportion.value_counts()[0]/advertiser_group_proportion.shape[0]:.4f}")
print(f"% advertisers pertenecientes a grupos: {advertiser_group_proportion.value_counts()[1]/advertiser_group_proportion.shape[0]:.4f}")
```

    % advertisers pertenecientes a grupos: 0.8117
    % advertisers pertenecientes a grupos: 0.1883



```python
advertiser_group_proportion.value_counts()[0]
```




    np.int64(5656)




```python
print(df_fct_merged["period_int"].min())
print(df_fct_merged["period_int"].max())
```

    2023-01-01 00:00:00
    2025-05-01 00:00:00


El período del que se tienen datos va desde el 2023-01 hasta 2025-05


```python
df_fct_merged[["period_int", "advertiser_zrive_id"]].isna().sum()
```




    period_int             0
    advertiser_zrive_id    0
    dtype: int64




```python
df_fct_merged.head()
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
      <th>monthly_unique_leads</th>
      <th>monthly_avg_ad_price</th>
      <th>has_active_contract</th>
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
      <td>15</td>
      <td>42667.86</td>
      <td>True</td>
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
      <td>4</td>
      <td>22742.40</td>
      <td>True</td>
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
      <td>0</td>
      <td>25420.83</td>
      <td>False</td>
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
      <td>12</td>
      <td>23259.52</td>
      <td>True</td>
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
      <td>15</td>
      <td>43439.00</td>
      <td>True</td>
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
<p>5 rows × 30 columns</p>
</div>




```python

```


```python

```


```python
df_fct_merged[df_fct_merged["advertiser_zrive_id"] == 1]

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
      <th>monthly_unique_leads</th>
      <th>monthly_avg_ad_price</th>
      <th>has_active_contract</th>
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
      <td>15</td>
      <td>42667.86</td>
      <td>True</td>
      <td>11</td>
      <td>2023-06-16 09:35:01</td>
      <td>Cádiz</td>
      <td>NaN</td>
      <td>2020-12-23</td>
      <td>NaT</td>
      <td>2023-05-31</td>
    </tr>
    <tr>
      <th>3542</th>
      <td>1</td>
      <td>2023-02-01</td>
      <td>50</td>
      <td>44</td>
      <td>44</td>
      <td>48.0</td>
      <td>3</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>9</td>
      <td>42691.67</td>
      <td>True</td>
      <td>11</td>
      <td>2023-06-16 09:35:01</td>
      <td>Cádiz</td>
      <td>NaN</td>
      <td>2020-12-23</td>
      <td>NaT</td>
      <td>2023-05-31</td>
    </tr>
    <tr>
      <th>6905</th>
      <td>1</td>
      <td>2023-03-01</td>
      <td>50</td>
      <td>39</td>
      <td>39</td>
      <td>48.0</td>
      <td>3</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>6</td>
      <td>44193.75</td>
      <td>True</td>
      <td>11</td>
      <td>2023-06-16 09:35:01</td>
      <td>Cádiz</td>
      <td>NaN</td>
      <td>2020-12-23</td>
      <td>NaT</td>
      <td>2023-05-31</td>
    </tr>
    <tr>
      <th>10328</th>
      <td>1</td>
      <td>2023-04-01</td>
      <td>50</td>
      <td>39</td>
      <td>39</td>
      <td>40.0</td>
      <td>3</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>6</td>
      <td>44452.50</td>
      <td>True</td>
      <td>11</td>
      <td>2023-06-16 09:35:01</td>
      <td>Cádiz</td>
      <td>NaN</td>
      <td>2020-12-23</td>
      <td>NaT</td>
      <td>2023-05-31</td>
    </tr>
    <tr>
      <th>13747</th>
      <td>1</td>
      <td>2023-05-01</td>
      <td>150</td>
      <td>0</td>
      <td>0</td>
      <td>39.0</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>8</td>
      <td>43033.33</td>
      <td>True</td>
      <td>11</td>
      <td>2023-06-16 09:35:01</td>
      <td>Cádiz</td>
      <td>NaN</td>
      <td>2020-12-23</td>
      <td>NaT</td>
      <td>2023-05-31</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 30 columns</p>
</div>



#### Merge completo

Se hace un **LEFT JOIN** para tener la foto completa

- df_fct
- df_dim
- df_wd


```python
df_full = df_fct_merged.merge(right=df_wd, how='left', on='advertiser_zrive_id')
```


```python
df_full
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
      <th>min_start_contrato_date</th>
      <th>max_start_contrato_nuevo_date</th>
      <th>contrato_churn_date</th>
      <th>withdrawal_id</th>
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
      <td>2020-12-23</td>
      <td>NaT</td>
      <td>2023-05-31</td>
      <td>6733.0</td>
      <td>Cerrada</td>
      <td>TOTAL</td>
      <td>2015-12-10</td>
      <td>NaT</td>
      <td>FALTA DE USO/TIEMPO</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1</th>
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
      <td>2020-12-23</td>
      <td>NaT</td>
      <td>2023-05-31</td>
      <td>22987.0</td>
      <td>Cerrada</td>
      <td>PARCIAL</td>
      <td>2018-09-27</td>
      <td>2018-10-01</td>
      <td>FALTA DE PRODUCTO</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
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
      <td>2020-12-23</td>
      <td>NaT</td>
      <td>2023-05-31</td>
      <td>23384.0</td>
      <td>Cerrada</td>
      <td>PARCIAL</td>
      <td>2018-12-03</td>
      <td>2018-12-01</td>
      <td>RAZONES ECONOMICAS</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
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
      <td>2020-12-23</td>
      <td>NaT</td>
      <td>2023-05-31</td>
      <td>33451.0</td>
      <td>Cerrada</td>
      <td>TOTAL</td>
      <td>2020-12-23</td>
      <td>2020-12-28</td>
      <td>Cambio a Bundle Online</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
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
      <td>2020-12-23</td>
      <td>NaT</td>
      <td>2023-05-31</td>
      <td>35720.0</td>
      <td>Cerrada</td>
      <td>TOTAL</td>
      <td>2021-07-09</td>
      <td>2021-07-30</td>
      <td>Cambio de Contrato/propuesta/producto</td>
      <td>0.0</td>
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
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
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
      <th>407656</th>
      <td>7047</td>
      <td>2025-05-01</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>...</td>
      <td>2025-05-28</td>
      <td>2025-05-28</td>
      <td>NaT</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaT</td>
      <td>NaT</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>407657</th>
      <td>7051</td>
      <td>2025-05-01</td>
      <td>20</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>2025-05-29</td>
      <td>2025-05-29</td>
      <td>NaT</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaT</td>
      <td>NaT</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>407658</th>
      <td>7052</td>
      <td>2025-05-01</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>...</td>
      <td>2025-05-29</td>
      <td>2025-05-29</td>
      <td>NaT</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaT</td>
      <td>NaT</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>407659</th>
      <td>7066</td>
      <td>2025-05-01</td>
      <td>20</td>
      <td>20</td>
      <td>20</td>
      <td>29.0</td>
      <td>3</td>
      <td>3</td>
      <td>7</td>
      <td>0</td>
      <td>...</td>
      <td>2025-06-10</td>
      <td>NaT</td>
      <td>NaT</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaT</td>
      <td>NaT</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>407660</th>
      <td>7069</td>
      <td>2025-05-01</td>
      <td>10</td>
      <td>5</td>
      <td>5</td>
      <td>6.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>2025-06-12</td>
      <td>NaT</td>
      <td>NaT</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaT</td>
      <td>NaT</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>407661 rows × 37 columns</p>
</div>




```python

print(df_full[df_full["advertiser_zrive_id"] == 1]["withdrawal_id"].nunique())
print(df_full[df_full["advertiser_zrive_id"] == 1]["period_int"].nunique())

```

    7
    5


Comportamiento usuarios: usuario(advertiser_zrive_id) = 1:
- Solicitó una baja **7** veces
- Estuvo en la plataforma durante **5 meses**


```python
df_full[df_full["advertiser_zrive_id"] == 1]["min_start_contrato_date"].max()
```




    Timestamp('2020-12-23 00:00:00')




```python
(
    df_full[df_full["advertiser_zrive_id"] == 1]
    ["min_start_contrato_date"]
    .nunique()
)
```




    1




```python
(
    df_full
    .groupby("advertiser_zrive_id")["max_start_contrato_nuevo_date"]
    .nunique()
    .eq(1)
    .sum()
)
```




    np.int64(5123)



Hay **5123** advertisers que se dieron de baja y volvieron al menos 1 vez


```python
(
    df_full
    .groupby("advertiser_zrive_id")["min_start_contrato_date"]
    .nunique()
    .eq(1)
    .sum()
)
```




    np.int64(6968)



Todos los advertisers tienen 1 fecha de contrato


```python
(
    df_full
    .groupby("advertiser_zrive_id")["period_int"]
    .nunique()
    .lt(3)
)
```




    advertiser_zrive_id
    1       False
    2       False
    3       False
    4       False
    5       False
            ...  
    7047     True
    7051     True
    7052     True
    7066    False
    7069     True
    Name: period_int, Length: 6968, dtype: bool



Sólo **571 advertisers** permanecieron menos de 3 meses en la plataforma


```python
mask = (
    df_full
    .groupby("advertiser_zrive_id")["period_int"]
    .nunique()
    .lt(3)
)

df_full[df_full["advertiser_zrive_id"].map(mask)]

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
      <th>min_start_contrato_date</th>
      <th>max_start_contrato_nuevo_date</th>
      <th>contrato_churn_date</th>
      <th>withdrawal_id</th>
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
      <th>241</th>
      <td>46</td>
      <td>2023-01-01</td>
      <td>100</td>
      <td>100</td>
      <td>100</td>
      <td>119.0</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>0</td>
      <td>...</td>
      <td>2021-04-23</td>
      <td>NaT</td>
      <td>2023-01-31</td>
      <td>35197.0</td>
      <td>Cerrada</td>
      <td>TOTAL</td>
      <td>2021-05-20</td>
      <td>2021-05-20</td>
      <td>OTROS</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>242</th>
      <td>46</td>
      <td>2023-01-01</td>
      <td>100</td>
      <td>100</td>
      <td>100</td>
      <td>119.0</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>0</td>
      <td>...</td>
      <td>2021-04-23</td>
      <td>NaT</td>
      <td>2023-01-31</td>
      <td>42271.0</td>
      <td>Cerrada</td>
      <td>TOTAL</td>
      <td>2023-01-31</td>
      <td>2023-02-01</td>
      <td>RESULTADOS</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>798</th>
      <td>142</td>
      <td>2023-01-01</td>
      <td>35</td>
      <td>35</td>
      <td>35</td>
      <td>54.0</td>
      <td>3</td>
      <td>2</td>
      <td>2</td>
      <td>0</td>
      <td>...</td>
      <td>2022-07-22</td>
      <td>2022-07-22</td>
      <td>2023-01-31</td>
      <td>5728.0</td>
      <td>Cerrada</td>
      <td>PARCIAL</td>
      <td>2015-06-30</td>
      <td>2015-07-01</td>
      <td>FIN DE CONTRATO</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>799</th>
      <td>142</td>
      <td>2023-01-01</td>
      <td>35</td>
      <td>35</td>
      <td>35</td>
      <td>54.0</td>
      <td>3</td>
      <td>2</td>
      <td>2</td>
      <td>0</td>
      <td>...</td>
      <td>2022-07-22</td>
      <td>2022-07-22</td>
      <td>2023-01-31</td>
      <td>6864.0</td>
      <td>Cerrada</td>
      <td>TOTAL</td>
      <td>2016-01-04</td>
      <td>2016-01-01</td>
      <td>FALTA DE USO/TIEMPO</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>800</th>
      <td>142</td>
      <td>2023-01-01</td>
      <td>35</td>
      <td>35</td>
      <td>35</td>
      <td>54.0</td>
      <td>3</td>
      <td>2</td>
      <td>2</td>
      <td>0</td>
      <td>...</td>
      <td>2022-07-22</td>
      <td>2022-07-22</td>
      <td>2023-01-31</td>
      <td>24953.0</td>
      <td>Cerrada</td>
      <td>TOTAL</td>
      <td>2019-05-06</td>
      <td>2019-06-01</td>
      <td>RESULTADOS</td>
      <td>1.0</td>
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
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
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
      <th>407654</th>
      <td>7045</td>
      <td>2025-05-01</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>...</td>
      <td>2025-05-29</td>
      <td>2025-05-29</td>
      <td>NaT</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaT</td>
      <td>NaT</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>407656</th>
      <td>7047</td>
      <td>2025-05-01</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>...</td>
      <td>2025-05-28</td>
      <td>2025-05-28</td>
      <td>NaT</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaT</td>
      <td>NaT</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>407657</th>
      <td>7051</td>
      <td>2025-05-01</td>
      <td>20</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>2025-05-29</td>
      <td>2025-05-29</td>
      <td>NaT</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaT</td>
      <td>NaT</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>407658</th>
      <td>7052</td>
      <td>2025-05-01</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>...</td>
      <td>2025-05-29</td>
      <td>2025-05-29</td>
      <td>NaT</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaT</td>
      <td>NaT</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>407660</th>
      <td>7069</td>
      <td>2025-05-01</td>
      <td>10</td>
      <td>5</td>
      <td>5</td>
      <td>6.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>2025-06-12</td>
      <td>NaT</td>
      <td>NaT</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaT</td>
      <td>NaT</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>1633 rows × 37 columns</p>
</div>




```python
df_full[df_full["advertiser_zrive_id"] == 1].groupby("withdrawal_id")["withdrawal_reason"].nunique()
```




    withdrawal_id
    6733.0     1
    22987.0    1
    23384.0    1
    33451.0    1
    35720.0    1
    40999.0    1
    43036.0    1
    Name: withdrawal_reason, dtype: int64




```python
df_full.head(5)
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
      <th>min_start_contrato_date</th>
      <th>max_start_contrato_nuevo_date</th>
      <th>contrato_churn_date</th>
      <th>withdrawal_id</th>
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
      <td>2020-12-23</td>
      <td>NaT</td>
      <td>2023-05-31</td>
      <td>6733.0</td>
      <td>Cerrada</td>
      <td>TOTAL</td>
      <td>2015-12-10</td>
      <td>NaT</td>
      <td>FALTA DE USO/TIEMPO</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1</th>
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
      <td>2020-12-23</td>
      <td>NaT</td>
      <td>2023-05-31</td>
      <td>22987.0</td>
      <td>Cerrada</td>
      <td>PARCIAL</td>
      <td>2018-09-27</td>
      <td>2018-10-01</td>
      <td>FALTA DE PRODUCTO</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
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
      <td>2020-12-23</td>
      <td>NaT</td>
      <td>2023-05-31</td>
      <td>23384.0</td>
      <td>Cerrada</td>
      <td>PARCIAL</td>
      <td>2018-12-03</td>
      <td>2018-12-01</td>
      <td>RAZONES ECONOMICAS</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
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
      <td>2020-12-23</td>
      <td>NaT</td>
      <td>2023-05-31</td>
      <td>33451.0</td>
      <td>Cerrada</td>
      <td>TOTAL</td>
      <td>2020-12-23</td>
      <td>2020-12-28</td>
      <td>Cambio a Bundle Online</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
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
      <td>2020-12-23</td>
      <td>NaT</td>
      <td>2023-05-31</td>
      <td>35720.0</td>
      <td>Cerrada</td>
      <td>TOTAL</td>
      <td>2021-07-09</td>
      <td>2021-07-30</td>
      <td>Cambio de Contrato/propuesta/producto</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 37 columns</p>
</div>



### Análisis permanencia por anunciante


```python
df_months = (
    df_full
    .groupby("advertiser_zrive_id")["period_int"]
    .nunique()
    .reset_index()
    .rename(columns={"period_int":"n_months"})
)

print(df_months["n_months"].describe())
```

    count    6968.000000
    mean       13.896240
    std         9.970446
    min         1.000000
    25%         5.000000
    50%        11.000000
    75%        25.000000
    max        29.000000
    Name: n_months, dtype: float64



```python
sns.histplot(df_months, x="n_months", bins=30, kde=True)

plt.title("Nº de meses por anunciante")
plt.xlabel("Nº meses")
plt.ylabel("Anunciantes")
```




    Text(0, 0.5, 'Anunciantes')




    
![png](EDA_jtc_files/EDA_jtc_99_1.png)
    



```python
sns.kdeplot(df_months["n_months"])
plt.title("Distribución de n_months")
```




    Text(0.5, 1.0, 'Distribución de n_months')




    
![png](EDA_jtc_files/EDA_jtc_100_1.png)
    



```python
plt.figure(figsize=(8,3))

ax = sns.boxplot(
    data=df_months,
    x="n_months",
    width=0.4,
    showfliers=False
)

ax.set_title("Distribución de n_months")
ax.set_xlabel("Valor")
plt.tight_layout()
```


    
![png](EDA_jtc_files/EDA_jtc_101_0.png)
    



```python
sns.kdeplot(df_months[df_months["n_months"] < 29]["n_months"])
plt.title("Distribución de n_months < 29")
```




    Text(0.5, 1.0, 'Distribución de n_months < 29')




    
![png](EDA_jtc_files/EDA_jtc_102_1.png)
    



```python
df_months[df_months["n_months"] == 29].count()
```




    advertiser_zrive_id    1387
    n_months               1387
    dtype: int64




```python
long_term_adv_pct = len(df_months[df_months["n_months"] == 29])/len(df_months)*100
round(long_term_adv_pct,2)
```




    19.91



De aquí podemos sacar insights:
- Los anunciantes permanecen en la plataforma una media de 13.9 meses
- Ditribución bimodal; dos poblaciones: 
    - Anunciantes corto-medio plazo, muchos se concentran a la izquierda, permaneciendo pocos meses activos
    - Anunciantes largo plazo: un total de **1387** anunciantes (casi un 20%) permanecieron el periodo completo (29 meses)
- Si quitamos anunciantes que permanecieron 29 meses, obtenemos la gráfica anterior


```python

```

### Análisis **estabilidad**


```python
df_advertiser = (
    df_full.groupby("advertiser_zrive_id")["period_int"]
    .nunique()
    .reset_index()
    .assign(is_stable=lambda df: (df["period_int"] >= 3).astype(int))
)
df_advertiser.is_stable.value_counts(normalize=True).plot(kind='bar')
plt.title("Proporción de clientes estables vs. no estables")

print(df_advertiser.is_stable.describe())
```

    count    6968.000000
    mean        0.918054
    std         0.274302
    min         0.000000
    25%         1.000000
    50%         1.000000
    75%         1.000000
    max         1.000000
    Name: is_stable, dtype: float64



    
![png](EDA_jtc_files/EDA_jtc_108_1.png)
    


Si definimos un cliente **no estable** como un cliente que permaneció en la plataforma **menos de 3 meses**, obtenemos esta gráfica que representa la distribución de clientes por estabilidad.
Observamos que está bastante desbalanceada, es decir, la mayoría de advertisers permanecieron más de 3 meses

### Análisis de los clientes **No Estables**


```python
unstable_ids = df_advertiser.loc[df_advertiser["is_stable"] == 0, "advertiser_zrive_id"]
```


```python
df_unstable = df_full[df_full["advertiser_zrive_id"].isin(unstable_ids)]
# df_unstable.head(20)
```

Ejemplo: usuario 142
- No estable: period_int únicos = 1 (solo permaneció 1 mes)
- Tiene 6 withdrawal_id diferentes: solicitó la baja en **6 ocasiones**
- Se dió de baja definitivamente ("is_definite_withdrawal") **5 veces**


## **LATEST approach**

Sabemos que un usuario es churn si:
1. withdrawal_type = 'TOTAL'
2. withdrawal_status != 'Denegada'
3. withdrawal_reason not in ('Upselling-cambio de contrato', 'Cambio a Bundle Online',
'Cambio de Contrato/propuesta/producto')

Por otro lado, de la tabla **zrive_dim_advertiser** tenemos los siguientes datos:
- **min_start_contrato_date**: primera: fecha inicio primer contrato (nuevo contrato)
- **max_start_contrato_nuevo_date**: última fecha de inicio de nuevo contrato
- **contrato_churn_date**: fecha fin de contrato

Según estas definiciones:

**duración del contrato** = end_date - max(max_start_contrato_nuevo_date, min_start_contrato_date)

**end_date** se define más abajo mediante una lógica


Para este approach, vamos a analizar el dataset de información general de los advertisers en la plataforma (df_dim) junto con el comportamiento de withdrawals (df_wd)


```python
dim_ids = set(df_dim.advertiser_zrive_id.unique())
wd_ids = set(df_wd.advertiser_zrive_id.unique())
dim_surplus = dim_ids - wd_ids
len(dim_surplus)
```




    1055



Hay 1055 advertisers de los que no se tiene información sobre withdrawals. Esto es, de los **7076** advertisers, sólo hay información de withdrawal de **6021** de ellos.

Hacemos un **LEFT JOIN** entre las dos tablas, y observamos los registros que no tienen información sobre **withdrawals**


```python
df_dim_wd = df_dim.merge(right=df_wd, how='left')
# df_dim_wd[df_dim_wd["withdrawal_id"].isna() & (df_dim_wd["is_active"] == 1)]
```

Ahora, vamos a crear una nueva columna donde calculamos el tiempo de permanencia en la plataforma (meses). Para ello vamos a realizar la siguiente operación:

$$
T^\circ_{\text{permanencia}} =
\text{end\_date}
-
\max(\text{max\_start\_contrato\_nuevo\_date},
     \text{min\_start\_contrato\_date})
$$

### Fecha de alta: **start_date**

Será la fecha más reciente del comienzo de un contrato, es decir:

    máx("max_start_contrato_nuevo_date", "min_start_contrato_date")


```python
# start_date = df_dim_wd[["max_start_contrato_nuevo_date", "min_start_contrato_date"]].max(axis=1)

df_dim_wd["start_date"] = df_dim_wd[["max_start_contrato_nuevo_date", "min_start_contrato_date"]].max(axis=1)
df_dim_wd["start_date"] = pd.to_datetime(df_dim_wd["start_date"])

df_dim_wd["snapshot_date"] = df_dim_wd["updated_at"]
df_dim_wd["snapshot_date"] = pd.to_datetime(df_dim_wd["snapshot_date"]).dt.date
            
df_dim_wd 
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
      <th>withdrawal_id</th>
      <th>withdrawal_status</th>
      <th>withdrawal_type</th>
      <th>withdrawal_creation_date</th>
      <th>withdrawal_effective_date</th>
      <th>withdrawal_reason</th>
      <th>is_definitive_withdrawal</th>
      <th>start_date</th>
      <th>snapshot_date</th>
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
      <td>4758.0</td>
      <td>Cerrada</td>
      <td>TOTAL</td>
      <td>2015-01-16</td>
      <td>2015-03-01</td>
      <td>RATIO RESULTADO-INVERSION</td>
      <td>1.0</td>
      <td>2025-01-07</td>
      <td>2025-05-01</td>
    </tr>
    <tr>
      <th>1</th>
      <td>986</td>
      <td>1</td>
      <td>2025-05-01 02:46:04</td>
      <td>Álava</td>
      <td>194.0</td>
      <td>2025-01-07</td>
      <td>2025-01-07</td>
      <td>2025-05-17</td>
      <td>23029.0</td>
      <td>Cerrada</td>
      <td>TOTAL</td>
      <td>2018-10-08</td>
      <td>2018-11-01</td>
      <td>RESULTADOS</td>
      <td>1.0</td>
      <td>2025-01-07</td>
      <td>2025-05-01</td>
    </tr>
    <tr>
      <th>2</th>
      <td>986</td>
      <td>1</td>
      <td>2025-05-01 02:46:04</td>
      <td>Álava</td>
      <td>194.0</td>
      <td>2025-01-07</td>
      <td>2025-01-07</td>
      <td>2025-05-17</td>
      <td>32827.0</td>
      <td>Cerrada</td>
      <td>TOTAL</td>
      <td>2020-11-17</td>
      <td>2020-11-20</td>
      <td>Cambio a Bundle Online</td>
      <td>0.0</td>
      <td>2025-01-07</td>
      <td>2025-05-01</td>
    </tr>
    <tr>
      <th>3</th>
      <td>986</td>
      <td>1</td>
      <td>2025-05-01 02:46:04</td>
      <td>Álava</td>
      <td>194.0</td>
      <td>2025-01-07</td>
      <td>2025-01-07</td>
      <td>2025-05-17</td>
      <td>33518.0</td>
      <td>Cerrada</td>
      <td>TOTAL</td>
      <td>2021-01-05</td>
      <td>2021-02-01</td>
      <td>RATIO RESULTADO-INVERSION</td>
      <td>1.0</td>
      <td>2025-01-07</td>
      <td>2025-05-01</td>
    </tr>
    <tr>
      <th>4</th>
      <td>986</td>
      <td>1</td>
      <td>2025-05-01 02:46:04</td>
      <td>Álava</td>
      <td>194.0</td>
      <td>2025-01-07</td>
      <td>2025-01-07</td>
      <td>2025-05-17</td>
      <td>54519.0</td>
      <td>Cerrada</td>
      <td>TOTAL</td>
      <td>2025-03-18</td>
      <td>2025-05-01</td>
      <td>RATIO RESULTADO-INVERSION</td>
      <td>1.0</td>
      <td>2025-01-07</td>
      <td>2025-05-01</td>
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
      <td>...</td>
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
      <th>23718</th>
      <td>4194</td>
      <td>50</td>
      <td>2023-10-10 11:21:22</td>
      <td>Zaragoza</td>
      <td>NaN</td>
      <td>2022-07-21</td>
      <td>NaT</td>
      <td>2025-06-30</td>
      <td>46264.0</td>
      <td>Cerrada</td>
      <td>TOTAL</td>
      <td>2023-10-09</td>
      <td>2023-10-09</td>
      <td>Cambio de Contrato/propuesta/producto</td>
      <td>0.0</td>
      <td>2022-07-21</td>
      <td>2023-10-10</td>
    </tr>
    <tr>
      <th>23719</th>
      <td>4194</td>
      <td>50</td>
      <td>2023-10-10 11:21:22</td>
      <td>Zaragoza</td>
      <td>NaN</td>
      <td>2022-07-21</td>
      <td>NaT</td>
      <td>2025-06-30</td>
      <td>47376.0</td>
      <td>Cerrada</td>
      <td>TOTAL</td>
      <td>2023-12-21</td>
      <td>NaT</td>
      <td>FALTA DE USO/TIEMPO</td>
      <td>1.0</td>
      <td>2022-07-21</td>
      <td>2023-10-10</td>
    </tr>
    <tr>
      <th>23720</th>
      <td>4194</td>
      <td>50</td>
      <td>2023-10-10 11:21:22</td>
      <td>Zaragoza</td>
      <td>NaN</td>
      <td>2022-07-21</td>
      <td>NaT</td>
      <td>2025-06-30</td>
      <td>48614.0</td>
      <td>Cerrada</td>
      <td>TOTAL</td>
      <td>2024-03-12</td>
      <td>NaT</td>
      <td>RAZONES ECONOMICAS</td>
      <td>1.0</td>
      <td>2022-07-21</td>
      <td>2023-10-10</td>
    </tr>
    <tr>
      <th>23721</th>
      <td>4194</td>
      <td>50</td>
      <td>2023-10-10 11:21:22</td>
      <td>Zaragoza</td>
      <td>NaN</td>
      <td>2022-07-21</td>
      <td>NaT</td>
      <td>2025-06-30</td>
      <td>51090.0</td>
      <td>Cerrada</td>
      <td>TOTAL</td>
      <td>2024-08-29</td>
      <td>NaT</td>
      <td>RESULTADOS</td>
      <td>1.0</td>
      <td>2022-07-21</td>
      <td>2023-10-10</td>
    </tr>
    <tr>
      <th>23722</th>
      <td>4194</td>
      <td>50</td>
      <td>2023-10-10 11:21:22</td>
      <td>Zaragoza</td>
      <td>NaN</td>
      <td>2022-07-21</td>
      <td>NaT</td>
      <td>2025-06-30</td>
      <td>53130.0</td>
      <td>Cerrada</td>
      <td>TOTAL</td>
      <td>2024-12-30</td>
      <td>NaT</td>
      <td>FALTA DE USO/TIEMPO</td>
      <td>1.0</td>
      <td>2022-07-21</td>
      <td>2023-10-10</td>
    </tr>
  </tbody>
</table>
<p>23723 rows × 17 columns</p>
</div>




```python
df_dim_wd[df_dim_wd["updated_at"]<(df_dim_wd["withdrawal_effective_date"]-pd.Timedelta(days=90))]
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
      <th>withdrawal_id</th>
      <th>withdrawal_status</th>
      <th>withdrawal_type</th>
      <th>withdrawal_creation_date</th>
      <th>withdrawal_effective_date</th>
      <th>withdrawal_reason</th>
      <th>is_definitive_withdrawal</th>
      <th>start_date</th>
      <th>snapshot_date</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>90</th>
      <td>1178</td>
      <td>9</td>
      <td>2019-12-28 20:14:49</td>
      <td>Burgos</td>
      <td>120.0</td>
      <td>2020-02-05</td>
      <td>2020-02-05</td>
      <td>2024-12-26</td>
      <td>43272.0</td>
      <td>Cerrada</td>
      <td>TOTAL</td>
      <td>2023-03-30</td>
      <td>2023-04-01</td>
      <td>RESULTADOS</td>
      <td>1.0</td>
      <td>2020-02-05</td>
      <td>2019-12-28</td>
    </tr>
    <tr>
      <th>91</th>
      <td>1178</td>
      <td>9</td>
      <td>2019-12-28 20:14:49</td>
      <td>Burgos</td>
      <td>120.0</td>
      <td>2020-02-05</td>
      <td>2020-02-05</td>
      <td>2024-12-26</td>
      <td>52509.0</td>
      <td>Cerrada</td>
      <td>TOTAL</td>
      <td>2024-11-18</td>
      <td>2025-01-01</td>
      <td>RESULTADOS</td>
      <td>1.0</td>
      <td>2020-02-05</td>
      <td>2019-12-28</td>
    </tr>
    <tr>
      <th>285</th>
      <td>7026</td>
      <td>1</td>
      <td>2025-05-14 12:14:57</td>
      <td>Álava</td>
      <td>NaN</td>
      <td>2025-05-14</td>
      <td>NaT</td>
      <td>NaT</td>
      <td>55418.0</td>
      <td>Cerrada</td>
      <td>TOTAL</td>
      <td>2025-05-16</td>
      <td>2025-12-01</td>
      <td>RESULTADOS</td>
      <td>1.0</td>
      <td>2025-05-14</td>
      <td>2025-05-14</td>
    </tr>
    <tr>
      <th>356</th>
      <td>1967</td>
      <td>2</td>
      <td>2022-12-21 10:46:18</td>
      <td>Albacete</td>
      <td>NaN</td>
      <td>2022-12-13</td>
      <td>NaT</td>
      <td>NaT</td>
      <td>47579.0</td>
      <td>Cerrada</td>
      <td>TOTAL</td>
      <td>2024-01-15</td>
      <td>2024-02-01</td>
      <td>Cambio de Contrato/propuesta/producto</td>
      <td>0.0</td>
      <td>2022-12-13</td>
      <td>2022-12-21</td>
    </tr>
    <tr>
      <th>357</th>
      <td>1967</td>
      <td>2</td>
      <td>2022-12-21 10:46:18</td>
      <td>Albacete</td>
      <td>NaN</td>
      <td>2022-12-13</td>
      <td>NaT</td>
      <td>NaT</td>
      <td>48562.0</td>
      <td>Cerrada</td>
      <td>TOTAL</td>
      <td>2024-03-11</td>
      <td>2024-03-11</td>
      <td>Upselling-cambio de contrato</td>
      <td>0.0</td>
      <td>2022-12-13</td>
      <td>2022-12-21</td>
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
      <td>...</td>
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
      <th>23673</th>
      <td>2718</td>
      <td>42</td>
      <td>2024-09-04 16:57:46</td>
      <td>Tarragona</td>
      <td>NaN</td>
      <td>2021-02-01</td>
      <td>NaT</td>
      <td>2025-06-30</td>
      <td>55336.0</td>
      <td>Cerrada</td>
      <td>TOTAL</td>
      <td>2025-05-12</td>
      <td>2025-07-01</td>
      <td>FALTA DE USO/TIEMPO</td>
      <td>1.0</td>
      <td>2021-02-01</td>
      <td>2024-09-04</td>
    </tr>
    <tr>
      <th>23681</th>
      <td>3396</td>
      <td>42</td>
      <td>2024-12-06 07:25:38</td>
      <td>Tarragona</td>
      <td>NaN</td>
      <td>2022-10-10</td>
      <td>NaT</td>
      <td>2025-06-30</td>
      <td>55335.0</td>
      <td>Cerrada</td>
      <td>TOTAL</td>
      <td>2025-05-12</td>
      <td>2025-07-01</td>
      <td>FALTA DE USO/TIEMPO</td>
      <td>1.0</td>
      <td>2022-10-10</td>
      <td>2024-12-06</td>
    </tr>
    <tr>
      <th>23683</th>
      <td>2970</td>
      <td>45</td>
      <td>2025-01-17 12:08:18</td>
      <td>Toledo</td>
      <td>NaN</td>
      <td>2022-08-01</td>
      <td>2024-10-17</td>
      <td>2025-06-30</td>
      <td>55308.0</td>
      <td>Cerrada</td>
      <td>TOTAL</td>
      <td>2025-05-09</td>
      <td>2025-07-01</td>
      <td>CESE DE ACTIVIDAD</td>
      <td>1.0</td>
      <td>2024-10-17</td>
      <td>2025-01-17</td>
    </tr>
    <tr>
      <th>23691</th>
      <td>3166</td>
      <td>46</td>
      <td>2025-02-27 14:21:20</td>
      <td>Valencia</td>
      <td>NaN</td>
      <td>2021-05-14</td>
      <td>2024-11-27</td>
      <td>2025-06-30</td>
      <td>55149.0</td>
      <td>Cerrada</td>
      <td>TOTAL</td>
      <td>2025-04-25</td>
      <td>2025-07-01</td>
      <td>FALTA DE USO/TIEMPO</td>
      <td>1.0</td>
      <td>2024-11-27</td>
      <td>2025-02-27</td>
    </tr>
    <tr>
      <th>23694</th>
      <td>6668</td>
      <td>46</td>
      <td>2025-03-23 12:51:13</td>
      <td>Valencia</td>
      <td>NaN</td>
      <td>2024-12-20</td>
      <td>2024-12-20</td>
      <td>2025-06-30</td>
      <td>55197.0</td>
      <td>Cerrada</td>
      <td>TOTAL</td>
      <td>2025-05-05</td>
      <td>2025-07-01</td>
      <td>FALTA DE USO/TIEMPO</td>
      <td>1.0</td>
      <td>2024-12-20</td>
      <td>2025-03-23</td>
    </tr>
  </tbody>
</table>
<p>467 rows × 17 columns</p>
</div>



### Fecha de baja: **end_date**

Para que la fecha de churn sea **withdrawal_effective_date** se deben cumplir simultáneamnete las siguientes condiciones:
- Que **is_definitive_withdrawal = 1**
- Que **withdrawal_effective_date** != NaT
- Que **withdrawal_effective_date** > **start_date**
- Que **contrato_churn_date** = NaT ó que sea más tarde que **withdrawal_effective_date**
- Que **withdrawal_effective_date** sea menor que **snapshot_date** (para no incurrir en **data leakage**)

Si no, habría dos escenarios:
- **contrato_churn_date** != NaT y menor que **snapshot_date** : 
-> Fecha de churn = **contrato_churn_date**

Y si no:
- Fecha de churn = **snapshot_date**



```python
df_dim_wd

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
      <th>withdrawal_id</th>
      <th>withdrawal_status</th>
      <th>withdrawal_type</th>
      <th>withdrawal_creation_date</th>
      <th>withdrawal_effective_date</th>
      <th>withdrawal_reason</th>
      <th>is_definitive_withdrawal</th>
      <th>start_date</th>
      <th>snapshot_date</th>
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
      <td>4758.0</td>
      <td>Cerrada</td>
      <td>TOTAL</td>
      <td>2015-01-16</td>
      <td>2015-03-01</td>
      <td>RATIO RESULTADO-INVERSION</td>
      <td>1.0</td>
      <td>2025-01-07</td>
      <td>2025-05-01</td>
    </tr>
    <tr>
      <th>1</th>
      <td>986</td>
      <td>1</td>
      <td>2025-05-01 02:46:04</td>
      <td>Álava</td>
      <td>194.0</td>
      <td>2025-01-07</td>
      <td>2025-01-07</td>
      <td>2025-05-17</td>
      <td>23029.0</td>
      <td>Cerrada</td>
      <td>TOTAL</td>
      <td>2018-10-08</td>
      <td>2018-11-01</td>
      <td>RESULTADOS</td>
      <td>1.0</td>
      <td>2025-01-07</td>
      <td>2025-05-01</td>
    </tr>
    <tr>
      <th>2</th>
      <td>986</td>
      <td>1</td>
      <td>2025-05-01 02:46:04</td>
      <td>Álava</td>
      <td>194.0</td>
      <td>2025-01-07</td>
      <td>2025-01-07</td>
      <td>2025-05-17</td>
      <td>32827.0</td>
      <td>Cerrada</td>
      <td>TOTAL</td>
      <td>2020-11-17</td>
      <td>2020-11-20</td>
      <td>Cambio a Bundle Online</td>
      <td>0.0</td>
      <td>2025-01-07</td>
      <td>2025-05-01</td>
    </tr>
    <tr>
      <th>3</th>
      <td>986</td>
      <td>1</td>
      <td>2025-05-01 02:46:04</td>
      <td>Álava</td>
      <td>194.0</td>
      <td>2025-01-07</td>
      <td>2025-01-07</td>
      <td>2025-05-17</td>
      <td>33518.0</td>
      <td>Cerrada</td>
      <td>TOTAL</td>
      <td>2021-01-05</td>
      <td>2021-02-01</td>
      <td>RATIO RESULTADO-INVERSION</td>
      <td>1.0</td>
      <td>2025-01-07</td>
      <td>2025-05-01</td>
    </tr>
    <tr>
      <th>4</th>
      <td>986</td>
      <td>1</td>
      <td>2025-05-01 02:46:04</td>
      <td>Álava</td>
      <td>194.0</td>
      <td>2025-01-07</td>
      <td>2025-01-07</td>
      <td>2025-05-17</td>
      <td>54519.0</td>
      <td>Cerrada</td>
      <td>TOTAL</td>
      <td>2025-03-18</td>
      <td>2025-05-01</td>
      <td>RATIO RESULTADO-INVERSION</td>
      <td>1.0</td>
      <td>2025-01-07</td>
      <td>2025-05-01</td>
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
      <td>...</td>
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
      <th>23718</th>
      <td>4194</td>
      <td>50</td>
      <td>2023-10-10 11:21:22</td>
      <td>Zaragoza</td>
      <td>NaN</td>
      <td>2022-07-21</td>
      <td>NaT</td>
      <td>2025-06-30</td>
      <td>46264.0</td>
      <td>Cerrada</td>
      <td>TOTAL</td>
      <td>2023-10-09</td>
      <td>2023-10-09</td>
      <td>Cambio de Contrato/propuesta/producto</td>
      <td>0.0</td>
      <td>2022-07-21</td>
      <td>2023-10-10</td>
    </tr>
    <tr>
      <th>23719</th>
      <td>4194</td>
      <td>50</td>
      <td>2023-10-10 11:21:22</td>
      <td>Zaragoza</td>
      <td>NaN</td>
      <td>2022-07-21</td>
      <td>NaT</td>
      <td>2025-06-30</td>
      <td>47376.0</td>
      <td>Cerrada</td>
      <td>TOTAL</td>
      <td>2023-12-21</td>
      <td>NaT</td>
      <td>FALTA DE USO/TIEMPO</td>
      <td>1.0</td>
      <td>2022-07-21</td>
      <td>2023-10-10</td>
    </tr>
    <tr>
      <th>23720</th>
      <td>4194</td>
      <td>50</td>
      <td>2023-10-10 11:21:22</td>
      <td>Zaragoza</td>
      <td>NaN</td>
      <td>2022-07-21</td>
      <td>NaT</td>
      <td>2025-06-30</td>
      <td>48614.0</td>
      <td>Cerrada</td>
      <td>TOTAL</td>
      <td>2024-03-12</td>
      <td>NaT</td>
      <td>RAZONES ECONOMICAS</td>
      <td>1.0</td>
      <td>2022-07-21</td>
      <td>2023-10-10</td>
    </tr>
    <tr>
      <th>23721</th>
      <td>4194</td>
      <td>50</td>
      <td>2023-10-10 11:21:22</td>
      <td>Zaragoza</td>
      <td>NaN</td>
      <td>2022-07-21</td>
      <td>NaT</td>
      <td>2025-06-30</td>
      <td>51090.0</td>
      <td>Cerrada</td>
      <td>TOTAL</td>
      <td>2024-08-29</td>
      <td>NaT</td>
      <td>RESULTADOS</td>
      <td>1.0</td>
      <td>2022-07-21</td>
      <td>2023-10-10</td>
    </tr>
    <tr>
      <th>23722</th>
      <td>4194</td>
      <td>50</td>
      <td>2023-10-10 11:21:22</td>
      <td>Zaragoza</td>
      <td>NaN</td>
      <td>2022-07-21</td>
      <td>NaT</td>
      <td>2025-06-30</td>
      <td>53130.0</td>
      <td>Cerrada</td>
      <td>TOTAL</td>
      <td>2024-12-30</td>
      <td>NaT</td>
      <td>FALTA DE USO/TIEMPO</td>
      <td>1.0</td>
      <td>2022-07-21</td>
      <td>2023-10-10</td>
    </tr>
  </tbody>
</table>
<p>23723 rows × 17 columns</p>
</div>



### ¿Cuál es realmente la fecha de fin de permanencia **end_date**?

- Si hay baja definitiva anticipada y observable antes del snapshot → **end_date = withdrawal_effective_date**
- Si no hay baja anticipada pero sí fin de contrato observable → **end_date = contrato_churn_date**
- Si ninguna de las anteriores → censurado → **end_date = snapshot_date** (Esto significa que no hay fecha anterior a la fecha última en la que se tomaron los datos, con lo cual se corta en snapshot_date = usuario activo)


```python
# CASO 1:
# Hay withdrawal definitivo y ocurre antes que fecha fin de contrato (o no hay fecha fin de contrato)
mask_withdrawal = (
    (df_dim_wd["is_definitive_withdrawal"] == 1) & # debe ser una baja DEFINITIVA
    (df_dim_wd["withdrawal_effective_date"].notna()) & # tiene que existir
    (df_dim_wd["withdrawal_effective_date"] > df_dim_wd["start_date"]) & # la fecha de comienzo de contrato debe ser, lógicamente, menor que la fecha de withdrawal
    (
        df_dim_wd["contrato_churn_date"].isna() | 
        (df_dim_wd["withdrawal_effective_date"] < df_dim_wd["contrato_churn_date"]) # No hay fecha fin de contrato o ésta es mayor a la withdrawal ==> se dió de baja antes de fecha fin de contrato
    ) & 
    (df_dim_wd["withdrawal_effective_date"] < df_dim_wd["snapshot_date"]) # Debe ser anterior a la fecha de snapshot (última fecha de actualización de ese advertiser)
)

# CASO 2: 
# No hay ruptura anticipada, pero hay fin de contrato observable
mask_contract_end = (
        (df_dim_wd["contrato_churn_date"].notna()) &
        (df_dim_wd["contrato_churn_date"] < df_dim_wd["snapshot_date"])
)

df_dim_wd["end_date"] = np.select(
    condlist=[
        mask_withdrawal, 
        mask_contract_end
    ],
    choicelist=[
        df_dim_wd["withdrawal_effective_date"],
        df_dim_wd["contrato_churn_date"]
    ],
    default=df_dim_wd["snapshot_date"] # Si no se cumple ninguna condición, se toma la fecha final como la última fecha de actualización de los datos
)
df_dim_wd["end_date"] = pd.to_datetime(df_dim_wd["end_date"])

# Nos quedamos únicamente con los registros donde updated_at > start_date
df_dim_wd = df_dim_wd.loc[df_dim_wd["updated_at"] > df_dim_wd["start_date"]]


```

### Días de permanencia


```python
df_dim_wd["permanencia"] = (df_dim_wd["end_date"] - df_dim_wd["start_date"])

```

    /tmp/ipykernel_160604/2279193302.py:1: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      df_dim_wd["permanencia"] = (df_dim_wd["end_date"] - df_dim_wd["start_date"])



```python
df_dim_wd["dias_permanencia"] = df_dim_wd["permanencia"].dt.days
df_dim_wd.head()
```

    /tmp/ipykernel_160604/2122577522.py:1: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      df_dim_wd["dias_permanencia"] = df_dim_wd["permanencia"].dt.days





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
      <th>withdrawal_id</th>
      <th>withdrawal_status</th>
      <th>withdrawal_type</th>
      <th>withdrawal_creation_date</th>
      <th>withdrawal_effective_date</th>
      <th>withdrawal_reason</th>
      <th>is_definitive_withdrawal</th>
      <th>start_date</th>
      <th>snapshot_date</th>
      <th>end_date</th>
      <th>permanencia</th>
      <th>dias_permanencia</th>
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
      <td>4758.0</td>
      <td>Cerrada</td>
      <td>TOTAL</td>
      <td>2015-01-16</td>
      <td>2015-03-01</td>
      <td>RATIO RESULTADO-INVERSION</td>
      <td>1.0</td>
      <td>2025-01-07</td>
      <td>2025-05-01</td>
      <td>2025-05-01</td>
      <td>114 days</td>
      <td>114</td>
    </tr>
    <tr>
      <th>1</th>
      <td>986</td>
      <td>1</td>
      <td>2025-05-01 02:46:04</td>
      <td>Álava</td>
      <td>194.0</td>
      <td>2025-01-07</td>
      <td>2025-01-07</td>
      <td>2025-05-17</td>
      <td>23029.0</td>
      <td>Cerrada</td>
      <td>TOTAL</td>
      <td>2018-10-08</td>
      <td>2018-11-01</td>
      <td>RESULTADOS</td>
      <td>1.0</td>
      <td>2025-01-07</td>
      <td>2025-05-01</td>
      <td>2025-05-01</td>
      <td>114 days</td>
      <td>114</td>
    </tr>
    <tr>
      <th>2</th>
      <td>986</td>
      <td>1</td>
      <td>2025-05-01 02:46:04</td>
      <td>Álava</td>
      <td>194.0</td>
      <td>2025-01-07</td>
      <td>2025-01-07</td>
      <td>2025-05-17</td>
      <td>32827.0</td>
      <td>Cerrada</td>
      <td>TOTAL</td>
      <td>2020-11-17</td>
      <td>2020-11-20</td>
      <td>Cambio a Bundle Online</td>
      <td>0.0</td>
      <td>2025-01-07</td>
      <td>2025-05-01</td>
      <td>2025-05-01</td>
      <td>114 days</td>
      <td>114</td>
    </tr>
    <tr>
      <th>3</th>
      <td>986</td>
      <td>1</td>
      <td>2025-05-01 02:46:04</td>
      <td>Álava</td>
      <td>194.0</td>
      <td>2025-01-07</td>
      <td>2025-01-07</td>
      <td>2025-05-17</td>
      <td>33518.0</td>
      <td>Cerrada</td>
      <td>TOTAL</td>
      <td>2021-01-05</td>
      <td>2021-02-01</td>
      <td>RATIO RESULTADO-INVERSION</td>
      <td>1.0</td>
      <td>2025-01-07</td>
      <td>2025-05-01</td>
      <td>2025-05-01</td>
      <td>114 days</td>
      <td>114</td>
    </tr>
    <tr>
      <th>4</th>
      <td>986</td>
      <td>1</td>
      <td>2025-05-01 02:46:04</td>
      <td>Álava</td>
      <td>194.0</td>
      <td>2025-01-07</td>
      <td>2025-01-07</td>
      <td>2025-05-17</td>
      <td>54519.0</td>
      <td>Cerrada</td>
      <td>TOTAL</td>
      <td>2025-03-18</td>
      <td>2025-05-01</td>
      <td>RATIO RESULTADO-INVERSION</td>
      <td>1.0</td>
      <td>2025-01-07</td>
      <td>2025-05-01</td>
      <td>2025-05-01</td>
      <td>114 days</td>
      <td>114</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_dim_wd["dias_permanencia"].describe()
```




    count    23568.000000
    mean       585.566870
    std        535.197471
    min          0.000000
    25%        159.000000
    50%        407.000000
    75%        910.000000
    max       5169.000000
    Name: dias_permanencia, dtype: float64



Para continuar el análisis, nos quedamos con las columnas siguientes


```python
df_churn = df_dim_wd.copy()
withdrawal_cols = [
    "advertiser_zrive_id",
    "advertiser_group_id", 
    "province_id", 
    "min_start_contrato_date", 
    "max_start_contrato_nuevo_date", 
    "contrato_churn_date",
    "withdrawal_id",
    "withdrawal_creation_date",
    "withdrawal_effective_date", 
    "is_definitive_withdrawal",  
    "start_date", 
    "snapshot_date", 
    "end_date", 
    "dias_permanencia"
    ]
df_churn = df_churn[withdrawal_cols]
df_churn
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
      <th>advertiser_group_id</th>
      <th>province_id</th>
      <th>min_start_contrato_date</th>
      <th>max_start_contrato_nuevo_date</th>
      <th>contrato_churn_date</th>
      <th>withdrawal_id</th>
      <th>withdrawal_creation_date</th>
      <th>withdrawal_effective_date</th>
      <th>is_definitive_withdrawal</th>
      <th>start_date</th>
      <th>snapshot_date</th>
      <th>end_date</th>
      <th>dias_permanencia</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>986</td>
      <td>194.0</td>
      <td>1</td>
      <td>2025-01-07</td>
      <td>2025-01-07</td>
      <td>2025-05-17</td>
      <td>4758.0</td>
      <td>2015-01-16</td>
      <td>2015-03-01</td>
      <td>1.0</td>
      <td>2025-01-07</td>
      <td>2025-05-01</td>
      <td>2025-05-01</td>
      <td>114</td>
    </tr>
    <tr>
      <th>1</th>
      <td>986</td>
      <td>194.0</td>
      <td>1</td>
      <td>2025-01-07</td>
      <td>2025-01-07</td>
      <td>2025-05-17</td>
      <td>23029.0</td>
      <td>2018-10-08</td>
      <td>2018-11-01</td>
      <td>1.0</td>
      <td>2025-01-07</td>
      <td>2025-05-01</td>
      <td>2025-05-01</td>
      <td>114</td>
    </tr>
    <tr>
      <th>2</th>
      <td>986</td>
      <td>194.0</td>
      <td>1</td>
      <td>2025-01-07</td>
      <td>2025-01-07</td>
      <td>2025-05-17</td>
      <td>32827.0</td>
      <td>2020-11-17</td>
      <td>2020-11-20</td>
      <td>0.0</td>
      <td>2025-01-07</td>
      <td>2025-05-01</td>
      <td>2025-05-01</td>
      <td>114</td>
    </tr>
    <tr>
      <th>3</th>
      <td>986</td>
      <td>194.0</td>
      <td>1</td>
      <td>2025-01-07</td>
      <td>2025-01-07</td>
      <td>2025-05-17</td>
      <td>33518.0</td>
      <td>2021-01-05</td>
      <td>2021-02-01</td>
      <td>1.0</td>
      <td>2025-01-07</td>
      <td>2025-05-01</td>
      <td>2025-05-01</td>
      <td>114</td>
    </tr>
    <tr>
      <th>4</th>
      <td>986</td>
      <td>194.0</td>
      <td>1</td>
      <td>2025-01-07</td>
      <td>2025-01-07</td>
      <td>2025-05-17</td>
      <td>54519.0</td>
      <td>2025-03-18</td>
      <td>2025-05-01</td>
      <td>1.0</td>
      <td>2025-01-07</td>
      <td>2025-05-01</td>
      <td>2025-05-01</td>
      <td>114</td>
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
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>23718</th>
      <td>4194</td>
      <td>NaN</td>
      <td>50</td>
      <td>2022-07-21</td>
      <td>NaT</td>
      <td>2025-06-30</td>
      <td>46264.0</td>
      <td>2023-10-09</td>
      <td>2023-10-09</td>
      <td>0.0</td>
      <td>2022-07-21</td>
      <td>2023-10-10</td>
      <td>2023-10-10</td>
      <td>446</td>
    </tr>
    <tr>
      <th>23719</th>
      <td>4194</td>
      <td>NaN</td>
      <td>50</td>
      <td>2022-07-21</td>
      <td>NaT</td>
      <td>2025-06-30</td>
      <td>47376.0</td>
      <td>2023-12-21</td>
      <td>NaT</td>
      <td>1.0</td>
      <td>2022-07-21</td>
      <td>2023-10-10</td>
      <td>2023-10-10</td>
      <td>446</td>
    </tr>
    <tr>
      <th>23720</th>
      <td>4194</td>
      <td>NaN</td>
      <td>50</td>
      <td>2022-07-21</td>
      <td>NaT</td>
      <td>2025-06-30</td>
      <td>48614.0</td>
      <td>2024-03-12</td>
      <td>NaT</td>
      <td>1.0</td>
      <td>2022-07-21</td>
      <td>2023-10-10</td>
      <td>2023-10-10</td>
      <td>446</td>
    </tr>
    <tr>
      <th>23721</th>
      <td>4194</td>
      <td>NaN</td>
      <td>50</td>
      <td>2022-07-21</td>
      <td>NaT</td>
      <td>2025-06-30</td>
      <td>51090.0</td>
      <td>2024-08-29</td>
      <td>NaT</td>
      <td>1.0</td>
      <td>2022-07-21</td>
      <td>2023-10-10</td>
      <td>2023-10-10</td>
      <td>446</td>
    </tr>
    <tr>
      <th>23722</th>
      <td>4194</td>
      <td>NaN</td>
      <td>50</td>
      <td>2022-07-21</td>
      <td>NaT</td>
      <td>2025-06-30</td>
      <td>53130.0</td>
      <td>2024-12-30</td>
      <td>NaT</td>
      <td>1.0</td>
      <td>2022-07-21</td>
      <td>2023-10-10</td>
      <td>2023-10-10</td>
      <td>446</td>
    </tr>
  </tbody>
</table>
<p>23568 rows × 14 columns</p>
</div>



### Data leakage prevention

Sabemos que **updated_at** es la última fecha en la que se actualizaron los datos, con lo cual más allá de esa fecha no podemos saber el comportamiento del advertiser, tanto si abandonó como si no abandonó; estos usuarios se consideran "**censurados**".

 No obstante, para el estudio y entrenamiento de churn de 90 días, sí podemos usar los usuarios censurados cuya duración es mayor a 90 días, ya que, aunque no sabemos si en algún momento hicieron withdrawal o no (no tenemos información posterior a su fecha de **updated_at**), lo que sí sabemos es que permanecieron al menos 90 días o más en la plataforma, con lo cual sí entran en el estudio.

Por tanto, el dataset final es:

- Todos los que ya han salido df[snapshot > end_date]
- Los que siguen activos df[snapshot < end_date] pero han durado ≥ 90 días


```python
df_churn = df_dim_wd.loc[
    (df_dim_wd["snapshot_date"]>df_dim_wd["end_date"]) | 
    (df_dim_wd["dias_permanencia"] >= 90)   
]
df_churn   
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
      <th>withdrawal_id</th>
      <th>withdrawal_status</th>
      <th>withdrawal_type</th>
      <th>withdrawal_creation_date</th>
      <th>withdrawal_effective_date</th>
      <th>withdrawal_reason</th>
      <th>is_definitive_withdrawal</th>
      <th>start_date</th>
      <th>snapshot_date</th>
      <th>end_date</th>
      <th>permanencia</th>
      <th>dias_permanencia</th>
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
      <td>4758.0</td>
      <td>Cerrada</td>
      <td>TOTAL</td>
      <td>2015-01-16</td>
      <td>2015-03-01</td>
      <td>RATIO RESULTADO-INVERSION</td>
      <td>1.0</td>
      <td>2025-01-07</td>
      <td>2025-05-01</td>
      <td>2025-05-01</td>
      <td>114 days</td>
      <td>114</td>
    </tr>
    <tr>
      <th>1</th>
      <td>986</td>
      <td>1</td>
      <td>2025-05-01 02:46:04</td>
      <td>Álava</td>
      <td>194.0</td>
      <td>2025-01-07</td>
      <td>2025-01-07</td>
      <td>2025-05-17</td>
      <td>23029.0</td>
      <td>Cerrada</td>
      <td>TOTAL</td>
      <td>2018-10-08</td>
      <td>2018-11-01</td>
      <td>RESULTADOS</td>
      <td>1.0</td>
      <td>2025-01-07</td>
      <td>2025-05-01</td>
      <td>2025-05-01</td>
      <td>114 days</td>
      <td>114</td>
    </tr>
    <tr>
      <th>2</th>
      <td>986</td>
      <td>1</td>
      <td>2025-05-01 02:46:04</td>
      <td>Álava</td>
      <td>194.0</td>
      <td>2025-01-07</td>
      <td>2025-01-07</td>
      <td>2025-05-17</td>
      <td>32827.0</td>
      <td>Cerrada</td>
      <td>TOTAL</td>
      <td>2020-11-17</td>
      <td>2020-11-20</td>
      <td>Cambio a Bundle Online</td>
      <td>0.0</td>
      <td>2025-01-07</td>
      <td>2025-05-01</td>
      <td>2025-05-01</td>
      <td>114 days</td>
      <td>114</td>
    </tr>
    <tr>
      <th>3</th>
      <td>986</td>
      <td>1</td>
      <td>2025-05-01 02:46:04</td>
      <td>Álava</td>
      <td>194.0</td>
      <td>2025-01-07</td>
      <td>2025-01-07</td>
      <td>2025-05-17</td>
      <td>33518.0</td>
      <td>Cerrada</td>
      <td>TOTAL</td>
      <td>2021-01-05</td>
      <td>2021-02-01</td>
      <td>RATIO RESULTADO-INVERSION</td>
      <td>1.0</td>
      <td>2025-01-07</td>
      <td>2025-05-01</td>
      <td>2025-05-01</td>
      <td>114 days</td>
      <td>114</td>
    </tr>
    <tr>
      <th>4</th>
      <td>986</td>
      <td>1</td>
      <td>2025-05-01 02:46:04</td>
      <td>Álava</td>
      <td>194.0</td>
      <td>2025-01-07</td>
      <td>2025-01-07</td>
      <td>2025-05-17</td>
      <td>54519.0</td>
      <td>Cerrada</td>
      <td>TOTAL</td>
      <td>2025-03-18</td>
      <td>2025-05-01</td>
      <td>RATIO RESULTADO-INVERSION</td>
      <td>1.0</td>
      <td>2025-01-07</td>
      <td>2025-05-01</td>
      <td>2025-05-01</td>
      <td>114 days</td>
      <td>114</td>
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
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
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
      <th>23718</th>
      <td>4194</td>
      <td>50</td>
      <td>2023-10-10 11:21:22</td>
      <td>Zaragoza</td>
      <td>NaN</td>
      <td>2022-07-21</td>
      <td>NaT</td>
      <td>2025-06-30</td>
      <td>46264.0</td>
      <td>Cerrada</td>
      <td>TOTAL</td>
      <td>2023-10-09</td>
      <td>2023-10-09</td>
      <td>Cambio de Contrato/propuesta/producto</td>
      <td>0.0</td>
      <td>2022-07-21</td>
      <td>2023-10-10</td>
      <td>2023-10-10</td>
      <td>446 days</td>
      <td>446</td>
    </tr>
    <tr>
      <th>23719</th>
      <td>4194</td>
      <td>50</td>
      <td>2023-10-10 11:21:22</td>
      <td>Zaragoza</td>
      <td>NaN</td>
      <td>2022-07-21</td>
      <td>NaT</td>
      <td>2025-06-30</td>
      <td>47376.0</td>
      <td>Cerrada</td>
      <td>TOTAL</td>
      <td>2023-12-21</td>
      <td>NaT</td>
      <td>FALTA DE USO/TIEMPO</td>
      <td>1.0</td>
      <td>2022-07-21</td>
      <td>2023-10-10</td>
      <td>2023-10-10</td>
      <td>446 days</td>
      <td>446</td>
    </tr>
    <tr>
      <th>23720</th>
      <td>4194</td>
      <td>50</td>
      <td>2023-10-10 11:21:22</td>
      <td>Zaragoza</td>
      <td>NaN</td>
      <td>2022-07-21</td>
      <td>NaT</td>
      <td>2025-06-30</td>
      <td>48614.0</td>
      <td>Cerrada</td>
      <td>TOTAL</td>
      <td>2024-03-12</td>
      <td>NaT</td>
      <td>RAZONES ECONOMICAS</td>
      <td>1.0</td>
      <td>2022-07-21</td>
      <td>2023-10-10</td>
      <td>2023-10-10</td>
      <td>446 days</td>
      <td>446</td>
    </tr>
    <tr>
      <th>23721</th>
      <td>4194</td>
      <td>50</td>
      <td>2023-10-10 11:21:22</td>
      <td>Zaragoza</td>
      <td>NaN</td>
      <td>2022-07-21</td>
      <td>NaT</td>
      <td>2025-06-30</td>
      <td>51090.0</td>
      <td>Cerrada</td>
      <td>TOTAL</td>
      <td>2024-08-29</td>
      <td>NaT</td>
      <td>RESULTADOS</td>
      <td>1.0</td>
      <td>2022-07-21</td>
      <td>2023-10-10</td>
      <td>2023-10-10</td>
      <td>446 days</td>
      <td>446</td>
    </tr>
    <tr>
      <th>23722</th>
      <td>4194</td>
      <td>50</td>
      <td>2023-10-10 11:21:22</td>
      <td>Zaragoza</td>
      <td>NaN</td>
      <td>2022-07-21</td>
      <td>NaT</td>
      <td>2025-06-30</td>
      <td>53130.0</td>
      <td>Cerrada</td>
      <td>TOTAL</td>
      <td>2024-12-30</td>
      <td>NaT</td>
      <td>FALTA DE USO/TIEMPO</td>
      <td>1.0</td>
      <td>2022-07-21</td>
      <td>2023-10-10</td>
      <td>2023-10-10</td>
      <td>446 days</td>
      <td>446</td>
    </tr>
  </tbody>
</table>
<p>21435 rows × 20 columns</p>
</div>



Para distinguir entre los usuarios activos en el momento del snapshot_date de los históricos, se crea la feature **is_active**


```python
df_churn["is_active"] = (
    df_churn["snapshot_date"] == df_churn["end_date"]
).astype(int)
df_churn
```

    /tmp/ipykernel_160604/1352573320.py:1: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      df_churn["is_active"] = (





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
      <th>withdrawal_id</th>
      <th>withdrawal_status</th>
      <th>...</th>
      <th>withdrawal_creation_date</th>
      <th>withdrawal_effective_date</th>
      <th>withdrawal_reason</th>
      <th>is_definitive_withdrawal</th>
      <th>start_date</th>
      <th>snapshot_date</th>
      <th>end_date</th>
      <th>permanencia</th>
      <th>dias_permanencia</th>
      <th>is_active</th>
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
      <td>4758.0</td>
      <td>Cerrada</td>
      <td>...</td>
      <td>2015-01-16</td>
      <td>2015-03-01</td>
      <td>RATIO RESULTADO-INVERSION</td>
      <td>1.0</td>
      <td>2025-01-07</td>
      <td>2025-05-01</td>
      <td>2025-05-01</td>
      <td>114 days</td>
      <td>114</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>986</td>
      <td>1</td>
      <td>2025-05-01 02:46:04</td>
      <td>Álava</td>
      <td>194.0</td>
      <td>2025-01-07</td>
      <td>2025-01-07</td>
      <td>2025-05-17</td>
      <td>23029.0</td>
      <td>Cerrada</td>
      <td>...</td>
      <td>2018-10-08</td>
      <td>2018-11-01</td>
      <td>RESULTADOS</td>
      <td>1.0</td>
      <td>2025-01-07</td>
      <td>2025-05-01</td>
      <td>2025-05-01</td>
      <td>114 days</td>
      <td>114</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>986</td>
      <td>1</td>
      <td>2025-05-01 02:46:04</td>
      <td>Álava</td>
      <td>194.0</td>
      <td>2025-01-07</td>
      <td>2025-01-07</td>
      <td>2025-05-17</td>
      <td>32827.0</td>
      <td>Cerrada</td>
      <td>...</td>
      <td>2020-11-17</td>
      <td>2020-11-20</td>
      <td>Cambio a Bundle Online</td>
      <td>0.0</td>
      <td>2025-01-07</td>
      <td>2025-05-01</td>
      <td>2025-05-01</td>
      <td>114 days</td>
      <td>114</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>986</td>
      <td>1</td>
      <td>2025-05-01 02:46:04</td>
      <td>Álava</td>
      <td>194.0</td>
      <td>2025-01-07</td>
      <td>2025-01-07</td>
      <td>2025-05-17</td>
      <td>33518.0</td>
      <td>Cerrada</td>
      <td>...</td>
      <td>2021-01-05</td>
      <td>2021-02-01</td>
      <td>RATIO RESULTADO-INVERSION</td>
      <td>1.0</td>
      <td>2025-01-07</td>
      <td>2025-05-01</td>
      <td>2025-05-01</td>
      <td>114 days</td>
      <td>114</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>986</td>
      <td>1</td>
      <td>2025-05-01 02:46:04</td>
      <td>Álava</td>
      <td>194.0</td>
      <td>2025-01-07</td>
      <td>2025-01-07</td>
      <td>2025-05-17</td>
      <td>54519.0</td>
      <td>Cerrada</td>
      <td>...</td>
      <td>2025-03-18</td>
      <td>2025-05-01</td>
      <td>RATIO RESULTADO-INVERSION</td>
      <td>1.0</td>
      <td>2025-01-07</td>
      <td>2025-05-01</td>
      <td>2025-05-01</td>
      <td>114 days</td>
      <td>114</td>
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
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
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
      <th>23718</th>
      <td>4194</td>
      <td>50</td>
      <td>2023-10-10 11:21:22</td>
      <td>Zaragoza</td>
      <td>NaN</td>
      <td>2022-07-21</td>
      <td>NaT</td>
      <td>2025-06-30</td>
      <td>46264.0</td>
      <td>Cerrada</td>
      <td>...</td>
      <td>2023-10-09</td>
      <td>2023-10-09</td>
      <td>Cambio de Contrato/propuesta/producto</td>
      <td>0.0</td>
      <td>2022-07-21</td>
      <td>2023-10-10</td>
      <td>2023-10-10</td>
      <td>446 days</td>
      <td>446</td>
      <td>1</td>
    </tr>
    <tr>
      <th>23719</th>
      <td>4194</td>
      <td>50</td>
      <td>2023-10-10 11:21:22</td>
      <td>Zaragoza</td>
      <td>NaN</td>
      <td>2022-07-21</td>
      <td>NaT</td>
      <td>2025-06-30</td>
      <td>47376.0</td>
      <td>Cerrada</td>
      <td>...</td>
      <td>2023-12-21</td>
      <td>NaT</td>
      <td>FALTA DE USO/TIEMPO</td>
      <td>1.0</td>
      <td>2022-07-21</td>
      <td>2023-10-10</td>
      <td>2023-10-10</td>
      <td>446 days</td>
      <td>446</td>
      <td>1</td>
    </tr>
    <tr>
      <th>23720</th>
      <td>4194</td>
      <td>50</td>
      <td>2023-10-10 11:21:22</td>
      <td>Zaragoza</td>
      <td>NaN</td>
      <td>2022-07-21</td>
      <td>NaT</td>
      <td>2025-06-30</td>
      <td>48614.0</td>
      <td>Cerrada</td>
      <td>...</td>
      <td>2024-03-12</td>
      <td>NaT</td>
      <td>RAZONES ECONOMICAS</td>
      <td>1.0</td>
      <td>2022-07-21</td>
      <td>2023-10-10</td>
      <td>2023-10-10</td>
      <td>446 days</td>
      <td>446</td>
      <td>1</td>
    </tr>
    <tr>
      <th>23721</th>
      <td>4194</td>
      <td>50</td>
      <td>2023-10-10 11:21:22</td>
      <td>Zaragoza</td>
      <td>NaN</td>
      <td>2022-07-21</td>
      <td>NaT</td>
      <td>2025-06-30</td>
      <td>51090.0</td>
      <td>Cerrada</td>
      <td>...</td>
      <td>2024-08-29</td>
      <td>NaT</td>
      <td>RESULTADOS</td>
      <td>1.0</td>
      <td>2022-07-21</td>
      <td>2023-10-10</td>
      <td>2023-10-10</td>
      <td>446 days</td>
      <td>446</td>
      <td>1</td>
    </tr>
    <tr>
      <th>23722</th>
      <td>4194</td>
      <td>50</td>
      <td>2023-10-10 11:21:22</td>
      <td>Zaragoza</td>
      <td>NaN</td>
      <td>2022-07-21</td>
      <td>NaT</td>
      <td>2025-06-30</td>
      <td>53130.0</td>
      <td>Cerrada</td>
      <td>...</td>
      <td>2024-12-30</td>
      <td>NaT</td>
      <td>FALTA DE USO/TIEMPO</td>
      <td>1.0</td>
      <td>2022-07-21</td>
      <td>2023-10-10</td>
      <td>2023-10-10</td>
      <td>446 days</td>
      <td>446</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>21435 rows × 21 columns</p>
</div>




```python
df_churn["is_active"].value_counts()
```




    is_active
    0    11591
    1     9844
    Name: count, dtype: int64



### Censored

Llamaremos usuarios censored aquellos cuya fecha **snapshot_date** sea igual o anterior a la **end_date** y tengan menos de 90 días de duración: al no saber si han hecho withdrawal o no, no podemos decir que se fueron antes de los 90 días



```python
df_censored = df_dim_wd.loc[
    (df_dim_wd["snapshot_date"] <= df_dim_wd["end_date"]) &
    (df_dim_wd["dias_permanencia"] < 90)     
]
df_censored
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
      <th>withdrawal_id</th>
      <th>withdrawal_status</th>
      <th>withdrawal_type</th>
      <th>withdrawal_creation_date</th>
      <th>withdrawal_effective_date</th>
      <th>withdrawal_reason</th>
      <th>is_definitive_withdrawal</th>
      <th>start_date</th>
      <th>snapshot_date</th>
      <th>end_date</th>
      <th>permanencia</th>
      <th>dias_permanencia</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>73</th>
      <td>1291</td>
      <td>8</td>
      <td>2024-07-05 11:30:50</td>
      <td>Barcelona</td>
      <td>41.0</td>
      <td>2024-04-29</td>
      <td>NaT</td>
      <td>2024-08-03</td>
      <td>22289.0</td>
      <td>Cerrada</td>
      <td>TOTAL</td>
      <td>2018-06-08</td>
      <td>2018-07-01</td>
      <td>RESULTADOS</td>
      <td>1.0</td>
      <td>2024-04-29</td>
      <td>2024-07-05</td>
      <td>2024-07-05</td>
      <td>67 days</td>
      <td>67</td>
    </tr>
    <tr>
      <th>74</th>
      <td>1291</td>
      <td>8</td>
      <td>2024-07-05 11:30:50</td>
      <td>Barcelona</td>
      <td>41.0</td>
      <td>2024-04-29</td>
      <td>NaT</td>
      <td>2024-08-03</td>
      <td>27431.0</td>
      <td>Cerrada</td>
      <td>TOTAL</td>
      <td>2019-12-20</td>
      <td>2019-12-31</td>
      <td>Cambio a Bundle Online</td>
      <td>0.0</td>
      <td>2024-04-29</td>
      <td>2024-07-05</td>
      <td>2024-07-05</td>
      <td>67 days</td>
      <td>67</td>
    </tr>
    <tr>
      <th>75</th>
      <td>1291</td>
      <td>8</td>
      <td>2024-07-05 11:30:50</td>
      <td>Barcelona</td>
      <td>41.0</td>
      <td>2024-04-29</td>
      <td>NaT</td>
      <td>2024-08-03</td>
      <td>27450.0</td>
      <td>Cerrada</td>
      <td>TOTAL</td>
      <td>2019-12-23</td>
      <td>2019-12-23</td>
      <td>Cambio a Bundle Online</td>
      <td>0.0</td>
      <td>2024-04-29</td>
      <td>2024-07-05</td>
      <td>2024-07-05</td>
      <td>67 days</td>
      <td>67</td>
    </tr>
    <tr>
      <th>76</th>
      <td>1291</td>
      <td>8</td>
      <td>2024-07-05 11:30:50</td>
      <td>Barcelona</td>
      <td>41.0</td>
      <td>2024-04-29</td>
      <td>NaT</td>
      <td>2024-08-03</td>
      <td>29285.0</td>
      <td>Cerrada</td>
      <td>TOTAL</td>
      <td>2020-03-25</td>
      <td>NaT</td>
      <td>CORONAVIRUS</td>
      <td>1.0</td>
      <td>2024-04-29</td>
      <td>2024-07-05</td>
      <td>2024-07-05</td>
      <td>67 days</td>
      <td>67</td>
    </tr>
    <tr>
      <th>77</th>
      <td>1291</td>
      <td>8</td>
      <td>2024-07-05 11:30:50</td>
      <td>Barcelona</td>
      <td>41.0</td>
      <td>2024-04-29</td>
      <td>NaT</td>
      <td>2024-08-03</td>
      <td>33087.0</td>
      <td>Cerrada</td>
      <td>TOTAL</td>
      <td>2020-11-30</td>
      <td>2021-02-01</td>
      <td>RESULTADOS</td>
      <td>1.0</td>
      <td>2024-04-29</td>
      <td>2024-07-05</td>
      <td>2024-07-05</td>
      <td>67 days</td>
      <td>67</td>
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
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
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
      <th>23609</th>
      <td>4802</td>
      <td>19</td>
      <td>2025-03-10 15:58:19</td>
      <td>Granada</td>
      <td>NaN</td>
      <td>2023-07-12</td>
      <td>2025-02-13</td>
      <td>2025-06-30</td>
      <td>47224.0</td>
      <td>Cerrada</td>
      <td>TOTAL</td>
      <td>2023-12-14</td>
      <td>2024-03-01</td>
      <td>RESULTADOS</td>
      <td>1.0</td>
      <td>2025-02-13</td>
      <td>2025-03-10</td>
      <td>2025-03-10</td>
      <td>25 days</td>
      <td>25</td>
    </tr>
    <tr>
      <th>23610</th>
      <td>4802</td>
      <td>19</td>
      <td>2025-03-10 15:58:19</td>
      <td>Granada</td>
      <td>NaN</td>
      <td>2023-07-12</td>
      <td>2025-02-13</td>
      <td>2025-06-30</td>
      <td>55074.0</td>
      <td>Cerrada</td>
      <td>TOTAL</td>
      <td>2025-04-16</td>
      <td>2025-07-01</td>
      <td>FALTA DE USO/TIEMPO</td>
      <td>1.0</td>
      <td>2025-02-13</td>
      <td>2025-03-10</td>
      <td>2025-03-10</td>
      <td>25 days</td>
      <td>25</td>
    </tr>
    <tr>
      <th>23616</th>
      <td>5738</td>
      <td>29</td>
      <td>2024-01-24 13:50:41</td>
      <td>Madrid</td>
      <td>NaN</td>
      <td>2024-01-17</td>
      <td>NaT</td>
      <td>2025-06-30</td>
      <td>54337.0</td>
      <td>Cerrada</td>
      <td>TOTAL</td>
      <td>2025-03-07</td>
      <td>2025-04-01</td>
      <td>Cambio de Contrato/propuesta/producto</td>
      <td>0.0</td>
      <td>2024-01-17</td>
      <td>2024-01-24</td>
      <td>2024-01-24</td>
      <td>7 days</td>
      <td>7</td>
    </tr>
    <tr>
      <th>23617</th>
      <td>5738</td>
      <td>29</td>
      <td>2024-01-24 13:50:41</td>
      <td>Madrid</td>
      <td>NaN</td>
      <td>2024-01-17</td>
      <td>NaT</td>
      <td>2025-06-30</td>
      <td>55212.0</td>
      <td>Cerrada</td>
      <td>TOTAL</td>
      <td>2025-05-05</td>
      <td>2025-07-01</td>
      <td>CESE DE ACTIVIDAD</td>
      <td>1.0</td>
      <td>2024-01-17</td>
      <td>2024-01-24</td>
      <td>2024-01-24</td>
      <td>7 days</td>
      <td>7</td>
    </tr>
    <tr>
      <th>23618</th>
      <td>3719</td>
      <td>30</td>
      <td>2024-06-20 17:32:53</td>
      <td>Málaga</td>
      <td>NaN</td>
      <td>2024-05-14</td>
      <td>2024-05-14</td>
      <td>2025-06-30</td>
      <td>55180.0</td>
      <td>Cerrada</td>
      <td>TOTAL</td>
      <td>2025-04-30</td>
      <td>2025-07-01</td>
      <td>FALTA DE PRODUCTO</td>
      <td>1.0</td>
      <td>2024-05-14</td>
      <td>2024-06-20</td>
      <td>2024-06-20</td>
      <td>37 days</td>
      <td>37</td>
    </tr>
  </tbody>
</table>
<p>2133 rows × 20 columns</p>
</div>




```python
df_churn[df_churn["dias_permanencia"] > 90]
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
      <th>withdrawal_id</th>
      <th>withdrawal_status</th>
      <th>...</th>
      <th>withdrawal_creation_date</th>
      <th>withdrawal_effective_date</th>
      <th>withdrawal_reason</th>
      <th>is_definitive_withdrawal</th>
      <th>start_date</th>
      <th>snapshot_date</th>
      <th>end_date</th>
      <th>permanencia</th>
      <th>dias_permanencia</th>
      <th>is_active</th>
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
      <td>4758.0</td>
      <td>Cerrada</td>
      <td>...</td>
      <td>2015-01-16</td>
      <td>2015-03-01</td>
      <td>RATIO RESULTADO-INVERSION</td>
      <td>1.0</td>
      <td>2025-01-07</td>
      <td>2025-05-01</td>
      <td>2025-05-01</td>
      <td>114 days</td>
      <td>114</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>986</td>
      <td>1</td>
      <td>2025-05-01 02:46:04</td>
      <td>Álava</td>
      <td>194.0</td>
      <td>2025-01-07</td>
      <td>2025-01-07</td>
      <td>2025-05-17</td>
      <td>23029.0</td>
      <td>Cerrada</td>
      <td>...</td>
      <td>2018-10-08</td>
      <td>2018-11-01</td>
      <td>RESULTADOS</td>
      <td>1.0</td>
      <td>2025-01-07</td>
      <td>2025-05-01</td>
      <td>2025-05-01</td>
      <td>114 days</td>
      <td>114</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>986</td>
      <td>1</td>
      <td>2025-05-01 02:46:04</td>
      <td>Álava</td>
      <td>194.0</td>
      <td>2025-01-07</td>
      <td>2025-01-07</td>
      <td>2025-05-17</td>
      <td>32827.0</td>
      <td>Cerrada</td>
      <td>...</td>
      <td>2020-11-17</td>
      <td>2020-11-20</td>
      <td>Cambio a Bundle Online</td>
      <td>0.0</td>
      <td>2025-01-07</td>
      <td>2025-05-01</td>
      <td>2025-05-01</td>
      <td>114 days</td>
      <td>114</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>986</td>
      <td>1</td>
      <td>2025-05-01 02:46:04</td>
      <td>Álava</td>
      <td>194.0</td>
      <td>2025-01-07</td>
      <td>2025-01-07</td>
      <td>2025-05-17</td>
      <td>33518.0</td>
      <td>Cerrada</td>
      <td>...</td>
      <td>2021-01-05</td>
      <td>2021-02-01</td>
      <td>RATIO RESULTADO-INVERSION</td>
      <td>1.0</td>
      <td>2025-01-07</td>
      <td>2025-05-01</td>
      <td>2025-05-01</td>
      <td>114 days</td>
      <td>114</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>986</td>
      <td>1</td>
      <td>2025-05-01 02:46:04</td>
      <td>Álava</td>
      <td>194.0</td>
      <td>2025-01-07</td>
      <td>2025-01-07</td>
      <td>2025-05-17</td>
      <td>54519.0</td>
      <td>Cerrada</td>
      <td>...</td>
      <td>2025-03-18</td>
      <td>2025-05-01</td>
      <td>RATIO RESULTADO-INVERSION</td>
      <td>1.0</td>
      <td>2025-01-07</td>
      <td>2025-05-01</td>
      <td>2025-05-01</td>
      <td>114 days</td>
      <td>114</td>
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
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
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
      <th>23718</th>
      <td>4194</td>
      <td>50</td>
      <td>2023-10-10 11:21:22</td>
      <td>Zaragoza</td>
      <td>NaN</td>
      <td>2022-07-21</td>
      <td>NaT</td>
      <td>2025-06-30</td>
      <td>46264.0</td>
      <td>Cerrada</td>
      <td>...</td>
      <td>2023-10-09</td>
      <td>2023-10-09</td>
      <td>Cambio de Contrato/propuesta/producto</td>
      <td>0.0</td>
      <td>2022-07-21</td>
      <td>2023-10-10</td>
      <td>2023-10-10</td>
      <td>446 days</td>
      <td>446</td>
      <td>1</td>
    </tr>
    <tr>
      <th>23719</th>
      <td>4194</td>
      <td>50</td>
      <td>2023-10-10 11:21:22</td>
      <td>Zaragoza</td>
      <td>NaN</td>
      <td>2022-07-21</td>
      <td>NaT</td>
      <td>2025-06-30</td>
      <td>47376.0</td>
      <td>Cerrada</td>
      <td>...</td>
      <td>2023-12-21</td>
      <td>NaT</td>
      <td>FALTA DE USO/TIEMPO</td>
      <td>1.0</td>
      <td>2022-07-21</td>
      <td>2023-10-10</td>
      <td>2023-10-10</td>
      <td>446 days</td>
      <td>446</td>
      <td>1</td>
    </tr>
    <tr>
      <th>23720</th>
      <td>4194</td>
      <td>50</td>
      <td>2023-10-10 11:21:22</td>
      <td>Zaragoza</td>
      <td>NaN</td>
      <td>2022-07-21</td>
      <td>NaT</td>
      <td>2025-06-30</td>
      <td>48614.0</td>
      <td>Cerrada</td>
      <td>...</td>
      <td>2024-03-12</td>
      <td>NaT</td>
      <td>RAZONES ECONOMICAS</td>
      <td>1.0</td>
      <td>2022-07-21</td>
      <td>2023-10-10</td>
      <td>2023-10-10</td>
      <td>446 days</td>
      <td>446</td>
      <td>1</td>
    </tr>
    <tr>
      <th>23721</th>
      <td>4194</td>
      <td>50</td>
      <td>2023-10-10 11:21:22</td>
      <td>Zaragoza</td>
      <td>NaN</td>
      <td>2022-07-21</td>
      <td>NaT</td>
      <td>2025-06-30</td>
      <td>51090.0</td>
      <td>Cerrada</td>
      <td>...</td>
      <td>2024-08-29</td>
      <td>NaT</td>
      <td>RESULTADOS</td>
      <td>1.0</td>
      <td>2022-07-21</td>
      <td>2023-10-10</td>
      <td>2023-10-10</td>
      <td>446 days</td>
      <td>446</td>
      <td>1</td>
    </tr>
    <tr>
      <th>23722</th>
      <td>4194</td>
      <td>50</td>
      <td>2023-10-10 11:21:22</td>
      <td>Zaragoza</td>
      <td>NaN</td>
      <td>2022-07-21</td>
      <td>NaT</td>
      <td>2025-06-30</td>
      <td>53130.0</td>
      <td>Cerrada</td>
      <td>...</td>
      <td>2024-12-30</td>
      <td>NaT</td>
      <td>FALTA DE USO/TIEMPO</td>
      <td>1.0</td>
      <td>2022-07-21</td>
      <td>2023-10-10</td>
      <td>2023-10-10</td>
      <td>446 days</td>
      <td>446</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>20644 rows × 21 columns</p>
</div>



Analizamos los advertisers que no son churn, al menos hasta la fecha estudiada en cada caso


```python
ids_no_definitive = (
    df_churn
    .groupby("advertiser_zrive_id")["is_definitive_withdrawal"]
    .max()
    .loc[lambda s: s == 0]
    .index
)


df_churn[df_churn["advertiser_zrive_id"].isin(ids_no_definitive)]["advertiser_zrive_id"].unique()
```




    array([4973, 6791, 1874, 1511, 1222, 2027, 2601, 4614, 1046, 6112, 3855,
           6588, 2634, 6876, 6591, 5464, 6584,  246, 3328, 6275, 5398,  212,
           5324, 1736, 2911, 3224, 4817, 2534, 2485, 6587, 2724, 2701, 5347,
           6705, 4183,  147, 6123, 2354, 5038, 5325,  801,  515, 6128, 2031,
           6126, 4746, 4348, 4751, 3924,  729, 6236, 3942, 6637, 4904, 5100,
            126, 6870, 3873, 4972, 5107, 2664, 5900, 2373, 5967, 6604, 5826,
           1756, 3863, 2604, 3636,  315, 6611, 3464, 6634, 5432, 6633, 6751,
           1162, 2671,  578, 4231,   60, 4533, 6477, 6624, 4826, 5577, 4685,
           5424, 3366, 6762, 3483, 5889, 6099, 6102, 6105, 6101, 6176, 5655,
           6709, 6652, 1219, 3391, 6755,  482, 2812, 6181, 6710, 6183, 4779,
           4260, 2151, 1637, 6186,  419, 1557, 5326, 3689, 3141, 1835, 4792,
           4401, 6017, 6168, 6165, 6555, 6579,  308, 6750, 5348, 6654, 6752,
            253, 3571, 6173, 2184, 5321, 6029, 6177, 6656, 6657,  847, 6659,
           5488, 5067, 1577, 6660, 5903, 1500, 2709, 1794, 1304, 5110, 5707,
           6589, 6599, 2711, 5829, 5225, 5346, 6170, 2725, 1771, 3448, 2010,
           2326, 2650, 4722, 6163, 1867, 4096,   69, 4697, 5751,  134,  328,
           1442,  886, 5445, 5694, 6530, 2648, 4038, 4350, 5840, 6164, 6351,
            267, 4280, 5924, 6097,   65, 2647, 2169, 2673, 4748, 6463, 4262,
           6719, 5846, 6483, 3891, 1784, 2117, 5288, 4864, 5598, 2926, 6724,
            103, 2646,  144, 3277, 4285, 5371, 1793, 2008, 6081, 1816, 1538,
           1931, 2339, 2428,  606, 1300, 4994, 5363, 4160, 6745, 1207, 2442,
           6161, 3555, 3196, 6687, 2541, 5669, 1879, 4847, 1091, 1194, 2028,
           2517, 3667, 5998, 4429, 2889, 3164, 5323, 2532, 5720, 6153, 5308,
           2721, 2193, 4295, 5977, 4753, 6150, 6802, 1747, 3359, 2551, 5839,
           6690, 1026, 2621,  270, 1812, 1956, 5322, 4180, 2188, 2584, 4564,
           6691, 5142, 6726, 2612, 1550, 2748,  463, 3384,  210, 2204, 4822,
           6693, 1339, 6187, 2744, 5320, 2716, 6448, 6663, 4703, 5520, 3989,
           6728,  452, 3520,  493, 5695, 4497, 6699, 3647, 6169, 2474,  910,
           6171, 5965, 6557, 6641, 3286, 5389, 6130, 2506,  151, 3897, 6730,
           1203, 1545, 3707, 4928,  467, 3649, 2276,  309,  101, 5680, 6612,
           4989, 6609, 5662, 2055,  872, 2199, 3896, 3725, 6731, 5319, 2224,
           2179, 2380, 2477, 5663, 6607,  158, 5457, 2372, 6389, 6606, 2185,
           2746, 5436, 6753, 2265, 1095, 5711, 2708, 2218, 4023, 4380, 3116,
           5127, 5013, 5129, 5130, 3284, 4998, 4882, 5533, 5983, 3129, 3138,
           6079, 4650, 5715, 3634, 3612, 4316, 4317, 3626, 5973, 4322, 4326,
           5966, 4649, 3639, 4330, 3628, 3615, 4333, 3640, 4335, 5406, 4336,
           3616, 4339, 4975, 3034, 4079, 4332, 4082, 4085, 5886, 3645, 4378,
           4648, 5716, 3651, 3646, 6381, 4458, 4976, 3853, 3623, 3650, 3614,
           3710,  773, 5945, 2351, 6087])




```python
df_churn.loc[df_churn["advertiser_zrive_id"] == 4333].sort_values(by="end_date", ascending=False)
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
      <th>withdrawal_id</th>
      <th>withdrawal_status</th>
      <th>...</th>
      <th>withdrawal_creation_date</th>
      <th>withdrawal_effective_date</th>
      <th>withdrawal_reason</th>
      <th>is_definitive_withdrawal</th>
      <th>start_date</th>
      <th>snapshot_date</th>
      <th>end_date</th>
      <th>permanencia</th>
      <th>dias_permanencia</th>
      <th>is_active</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>20504</th>
      <td>4333</td>
      <td>18</td>
      <td>2025-01-23 12:02:53</td>
      <td>Girona</td>
      <td>125.0</td>
      <td>2022-12-16</td>
      <td>NaT</td>
      <td>2024-12-31</td>
      <td>41537.0</td>
      <td>Cerrada</td>
      <td>...</td>
      <td>2022-12-16</td>
      <td>2022-12-19</td>
      <td>Upselling-cambio de contrato</td>
      <td>0.0</td>
      <td>2022-12-16</td>
      <td>2025-01-23</td>
      <td>2024-12-31</td>
      <td>746 days</td>
      <td>746</td>
      <td>0</td>
    </tr>
    <tr>
      <th>20505</th>
      <td>4333</td>
      <td>18</td>
      <td>2025-01-23 12:02:53</td>
      <td>Girona</td>
      <td>125.0</td>
      <td>2022-12-16</td>
      <td>NaT</td>
      <td>2024-12-31</td>
      <td>43578.0</td>
      <td>Cerrada</td>
      <td>...</td>
      <td>2023-04-14</td>
      <td>2023-04-14</td>
      <td>Cambio de Contrato/propuesta/producto</td>
      <td>0.0</td>
      <td>2022-12-16</td>
      <td>2025-01-23</td>
      <td>2024-12-31</td>
      <td>746 days</td>
      <td>746</td>
      <td>0</td>
    </tr>
    <tr>
      <th>20506</th>
      <td>4333</td>
      <td>18</td>
      <td>2025-01-23 12:02:53</td>
      <td>Girona</td>
      <td>125.0</td>
      <td>2022-12-16</td>
      <td>NaT</td>
      <td>2024-12-31</td>
      <td>45897.0</td>
      <td>Cerrada</td>
      <td>...</td>
      <td>2023-09-14</td>
      <td>2023-09-14</td>
      <td>Upselling-cambio de contrato</td>
      <td>0.0</td>
      <td>2022-12-16</td>
      <td>2025-01-23</td>
      <td>2024-12-31</td>
      <td>746 days</td>
      <td>746</td>
      <td>0</td>
    </tr>
    <tr>
      <th>20507</th>
      <td>4333</td>
      <td>18</td>
      <td>2025-01-23 12:02:53</td>
      <td>Girona</td>
      <td>125.0</td>
      <td>2022-12-16</td>
      <td>NaT</td>
      <td>2024-12-31</td>
      <td>51907.0</td>
      <td>Cerrada</td>
      <td>...</td>
      <td>2024-10-10</td>
      <td>2024-10-10</td>
      <td>Reestructuración cuentas de grupo</td>
      <td>0.0</td>
      <td>2022-12-16</td>
      <td>2025-01-23</td>
      <td>2024-12-31</td>
      <td>746 days</td>
      <td>746</td>
      <td>0</td>
    </tr>
    <tr>
      <th>20508</th>
      <td>4333</td>
      <td>18</td>
      <td>2025-01-23 12:02:53</td>
      <td>Girona</td>
      <td>125.0</td>
      <td>2022-12-16</td>
      <td>NaT</td>
      <td>2024-12-31</td>
      <td>51927.0</td>
      <td>Cerrada</td>
      <td>...</td>
      <td>2024-10-11</td>
      <td>2024-10-16</td>
      <td>Upselling-cambio de contrato</td>
      <td>0.0</td>
      <td>2022-12-16</td>
      <td>2025-01-23</td>
      <td>2024-12-31</td>
      <td>746 days</td>
      <td>746</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 21 columns</p>
</div>



Ahora creamos una variable nueva que representa si un advertiser ya ha estado con anterioridad, es decir, si más de una fecha de inicio de contrato 
- **Usuario nuevo**: sólo tiene **min_start_contrato_date** ó tiene **max_start_contrato_nuevo_date** pero es igual a **min_start_contrato_date**
- **Usuario antiguo**: tiene un **max_start_contrato_nuevo_date** != **min_start_contrato_date**


```python
df_churn["is_new"] = (
    df_churn["max_start_contrato_nuevo_date"].isna() |
    (df_churn["max_start_contrato_nuevo_date"] == df_churn["min_start_contrato_date"]) # Si fuera NaT, siempre daría False
).astype(int)

df_churn.groupby("advertiser_zrive_id")["is_new"].max().value_counts().plot(kind='bar')
plt.title("Proporción de usuarios nuevos vs. antiguos")
```

    /tmp/ipykernel_160604/2797343098.py:1: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      df_churn["is_new"] = (





    Text(0.5, 1.0, 'Proporción de usuarios nuevos vs. antiguos')




    
![png](EDA_jtc_files/EDA_jtc_145_2.png)
    


Casi todos los usuarios son nuevos

#### Nueva feature **n_withdrawal_attempts**

**n_withdrawal_attempts**: se define como el número de intentos de withdrawals, obtenido con los valores únicos de **withdrawal_id** de cada advertiser. Si un advertiser tiene 6 withdrawal_id distintos, se cuentan únicamente aquellos donde el withdrawal no llegó a ser definitivo (is_definitive_withdrawal = 0)



```python
df_churn.head()
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
      <th>withdrawal_id</th>
      <th>withdrawal_status</th>
      <th>...</th>
      <th>withdrawal_effective_date</th>
      <th>withdrawal_reason</th>
      <th>is_definitive_withdrawal</th>
      <th>start_date</th>
      <th>snapshot_date</th>
      <th>end_date</th>
      <th>permanencia</th>
      <th>dias_permanencia</th>
      <th>is_active</th>
      <th>is_new</th>
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
      <td>4758.0</td>
      <td>Cerrada</td>
      <td>...</td>
      <td>2015-03-01</td>
      <td>RATIO RESULTADO-INVERSION</td>
      <td>1.0</td>
      <td>2025-01-07</td>
      <td>2025-05-01</td>
      <td>2025-05-01</td>
      <td>114 days</td>
      <td>114</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>986</td>
      <td>1</td>
      <td>2025-05-01 02:46:04</td>
      <td>Álava</td>
      <td>194.0</td>
      <td>2025-01-07</td>
      <td>2025-01-07</td>
      <td>2025-05-17</td>
      <td>23029.0</td>
      <td>Cerrada</td>
      <td>...</td>
      <td>2018-11-01</td>
      <td>RESULTADOS</td>
      <td>1.0</td>
      <td>2025-01-07</td>
      <td>2025-05-01</td>
      <td>2025-05-01</td>
      <td>114 days</td>
      <td>114</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>986</td>
      <td>1</td>
      <td>2025-05-01 02:46:04</td>
      <td>Álava</td>
      <td>194.0</td>
      <td>2025-01-07</td>
      <td>2025-01-07</td>
      <td>2025-05-17</td>
      <td>32827.0</td>
      <td>Cerrada</td>
      <td>...</td>
      <td>2020-11-20</td>
      <td>Cambio a Bundle Online</td>
      <td>0.0</td>
      <td>2025-01-07</td>
      <td>2025-05-01</td>
      <td>2025-05-01</td>
      <td>114 days</td>
      <td>114</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>986</td>
      <td>1</td>
      <td>2025-05-01 02:46:04</td>
      <td>Álava</td>
      <td>194.0</td>
      <td>2025-01-07</td>
      <td>2025-01-07</td>
      <td>2025-05-17</td>
      <td>33518.0</td>
      <td>Cerrada</td>
      <td>...</td>
      <td>2021-02-01</td>
      <td>RATIO RESULTADO-INVERSION</td>
      <td>1.0</td>
      <td>2025-01-07</td>
      <td>2025-05-01</td>
      <td>2025-05-01</td>
      <td>114 days</td>
      <td>114</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>986</td>
      <td>1</td>
      <td>2025-05-01 02:46:04</td>
      <td>Álava</td>
      <td>194.0</td>
      <td>2025-01-07</td>
      <td>2025-01-07</td>
      <td>2025-05-17</td>
      <td>54519.0</td>
      <td>Cerrada</td>
      <td>...</td>
      <td>2025-05-01</td>
      <td>RATIO RESULTADO-INVERSION</td>
      <td>1.0</td>
      <td>2025-01-07</td>
      <td>2025-05-01</td>
      <td>2025-05-01</td>
      <td>114 days</td>
      <td>114</td>
      <td>1</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 22 columns</p>
</div>




```python
df_churn = df_churn.copy()     
df_churn["n_withdrawal_attempts"] = (
    df_churn["withdrawal_id"]
      .where(df_churn["is_definitive_withdrawal"].eq(0))
      .groupby(df_churn["advertiser_zrive_id"])
      .transform("nunique")
      .fillna(0)
      .astype(int)
)
```


```python
df_churn[df_churn["snapshot_date"]>=df_churn["end_date"]]
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
      <th>withdrawal_id</th>
      <th>withdrawal_status</th>
      <th>...</th>
      <th>withdrawal_reason</th>
      <th>is_definitive_withdrawal</th>
      <th>start_date</th>
      <th>snapshot_date</th>
      <th>end_date</th>
      <th>permanencia</th>
      <th>dias_permanencia</th>
      <th>is_active</th>
      <th>is_new</th>
      <th>n_withdrawal_attempts</th>
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
      <td>4758.0</td>
      <td>Cerrada</td>
      <td>...</td>
      <td>RATIO RESULTADO-INVERSION</td>
      <td>1.0</td>
      <td>2025-01-07</td>
      <td>2025-05-01</td>
      <td>2025-05-01</td>
      <td>114 days</td>
      <td>114</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>986</td>
      <td>1</td>
      <td>2025-05-01 02:46:04</td>
      <td>Álava</td>
      <td>194.0</td>
      <td>2025-01-07</td>
      <td>2025-01-07</td>
      <td>2025-05-17</td>
      <td>23029.0</td>
      <td>Cerrada</td>
      <td>...</td>
      <td>RESULTADOS</td>
      <td>1.0</td>
      <td>2025-01-07</td>
      <td>2025-05-01</td>
      <td>2025-05-01</td>
      <td>114 days</td>
      <td>114</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>986</td>
      <td>1</td>
      <td>2025-05-01 02:46:04</td>
      <td>Álava</td>
      <td>194.0</td>
      <td>2025-01-07</td>
      <td>2025-01-07</td>
      <td>2025-05-17</td>
      <td>32827.0</td>
      <td>Cerrada</td>
      <td>...</td>
      <td>Cambio a Bundle Online</td>
      <td>0.0</td>
      <td>2025-01-07</td>
      <td>2025-05-01</td>
      <td>2025-05-01</td>
      <td>114 days</td>
      <td>114</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>986</td>
      <td>1</td>
      <td>2025-05-01 02:46:04</td>
      <td>Álava</td>
      <td>194.0</td>
      <td>2025-01-07</td>
      <td>2025-01-07</td>
      <td>2025-05-17</td>
      <td>33518.0</td>
      <td>Cerrada</td>
      <td>...</td>
      <td>RATIO RESULTADO-INVERSION</td>
      <td>1.0</td>
      <td>2025-01-07</td>
      <td>2025-05-01</td>
      <td>2025-05-01</td>
      <td>114 days</td>
      <td>114</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>986</td>
      <td>1</td>
      <td>2025-05-01 02:46:04</td>
      <td>Álava</td>
      <td>194.0</td>
      <td>2025-01-07</td>
      <td>2025-01-07</td>
      <td>2025-05-17</td>
      <td>54519.0</td>
      <td>Cerrada</td>
      <td>...</td>
      <td>RATIO RESULTADO-INVERSION</td>
      <td>1.0</td>
      <td>2025-01-07</td>
      <td>2025-05-01</td>
      <td>2025-05-01</td>
      <td>114 days</td>
      <td>114</td>
      <td>1</td>
      <td>1</td>
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
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
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
      <th>23718</th>
      <td>4194</td>
      <td>50</td>
      <td>2023-10-10 11:21:22</td>
      <td>Zaragoza</td>
      <td>NaN</td>
      <td>2022-07-21</td>
      <td>NaT</td>
      <td>2025-06-30</td>
      <td>46264.0</td>
      <td>Cerrada</td>
      <td>...</td>
      <td>Cambio de Contrato/propuesta/producto</td>
      <td>0.0</td>
      <td>2022-07-21</td>
      <td>2023-10-10</td>
      <td>2023-10-10</td>
      <td>446 days</td>
      <td>446</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>23719</th>
      <td>4194</td>
      <td>50</td>
      <td>2023-10-10 11:21:22</td>
      <td>Zaragoza</td>
      <td>NaN</td>
      <td>2022-07-21</td>
      <td>NaT</td>
      <td>2025-06-30</td>
      <td>47376.0</td>
      <td>Cerrada</td>
      <td>...</td>
      <td>FALTA DE USO/TIEMPO</td>
      <td>1.0</td>
      <td>2022-07-21</td>
      <td>2023-10-10</td>
      <td>2023-10-10</td>
      <td>446 days</td>
      <td>446</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>23720</th>
      <td>4194</td>
      <td>50</td>
      <td>2023-10-10 11:21:22</td>
      <td>Zaragoza</td>
      <td>NaN</td>
      <td>2022-07-21</td>
      <td>NaT</td>
      <td>2025-06-30</td>
      <td>48614.0</td>
      <td>Cerrada</td>
      <td>...</td>
      <td>RAZONES ECONOMICAS</td>
      <td>1.0</td>
      <td>2022-07-21</td>
      <td>2023-10-10</td>
      <td>2023-10-10</td>
      <td>446 days</td>
      <td>446</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>23721</th>
      <td>4194</td>
      <td>50</td>
      <td>2023-10-10 11:21:22</td>
      <td>Zaragoza</td>
      <td>NaN</td>
      <td>2022-07-21</td>
      <td>NaT</td>
      <td>2025-06-30</td>
      <td>51090.0</td>
      <td>Cerrada</td>
      <td>...</td>
      <td>RESULTADOS</td>
      <td>1.0</td>
      <td>2022-07-21</td>
      <td>2023-10-10</td>
      <td>2023-10-10</td>
      <td>446 days</td>
      <td>446</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>23722</th>
      <td>4194</td>
      <td>50</td>
      <td>2023-10-10 11:21:22</td>
      <td>Zaragoza</td>
      <td>NaN</td>
      <td>2022-07-21</td>
      <td>NaT</td>
      <td>2025-06-30</td>
      <td>53130.0</td>
      <td>Cerrada</td>
      <td>...</td>
      <td>FALTA DE USO/TIEMPO</td>
      <td>1.0</td>
      <td>2022-07-21</td>
      <td>2023-10-10</td>
      <td>2023-10-10</td>
      <td>446 days</td>
      <td>446</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>21435 rows × 23 columns</p>
</div>




```python
df_churn[df_churn["n_withdrawal_attempts"]>1]
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
      <th>withdrawal_id</th>
      <th>withdrawal_status</th>
      <th>...</th>
      <th>withdrawal_reason</th>
      <th>is_definitive_withdrawal</th>
      <th>start_date</th>
      <th>snapshot_date</th>
      <th>end_date</th>
      <th>permanencia</th>
      <th>dias_permanencia</th>
      <th>is_active</th>
      <th>is_new</th>
      <th>n_withdrawal_attempts</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>44</th>
      <td>534</td>
      <td>8</td>
      <td>2024-07-05 11:33:04</td>
      <td>Barcelona</td>
      <td>41.0</td>
      <td>2024-04-30</td>
      <td>NaT</td>
      <td>2024-06-29</td>
      <td>4892.0</td>
      <td>Cerrada</td>
      <td>...</td>
      <td>RESULTADOS</td>
      <td>1.0</td>
      <td>2024-04-30</td>
      <td>2024-07-05</td>
      <td>2024-06-29</td>
      <td>60 days</td>
      <td>60</td>
      <td>0</td>
      <td>1</td>
      <td>5</td>
    </tr>
    <tr>
      <th>45</th>
      <td>534</td>
      <td>8</td>
      <td>2024-07-05 11:33:04</td>
      <td>Barcelona</td>
      <td>41.0</td>
      <td>2024-04-30</td>
      <td>NaT</td>
      <td>2024-06-29</td>
      <td>14657.0</td>
      <td>Cerrada</td>
      <td>...</td>
      <td>FALTA DE PRODUCTO</td>
      <td>1.0</td>
      <td>2024-04-30</td>
      <td>2024-07-05</td>
      <td>2024-06-29</td>
      <td>60 days</td>
      <td>60</td>
      <td>0</td>
      <td>1</td>
      <td>5</td>
    </tr>
    <tr>
      <th>46</th>
      <td>534</td>
      <td>8</td>
      <td>2024-07-05 11:33:04</td>
      <td>Barcelona</td>
      <td>41.0</td>
      <td>2024-04-30</td>
      <td>NaT</td>
      <td>2024-06-29</td>
      <td>28025.0</td>
      <td>Cerrada</td>
      <td>...</td>
      <td>Cambio a Bundle Online</td>
      <td>0.0</td>
      <td>2024-04-30</td>
      <td>2024-07-05</td>
      <td>2024-06-29</td>
      <td>60 days</td>
      <td>60</td>
      <td>0</td>
      <td>1</td>
      <td>5</td>
    </tr>
    <tr>
      <th>47</th>
      <td>534</td>
      <td>8</td>
      <td>2024-07-05 11:33:04</td>
      <td>Barcelona</td>
      <td>41.0</td>
      <td>2024-04-30</td>
      <td>NaT</td>
      <td>2024-06-29</td>
      <td>28175.0</td>
      <td>Cerrada</td>
      <td>...</td>
      <td>Cambio a Bundle Online</td>
      <td>0.0</td>
      <td>2024-04-30</td>
      <td>2024-07-05</td>
      <td>2024-06-29</td>
      <td>60 days</td>
      <td>60</td>
      <td>0</td>
      <td>1</td>
      <td>5</td>
    </tr>
    <tr>
      <th>48</th>
      <td>534</td>
      <td>8</td>
      <td>2024-07-05 11:33:04</td>
      <td>Barcelona</td>
      <td>41.0</td>
      <td>2024-04-30</td>
      <td>NaT</td>
      <td>2024-06-29</td>
      <td>28220.0</td>
      <td>Cerrada</td>
      <td>...</td>
      <td>Upselling-cambio de contrato</td>
      <td>0.0</td>
      <td>2024-04-30</td>
      <td>2024-07-05</td>
      <td>2024-06-29</td>
      <td>60 days</td>
      <td>60</td>
      <td>0</td>
      <td>1</td>
      <td>5</td>
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
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
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
      <th>23687</th>
      <td>3166</td>
      <td>46</td>
      <td>2025-02-27 14:21:20</td>
      <td>Valencia</td>
      <td>NaN</td>
      <td>2021-05-14</td>
      <td>2024-11-27</td>
      <td>2025-06-30</td>
      <td>45070.0</td>
      <td>Cerrada</td>
      <td>...</td>
      <td>MOROSIDAD</td>
      <td>1.0</td>
      <td>2024-11-27</td>
      <td>2025-02-27</td>
      <td>2025-02-27</td>
      <td>92 days</td>
      <td>92</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>23688</th>
      <td>3166</td>
      <td>46</td>
      <td>2025-02-27 14:21:20</td>
      <td>Valencia</td>
      <td>NaN</td>
      <td>2021-05-14</td>
      <td>2024-11-27</td>
      <td>2025-06-30</td>
      <td>46400.0</td>
      <td>Cerrada</td>
      <td>...</td>
      <td>Upselling-cambio de contrato</td>
      <td>0.0</td>
      <td>2024-11-27</td>
      <td>2025-02-27</td>
      <td>2025-02-27</td>
      <td>92 days</td>
      <td>92</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>23689</th>
      <td>3166</td>
      <td>46</td>
      <td>2025-02-27 14:21:20</td>
      <td>Valencia</td>
      <td>NaN</td>
      <td>2021-05-14</td>
      <td>2024-11-27</td>
      <td>2025-06-30</td>
      <td>48906.0</td>
      <td>Cerrada</td>
      <td>...</td>
      <td>RESULTADOS</td>
      <td>1.0</td>
      <td>2024-11-27</td>
      <td>2025-02-27</td>
      <td>2025-02-27</td>
      <td>92 days</td>
      <td>92</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>23690</th>
      <td>3166</td>
      <td>46</td>
      <td>2025-02-27 14:21:20</td>
      <td>Valencia</td>
      <td>NaN</td>
      <td>2021-05-14</td>
      <td>2024-11-27</td>
      <td>2025-06-30</td>
      <td>50079.0</td>
      <td>Cerrada</td>
      <td>...</td>
      <td>RESULTADOS</td>
      <td>1.0</td>
      <td>2024-11-27</td>
      <td>2025-02-27</td>
      <td>2025-02-27</td>
      <td>92 days</td>
      <td>92</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>23691</th>
      <td>3166</td>
      <td>46</td>
      <td>2025-02-27 14:21:20</td>
      <td>Valencia</td>
      <td>NaN</td>
      <td>2021-05-14</td>
      <td>2024-11-27</td>
      <td>2025-06-30</td>
      <td>55149.0</td>
      <td>Cerrada</td>
      <td>...</td>
      <td>FALTA DE USO/TIEMPO</td>
      <td>1.0</td>
      <td>2024-11-27</td>
      <td>2025-02-27</td>
      <td>2025-02-27</td>
      <td>92 days</td>
      <td>92</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
<p>10516 rows × 23 columns</p>
</div>



### Ejemplo:
- Vemos el comportamiento de un usuario y el nº de intentos de withdrawal (nótese que los intentos que fueron definitivos (is_definitive_withdrawal=1) lógicamente no se cuentan). 
- Advertiser **534**:
¿Cuántos is_definitive_withdrawal == 0 únicos tiene? --> 5


```python
df_churn[df_churn["advertiser_zrive_id"]==534]
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
      <th>withdrawal_id</th>
      <th>withdrawal_status</th>
      <th>...</th>
      <th>withdrawal_reason</th>
      <th>is_definitive_withdrawal</th>
      <th>start_date</th>
      <th>snapshot_date</th>
      <th>end_date</th>
      <th>permanencia</th>
      <th>dias_permanencia</th>
      <th>is_active</th>
      <th>is_new</th>
      <th>n_withdrawal_attempts</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>44</th>
      <td>534</td>
      <td>8</td>
      <td>2024-07-05 11:33:04</td>
      <td>Barcelona</td>
      <td>41.0</td>
      <td>2024-04-30</td>
      <td>NaT</td>
      <td>2024-06-29</td>
      <td>4892.0</td>
      <td>Cerrada</td>
      <td>...</td>
      <td>RESULTADOS</td>
      <td>1.0</td>
      <td>2024-04-30</td>
      <td>2024-07-05</td>
      <td>2024-06-29</td>
      <td>60 days</td>
      <td>60</td>
      <td>0</td>
      <td>1</td>
      <td>5</td>
    </tr>
    <tr>
      <th>45</th>
      <td>534</td>
      <td>8</td>
      <td>2024-07-05 11:33:04</td>
      <td>Barcelona</td>
      <td>41.0</td>
      <td>2024-04-30</td>
      <td>NaT</td>
      <td>2024-06-29</td>
      <td>14657.0</td>
      <td>Cerrada</td>
      <td>...</td>
      <td>FALTA DE PRODUCTO</td>
      <td>1.0</td>
      <td>2024-04-30</td>
      <td>2024-07-05</td>
      <td>2024-06-29</td>
      <td>60 days</td>
      <td>60</td>
      <td>0</td>
      <td>1</td>
      <td>5</td>
    </tr>
    <tr>
      <th>46</th>
      <td>534</td>
      <td>8</td>
      <td>2024-07-05 11:33:04</td>
      <td>Barcelona</td>
      <td>41.0</td>
      <td>2024-04-30</td>
      <td>NaT</td>
      <td>2024-06-29</td>
      <td>28025.0</td>
      <td>Cerrada</td>
      <td>...</td>
      <td>Cambio a Bundle Online</td>
      <td>0.0</td>
      <td>2024-04-30</td>
      <td>2024-07-05</td>
      <td>2024-06-29</td>
      <td>60 days</td>
      <td>60</td>
      <td>0</td>
      <td>1</td>
      <td>5</td>
    </tr>
    <tr>
      <th>47</th>
      <td>534</td>
      <td>8</td>
      <td>2024-07-05 11:33:04</td>
      <td>Barcelona</td>
      <td>41.0</td>
      <td>2024-04-30</td>
      <td>NaT</td>
      <td>2024-06-29</td>
      <td>28175.0</td>
      <td>Cerrada</td>
      <td>...</td>
      <td>Cambio a Bundle Online</td>
      <td>0.0</td>
      <td>2024-04-30</td>
      <td>2024-07-05</td>
      <td>2024-06-29</td>
      <td>60 days</td>
      <td>60</td>
      <td>0</td>
      <td>1</td>
      <td>5</td>
    </tr>
    <tr>
      <th>48</th>
      <td>534</td>
      <td>8</td>
      <td>2024-07-05 11:33:04</td>
      <td>Barcelona</td>
      <td>41.0</td>
      <td>2024-04-30</td>
      <td>NaT</td>
      <td>2024-06-29</td>
      <td>28220.0</td>
      <td>Cerrada</td>
      <td>...</td>
      <td>Upselling-cambio de contrato</td>
      <td>0.0</td>
      <td>2024-04-30</td>
      <td>2024-07-05</td>
      <td>2024-06-29</td>
      <td>60 days</td>
      <td>60</td>
      <td>0</td>
      <td>1</td>
      <td>5</td>
    </tr>
    <tr>
      <th>49</th>
      <td>534</td>
      <td>8</td>
      <td>2024-07-05 11:33:04</td>
      <td>Barcelona</td>
      <td>41.0</td>
      <td>2024-04-30</td>
      <td>NaT</td>
      <td>2024-06-29</td>
      <td>28966.0</td>
      <td>Cerrada</td>
      <td>...</td>
      <td>CORONAVIRUS</td>
      <td>1.0</td>
      <td>2024-04-30</td>
      <td>2024-07-05</td>
      <td>2024-06-29</td>
      <td>60 days</td>
      <td>60</td>
      <td>0</td>
      <td>1</td>
      <td>5</td>
    </tr>
    <tr>
      <th>50</th>
      <td>534</td>
      <td>8</td>
      <td>2024-07-05 11:33:04</td>
      <td>Barcelona</td>
      <td>41.0</td>
      <td>2024-04-30</td>
      <td>NaT</td>
      <td>2024-06-29</td>
      <td>31242.0</td>
      <td>Cerrada</td>
      <td>...</td>
      <td>MOROSIDAD</td>
      <td>0.0</td>
      <td>2024-04-30</td>
      <td>2024-07-05</td>
      <td>2024-06-29</td>
      <td>60 days</td>
      <td>60</td>
      <td>0</td>
      <td>1</td>
      <td>5</td>
    </tr>
    <tr>
      <th>51</th>
      <td>534</td>
      <td>8</td>
      <td>2024-07-05 11:33:04</td>
      <td>Barcelona</td>
      <td>41.0</td>
      <td>2024-04-30</td>
      <td>NaT</td>
      <td>2024-06-29</td>
      <td>32327.0</td>
      <td>Cerrada</td>
      <td>...</td>
      <td>MOROSIDAD</td>
      <td>1.0</td>
      <td>2024-04-30</td>
      <td>2024-07-05</td>
      <td>2024-06-29</td>
      <td>60 days</td>
      <td>60</td>
      <td>0</td>
      <td>1</td>
      <td>5</td>
    </tr>
    <tr>
      <th>52</th>
      <td>534</td>
      <td>8</td>
      <td>2024-07-05 11:33:04</td>
      <td>Barcelona</td>
      <td>41.0</td>
      <td>2024-04-30</td>
      <td>NaT</td>
      <td>2024-06-29</td>
      <td>33360.0</td>
      <td>Cerrada</td>
      <td>...</td>
      <td>MOROSIDAD</td>
      <td>1.0</td>
      <td>2024-04-30</td>
      <td>2024-07-05</td>
      <td>2024-06-29</td>
      <td>60 days</td>
      <td>60</td>
      <td>0</td>
      <td>1</td>
      <td>5</td>
    </tr>
    <tr>
      <th>53</th>
      <td>534</td>
      <td>8</td>
      <td>2024-07-05 11:33:04</td>
      <td>Barcelona</td>
      <td>41.0</td>
      <td>2024-04-30</td>
      <td>NaT</td>
      <td>2024-06-29</td>
      <td>38879.0</td>
      <td>Cerrada</td>
      <td>...</td>
      <td>MOROSIDAD</td>
      <td>1.0</td>
      <td>2024-04-30</td>
      <td>2024-07-05</td>
      <td>2024-06-29</td>
      <td>60 days</td>
      <td>60</td>
      <td>0</td>
      <td>1</td>
      <td>5</td>
    </tr>
    <tr>
      <th>54</th>
      <td>534</td>
      <td>8</td>
      <td>2024-07-05 11:33:04</td>
      <td>Barcelona</td>
      <td>41.0</td>
      <td>2024-04-30</td>
      <td>NaT</td>
      <td>2024-06-29</td>
      <td>40013.0</td>
      <td>Cerrada</td>
      <td>...</td>
      <td>RESULTADOS</td>
      <td>0.0</td>
      <td>2024-04-30</td>
      <td>2024-07-05</td>
      <td>2024-06-29</td>
      <td>60 days</td>
      <td>60</td>
      <td>0</td>
      <td>1</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
<p>11 rows × 23 columns</p>
</div>



### Agregación final

Ahora, como tenemos varios registros e información de withdrawal a nivel advertiser_id, aplicamos una lógica para quedarnos con 1 fila = 1 advertiser


```python
df_churn[df_churn["advertiser_zrive_id"] == 417]
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
      <th>withdrawal_id</th>
      <th>withdrawal_status</th>
      <th>...</th>
      <th>withdrawal_reason</th>
      <th>is_definitive_withdrawal</th>
      <th>start_date</th>
      <th>snapshot_date</th>
      <th>end_date</th>
      <th>permanencia</th>
      <th>dias_permanencia</th>
      <th>is_active</th>
      <th>is_new</th>
      <th>n_withdrawal_attempts</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2917</th>
      <td>417</td>
      <td>9</td>
      <td>2025-06-11 09:38:30</td>
      <td>Burgos</td>
      <td>NaN</td>
      <td>2020-03-17</td>
      <td>2020-03-17</td>
      <td>NaT</td>
      <td>10570.0</td>
      <td>Cerrada</td>
      <td>...</td>
      <td>RATIO RESULTADO-INVERSION</td>
      <td>0.0</td>
      <td>2020-03-17</td>
      <td>2025-06-11</td>
      <td>2025-06-11</td>
      <td>1912 days</td>
      <td>1912</td>
      <td>1</td>
      <td>1</td>
      <td>7</td>
    </tr>
    <tr>
      <th>2918</th>
      <td>417</td>
      <td>9</td>
      <td>2025-06-11 09:38:30</td>
      <td>Burgos</td>
      <td>NaN</td>
      <td>2020-03-17</td>
      <td>2020-03-17</td>
      <td>NaT</td>
      <td>19933.0</td>
      <td>Cerrada</td>
      <td>...</td>
      <td>RATIO RESULTADO-INVERSION</td>
      <td>0.0</td>
      <td>2020-03-17</td>
      <td>2025-06-11</td>
      <td>2025-06-11</td>
      <td>1912 days</td>
      <td>1912</td>
      <td>1</td>
      <td>1</td>
      <td>7</td>
    </tr>
    <tr>
      <th>2919</th>
      <td>417</td>
      <td>9</td>
      <td>2025-06-11 09:38:30</td>
      <td>Burgos</td>
      <td>NaN</td>
      <td>2020-03-17</td>
      <td>2020-03-17</td>
      <td>NaT</td>
      <td>24455.0</td>
      <td>Cerrada</td>
      <td>...</td>
      <td>OTROS</td>
      <td>0.0</td>
      <td>2020-03-17</td>
      <td>2025-06-11</td>
      <td>2025-06-11</td>
      <td>1912 days</td>
      <td>1912</td>
      <td>1</td>
      <td>1</td>
      <td>7</td>
    </tr>
    <tr>
      <th>2920</th>
      <td>417</td>
      <td>9</td>
      <td>2025-06-11 09:38:30</td>
      <td>Burgos</td>
      <td>NaN</td>
      <td>2020-03-17</td>
      <td>2020-03-17</td>
      <td>NaT</td>
      <td>28731.0</td>
      <td>Cerrada</td>
      <td>...</td>
      <td>Cambio a Bundle Online</td>
      <td>0.0</td>
      <td>2020-03-17</td>
      <td>2025-06-11</td>
      <td>2025-06-11</td>
      <td>1912 days</td>
      <td>1912</td>
      <td>1</td>
      <td>1</td>
      <td>7</td>
    </tr>
    <tr>
      <th>2921</th>
      <td>417</td>
      <td>9</td>
      <td>2025-06-11 09:38:30</td>
      <td>Burgos</td>
      <td>NaN</td>
      <td>2020-03-17</td>
      <td>2020-03-17</td>
      <td>NaT</td>
      <td>44849.0</td>
      <td>Cerrada</td>
      <td>...</td>
      <td>RESULTADOS</td>
      <td>1.0</td>
      <td>2020-03-17</td>
      <td>2025-06-11</td>
      <td>2023-07-01</td>
      <td>1201 days</td>
      <td>1201</td>
      <td>0</td>
      <td>1</td>
      <td>7</td>
    </tr>
    <tr>
      <th>2922</th>
      <td>417</td>
      <td>9</td>
      <td>2025-06-11 09:38:30</td>
      <td>Burgos</td>
      <td>NaN</td>
      <td>2020-03-17</td>
      <td>2020-03-17</td>
      <td>NaT</td>
      <td>44852.0</td>
      <td>Cerrada</td>
      <td>...</td>
      <td>Cambio de Contrato/propuesta/producto</td>
      <td>0.0</td>
      <td>2020-03-17</td>
      <td>2025-06-11</td>
      <td>2025-06-11</td>
      <td>1912 days</td>
      <td>1912</td>
      <td>1</td>
      <td>1</td>
      <td>7</td>
    </tr>
    <tr>
      <th>2923</th>
      <td>417</td>
      <td>9</td>
      <td>2025-06-11 09:38:30</td>
      <td>Burgos</td>
      <td>NaN</td>
      <td>2020-03-17</td>
      <td>2020-03-17</td>
      <td>NaT</td>
      <td>46601.0</td>
      <td>Cerrada</td>
      <td>...</td>
      <td>RESULTADOS</td>
      <td>1.0</td>
      <td>2020-03-17</td>
      <td>2025-06-11</td>
      <td>2023-11-01</td>
      <td>1324 days</td>
      <td>1324</td>
      <td>0</td>
      <td>1</td>
      <td>7</td>
    </tr>
    <tr>
      <th>2924</th>
      <td>417</td>
      <td>9</td>
      <td>2025-06-11 09:38:30</td>
      <td>Burgos</td>
      <td>NaN</td>
      <td>2020-03-17</td>
      <td>2020-03-17</td>
      <td>NaT</td>
      <td>46633.0</td>
      <td>Cerrada</td>
      <td>...</td>
      <td>RESULTADOS</td>
      <td>0.0</td>
      <td>2020-03-17</td>
      <td>2025-06-11</td>
      <td>2025-06-11</td>
      <td>1912 days</td>
      <td>1912</td>
      <td>1</td>
      <td>1</td>
      <td>7</td>
    </tr>
    <tr>
      <th>2925</th>
      <td>417</td>
      <td>9</td>
      <td>2025-06-11 09:38:30</td>
      <td>Burgos</td>
      <td>NaN</td>
      <td>2020-03-17</td>
      <td>2020-03-17</td>
      <td>NaT</td>
      <td>47574.0</td>
      <td>Cerrada</td>
      <td>...</td>
      <td>RAZONES ECONOMICAS</td>
      <td>1.0</td>
      <td>2020-03-17</td>
      <td>2025-06-11</td>
      <td>2024-02-01</td>
      <td>1416 days</td>
      <td>1416</td>
      <td>0</td>
      <td>1</td>
      <td>7</td>
    </tr>
    <tr>
      <th>2926</th>
      <td>417</td>
      <td>9</td>
      <td>2025-06-11 09:38:30</td>
      <td>Burgos</td>
      <td>NaN</td>
      <td>2020-03-17</td>
      <td>2020-03-17</td>
      <td>NaT</td>
      <td>47964.0</td>
      <td>Cerrada</td>
      <td>...</td>
      <td>RESULTADOS</td>
      <td>1.0</td>
      <td>2020-03-17</td>
      <td>2025-06-11</td>
      <td>2025-06-11</td>
      <td>1912 days</td>
      <td>1912</td>
      <td>1</td>
      <td>1</td>
      <td>7</td>
    </tr>
    <tr>
      <th>2927</th>
      <td>417</td>
      <td>9</td>
      <td>2025-06-11 09:38:30</td>
      <td>Burgos</td>
      <td>NaN</td>
      <td>2020-03-17</td>
      <td>2020-03-17</td>
      <td>NaT</td>
      <td>50813.0</td>
      <td>Cerrada</td>
      <td>...</td>
      <td>RESULTADOS</td>
      <td>1.0</td>
      <td>2020-03-17</td>
      <td>2025-06-11</td>
      <td>2025-06-11</td>
      <td>1912 days</td>
      <td>1912</td>
      <td>1</td>
      <td>1</td>
      <td>7</td>
    </tr>
    <tr>
      <th>2928</th>
      <td>417</td>
      <td>9</td>
      <td>2025-06-11 09:38:30</td>
      <td>Burgos</td>
      <td>NaN</td>
      <td>2020-03-17</td>
      <td>2020-03-17</td>
      <td>NaT</td>
      <td>50834.0</td>
      <td>Cerrada</td>
      <td>...</td>
      <td>RESULTADOS</td>
      <td>1.0</td>
      <td>2020-03-17</td>
      <td>2025-06-11</td>
      <td>2025-06-11</td>
      <td>1912 days</td>
      <td>1912</td>
      <td>1</td>
      <td>1</td>
      <td>7</td>
    </tr>
    <tr>
      <th>2929</th>
      <td>417</td>
      <td>9</td>
      <td>2025-06-11 09:38:30</td>
      <td>Burgos</td>
      <td>NaN</td>
      <td>2020-03-17</td>
      <td>2020-03-17</td>
      <td>NaT</td>
      <td>52944.0</td>
      <td>Cerrada</td>
      <td>...</td>
      <td>RESULTADOS</td>
      <td>0.0</td>
      <td>2020-03-17</td>
      <td>2025-06-11</td>
      <td>2025-06-11</td>
      <td>1912 days</td>
      <td>1912</td>
      <td>1</td>
      <td>1</td>
      <td>7</td>
    </tr>
    <tr>
      <th>2930</th>
      <td>417</td>
      <td>9</td>
      <td>2025-06-11 09:38:30</td>
      <td>Burgos</td>
      <td>NaN</td>
      <td>2020-03-17</td>
      <td>2020-03-17</td>
      <td>NaT</td>
      <td>52990.0</td>
      <td>Cerrada</td>
      <td>...</td>
      <td>RESULTADOS</td>
      <td>1.0</td>
      <td>2020-03-17</td>
      <td>2025-06-11</td>
      <td>2025-06-11</td>
      <td>1912 days</td>
      <td>1912</td>
      <td>1</td>
      <td>1</td>
      <td>7</td>
    </tr>
  </tbody>
</table>
<p>14 rows × 23 columns</p>
</div>



Detectamos un "bug": por ejemplo, el usuario **417** que inició un contrato el 2020-03-17 pero se dió de baja definitiva en 3 fechas distintas... ¿con qué fecha de salida nos quedamos?
-> Decidimos que con la mínima: el churn es el PRIMER withdrawal definitivo efectivo del contrato actual. Para ello, creamos el dataframe **df_churn_first**, es decir, nos quedamos con la primera fecha de withdrawal


```python
df_churn_first = df_churn.loc[
    df_churn.groupby("advertiser_zrive_id")["end_date"].idxmin()
]

df_churn_first[df_churn_first["advertiser_zrive_id"] == 417]
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
      <th>withdrawal_id</th>
      <th>withdrawal_status</th>
      <th>...</th>
      <th>withdrawal_reason</th>
      <th>is_definitive_withdrawal</th>
      <th>start_date</th>
      <th>snapshot_date</th>
      <th>end_date</th>
      <th>permanencia</th>
      <th>dias_permanencia</th>
      <th>is_active</th>
      <th>is_new</th>
      <th>n_withdrawal_attempts</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2921</th>
      <td>417</td>
      <td>9</td>
      <td>2025-06-11 09:38:30</td>
      <td>Burgos</td>
      <td>NaN</td>
      <td>2020-03-17</td>
      <td>2020-03-17</td>
      <td>NaT</td>
      <td>44849.0</td>
      <td>Cerrada</td>
      <td>...</td>
      <td>RESULTADOS</td>
      <td>1.0</td>
      <td>2020-03-17</td>
      <td>2025-06-11</td>
      <td>2023-07-01</td>
      <td>1201 days</td>
      <td>1201</td>
      <td>0</td>
      <td>1</td>
      <td>7</td>
    </tr>
  </tbody>
</table>
<p>1 rows × 23 columns</p>
</div>




```python
df_churn_first.head()
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
      <th>withdrawal_id</th>
      <th>withdrawal_status</th>
      <th>...</th>
      <th>withdrawal_reason</th>
      <th>is_definitive_withdrawal</th>
      <th>start_date</th>
      <th>snapshot_date</th>
      <th>end_date</th>
      <th>permanencia</th>
      <th>dias_permanencia</th>
      <th>is_active</th>
      <th>is_new</th>
      <th>n_withdrawal_attempts</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>13704</th>
      <td>1</td>
      <td>11</td>
      <td>2023-06-16 09:35:01</td>
      <td>Cádiz</td>
      <td>NaN</td>
      <td>2020-12-23</td>
      <td>NaT</td>
      <td>2023-05-31</td>
      <td>6733.0</td>
      <td>Cerrada</td>
      <td>...</td>
      <td>FALTA DE USO/TIEMPO</td>
      <td>1.0</td>
      <td>2020-12-23</td>
      <td>2023-06-16</td>
      <td>2023-05-31</td>
      <td>889 days</td>
      <td>889</td>
      <td>0</td>
      <td>1</td>
      <td>5</td>
    </tr>
    <tr>
      <th>3217</th>
      <td>2</td>
      <td>11</td>
      <td>2023-11-06 16:01:14</td>
      <td>Cádiz</td>
      <td>100.0</td>
      <td>2020-09-25</td>
      <td>2020-09-25</td>
      <td>NaT</td>
      <td>4770.0</td>
      <td>Cerrada</td>
      <td>...</td>
      <td>RESULTADOS</td>
      <td>1.0</td>
      <td>2020-09-25</td>
      <td>2023-11-06</td>
      <td>2023-11-06</td>
      <td>1137 days</td>
      <td>1137</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>79</th>
      <td>3</td>
      <td>8</td>
      <td>2024-08-07 13:37:09</td>
      <td>Barcelona</td>
      <td>41.0</td>
      <td>2024-05-06</td>
      <td>NaT</td>
      <td>2024-07-15</td>
      <td>41236.0</td>
      <td>Cerrada</td>
      <td>...</td>
      <td>RESULTADOS</td>
      <td>1.0</td>
      <td>2024-05-06</td>
      <td>2024-08-07</td>
      <td>2024-07-15</td>
      <td>70 days</td>
      <td>70</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>11568</th>
      <td>4</td>
      <td>48</td>
      <td>2023-05-03 16:37:02</td>
      <td>Vizcaya</td>
      <td>NaN</td>
      <td>2022-09-30</td>
      <td>NaT</td>
      <td>NaT</td>
      <td>148.0</td>
      <td>Cerrada</td>
      <td>...</td>
      <td>OTROS</td>
      <td>1.0</td>
      <td>2022-09-30</td>
      <td>2023-05-03</td>
      <td>2023-05-03</td>
      <td>215 days</td>
      <td>215</td>
      <td>1</td>
      <td>1</td>
      <td>7</td>
    </tr>
    <tr>
      <th>11520</th>
      <td>5</td>
      <td>48</td>
      <td>2025-01-07 17:47:31</td>
      <td>Vizcaya</td>
      <td>193.0</td>
      <td>2024-07-30</td>
      <td>2024-09-30</td>
      <td>NaT</td>
      <td>3614.0</td>
      <td>Cerrada</td>
      <td>...</td>
      <td>FALTA DE PRODUCTO</td>
      <td>0.0</td>
      <td>2024-09-30</td>
      <td>2025-01-07</td>
      <td>2025-01-07</td>
      <td>99 days</td>
      <td>99</td>
      <td>1</td>
      <td>0</td>
      <td>10</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 23 columns</p>
</div>



Creamos una nueva variable para identificar advertisers individuales de aquellos que pertenecen a un grupo


```python
largest_group = df_churn_first.groupby("advertiser_group_id")["advertiser_zrive_id"].nunique().idxmax()
n_adv_largest_group = df_churn_first[df_churn_first["advertiser_group_id"] == largest_group]["advertiser_zrive_id"].nunique()

print(f"El grupo/empresa más grande en volúmen de advertisers es el: {largest_group}, con {n_adv_largest_group} advertisers distintos")

```

    El grupo/empresa más grande en volúmen de advertisers es el: 125.0, con 244 advertisers distintos



```python
df_churn_first[df_churn_first["is_definitive_withdrawal"] == 1]
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
      <th>withdrawal_id</th>
      <th>withdrawal_status</th>
      <th>...</th>
      <th>withdrawal_reason</th>
      <th>is_definitive_withdrawal</th>
      <th>start_date</th>
      <th>snapshot_date</th>
      <th>end_date</th>
      <th>permanencia</th>
      <th>dias_permanencia</th>
      <th>is_active</th>
      <th>is_new</th>
      <th>n_withdrawal_attempts</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>13704</th>
      <td>1</td>
      <td>11</td>
      <td>2023-06-16 09:35:01</td>
      <td>Cádiz</td>
      <td>NaN</td>
      <td>2020-12-23</td>
      <td>NaT</td>
      <td>2023-05-31</td>
      <td>6733.0</td>
      <td>Cerrada</td>
      <td>...</td>
      <td>FALTA DE USO/TIEMPO</td>
      <td>1.0</td>
      <td>2020-12-23</td>
      <td>2023-06-16</td>
      <td>2023-05-31</td>
      <td>889 days</td>
      <td>889</td>
      <td>0</td>
      <td>1</td>
      <td>5</td>
    </tr>
    <tr>
      <th>3217</th>
      <td>2</td>
      <td>11</td>
      <td>2023-11-06 16:01:14</td>
      <td>Cádiz</td>
      <td>100.0</td>
      <td>2020-09-25</td>
      <td>2020-09-25</td>
      <td>NaT</td>
      <td>4770.0</td>
      <td>Cerrada</td>
      <td>...</td>
      <td>RESULTADOS</td>
      <td>1.0</td>
      <td>2020-09-25</td>
      <td>2023-11-06</td>
      <td>2023-11-06</td>
      <td>1137 days</td>
      <td>1137</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>79</th>
      <td>3</td>
      <td>8</td>
      <td>2024-08-07 13:37:09</td>
      <td>Barcelona</td>
      <td>41.0</td>
      <td>2024-05-06</td>
      <td>NaT</td>
      <td>2024-07-15</td>
      <td>41236.0</td>
      <td>Cerrada</td>
      <td>...</td>
      <td>RESULTADOS</td>
      <td>1.0</td>
      <td>2024-05-06</td>
      <td>2024-08-07</td>
      <td>2024-07-15</td>
      <td>70 days</td>
      <td>70</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>11568</th>
      <td>4</td>
      <td>48</td>
      <td>2023-05-03 16:37:02</td>
      <td>Vizcaya</td>
      <td>NaN</td>
      <td>2022-09-30</td>
      <td>NaT</td>
      <td>NaT</td>
      <td>148.0</td>
      <td>Cerrada</td>
      <td>...</td>
      <td>OTROS</td>
      <td>1.0</td>
      <td>2022-09-30</td>
      <td>2023-05-03</td>
      <td>2023-05-03</td>
      <td>215 days</td>
      <td>215</td>
      <td>1</td>
      <td>1</td>
      <td>7</td>
    </tr>
    <tr>
      <th>11717</th>
      <td>6</td>
      <td>48</td>
      <td>2025-03-12 13:48:02</td>
      <td>Vizcaya</td>
      <td>NaN</td>
      <td>2022-10-10</td>
      <td>NaT</td>
      <td>NaT</td>
      <td>446.0</td>
      <td>Cerrada</td>
      <td>...</td>
      <td>RESULTADOS</td>
      <td>1.0</td>
      <td>2022-10-10</td>
      <td>2025-03-12</td>
      <td>2025-03-12</td>
      <td>884 days</td>
      <td>884</td>
      <td>1</td>
      <td>1</td>
      <td>7</td>
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
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
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
      <th>6855</th>
      <td>6897</td>
      <td>29</td>
      <td>2025-06-04 13:31:07</td>
      <td>Madrid</td>
      <td>NaN</td>
      <td>2025-03-04</td>
      <td>2025-03-04</td>
      <td>NaT</td>
      <td>55470.0</td>
      <td>Cerrada</td>
      <td>...</td>
      <td>MOROSIDAD</td>
      <td>1.0</td>
      <td>2025-03-04</td>
      <td>2025-06-04</td>
      <td>2025-06-04</td>
      <td>92 days</td>
      <td>92</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4766</th>
      <td>6900</td>
      <td>24</td>
      <td>2025-06-06 13:37:22</td>
      <td>Islas Baleares</td>
      <td>NaN</td>
      <td>2025-03-05</td>
      <td>2025-03-05</td>
      <td>NaT</td>
      <td>55591.0</td>
      <td>Cerrada</td>
      <td>...</td>
      <td>MOROSIDAD</td>
      <td>1.0</td>
      <td>2025-03-05</td>
      <td>2025-06-06</td>
      <td>2025-06-06</td>
      <td>93 days</td>
      <td>93</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4339</th>
      <td>6901</td>
      <td>19</td>
      <td>2025-06-05 19:00:08</td>
      <td>Granada</td>
      <td>NaN</td>
      <td>2025-03-05</td>
      <td>2025-03-05</td>
      <td>NaT</td>
      <td>55194.0</td>
      <td>Cerrada</td>
      <td>...</td>
      <td>RESULTADOS</td>
      <td>1.0</td>
      <td>2025-03-05</td>
      <td>2025-06-05</td>
      <td>2025-06-05</td>
      <td>92 days</td>
      <td>92</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>23050</th>
      <td>6931</td>
      <td>3</td>
      <td>2025-06-01 03:58:05</td>
      <td>Alicante</td>
      <td>NaN</td>
      <td>2025-03-28</td>
      <td>2025-03-28</td>
      <td>2025-05-31</td>
      <td>55483.0</td>
      <td>Cerrada</td>
      <td>...</td>
      <td>MOROSIDAD</td>
      <td>1.0</td>
      <td>2025-03-28</td>
      <td>2025-06-01</td>
      <td>2025-05-31</td>
      <td>64 days</td>
      <td>64</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>22694</th>
      <td>6967</td>
      <td>11</td>
      <td>2025-05-10 01:02:07</td>
      <td>Cádiz</td>
      <td>NaN</td>
      <td>2025-04-08</td>
      <td>2025-04-08</td>
      <td>2025-04-30</td>
      <td>55299.0</td>
      <td>Cerrada</td>
      <td>...</td>
      <td>OTROS</td>
      <td>1.0</td>
      <td>2025-04-08</td>
      <td>2025-05-10</td>
      <td>2025-04-30</td>
      <td>22 days</td>
      <td>22</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>4067 rows × 23 columns</p>
</div>




```python
df_churn_first.head()
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
      <th>withdrawal_id</th>
      <th>withdrawal_status</th>
      <th>...</th>
      <th>withdrawal_reason</th>
      <th>is_definitive_withdrawal</th>
      <th>start_date</th>
      <th>snapshot_date</th>
      <th>end_date</th>
      <th>permanencia</th>
      <th>dias_permanencia</th>
      <th>is_active</th>
      <th>is_new</th>
      <th>n_withdrawal_attempts</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>13704</th>
      <td>1</td>
      <td>11</td>
      <td>2023-06-16 09:35:01</td>
      <td>Cádiz</td>
      <td>NaN</td>
      <td>2020-12-23</td>
      <td>NaT</td>
      <td>2023-05-31</td>
      <td>6733.0</td>
      <td>Cerrada</td>
      <td>...</td>
      <td>FALTA DE USO/TIEMPO</td>
      <td>1.0</td>
      <td>2020-12-23</td>
      <td>2023-06-16</td>
      <td>2023-05-31</td>
      <td>889 days</td>
      <td>889</td>
      <td>0</td>
      <td>1</td>
      <td>5</td>
    </tr>
    <tr>
      <th>3217</th>
      <td>2</td>
      <td>11</td>
      <td>2023-11-06 16:01:14</td>
      <td>Cádiz</td>
      <td>100.0</td>
      <td>2020-09-25</td>
      <td>2020-09-25</td>
      <td>NaT</td>
      <td>4770.0</td>
      <td>Cerrada</td>
      <td>...</td>
      <td>RESULTADOS</td>
      <td>1.0</td>
      <td>2020-09-25</td>
      <td>2023-11-06</td>
      <td>2023-11-06</td>
      <td>1137 days</td>
      <td>1137</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>79</th>
      <td>3</td>
      <td>8</td>
      <td>2024-08-07 13:37:09</td>
      <td>Barcelona</td>
      <td>41.0</td>
      <td>2024-05-06</td>
      <td>NaT</td>
      <td>2024-07-15</td>
      <td>41236.0</td>
      <td>Cerrada</td>
      <td>...</td>
      <td>RESULTADOS</td>
      <td>1.0</td>
      <td>2024-05-06</td>
      <td>2024-08-07</td>
      <td>2024-07-15</td>
      <td>70 days</td>
      <td>70</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>11568</th>
      <td>4</td>
      <td>48</td>
      <td>2023-05-03 16:37:02</td>
      <td>Vizcaya</td>
      <td>NaN</td>
      <td>2022-09-30</td>
      <td>NaT</td>
      <td>NaT</td>
      <td>148.0</td>
      <td>Cerrada</td>
      <td>...</td>
      <td>OTROS</td>
      <td>1.0</td>
      <td>2022-09-30</td>
      <td>2023-05-03</td>
      <td>2023-05-03</td>
      <td>215 days</td>
      <td>215</td>
      <td>1</td>
      <td>1</td>
      <td>7</td>
    </tr>
    <tr>
      <th>11520</th>
      <td>5</td>
      <td>48</td>
      <td>2025-01-07 17:47:31</td>
      <td>Vizcaya</td>
      <td>193.0</td>
      <td>2024-07-30</td>
      <td>2024-09-30</td>
      <td>NaT</td>
      <td>3614.0</td>
      <td>Cerrada</td>
      <td>...</td>
      <td>FALTA DE PRODUCTO</td>
      <td>0.0</td>
      <td>2024-09-30</td>
      <td>2025-01-07</td>
      <td>2025-01-07</td>
      <td>99 days</td>
      <td>99</td>
      <td>1</td>
      <td>0</td>
      <td>10</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 23 columns</p>
</div>




```python
df_churn_first = df_churn_first.copy()
df_churn_first["is_individual"] = (
    df_churn_first["advertiser_group_id"].isna()
).astype(int)

# df_churn_first["dias_permanencia"] = df_churn_first["dias_permanencia"].dt.days
df = df_churn_first.copy()
```

### **Early churn**

Ahora definimos el evento de early churn, es decir si los dias de permanencia en la plataforma fueron menos de 90 días o más

- usuarios con churn_3m=1
- usuarios con churn_3m=0
- usuarios censurados (NA): estos son aquellos cuya end_date = snapshot_date, es decir, no podemos certificar que end_date es la fecha de baja ya que realmente es la última fecha de actualización de información de ese advertiser, con lo cual en esa fecha el usuario aun seguía activo y no sabemos si en el futuro hizo churn o no

OJO: en los casos en los que end_date >= snapshot_date, no podemos afirmar si es churn o no

Desde **start_date**, ¿ocurre algún evento de churn en los siguientes 90 días?


```python
df[df["dias_permanencia"] < 90]
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
      <th>withdrawal_id</th>
      <th>withdrawal_status</th>
      <th>...</th>
      <th>is_definitive_withdrawal</th>
      <th>start_date</th>
      <th>snapshot_date</th>
      <th>end_date</th>
      <th>permanencia</th>
      <th>dias_permanencia</th>
      <th>is_active</th>
      <th>is_new</th>
      <th>n_withdrawal_attempts</th>
      <th>is_individual</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>79</th>
      <td>3</td>
      <td>8</td>
      <td>2024-08-07 13:37:09</td>
      <td>Barcelona</td>
      <td>41.0</td>
      <td>2024-05-06</td>
      <td>NaT</td>
      <td>2024-07-15</td>
      <td>41236.0</td>
      <td>Cerrada</td>
      <td>...</td>
      <td>1.0</td>
      <td>2024-05-06</td>
      <td>2024-08-07</td>
      <td>2024-07-15</td>
      <td>70 days</td>
      <td>70</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>231</th>
      <td>17</td>
      <td>46</td>
      <td>2023-03-29 11:00:19</td>
      <td>Valencia</td>
      <td>NaN</td>
      <td>2023-01-10</td>
      <td>NaT</td>
      <td>2023-03-12</td>
      <td>41579.0</td>
      <td>Cerrada</td>
      <td>...</td>
      <td>1.0</td>
      <td>2023-01-10</td>
      <td>2023-03-29</td>
      <td>2023-03-12</td>
      <td>61 days</td>
      <td>61</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>239</th>
      <td>19</td>
      <td>46</td>
      <td>2023-07-16 01:00:47</td>
      <td>Valencia</td>
      <td>NaN</td>
      <td>2023-05-16</td>
      <td>NaT</td>
      <td>2023-07-15</td>
      <td>6372.0</td>
      <td>Cerrada</td>
      <td>...</td>
      <td>1.0</td>
      <td>2023-05-16</td>
      <td>2023-07-16</td>
      <td>2023-07-15</td>
      <td>60 days</td>
      <td>60</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>12287</th>
      <td>46</td>
      <td>29</td>
      <td>2023-02-09 10:03:17</td>
      <td>Madrid</td>
      <td>NaN</td>
      <td>2021-04-23</td>
      <td>NaT</td>
      <td>2023-01-31</td>
      <td>35197.0</td>
      <td>Cerrada</td>
      <td>...</td>
      <td>1.0</td>
      <td>2021-04-23</td>
      <td>2023-02-09</td>
      <td>2021-05-20</td>
      <td>27 days</td>
      <td>27</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>13660</th>
      <td>76</td>
      <td>8</td>
      <td>2023-09-21 17:00:56</td>
      <td>Barcelona</td>
      <td>115.0</td>
      <td>2023-04-19</td>
      <td>NaT</td>
      <td>2023-05-31</td>
      <td>4988.0</td>
      <td>Cerrada</td>
      <td>...</td>
      <td>0.0</td>
      <td>2023-04-19</td>
      <td>2023-09-21</td>
      <td>2023-05-31</td>
      <td>42 days</td>
      <td>42</td>
      <td>0</td>
      <td>1</td>
      <td>5</td>
      <td>0</td>
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
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
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
      <th>23378</th>
      <td>6864</td>
      <td>30</td>
      <td>2025-06-01 03:54:11</td>
      <td>Málaga</td>
      <td>NaN</td>
      <td>2025-03-07</td>
      <td>2025-03-07</td>
      <td>2025-05-31</td>
      <td>55301.0</td>
      <td>Cerrada</td>
      <td>...</td>
      <td>1.0</td>
      <td>2025-03-07</td>
      <td>2025-06-01</td>
      <td>2025-05-31</td>
      <td>85 days</td>
      <td>85</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>22914</th>
      <td>6868</td>
      <td>31</td>
      <td>2025-05-14 10:00:10</td>
      <td>Murcia</td>
      <td>NaN</td>
      <td>2025-02-20</td>
      <td>2025-02-20</td>
      <td>2025-04-30</td>
      <td>55126.0</td>
      <td>Cerrada</td>
      <td>...</td>
      <td>1.0</td>
      <td>2025-02-20</td>
      <td>2025-05-14</td>
      <td>2025-04-30</td>
      <td>69 days</td>
      <td>69</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>22902</th>
      <td>6878</td>
      <td>30</td>
      <td>2025-05-01 05:16:06</td>
      <td>Málaga</td>
      <td>NaN</td>
      <td>2025-02-26</td>
      <td>2025-02-26</td>
      <td>2025-04-30</td>
      <td>54702.0</td>
      <td>Cerrada</td>
      <td>...</td>
      <td>1.0</td>
      <td>2025-02-26</td>
      <td>2025-05-01</td>
      <td>2025-04-30</td>
      <td>63 days</td>
      <td>63</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>23050</th>
      <td>6931</td>
      <td>3</td>
      <td>2025-06-01 03:58:05</td>
      <td>Alicante</td>
      <td>NaN</td>
      <td>2025-03-28</td>
      <td>2025-03-28</td>
      <td>2025-05-31</td>
      <td>55483.0</td>
      <td>Cerrada</td>
      <td>...</td>
      <td>1.0</td>
      <td>2025-03-28</td>
      <td>2025-06-01</td>
      <td>2025-05-31</td>
      <td>64 days</td>
      <td>64</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>22694</th>
      <td>6967</td>
      <td>11</td>
      <td>2025-05-10 01:02:07</td>
      <td>Cádiz</td>
      <td>NaN</td>
      <td>2025-04-08</td>
      <td>2025-04-08</td>
      <td>2025-04-30</td>
      <td>55299.0</td>
      <td>Cerrada</td>
      <td>...</td>
      <td>1.0</td>
      <td>2025-04-08</td>
      <td>2025-05-10</td>
      <td>2025-04-30</td>
      <td>22 days</td>
      <td>22</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>334 rows × 24 columns</p>
</div>




```python
df["is_early_churn"] = (
    df["dias_permanencia"] < 90
).astype(int)
```


```python
df[df["withdrawal_creation_date"]<df["start_date"]]
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
      <th>withdrawal_id</th>
      <th>withdrawal_status</th>
      <th>...</th>
      <th>start_date</th>
      <th>snapshot_date</th>
      <th>end_date</th>
      <th>permanencia</th>
      <th>dias_permanencia</th>
      <th>is_active</th>
      <th>is_new</th>
      <th>n_withdrawal_attempts</th>
      <th>is_individual</th>
      <th>is_early_churn</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>13704</th>
      <td>1</td>
      <td>11</td>
      <td>2023-06-16 09:35:01</td>
      <td>Cádiz</td>
      <td>NaN</td>
      <td>2020-12-23</td>
      <td>NaT</td>
      <td>2023-05-31</td>
      <td>6733.0</td>
      <td>Cerrada</td>
      <td>...</td>
      <td>2020-12-23</td>
      <td>2023-06-16</td>
      <td>2023-05-31</td>
      <td>889 days</td>
      <td>889</td>
      <td>0</td>
      <td>1</td>
      <td>5</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3217</th>
      <td>2</td>
      <td>11</td>
      <td>2023-11-06 16:01:14</td>
      <td>Cádiz</td>
      <td>100.0</td>
      <td>2020-09-25</td>
      <td>2020-09-25</td>
      <td>NaT</td>
      <td>4770.0</td>
      <td>Cerrada</td>
      <td>...</td>
      <td>2020-09-25</td>
      <td>2023-11-06</td>
      <td>2023-11-06</td>
      <td>1137 days</td>
      <td>1137</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>79</th>
      <td>3</td>
      <td>8</td>
      <td>2024-08-07 13:37:09</td>
      <td>Barcelona</td>
      <td>41.0</td>
      <td>2024-05-06</td>
      <td>NaT</td>
      <td>2024-07-15</td>
      <td>41236.0</td>
      <td>Cerrada</td>
      <td>...</td>
      <td>2024-05-06</td>
      <td>2024-08-07</td>
      <td>2024-07-15</td>
      <td>70 days</td>
      <td>70</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>11568</th>
      <td>4</td>
      <td>48</td>
      <td>2023-05-03 16:37:02</td>
      <td>Vizcaya</td>
      <td>NaN</td>
      <td>2022-09-30</td>
      <td>NaT</td>
      <td>NaT</td>
      <td>148.0</td>
      <td>Cerrada</td>
      <td>...</td>
      <td>2022-09-30</td>
      <td>2023-05-03</td>
      <td>2023-05-03</td>
      <td>215 days</td>
      <td>215</td>
      <td>1</td>
      <td>1</td>
      <td>7</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>11520</th>
      <td>5</td>
      <td>48</td>
      <td>2025-01-07 17:47:31</td>
      <td>Vizcaya</td>
      <td>193.0</td>
      <td>2024-07-30</td>
      <td>2024-09-30</td>
      <td>NaT</td>
      <td>3614.0</td>
      <td>Cerrada</td>
      <td>...</td>
      <td>2024-09-30</td>
      <td>2025-01-07</td>
      <td>2025-01-07</td>
      <td>99 days</td>
      <td>99</td>
      <td>1</td>
      <td>0</td>
      <td>10</td>
      <td>0</td>
      <td>0</td>
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
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
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
      <th>19584</th>
      <td>5845</td>
      <td>29</td>
      <td>2024-11-07 12:43:00</td>
      <td>Madrid</td>
      <td>193.0</td>
      <td>2024-02-21</td>
      <td>2024-06-03</td>
      <td>2024-10-31</td>
      <td>49191.0</td>
      <td>Cerrada</td>
      <td>...</td>
      <td>2024-06-03</td>
      <td>2024-11-07</td>
      <td>2024-10-31</td>
      <td>150 days</td>
      <td>150</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>21272</th>
      <td>5917</td>
      <td>31</td>
      <td>2025-02-01 03:58:04</td>
      <td>Murcia</td>
      <td>NaN</td>
      <td>2024-03-11</td>
      <td>2024-11-22</td>
      <td>2025-01-31</td>
      <td>49430.0</td>
      <td>Cerrada</td>
      <td>...</td>
      <td>2024-11-22</td>
      <td>2025-02-01</td>
      <td>2025-01-31</td>
      <td>70 days</td>
      <td>70</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4061</th>
      <td>6013</td>
      <td>17</td>
      <td>2025-06-06 19:02:06</td>
      <td>Cuenca</td>
      <td>NaN</td>
      <td>2024-05-22</td>
      <td>2025-03-06</td>
      <td>NaT</td>
      <td>50444.0</td>
      <td>Cerrada</td>
      <td>...</td>
      <td>2025-03-06</td>
      <td>2025-06-06</td>
      <td>2025-06-06</td>
      <td>92 days</td>
      <td>92</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>10324</th>
      <td>6244</td>
      <td>43</td>
      <td>2025-05-11 11:37:06</td>
      <td>Tenerife</td>
      <td>NaN</td>
      <td>2024-06-18</td>
      <td>2025-02-10</td>
      <td>NaT</td>
      <td>51222.0</td>
      <td>Cerrada</td>
      <td>...</td>
      <td>2025-02-10</td>
      <td>2025-05-11</td>
      <td>2025-05-11</td>
      <td>90 days</td>
      <td>90</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3239</th>
      <td>6396</td>
      <td>11</td>
      <td>2025-06-01 03:26:10</td>
      <td>Cádiz</td>
      <td>NaN</td>
      <td>2024-09-05</td>
      <td>2025-02-11</td>
      <td>NaT</td>
      <td>51251.0</td>
      <td>Cerrada</td>
      <td>...</td>
      <td>2025-02-11</td>
      <td>2025-06-01</td>
      <td>2025-06-01</td>
      <td>110 days</td>
      <td>110</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>2345 rows × 25 columns</p>
</div>




```python
df["is_early_churn"].value_counts().plot(kind='bar')
```




    <Axes: xlabel='is_early_churn'>




    
![png](EDA_jtc_files/EDA_jtc_168_1.png)
    


Dataset muy desbalanceado


```python
df["dias_permanencia"].plot(kind='hist', bins=50)
```




    <Axes: ylabel='Frequency'>




    
![png](EDA_jtc_files/EDA_jtc_170_1.png)
    



```python
(df["dias_permanencia"] < 90).mean()

```




    np.float64(0.053568564554931836)




```python
df["dias_permanencia"].describe()

```




    count    6235.000000
    mean      521.690136
    std       520.273778
    min         0.000000
    25%       151.000000
    50%       299.000000
    75%       771.000000
    max      5169.000000
    Name: dias_permanencia, dtype: float64




```python
sns.boxplot(df["dias_permanencia"])
```




    <Axes: ylabel='dias_permanencia'>




    
![png](EDA_jtc_files/EDA_jtc_173_1.png)
    



```python
df["dias_permanencia"].value_counts().sort_values().reset_index()
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
      <th>dias_permanencia</th>
      <th>count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>611</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>473</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>468</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>581</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>568</td>
      <td>1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1499</th>
      <td>93</td>
      <td>50</td>
    </tr>
    <tr>
      <th>1500</th>
      <td>152</td>
      <td>62</td>
    </tr>
    <tr>
      <th>1501</th>
      <td>151</td>
      <td>70</td>
    </tr>
    <tr>
      <th>1502</th>
      <td>92</td>
      <td>81</td>
    </tr>
    <tr>
      <th>1503</th>
      <td>90</td>
      <td>83</td>
    </tr>
  </tbody>
</table>
<p>1504 rows × 2 columns</p>
</div>




```python
df.shape
```




    (6235, 25)




```python
df[df["advertiser_zrive_id"]==1]
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
      <th>withdrawal_id</th>
      <th>withdrawal_status</th>
      <th>...</th>
      <th>start_date</th>
      <th>snapshot_date</th>
      <th>end_date</th>
      <th>permanencia</th>
      <th>dias_permanencia</th>
      <th>is_active</th>
      <th>is_new</th>
      <th>n_withdrawal_attempts</th>
      <th>is_individual</th>
      <th>is_early_churn</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>13704</th>
      <td>1</td>
      <td>11</td>
      <td>2023-06-16 09:35:01</td>
      <td>Cádiz</td>
      <td>NaN</td>
      <td>2020-12-23</td>
      <td>NaT</td>
      <td>2023-05-31</td>
      <td>6733.0</td>
      <td>Cerrada</td>
      <td>...</td>
      <td>2020-12-23</td>
      <td>2023-06-16</td>
      <td>2023-05-31</td>
      <td>889 days</td>
      <td>889</td>
      <td>0</td>
      <td>1</td>
      <td>5</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>1 rows × 25 columns</p>
</div>




```python
df_period_check = df.copy()
```


```python
df_fct_max_period = (
    df_fct
    .groupby("advertiser_zrive_id")["period_int"]
    .max()
    .reset_index()
    .rename(columns={'period_int':'max_period_int'})
)
df_fct_max_period
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
      <th>max_period_int</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>2023-05-01</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>2025-05-01</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>2024-07-01</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>2025-05-01</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>2025-05-01</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>6963</th>
      <td>7047</td>
      <td>2025-05-01</td>
    </tr>
    <tr>
      <th>6964</th>
      <td>7051</td>
      <td>2025-05-01</td>
    </tr>
    <tr>
      <th>6965</th>
      <td>7052</td>
      <td>2025-05-01</td>
    </tr>
    <tr>
      <th>6966</th>
      <td>7066</td>
      <td>2025-05-01</td>
    </tr>
    <tr>
      <th>6967</th>
      <td>7069</td>
      <td>2025-05-01</td>
    </tr>
  </tbody>
</table>
<p>6968 rows × 2 columns</p>
</div>




```python
df_period_check = df_period_check.merge(df_fct_max_period, how='inner', on='advertiser_zrive_id')
df_period_check
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
      <th>withdrawal_id</th>
      <th>withdrawal_status</th>
      <th>...</th>
      <th>snapshot_date</th>
      <th>end_date</th>
      <th>permanencia</th>
      <th>dias_permanencia</th>
      <th>is_active</th>
      <th>is_new</th>
      <th>n_withdrawal_attempts</th>
      <th>is_individual</th>
      <th>is_early_churn</th>
      <th>max_period_int</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>11</td>
      <td>2023-06-16 09:35:01</td>
      <td>Cádiz</td>
      <td>NaN</td>
      <td>2020-12-23</td>
      <td>NaT</td>
      <td>2023-05-31</td>
      <td>6733.0</td>
      <td>Cerrada</td>
      <td>...</td>
      <td>2023-06-16</td>
      <td>2023-05-31</td>
      <td>889 days</td>
      <td>889</td>
      <td>0</td>
      <td>1</td>
      <td>5</td>
      <td>1</td>
      <td>0</td>
      <td>2023-05-01</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>11</td>
      <td>2023-11-06 16:01:14</td>
      <td>Cádiz</td>
      <td>100.0</td>
      <td>2020-09-25</td>
      <td>2020-09-25</td>
      <td>NaT</td>
      <td>4770.0</td>
      <td>Cerrada</td>
      <td>...</td>
      <td>2023-11-06</td>
      <td>2023-11-06</td>
      <td>1137 days</td>
      <td>1137</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>2025-05-01</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>8</td>
      <td>2024-08-07 13:37:09</td>
      <td>Barcelona</td>
      <td>41.0</td>
      <td>2024-05-06</td>
      <td>NaT</td>
      <td>2024-07-15</td>
      <td>41236.0</td>
      <td>Cerrada</td>
      <td>...</td>
      <td>2024-08-07</td>
      <td>2024-07-15</td>
      <td>70 days</td>
      <td>70</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>2024-07-01</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>48</td>
      <td>2023-05-03 16:37:02</td>
      <td>Vizcaya</td>
      <td>NaN</td>
      <td>2022-09-30</td>
      <td>NaT</td>
      <td>NaT</td>
      <td>148.0</td>
      <td>Cerrada</td>
      <td>...</td>
      <td>2023-05-03</td>
      <td>2023-05-03</td>
      <td>215 days</td>
      <td>215</td>
      <td>1</td>
      <td>1</td>
      <td>7</td>
      <td>1</td>
      <td>0</td>
      <td>2025-05-01</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>48</td>
      <td>2025-01-07 17:47:31</td>
      <td>Vizcaya</td>
      <td>193.0</td>
      <td>2024-07-30</td>
      <td>2024-09-30</td>
      <td>NaT</td>
      <td>3614.0</td>
      <td>Cerrada</td>
      <td>...</td>
      <td>2025-01-07</td>
      <td>2025-01-07</td>
      <td>99 days</td>
      <td>99</td>
      <td>1</td>
      <td>0</td>
      <td>10</td>
      <td>0</td>
      <td>0</td>
      <td>2025-05-01</td>
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
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
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
      <th>6201</th>
      <td>6909</td>
      <td>5</td>
      <td>2025-06-11 13:25:09</td>
      <td>Asturias</td>
      <td>NaN</td>
      <td>2025-03-10</td>
      <td>NaT</td>
      <td>NaT</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>2025-06-11</td>
      <td>2025-06-11</td>
      <td>93 days</td>
      <td>93</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>2025-05-01</td>
    </tr>
    <tr>
      <th>6202</th>
      <td>6913</td>
      <td>31</td>
      <td>2025-06-12 13:27:01</td>
      <td>Murcia</td>
      <td>NaN</td>
      <td>2025-03-12</td>
      <td>2025-03-12</td>
      <td>NaT</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>2025-06-12</td>
      <td>2025-06-12</td>
      <td>92 days</td>
      <td>92</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>2025-05-01</td>
    </tr>
    <tr>
      <th>6203</th>
      <td>6923</td>
      <td>40</td>
      <td>2025-06-18 12:12:19</td>
      <td>Sevilla</td>
      <td>55.0</td>
      <td>2025-03-17</td>
      <td>2025-03-17</td>
      <td>NaT</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>2025-06-18</td>
      <td>2025-06-18</td>
      <td>93 days</td>
      <td>93</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2025-05-01</td>
    </tr>
    <tr>
      <th>6204</th>
      <td>6931</td>
      <td>3</td>
      <td>2025-06-01 03:58:05</td>
      <td>Alicante</td>
      <td>NaN</td>
      <td>2025-03-28</td>
      <td>2025-03-28</td>
      <td>2025-05-31</td>
      <td>55483.0</td>
      <td>Cerrada</td>
      <td>...</td>
      <td>2025-06-01</td>
      <td>2025-05-31</td>
      <td>64 days</td>
      <td>64</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>2025-05-01</td>
    </tr>
    <tr>
      <th>6205</th>
      <td>6967</td>
      <td>11</td>
      <td>2025-05-10 01:02:07</td>
      <td>Cádiz</td>
      <td>NaN</td>
      <td>2025-04-08</td>
      <td>2025-04-08</td>
      <td>2025-04-30</td>
      <td>55299.0</td>
      <td>Cerrada</td>
      <td>...</td>
      <td>2025-05-10</td>
      <td>2025-04-30</td>
      <td>22 days</td>
      <td>22</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>2025-05-01</td>
    </tr>
  </tbody>
</table>
<p>6206 rows × 26 columns</p>
</div>




```python

```

## FCT analysis


```python
df["advertiser_zrive_id"].nunique()
```




    6235




```python
df_fct = df_fct.drop_duplicates()
```


```python
df_fct["advertiser_zrive_id"].nunique()
```




    6968




```python
df_fct["monthly_total_invoice"].describe()
```




    count    96829.000000
    mean       502.381491
    std       1048.326272
    min      -1833.300000
    25%         71.600000
    50%        235.000000
    75%        545.000000
    max      33500.000000
    Name: monthly_total_invoice, dtype: float64




```python
df_fct["monthly_total_invoice"].plot(kind='box')
```




    <Axes: >




    
![png](EDA_jtc_files/EDA_jtc_186_1.png)
    



```python
df_dim
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
      <th>7071</th>
      <td>5179</td>
      <td>46</td>
      <td>2025-06-03 12:47:06</td>
      <td>Valencia</td>
      <td>NaN</td>
      <td>2024-09-23</td>
      <td>2024-09-23</td>
      <td>2025-06-30</td>
    </tr>
    <tr>
      <th>7072</th>
      <td>789</td>
      <td>48</td>
      <td>2025-04-30 13:40:11</td>
      <td>Vizcaya</td>
      <td>NaN</td>
      <td>2022-10-01</td>
      <td>2025-01-30</td>
      <td>2025-06-30</td>
    </tr>
    <tr>
      <th>7073</th>
      <td>4284</td>
      <td>48</td>
      <td>2025-04-22 14:33:13</td>
      <td>Vizcaya</td>
      <td>NaN</td>
      <td>2024-03-05</td>
      <td>2025-01-22</td>
      <td>2025-06-30</td>
    </tr>
    <tr>
      <th>7074</th>
      <td>4397</td>
      <td>50</td>
      <td>2025-04-01 04:08:33</td>
      <td>Zaragoza</td>
      <td>NaN</td>
      <td>2022-10-10</td>
      <td>2022-10-10</td>
      <td>2025-06-30</td>
    </tr>
    <tr>
      <th>7075</th>
      <td>4194</td>
      <td>50</td>
      <td>2023-10-10 11:21:22</td>
      <td>Zaragoza</td>
      <td>NaN</td>
      <td>2022-07-21</td>
      <td>NaT</td>
      <td>2025-06-30</td>
    </tr>
  </tbody>
</table>
<p>7076 rows × 8 columns</p>
</div>




```python
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
      <td>2012-06-19</td>
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
      <td>2012-06-19</td>
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
      <td>2012-06-20</td>
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
      <td>2012-06-20</td>
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
      <td>2012-06-20</td>
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
      <td>2025-05-30</td>
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
      <td>2025-05-30</td>
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
      <td>2025-05-30</td>
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
      <td>2025-05-30</td>
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
      <td>2025-05-30</td>
      <td>NaT</td>
      <td>MOROSIDAD</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>22668 rows × 8 columns</p>
</div>




```python
df_fct
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
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
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
      <th>96824</th>
      <td>7047</td>
      <td>2025-05-01</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
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
      <td>NaN</td>
      <td>True</td>
    </tr>
    <tr>
      <th>96825</th>
      <td>7051</td>
      <td>2025-05-01</td>
      <td>20</td>
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
      <td>NaN</td>
      <td>True</td>
    </tr>
    <tr>
      <th>96826</th>
      <td>7052</td>
      <td>2025-05-01</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
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
      <td>NaN</td>
      <td>True</td>
    </tr>
    <tr>
      <th>96827</th>
      <td>7066</td>
      <td>2025-05-01</td>
      <td>20</td>
      <td>20</td>
      <td>20</td>
      <td>29.0</td>
      <td>3</td>
      <td>3</td>
      <td>7</td>
      <td>0</td>
      <td>...</td>
      <td>6</td>
      <td>24</td>
      <td>4</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>17</td>
      <td>26</td>
      <td>43</td>
      <td>9859.33</td>
      <td>False</td>
    </tr>
    <tr>
      <th>96828</th>
      <td>7069</td>
      <td>2025-05-01</td>
      <td>10</td>
      <td>5</td>
      <td>5</td>
      <td>6.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>4</td>
      <td>4</td>
      <td>3</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>3</td>
      <td>11</td>
      <td>14</td>
      <td>7495.00</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
<p>96827 rows × 23 columns</p>
</div>




```python
df_fct = df_fct[df_fct["monthly_total_invoice"] >= 0]
df_fct["monthly_total_invoice"].describe()
```




    count    96827.000000
    mean       502.412151
    std       1048.308252
    min          0.000000
    25%         71.600000
    50%        235.000000
    75%        545.000000
    max      33500.000000
    Name: monthly_total_invoice, dtype: float64




```python
from scipy.stats import variation

variation(df_fct["monthly_total_invoice"])
```




    np.float64(2.086539581807757)




```python
df_fct["monthly_total_invoice"].plot(kind="hist", bins=100)
plt.title("Distribución del precio real")
```




    Text(0.5, 1.0, 'Distribución del precio real')




    
![png](EDA_jtc_files/EDA_jtc_192_1.png)
    



```python
df_fct.describe()
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
      <th>monthly_leads</th>
      <th>monthly_total_phone_views</th>
      <th>monthly_total_calls</th>
      <th>monthly_total_emails</th>
      <th>monthly_total_invoice</th>
      <th>monthly_total_reference_price</th>
      <th>monthly_unique_calls</th>
      <th>monthly_unique_emails</th>
      <th>monthly_unique_leads</th>
      <th>monthly_avg_ad_price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>96827.000000</td>
      <td>96827</td>
      <td>96827.000000</td>
      <td>96827.000000</td>
      <td>96827.00000</td>
      <td>86775.000000</td>
      <td>96827.000000</td>
      <td>96827.000000</td>
      <td>96827.000000</td>
      <td>96827.000000</td>
      <td>...</td>
      <td>96827.000000</td>
      <td>96827.000000</td>
      <td>96827.000000</td>
      <td>96827.000000</td>
      <td>96827.000000</td>
      <td>84498.000000</td>
      <td>96827.000000</td>
      <td>96827.000000</td>
      <td>96827.000000</td>
      <td>8.836000e+04</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>2994.207866</td>
      <td>2024-03-01 02:55:41.786898176</td>
      <td>108.101191</td>
      <td>93.381774</td>
      <td>50.39962</td>
      <td>181.334947</td>
      <td>3.057453</td>
      <td>1.577008</td>
      <td>1.654972</td>
      <td>0.389984</td>
      <td>...</td>
      <td>19.562261</td>
      <td>12.317257</td>
      <td>8.584279</td>
      <td>4.729673</td>
      <td>502.412151</td>
      <td>1325.395421</td>
      <td>6.134973</td>
      <td>9.332624</td>
      <td>15.467597</td>
      <td>2.869974e+04</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>2023-01-01 00:00:00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>3.300000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000e+00</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>1364.000000</td>
      <td>2023-08-01 00:00:00</td>
      <td>6.000000</td>
      <td>3.000000</td>
      <td>3.00000</td>
      <td>6.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>2.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>71.600000</td>
      <td>361.700000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.433743e+04</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>2782.000000</td>
      <td>2024-03-01 00:00:00</td>
      <td>20.000000</td>
      <td>12.000000</td>
      <td>10.00000</td>
      <td>18.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>6.000000</td>
      <td>4.000000</td>
      <td>3.000000</td>
      <td>1.000000</td>
      <td>235.000000</td>
      <td>548.300000</td>
      <td>2.000000</td>
      <td>3.000000</td>
      <td>5.000000</td>
      <td>2.042249e+04</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>4514.000000</td>
      <td>2024-10-01 00:00:00</td>
      <td>50.000000</td>
      <td>38.000000</td>
      <td>34.00000</td>
      <td>60.000000</td>
      <td>2.000000</td>
      <td>2.000000</td>
      <td>2.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>18.000000</td>
      <td>13.000000</td>
      <td>8.000000</td>
      <td>5.000000</td>
      <td>545.000000</td>
      <td>1105.000000</td>
      <td>6.000000</td>
      <td>9.000000</td>
      <td>15.000000</td>
      <td>2.799000e+04</td>
    </tr>
    <tr>
      <th>max</th>
      <td>7069.000000</td>
      <td>2025-05-01 00:00:00</td>
      <td>8200.000000</td>
      <td>8200.000000</td>
      <td>5400.00000</td>
      <td>133865.000000</td>
      <td>900.000000</td>
      <td>165.000000</td>
      <td>120.000000</td>
      <td>900.000000</td>
      <td>...</td>
      <td>1983.000000</td>
      <td>926.000000</td>
      <td>1098.000000</td>
      <td>406.000000</td>
      <td>33500.000000</td>
      <td>696546.700000</td>
      <td>799.000000</td>
      <td>716.000000</td>
      <td>1515.000000</td>
      <td>3.705341e+07</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1902.383463</td>
      <td>NaN</td>
      <td>490.209838</td>
      <td>478.776149</td>
      <td>243.68727</td>
      <td>1250.878986</td>
      <td>19.359616</td>
      <td>4.308079</td>
      <td>4.098040</td>
      <td>8.207100</td>
      <td>...</td>
      <td>52.583756</td>
      <td>28.280672</td>
      <td>24.388530</td>
      <td>12.075520</td>
      <td>1048.308252</td>
      <td>5554.633364</td>
      <td>16.616800</td>
      <td>23.543399</td>
      <td>39.382878</td>
      <td>1.460678e+05</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 22 columns</p>
</div>




```python
try:
    df_fct["period_int"] = pd.to_datetime(df_fct['period_int'].astype(str), format="%Y%m")
except ValueError:
    pass

df_fct
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
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
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
      <th>96824</th>
      <td>7047</td>
      <td>2025-05-01</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
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
      <td>NaN</td>
      <td>True</td>
    </tr>
    <tr>
      <th>96825</th>
      <td>7051</td>
      <td>2025-05-01</td>
      <td>20</td>
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
      <td>NaN</td>
      <td>True</td>
    </tr>
    <tr>
      <th>96826</th>
      <td>7052</td>
      <td>2025-05-01</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
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
      <td>NaN</td>
      <td>True</td>
    </tr>
    <tr>
      <th>96827</th>
      <td>7066</td>
      <td>2025-05-01</td>
      <td>20</td>
      <td>20</td>
      <td>20</td>
      <td>29.0</td>
      <td>3</td>
      <td>3</td>
      <td>7</td>
      <td>0</td>
      <td>...</td>
      <td>6</td>
      <td>24</td>
      <td>4</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>17</td>
      <td>26</td>
      <td>43</td>
      <td>9859.33</td>
      <td>False</td>
    </tr>
    <tr>
      <th>96828</th>
      <td>7069</td>
      <td>2025-05-01</td>
      <td>10</td>
      <td>5</td>
      <td>5</td>
      <td>6.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>4</td>
      <td>4</td>
      <td>3</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>3</td>
      <td>11</td>
      <td>14</td>
      <td>7495.00</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
<p>96827 rows × 23 columns</p>
</div>




```python
df_fct["period_int"].describe()
```




    count                            96827
    mean     2024-03-01 02:55:41.786898176
    min                2023-01-01 00:00:00
    25%                2023-08-01 00:00:00
    50%                2024-03-01 00:00:00
    75%                2024-10-01 00:00:00
    max                2025-05-01 00:00:00
    Name: period_int, dtype: object




```python
(
    df_fct
    .groupby("advertiser_zrive_id")["period_int"]
    .nunique()
).plot(kind='hist', bins=30)
```




    <Axes: ylabel='Frequency'>




    
![png](EDA_jtc_files/EDA_jtc_196_1.png)
    


Distribución bimodal: hay un segmento de clientes de corto plazo (al rededor de 5 meses de estancia) y otro de clientes de largo plazo (29 meses de estancia)


```python
months = df_fct.groupby("advertiser_zrive_id")["period_int"].nunique()

df_clients = df_fct.groupby("advertiser_zrive_id").agg({
    "monthly_total_invoice": "sum",
    "monthly_contracted_ads": "median"
}).join(months.rename("n_months"))

import seaborn as sns
sns.scatterplot(data=df_clients, x="monthly_total_invoice", y="n_months")

```




    <Axes: xlabel='monthly_total_invoice', ylabel='n_months'>




    
![png](EDA_jtc_files/EDA_jtc_198_1.png)
    



```python
from sklearn.cluster import KMeans

X = df_clients[["n_months"]]
kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
df_clients["cluster"] = kmeans.labels_

```


```python
df_clients
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
      <th>monthly_total_invoice</th>
      <th>monthly_contracted_ads</th>
      <th>n_months</th>
      <th>cluster</th>
    </tr>
    <tr>
      <th>advertiser_zrive_id</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>8944.2</td>
      <td>50.0</td>
      <td>5</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5519.9</td>
      <td>150.0</td>
      <td>29</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.0</td>
      <td>35.0</td>
      <td>4</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>51437.6</td>
      <td>100.0</td>
      <td>29</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>7618.2</td>
      <td>100.0</td>
      <td>11</td>
      <td>1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>7047</th>
      <td>0.0</td>
      <td>3.0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>7051</th>
      <td>0.0</td>
      <td>20.0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>7052</th>
      <td>0.0</td>
      <td>3.0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>7066</th>
      <td>0.0</td>
      <td>20.0</td>
      <td>5</td>
      <td>1</td>
    </tr>
    <tr>
      <th>7069</th>
      <td>0.0</td>
      <td>10.0</td>
      <td>2</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>6968 rows × 4 columns</p>
</div>




```python
df_clients["cluster"].value_counts().plot(kind='bar')
plt.title("Clientes a corto vs largo plazo")
```




    Text(0.5, 1.0, 'Clientes a corto vs largo plazo')




    
![png](EDA_jtc_files/EDA_jtc_201_1.png)
    



```python
df_clients[df_clients["cluster"]==1]['n_months'].describe()
```




    count    4429.000000
    mean        7.011967
    std         4.071192
    min         1.000000
    25%         4.000000
    50%         6.000000
    75%        10.000000
    max        16.000000
    Name: n_months, dtype: float64




```python
df_clients[df_clients["cluster"]==0]['n_months'].describe()
```




    count    2539.000000
    mean       25.904293
    std         4.123858
    min        17.000000
    25%        23.000000
    50%        29.000000
    75%        29.000000
    max        29.000000
    Name: n_months, dtype: float64




```python
df_fct[df_fct.advertiser_zrive_id == 316][["monthly_total_reference_price","monthly_leads", "monthly_total_invoice"]]
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
      <th>monthly_total_reference_price</th>
      <th>monthly_leads</th>
      <th>monthly_total_invoice</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>231</th>
      <td>358.3</td>
      <td>0</td>
      <td>379.5</td>
    </tr>
    <tr>
      <th>3762</th>
      <td>358.3</td>
      <td>0</td>
      <td>379.5</td>
    </tr>
    <tr>
      <th>7123</th>
      <td>358.3</td>
      <td>0</td>
      <td>379.5</td>
    </tr>
    <tr>
      <th>10538</th>
      <td>358.3</td>
      <td>1</td>
      <td>379.5</td>
    </tr>
    <tr>
      <th>13955</th>
      <td>358.3</td>
      <td>1</td>
      <td>379.5</td>
    </tr>
    <tr>
      <th>17343</th>
      <td>358.3</td>
      <td>2</td>
      <td>379.5</td>
    </tr>
    <tr>
      <th>20705</th>
      <td>358.3</td>
      <td>1</td>
      <td>379.5</td>
    </tr>
    <tr>
      <th>24133</th>
      <td>358.3</td>
      <td>0</td>
      <td>379.5</td>
    </tr>
    <tr>
      <th>27352</th>
      <td>358.3</td>
      <td>2</td>
      <td>379.5</td>
    </tr>
    <tr>
      <th>30549</th>
      <td>358.3</td>
      <td>0</td>
      <td>379.5</td>
    </tr>
    <tr>
      <th>33699</th>
      <td>358.3</td>
      <td>0</td>
      <td>379.5</td>
    </tr>
    <tr>
      <th>36913</th>
      <td>358.3</td>
      <td>2</td>
      <td>379.5</td>
    </tr>
    <tr>
      <th>40121</th>
      <td>358.3</td>
      <td>0</td>
      <td>455.4</td>
    </tr>
    <tr>
      <th>43314</th>
      <td>358.3</td>
      <td>0</td>
      <td>455.4</td>
    </tr>
    <tr>
      <th>46553</th>
      <td>358.3</td>
      <td>0</td>
      <td>455.4</td>
    </tr>
    <tr>
      <th>49851</th>
      <td>358.3</td>
      <td>0</td>
      <td>455.4</td>
    </tr>
    <tr>
      <th>53208</th>
      <td>358.3</td>
      <td>0</td>
      <td>455.4</td>
    </tr>
    <tr>
      <th>56647</th>
      <td>358.3</td>
      <td>0</td>
      <td>455.4</td>
    </tr>
    <tr>
      <th>60172</th>
      <td>358.3</td>
      <td>1</td>
      <td>455.4</td>
    </tr>
    <tr>
      <th>63556</th>
      <td>358.3</td>
      <td>0</td>
      <td>455.4</td>
    </tr>
    <tr>
      <th>66880</th>
      <td>358.3</td>
      <td>0</td>
      <td>455.4</td>
    </tr>
    <tr>
      <th>70233</th>
      <td>358.3</td>
      <td>0</td>
      <td>455.4</td>
    </tr>
    <tr>
      <th>73540</th>
      <td>358.3</td>
      <td>1</td>
      <td>455.4</td>
    </tr>
    <tr>
      <th>76867</th>
      <td>358.3</td>
      <td>0</td>
      <td>455.4</td>
    </tr>
    <tr>
      <th>80316</th>
      <td>358.3</td>
      <td>3</td>
      <td>491.8</td>
    </tr>
    <tr>
      <th>83720</th>
      <td>358.3</td>
      <td>0</td>
      <td>491.8</td>
    </tr>
    <tr>
      <th>87133</th>
      <td>358.3</td>
      <td>0</td>
      <td>491.8</td>
    </tr>
    <tr>
      <th>90476</th>
      <td>358.3</td>
      <td>0</td>
      <td>491.8</td>
    </tr>
    <tr>
      <th>93767</th>
      <td>358.3</td>
      <td>0</td>
      <td>491.8</td>
    </tr>
  </tbody>
</table>
</div>




```python
mask = (
    df_fct.groupby("advertiser_zrive_id")["period_int"].nunique() == 29
)

advertisers_29 = mask[mask].index
df_fct[df_fct["advertiser_zrive_id"].isin(advertisers_29)]
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
    <tr>
      <th>5</th>
      <td>7</td>
      <td>2023-01-01</td>
      <td>50</td>
      <td>42</td>
      <td>42</td>
      <td>49.0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>18</td>
      <td>5</td>
      <td>3</td>
      <td>763.3</td>
      <td>1006.7</td>
      <td>4</td>
      <td>7</td>
      <td>11</td>
      <td>19495.74</td>
      <td>True</td>
    </tr>
    <tr>
      <th>6</th>
      <td>10</td>
      <td>2023-01-01</td>
      <td>49</td>
      <td>37</td>
      <td>37</td>
      <td>56.0</td>
      <td>11</td>
      <td>5</td>
      <td>8</td>
      <td>0</td>
      <td>...</td>
      <td>8</td>
      <td>3</td>
      <td>0</td>
      <td>75.0</td>
      <td>1645.0</td>
      <td>3</td>
      <td>3</td>
      <td>6</td>
      <td>37954.00</td>
      <td>True</td>
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
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
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
      <th>96756</th>
      <td>6975</td>
      <td>2025-05-01</td>
      <td>150</td>
      <td>109</td>
      <td>109</td>
      <td>198.0</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>0</td>
      <td>...</td>
      <td>5</td>
      <td>21</td>
      <td>8</td>
      <td>1331.8</td>
      <td>1550.0</td>
      <td>14</td>
      <td>23</td>
      <td>37</td>
      <td>20569.58</td>
      <td>True</td>
    </tr>
    <tr>
      <th>96783</th>
      <td>7003</td>
      <td>2025-05-01</td>
      <td>300</td>
      <td>237</td>
      <td>237</td>
      <td>267.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>85</td>
      <td>116</td>
      <td>26</td>
      <td>2976.7</td>
      <td>2976.7</td>
      <td>60</td>
      <td>94</td>
      <td>154</td>
      <td>64443.55</td>
      <td>True</td>
    </tr>
    <tr>
      <th>96801</th>
      <td>7022</td>
      <td>2025-05-01</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>14.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>3</td>
      <td>1</td>
      <td>2</td>
      <td>11.4</td>
      <td>548.3</td>
      <td>1</td>
      <td>4</td>
      <td>5</td>
      <td>6640.59</td>
      <td>True</td>
    </tr>
    <tr>
      <th>96803</th>
      <td>7024</td>
      <td>2025-05-01</td>
      <td>6</td>
      <td>5</td>
      <td>5</td>
      <td>7.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>3</td>
      <td>4</td>
      <td>1</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>2</td>
      <td>2</td>
      <td>4</td>
      <td>13257.14</td>
      <td>True</td>
    </tr>
    <tr>
      <th>96805</th>
      <td>7026</td>
      <td>2025-05-01</td>
      <td>75</td>
      <td>74</td>
      <td>74</td>
      <td>115.0</td>
      <td>4</td>
      <td>4</td>
      <td>15</td>
      <td>60</td>
      <td>...</td>
      <td>29</td>
      <td>13</td>
      <td>11</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>9</td>
      <td>15</td>
      <td>24</td>
      <td>36636.83</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
<p>40194 rows × 23 columns</p>
</div>




```python
df_fct[df_fct["monthly_total_reference_price"] < df_fct["monthly_total_invoice"]]
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
      <th>25</th>
      <td>37</td>
      <td>2023-01-01</td>
      <td>75</td>
      <td>42</td>
      <td>42</td>
      <td>45.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>63</td>
      <td>19</td>
      <td>29</td>
      <td>958.9</td>
      <td>765.0</td>
      <td>14</td>
      <td>28</td>
      <td>42</td>
      <td>11227.23</td>
      <td>True</td>
    </tr>
    <tr>
      <th>231</th>
      <td>316</td>
      <td>2023-01-01</td>
      <td>20</td>
      <td>20</td>
      <td>20</td>
      <td>27.0</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>379.5</td>
      <td>358.3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>31658.97</td>
      <td>True</td>
    </tr>
    <tr>
      <th>321</th>
      <td>448</td>
      <td>2023-01-01</td>
      <td>250</td>
      <td>247</td>
      <td>247</td>
      <td>460.0</td>
      <td>150</td>
      <td>15</td>
      <td>15</td>
      <td>0</td>
      <td>...</td>
      <td>133</td>
      <td>25</td>
      <td>12</td>
      <td>8135.0</td>
      <td>6608.3</td>
      <td>25</td>
      <td>34</td>
      <td>59</td>
      <td>22943.52</td>
      <td>True</td>
    </tr>
    <tr>
      <th>706</th>
      <td>994</td>
      <td>2023-01-01</td>
      <td>20</td>
      <td>13</td>
      <td>13</td>
      <td>2.0</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>...</td>
      <td>5</td>
      <td>5</td>
      <td>4</td>
      <td>382.4</td>
      <td>351.7</td>
      <td>5</td>
      <td>4</td>
      <td>9</td>
      <td>5000.00</td>
      <td>True</td>
    </tr>
    <tr>
      <th>847</th>
      <td>1195</td>
      <td>2023-01-01</td>
      <td>300</td>
      <td>276</td>
      <td>276</td>
      <td>396.0</td>
      <td>5</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>41</td>
      <td>31</td>
      <td>15</td>
      <td>2449.3</td>
      <td>2363.3</td>
      <td>20</td>
      <td>30</td>
      <td>50</td>
      <td>21321.58</td>
      <td>True</td>
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
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
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
      <th>95889</th>
      <td>5477</td>
      <td>2025-05-01</td>
      <td>100</td>
      <td>92</td>
      <td>92</td>
      <td>119.0</td>
      <td>8</td>
      <td>8</td>
      <td>3</td>
      <td>0</td>
      <td>...</td>
      <td>25</td>
      <td>16</td>
      <td>6</td>
      <td>2330.0</td>
      <td>1426.7</td>
      <td>9</td>
      <td>20</td>
      <td>29</td>
      <td>22807.30</td>
      <td>True</td>
    </tr>
    <tr>
      <th>95974</th>
      <td>5776</td>
      <td>2025-05-01</td>
      <td>1300</td>
      <td>951</td>
      <td>0</td>
      <td>1056.0</td>
      <td>7</td>
      <td>7</td>
      <td>3</td>
      <td>0</td>
      <td>...</td>
      <td>19</td>
      <td>13</td>
      <td>1</td>
      <td>8866.7</td>
      <td>4453.3</td>
      <td>7</td>
      <td>8</td>
      <td>15</td>
      <td>28053.47</td>
      <td>True</td>
    </tr>
    <tr>
      <th>95991</th>
      <td>5814</td>
      <td>2025-05-01</td>
      <td>1</td>
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
      <td>2448.3</td>
      <td>1666.7</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>True</td>
    </tr>
    <tr>
      <th>96249</th>
      <td>6328</td>
      <td>2025-05-01</td>
      <td>20</td>
      <td>20</td>
      <td>20</td>
      <td>27.0</td>
      <td>16</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>17</td>
      <td>10</td>
      <td>3</td>
      <td>1000.0</td>
      <td>551.7</td>
      <td>6</td>
      <td>10</td>
      <td>16</td>
      <td>45378.52</td>
      <td>True</td>
    </tr>
    <tr>
      <th>96582</th>
      <td>6791</td>
      <td>2025-05-01</td>
      <td>500</td>
      <td>493</td>
      <td>493</td>
      <td>1719.0</td>
      <td>15</td>
      <td>15</td>
      <td>15</td>
      <td>0</td>
      <td>...</td>
      <td>41</td>
      <td>45</td>
      <td>13</td>
      <td>23201.7</td>
      <td>6111.7</td>
      <td>15</td>
      <td>20</td>
      <td>35</td>
      <td>21567.03</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
<p>640 rows × 23 columns</p>
</div>




```python
df_fct.describe()
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
      <th>monthly_leads</th>
      <th>monthly_total_phone_views</th>
      <th>monthly_total_calls</th>
      <th>monthly_total_emails</th>
      <th>monthly_total_invoice</th>
      <th>monthly_total_reference_price</th>
      <th>monthly_unique_calls</th>
      <th>monthly_unique_emails</th>
      <th>monthly_unique_leads</th>
      <th>monthly_avg_ad_price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>96827.000000</td>
      <td>96827</td>
      <td>96827.000000</td>
      <td>96827.000000</td>
      <td>96827.00000</td>
      <td>86775.000000</td>
      <td>96827.000000</td>
      <td>96827.000000</td>
      <td>96827.000000</td>
      <td>96827.000000</td>
      <td>...</td>
      <td>96827.000000</td>
      <td>96827.000000</td>
      <td>96827.000000</td>
      <td>96827.000000</td>
      <td>96827.000000</td>
      <td>84498.000000</td>
      <td>96827.000000</td>
      <td>96827.000000</td>
      <td>96827.000000</td>
      <td>8.836000e+04</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>2994.207866</td>
      <td>2024-03-01 02:55:41.786898176</td>
      <td>108.101191</td>
      <td>93.381774</td>
      <td>50.39962</td>
      <td>181.334947</td>
      <td>3.057453</td>
      <td>1.577008</td>
      <td>1.654972</td>
      <td>0.389984</td>
      <td>...</td>
      <td>19.562261</td>
      <td>12.317257</td>
      <td>8.584279</td>
      <td>4.729673</td>
      <td>502.412151</td>
      <td>1325.395421</td>
      <td>6.134973</td>
      <td>9.332624</td>
      <td>15.467597</td>
      <td>2.869974e+04</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>2023-01-01 00:00:00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>3.300000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000e+00</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>1364.000000</td>
      <td>2023-08-01 00:00:00</td>
      <td>6.000000</td>
      <td>3.000000</td>
      <td>3.00000</td>
      <td>6.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>2.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>71.600000</td>
      <td>361.700000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.433743e+04</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>2782.000000</td>
      <td>2024-03-01 00:00:00</td>
      <td>20.000000</td>
      <td>12.000000</td>
      <td>10.00000</td>
      <td>18.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>6.000000</td>
      <td>4.000000</td>
      <td>3.000000</td>
      <td>1.000000</td>
      <td>235.000000</td>
      <td>548.300000</td>
      <td>2.000000</td>
      <td>3.000000</td>
      <td>5.000000</td>
      <td>2.042249e+04</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>4514.000000</td>
      <td>2024-10-01 00:00:00</td>
      <td>50.000000</td>
      <td>38.000000</td>
      <td>34.00000</td>
      <td>60.000000</td>
      <td>2.000000</td>
      <td>2.000000</td>
      <td>2.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>18.000000</td>
      <td>13.000000</td>
      <td>8.000000</td>
      <td>5.000000</td>
      <td>545.000000</td>
      <td>1105.000000</td>
      <td>6.000000</td>
      <td>9.000000</td>
      <td>15.000000</td>
      <td>2.799000e+04</td>
    </tr>
    <tr>
      <th>max</th>
      <td>7069.000000</td>
      <td>2025-05-01 00:00:00</td>
      <td>8200.000000</td>
      <td>8200.000000</td>
      <td>5400.00000</td>
      <td>133865.000000</td>
      <td>900.000000</td>
      <td>165.000000</td>
      <td>120.000000</td>
      <td>900.000000</td>
      <td>...</td>
      <td>1983.000000</td>
      <td>926.000000</td>
      <td>1098.000000</td>
      <td>406.000000</td>
      <td>33500.000000</td>
      <td>696546.700000</td>
      <td>799.000000</td>
      <td>716.000000</td>
      <td>1515.000000</td>
      <td>3.705341e+07</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1902.383463</td>
      <td>NaN</td>
      <td>490.209838</td>
      <td>478.776149</td>
      <td>243.68727</td>
      <td>1250.878986</td>
      <td>19.359616</td>
      <td>4.308079</td>
      <td>4.098040</td>
      <td>8.207100</td>
      <td>...</td>
      <td>52.583756</td>
      <td>28.280672</td>
      <td>24.388530</td>
      <td>12.075520</td>
      <td>1048.308252</td>
      <td>5554.633364</td>
      <td>16.616800</td>
      <td>23.543399</td>
      <td>39.382878</td>
      <td>1.460678e+05</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 22 columns</p>
</div>




```python
df_fct.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Index: 96827 entries, 0 to 96828
    Data columns (total 23 columns):
     #   Column                         Non-Null Count  Dtype         
    ---  ------                         --------------  -----         
     0   advertiser_zrive_id            96827 non-null  int64         
     1   period_int                     96827 non-null  datetime64[ns]
     2   monthly_contracted_ads         96827 non-null  int64         
     3   monthly_published_ads          96827 non-null  int64         
     4   monthly_unique_published_ads   96827 non-null  int64         
     5   monthly_distinct_ads           86775 non-null  float64       
     6   monthly_oro_ads                96827 non-null  int64         
     7   monthly_plata_ads              96827 non-null  int64         
     8   monthly_destacados_ads         96827 non-null  int64         
     9   monthly_pepitas_ads            96827 non-null  int64         
     10  monthly_shows                  96827 non-null  float64       
     11  monthly_visits                 96827 non-null  float64       
     12  monthly_leads                  96827 non-null  int64         
     13  monthly_total_phone_views      96827 non-null  int64         
     14  monthly_total_calls            96827 non-null  int64         
     15  monthly_total_emails           96827 non-null  int64         
     16  monthly_total_invoice          96827 non-null  float64       
     17  monthly_total_reference_price  84498 non-null  float64       
     18  monthly_unique_calls           96827 non-null  int64         
     19  monthly_unique_emails          96827 non-null  int64         
     20  monthly_unique_leads           96827 non-null  int64         
     21  monthly_avg_ad_price           88360 non-null  float64       
     22  has_active_contract            96827 non-null  bool          
    dtypes: bool(1), datetime64[ns](1), float64(6), int64(15)
    memory usage: 17.1 MB


### FCT - Agregación

Agregamos los valores mensuales para los meses de onboarding (3 primeros meses) para quedarnos con 1 fila = 1 registro

#### Identificadores
- 🔑 **advertiser_zive_id**: Identificador único del anunciante.
- 🔑 **period_int**: contamos el nº de meses.

#### Métricas de anuncios -> media
- **monthly_contracted_ads**: Número total de anuncios contratados.
- **monthly_published_ads**: Número total de anuncios publicados.

#### Métricas de anuncios únicas
- **monthly_unique_published_ads**: Número total de anuncios únicos publicados, en caso de publicar anuncios en diferentes provincias.
- **monthly_distinct_ads**: Número total de anuncios diferentes publicados en un mes (rotación total mensual).

#### Anuncios premium -> media
- **monthly_oro_ads**: Número total de anuncios Oro contratados.
- **monthly_plata_ads**: Número total de anuncios Plata contratados.
- **monthly_destacados_ads**: Número total de anuncios Destacados contratados.
- **monthly_pepitas_ads**: Número total de anuncios Pepitas contratados.

#### Métricas de interacción 
- **monthly_shows_ads**: Número total de búsquedas de anuncios publicados.
- **monthly_visits_ads**: Número total de visitas de anuncios publicados.
- **monthly_leads**: Número total de leads de anuncios publicados.
- **monthly_total_phone_views**: Número total de vistas del teléfono del anunciante de anuncios publicados.
- **monthly_total_calls**: Número total de llamadas al anunciante de anuncios publicados.
- **monthly_total_emails**: Número total de emails al anunciante de anuncios publicados.

#### Métricas económicas -> media
- **monthly_total_invoice**: Facturación mensual.
- **monthly_total_reference_price**: Precio tarifa mensual.
- **monthly_avg_ad_price**: Precio medio de los anuncios publicados.

#### Métricas únicas -> suma, max ?
- **monthly_unique_calls**: Número de llamadas al anunciante de anuncios únicos publicados.
- **monthly_unique_emails**: Número de emails al anunciante de anuncios únicos publicados.
- **monthly_unique_leads**: Número de leads de anuncios únicos publicados.

#### Estado del contrato
- **has_active_contract**: Indica si el anunciante tiene una propuesta activa en el mes.

Como vamos a estudiar el periodo de onboarding (3 primeros meses), no nos interesan los advertisers que hayan estado menos de 3 meses en la plataforma


```python
ids_valid_onboarding = (
    df_fct
    .groupby("advertiser_zrive_id")["period_int"]
    .nunique()
    .loc[lambda s: s >= 3] # Filtramos los registros >= 3 
    .index
)
ids_valid_onboarding
```




    Index([   1,    2,    3,    4,    5,    6,    7,    8,    9,   10,
           ...
           7001, 7003, 7011, 7022, 7024, 7026, 7027, 7036, 7046, 7066],
          dtype='int64', name='advertiser_zrive_id', length=6397)




```python
df_onboarding = df_fct[df_fct["advertiser_zrive_id"].isin(ids_valid_onboarding)].copy()
```


```python
df_onboarding
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
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
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
      <th>96805</th>
      <td>7026</td>
      <td>2025-05-01</td>
      <td>75</td>
      <td>74</td>
      <td>74</td>
      <td>115.0</td>
      <td>4</td>
      <td>4</td>
      <td>15</td>
      <td>60</td>
      <td>...</td>
      <td>29</td>
      <td>13</td>
      <td>11</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>9</td>
      <td>15</td>
      <td>24</td>
      <td>36636.83</td>
      <td>True</td>
    </tr>
    <tr>
      <th>96806</th>
      <td>7027</td>
      <td>2025-05-01</td>
      <td>50</td>
      <td>45</td>
      <td>45</td>
      <td>63.0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>36</td>
      <td>27</td>
      <td>7</td>
      <td>550.0</td>
      <td>1006.7</td>
      <td>17</td>
      <td>16</td>
      <td>33</td>
      <td>17110.32</td>
      <td>True</td>
    </tr>
    <tr>
      <th>96815</th>
      <td>7036</td>
      <td>2025-05-01</td>
      <td>14</td>
      <td>17</td>
      <td>17</td>
      <td>17.0</td>
      <td>0</td>
      <td>2</td>
      <td>4</td>
      <td>0</td>
      <td>...</td>
      <td>7</td>
      <td>10</td>
      <td>2</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>9</td>
      <td>6</td>
      <td>15</td>
      <td>19947.06</td>
      <td>False</td>
    </tr>
    <tr>
      <th>96823</th>
      <td>7046</td>
      <td>2025-05-01</td>
      <td>35</td>
      <td>35</td>
      <td>35</td>
      <td>42.0</td>
      <td>3</td>
      <td>2</td>
      <td>9</td>
      <td>0</td>
      <td>...</td>
      <td>22</td>
      <td>4</td>
      <td>7</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>4</td>
      <td>15</td>
      <td>19</td>
      <td>19848.81</td>
      <td>True</td>
    </tr>
    <tr>
      <th>96827</th>
      <td>7066</td>
      <td>2025-05-01</td>
      <td>20</td>
      <td>20</td>
      <td>20</td>
      <td>29.0</td>
      <td>3</td>
      <td>3</td>
      <td>7</td>
      <td>0</td>
      <td>...</td>
      <td>6</td>
      <td>24</td>
      <td>4</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>17</td>
      <td>26</td>
      <td>43</td>
      <td>9859.33</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
<p>96011 rows × 23 columns</p>
</div>




```python
df_onboarding.duplicated(subset=["advertiser_zrive_id", "period_int"]).any()
```




    np.False_




```python
df_onboarding.columns
```




    Index(['advertiser_zrive_id', 'period_int', 'monthly_contracted_ads',
           'monthly_published_ads', 'monthly_unique_published_ads',
           'monthly_distinct_ads', 'monthly_oro_ads', 'monthly_plata_ads',
           'monthly_destacados_ads', 'monthly_pepitas_ads', 'monthly_shows',
           'monthly_visits', 'monthly_leads', 'monthly_total_phone_views',
           'monthly_total_calls', 'monthly_total_emails', 'monthly_total_invoice',
           'monthly_total_reference_price', 'monthly_unique_calls',
           'monthly_unique_emails', 'monthly_unique_leads', 'monthly_avg_ad_price',
           'has_active_contract'],
          dtype='object')




```python
df_onboarding.shape
```




    (96011, 23)




```python
df_onboarding.has_active_contract.mean()
```




    np.float64(0.97285727677037)



### Imputación de valores nulos

- **monthly_distinct_ads**: los rellenamos con 0. Si no existe, es que no hay rotación de anuncios, por lo tanto tiene sentido que sean 0


```python
df_onboarding.isna().sum()
```




    advertiser_zrive_id                  0
    period_int                           0
    monthly_contracted_ads               0
    monthly_published_ads                0
    monthly_unique_published_ads         0
    monthly_distinct_ads              9857
    monthly_oro_ads                      0
    monthly_plata_ads                    0
    monthly_destacados_ads               0
    monthly_pepitas_ads                  0
    monthly_shows                        0
    monthly_visits                       0
    monthly_leads                        0
    monthly_total_phone_views            0
    monthly_total_calls                  0
    monthly_total_emails                 0
    monthly_total_invoice                0
    monthly_total_reference_price    12003
    monthly_unique_calls                 0
    monthly_unique_emails                0
    monthly_unique_leads                 0
    monthly_avg_ad_price              8292
    has_active_contract                  0
    dtype: int64




```python
df_onboarding[df_onboarding["monthly_total_reference_price"].isna()].advertiser_zrive_id.unique()
```




    array([   3,   11,   15, ..., 6583, 6868, 6904], shape=(3305,))




```python
df_onboarding["monthly_distinct_ads"] = df_onboarding["monthly_distinct_ads"].fillna(0)
```


```python
df_onboarding[df_onboarding["advertiser_zrive_id"]==4422]
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
      <th>3258</th>
      <td>4422</td>
      <td>2023-01-01</td>
      <td>14</td>
      <td>3</td>
      <td>3</td>
      <td>3.0</td>
      <td>6</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>17</td>
      <td>1</td>
      <td>0</td>
      <td>505.0</td>
      <td>2376.7</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>43544.00</td>
      <td>True</td>
    </tr>
    <tr>
      <th>6559</th>
      <td>4422</td>
      <td>2023-02-01</td>
      <td>14</td>
      <td>2</td>
      <td>2</td>
      <td>8.0</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>13</td>
      <td>5</td>
      <td>2</td>
      <td>911.6</td>
      <td>2376.7</td>
      <td>5</td>
      <td>8</td>
      <td>13</td>
      <td>42856.67</td>
      <td>True</td>
    </tr>
    <tr>
      <th>9882</th>
      <td>4422</td>
      <td>2023-03-01</td>
      <td>14</td>
      <td>3</td>
      <td>3</td>
      <td>5.0</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>9</td>
      <td>2</td>
      <td>2</td>
      <td>0.0</td>
      <td>1650.0</td>
      <td>2</td>
      <td>6</td>
      <td>8</td>
      <td>45146.36</td>
      <td>True</td>
    </tr>
    <tr>
      <th>13248</th>
      <td>4422</td>
      <td>2023-04-01</td>
      <td>14</td>
      <td>3</td>
      <td>3</td>
      <td>4.0</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>32</td>
      <td>8</td>
      <td>3</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>5</td>
      <td>7</td>
      <td>12</td>
      <td>44533.33</td>
      <td>True</td>
    </tr>
    <tr>
      <th>16589</th>
      <td>4422</td>
      <td>2023-05-01</td>
      <td>14</td>
      <td>0</td>
      <td>0</td>
      <td>3.0</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>1</td>
      <td>2</td>
      <td>3</td>
      <td>44533.33</td>
      <td>True</td>
    </tr>
    <tr>
      <th>19918</th>
      <td>4422</td>
      <td>2023-06-01</td>
      <td>6</td>
      <td>1</td>
      <td>1</td>
      <td>1.0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2444.3</td>
      <td>2376.7</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>44533.33</td>
      <td>True</td>
    </tr>
    <tr>
      <th>23302</th>
      <td>4422</td>
      <td>2023-07-01</td>
      <td>6</td>
      <td>2</td>
      <td>2</td>
      <td>3.0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>1</td>
      <td>3</td>
      <td>4</td>
      <td>51050.00</td>
      <td>True</td>
    </tr>
    <tr>
      <th>26559</th>
      <td>4422</td>
      <td>2023-08-01</td>
      <td>6</td>
      <td>2</td>
      <td>2</td>
      <td>2.0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>7</td>
      <td>0</td>
      <td>0</td>
      <td>1515.0</td>
      <td>2376.7</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>51050.00</td>
      <td>True</td>
    </tr>
    <tr>
      <th>29728</th>
      <td>4422</td>
      <td>2023-09-01</td>
      <td>6</td>
      <td>6</td>
      <td>6</td>
      <td>13.0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>1</td>
      <td>3</td>
      <td>4</td>
      <td>47711.60</td>
      <td>True</td>
    </tr>
    <tr>
      <th>32835</th>
      <td>4422</td>
      <td>2023-10-01</td>
      <td>6</td>
      <td>3</td>
      <td>3</td>
      <td>13.0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>4</td>
      <td>3</td>
      <td>6</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>3</td>
      <td>8</td>
      <td>11</td>
      <td>45772.42</td>
      <td>True</td>
    </tr>
    <tr>
      <th>36002</th>
      <td>4422</td>
      <td>2023-11-01</td>
      <td>6</td>
      <td>5</td>
      <td>5</td>
      <td>9.0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>9</td>
      <td>4</td>
      <td>5</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>3</td>
      <td>5</td>
      <td>8</td>
      <td>44951.08</td>
      <td>True</td>
    </tr>
    <tr>
      <th>39164</th>
      <td>4422</td>
      <td>2023-12-01</td>
      <td>6</td>
      <td>6</td>
      <td>6</td>
      <td>9.0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>1</td>
      <td>2</td>
      <td>3</td>
      <td>44607.00</td>
      <td>True</td>
    </tr>
    <tr>
      <th>42342</th>
      <td>4422</td>
      <td>2024-01-01</td>
      <td>6</td>
      <td>3</td>
      <td>3</td>
      <td>11.0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>2</td>
      <td>7</td>
      <td>2</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>5</td>
      <td>5</td>
      <td>10</td>
      <td>43800.91</td>
      <td>True</td>
    </tr>
    <tr>
      <th>45527</th>
      <td>4422</td>
      <td>2024-02-01</td>
      <td>6</td>
      <td>3</td>
      <td>3</td>
      <td>8.0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>4</td>
      <td>3</td>
      <td>2</td>
      <td>2225.0</td>
      <td>3706.7</td>
      <td>2</td>
      <td>3</td>
      <td>5</td>
      <td>44095.83</td>
      <td>True</td>
    </tr>
    <tr>
      <th>48770</th>
      <td>4422</td>
      <td>2024-03-01</td>
      <td>6</td>
      <td>2</td>
      <td>2</td>
      <td>4.0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>2</td>
      <td>2</td>
      <td>4</td>
      <td>43854.90</td>
      <td>True</td>
    </tr>
    <tr>
      <th>52077</th>
      <td>4422</td>
      <td>2024-04-01</td>
      <td>6</td>
      <td>6</td>
      <td>6</td>
      <td>8.0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>1</td>
      <td>5</td>
      <td>6</td>
      <td>44800.55</td>
      <td>True</td>
    </tr>
    <tr>
      <th>55464</th>
      <td>4422</td>
      <td>2024-05-01</td>
      <td>6</td>
      <td>6</td>
      <td>6</td>
      <td>6.0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>3</td>
      <td>0</td>
      <td>4</td>
      <td>710.0</td>
      <td>910.0</td>
      <td>0</td>
      <td>4</td>
      <td>4</td>
      <td>44800.55</td>
      <td>True</td>
    </tr>
    <tr>
      <th>58887</th>
      <td>4422</td>
      <td>2024-06-01</td>
      <td>6</td>
      <td>6</td>
      <td>6</td>
      <td>6.0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>44800.55</td>
      <td>True</td>
    </tr>
    <tr>
      <th>62309</th>
      <td>4422</td>
      <td>2024-07-01</td>
      <td>6</td>
      <td>6</td>
      <td>6</td>
      <td>6.0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>4</td>
      <td>0</td>
      <td>1</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>44800.55</td>
      <td>True</td>
    </tr>
    <tr>
      <th>65648</th>
      <td>4422</td>
      <td>2024-08-01</td>
      <td>6</td>
      <td>6</td>
      <td>6</td>
      <td>6.0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>2</td>
      <td>3</td>
      <td>2</td>
      <td>710.0</td>
      <td>910.0</td>
      <td>2</td>
      <td>2</td>
      <td>4</td>
      <td>44800.55</td>
      <td>True</td>
    </tr>
    <tr>
      <th>68967</th>
      <td>4422</td>
      <td>2024-09-01</td>
      <td>6</td>
      <td>5</td>
      <td>5</td>
      <td>23.0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>3</td>
      <td>4</td>
      <td>1</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>3</td>
      <td>6</td>
      <td>9</td>
      <td>43109.68</td>
      <td>True</td>
    </tr>
    <tr>
      <th>72278</th>
      <td>4422</td>
      <td>2024-10-01</td>
      <td>6</td>
      <td>3</td>
      <td>3</td>
      <td>5.0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>42061.46</td>
      <td>True</td>
    </tr>
    <tr>
      <th>75563</th>
      <td>4422</td>
      <td>2024-11-01</td>
      <td>6</td>
      <td>3</td>
      <td>3</td>
      <td>5.0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>41585.00</td>
      <td>True</td>
    </tr>
    <tr>
      <th>78867</th>
      <td>4422</td>
      <td>2024-12-01</td>
      <td>6</td>
      <td>6</td>
      <td>6</td>
      <td>7.0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>2</td>
      <td>6</td>
      <td>1</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>3</td>
      <td>1</td>
      <td>4</td>
      <td>41147.22</td>
      <td>True</td>
    </tr>
    <tr>
      <th>95576</th>
      <td>4422</td>
      <td>2025-05-01</td>
      <td>20</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>2</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>NaN</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
<p>25 rows × 23 columns</p>
</div>




```python
ads_cols = ['monthly_contracted_ads', 'monthly_published_ads', 'monthly_distinct_ads']
premium_ads_cols = ['monthly_oro_ads', 'monthly_plata_ads', 'monthly_destacados_ads', 'monthly_pepitas_ads']
funnel_cols = ['monthly_shows', 'monthly_visits', 'monthly_leads']
interaction_cols = ['monthly_total_phone_views', 'monthly_total_calls', 'monthly_total_emails']
unique_cols = ['monthly_unique_published_ads', 'monthly_unique_emails', 'monthly_unique_leads']
economic_cols = ['monthly_total_invoice', 'monthly_total_reference_price']
```


```python
df_onboarding[['monthly_distinct_ads', 'monthly_unique_published_ads']]
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
      <th>monthly_distinct_ads</th>
      <th>monthly_unique_published_ads</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>56.0</td>
      <td>47</td>
    </tr>
    <tr>
      <th>1</th>
      <td>55.0</td>
      <td>31</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>86.0</td>
      <td>79</td>
    </tr>
    <tr>
      <th>4</th>
      <td>25.0</td>
      <td>20</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>96805</th>
      <td>115.0</td>
      <td>74</td>
    </tr>
    <tr>
      <th>96806</th>
      <td>63.0</td>
      <td>45</td>
    </tr>
    <tr>
      <th>96815</th>
      <td>17.0</td>
      <td>17</td>
    </tr>
    <tr>
      <th>96823</th>
      <td>42.0</td>
      <td>35</td>
    </tr>
    <tr>
      <th>96827</th>
      <td>29.0</td>
      <td>20</td>
    </tr>
  </tbody>
</table>
<p>96011 rows × 2 columns</p>
</div>




```python
df_onboarding[df_onboarding.advertiser_zrive_id == 1]
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
      <th>3542</th>
      <td>1</td>
      <td>2023-02-01</td>
      <td>50</td>
      <td>44</td>
      <td>44</td>
      <td>48.0</td>
      <td>3</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>18</td>
      <td>6</td>
      <td>3</td>
      <td>1807.5</td>
      <td>3948.3</td>
      <td>5</td>
      <td>4</td>
      <td>9</td>
      <td>42691.67</td>
      <td>True</td>
    </tr>
    <tr>
      <th>6905</th>
      <td>1</td>
      <td>2023-03-01</td>
      <td>50</td>
      <td>39</td>
      <td>39</td>
      <td>48.0</td>
      <td>3</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>5</td>
      <td>4</td>
      <td>2</td>
      <td>1807.5</td>
      <td>3948.3</td>
      <td>3</td>
      <td>3</td>
      <td>6</td>
      <td>44193.75</td>
      <td>True</td>
    </tr>
    <tr>
      <th>10328</th>
      <td>1</td>
      <td>2023-04-01</td>
      <td>50</td>
      <td>39</td>
      <td>39</td>
      <td>40.0</td>
      <td>3</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>7</td>
      <td>3</td>
      <td>2</td>
      <td>1807.5</td>
      <td>3948.3</td>
      <td>3</td>
      <td>3</td>
      <td>6</td>
      <td>44452.50</td>
      <td>True</td>
    </tr>
    <tr>
      <th>13747</th>
      <td>1</td>
      <td>2023-05-01</td>
      <td>150</td>
      <td>0</td>
      <td>0</td>
      <td>39.0</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>13</td>
      <td>4</td>
      <td>2</td>
      <td>1807.5</td>
      <td>3948.3</td>
      <td>4</td>
      <td>4</td>
      <td>8</td>
      <td>43033.33</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 23 columns</p>
</div>




```python
df_fct[df_fct.advertiser_zrive_id == 1]
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
      <th>3542</th>
      <td>1</td>
      <td>2023-02-01</td>
      <td>50</td>
      <td>44</td>
      <td>44</td>
      <td>48.0</td>
      <td>3</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>18</td>
      <td>6</td>
      <td>3</td>
      <td>1807.5</td>
      <td>3948.3</td>
      <td>5</td>
      <td>4</td>
      <td>9</td>
      <td>42691.67</td>
      <td>True</td>
    </tr>
    <tr>
      <th>6905</th>
      <td>1</td>
      <td>2023-03-01</td>
      <td>50</td>
      <td>39</td>
      <td>39</td>
      <td>48.0</td>
      <td>3</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>5</td>
      <td>4</td>
      <td>2</td>
      <td>1807.5</td>
      <td>3948.3</td>
      <td>3</td>
      <td>3</td>
      <td>6</td>
      <td>44193.75</td>
      <td>True</td>
    </tr>
    <tr>
      <th>10328</th>
      <td>1</td>
      <td>2023-04-01</td>
      <td>50</td>
      <td>39</td>
      <td>39</td>
      <td>40.0</td>
      <td>3</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>7</td>
      <td>3</td>
      <td>2</td>
      <td>1807.5</td>
      <td>3948.3</td>
      <td>3</td>
      <td>3</td>
      <td>6</td>
      <td>44452.50</td>
      <td>True</td>
    </tr>
    <tr>
      <th>13747</th>
      <td>1</td>
      <td>2023-05-01</td>
      <td>150</td>
      <td>0</td>
      <td>0</td>
      <td>39.0</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>13</td>
      <td>4</td>
      <td>2</td>
      <td>1807.5</td>
      <td>3948.3</td>
      <td>4</td>
      <td>4</td>
      <td>8</td>
      <td>43033.33</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 23 columns</p>
</div>




```python
df_fct[df_fct["advertiser_zrive_id"] == 210]
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
      <th>157</th>
      <td>210</td>
      <td>2023-01-01</td>
      <td>20</td>
      <td>4</td>
      <td>0</td>
      <td>4.0</td>
      <td>3</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>3</td>
      <td>2</td>
      <td>0</td>
      <td>757.6</td>
      <td>938.3</td>
      <td>1</td>
      <td>2</td>
      <td>3</td>
      <td>90925.00</td>
      <td>True</td>
    </tr>
    <tr>
      <th>3692</th>
      <td>210</td>
      <td>2023-02-01</td>
      <td>20</td>
      <td>4</td>
      <td>0</td>
      <td>4.0</td>
      <td>3</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>5</td>
      <td>3</td>
      <td>0</td>
      <td>757.6</td>
      <td>938.3</td>
      <td>3</td>
      <td>0</td>
      <td>3</td>
      <td>90925.00</td>
      <td>True</td>
    </tr>
    <tr>
      <th>7056</th>
      <td>210</td>
      <td>2023-03-01</td>
      <td>20</td>
      <td>7</td>
      <td>0</td>
      <td>8.0</td>
      <td>3</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>757.6</td>
      <td>938.3</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>96425.00</td>
      <td>True</td>
    </tr>
    <tr>
      <th>10472</th>
      <td>210</td>
      <td>2023-04-01</td>
      <td>20</td>
      <td>7</td>
      <td>0</td>
      <td>7.0</td>
      <td>3</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>9</td>
      <td>3</td>
      <td>0</td>
      <td>757.6</td>
      <td>938.3</td>
      <td>3</td>
      <td>1</td>
      <td>4</td>
      <td>97914.29</td>
      <td>True</td>
    </tr>
    <tr>
      <th>13887</th>
      <td>210</td>
      <td>2023-05-01</td>
      <td>20</td>
      <td>9</td>
      <td>0</td>
      <td>9.0</td>
      <td>3</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>757.6</td>
      <td>938.3</td>
      <td>0</td>
      <td>4</td>
      <td>4</td>
      <td>98244.44</td>
      <td>True</td>
    </tr>
    <tr>
      <th>17276</th>
      <td>210</td>
      <td>2023-06-01</td>
      <td>20</td>
      <td>9</td>
      <td>0</td>
      <td>9.0</td>
      <td>3</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>5</td>
      <td>1</td>
      <td>0</td>
      <td>757.6</td>
      <td>938.3</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>98244.44</td>
      <td>True</td>
    </tr>
    <tr>
      <th>20638</th>
      <td>210</td>
      <td>2023-07-01</td>
      <td>20</td>
      <td>9</td>
      <td>0</td>
      <td>9.0</td>
      <td>3</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>8</td>
      <td>3</td>
      <td>4</td>
      <td>757.6</td>
      <td>938.3</td>
      <td>3</td>
      <td>4</td>
      <td>7</td>
      <td>98244.44</td>
      <td>True</td>
    </tr>
    <tr>
      <th>24070</th>
      <td>210</td>
      <td>2023-08-01</td>
      <td>20</td>
      <td>9</td>
      <td>0</td>
      <td>9.0</td>
      <td>3</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>8</td>
      <td>4</td>
      <td>1</td>
      <td>757.6</td>
      <td>938.3</td>
      <td>4</td>
      <td>3</td>
      <td>7</td>
      <td>98244.44</td>
      <td>True</td>
    </tr>
    <tr>
      <th>27289</th>
      <td>210</td>
      <td>2023-09-01</td>
      <td>20</td>
      <td>9</td>
      <td>0</td>
      <td>9.0</td>
      <td>3</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>4</td>
      <td>1</td>
      <td>0</td>
      <td>757.6</td>
      <td>938.3</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>98244.44</td>
      <td>True</td>
    </tr>
    <tr>
      <th>30487</th>
      <td>210</td>
      <td>2023-10-01</td>
      <td>20</td>
      <td>9</td>
      <td>0</td>
      <td>9.0</td>
      <td>3</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>6</td>
      <td>1</td>
      <td>0</td>
      <td>757.6</td>
      <td>938.3</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>98244.44</td>
      <td>True</td>
    </tr>
    <tr>
      <th>33638</th>
      <td>210</td>
      <td>2023-11-01</td>
      <td>20</td>
      <td>9</td>
      <td>0</td>
      <td>9.0</td>
      <td>3</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>757.6</td>
      <td>938.3</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>98244.44</td>
      <td>True</td>
    </tr>
    <tr>
      <th>36852</th>
      <td>210</td>
      <td>2023-12-01</td>
      <td>20</td>
      <td>9</td>
      <td>0</td>
      <td>9.0</td>
      <td>3</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>7</td>
      <td>2</td>
      <td>1</td>
      <td>757.6</td>
      <td>938.3</td>
      <td>2</td>
      <td>2</td>
      <td>4</td>
      <td>98244.44</td>
      <td>True</td>
    </tr>
    <tr>
      <th>40058</th>
      <td>210</td>
      <td>2024-01-01</td>
      <td>20</td>
      <td>9</td>
      <td>0</td>
      <td>9.0</td>
      <td>3</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>2</td>
      <td>2</td>
      <td>5</td>
      <td>786.4</td>
      <td>938.3</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>98244.44</td>
      <td>True</td>
    </tr>
    <tr>
      <th>43251</th>
      <td>210</td>
      <td>2024-02-01</td>
      <td>20</td>
      <td>9</td>
      <td>0</td>
      <td>9.0</td>
      <td>3</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>786.4</td>
      <td>938.3</td>
      <td>0</td>
      <td>2</td>
      <td>2</td>
      <td>98244.44</td>
      <td>True</td>
    </tr>
    <tr>
      <th>46492</th>
      <td>210</td>
      <td>2024-03-01</td>
      <td>20</td>
      <td>9</td>
      <td>0</td>
      <td>9.0</td>
      <td>3</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>3</td>
      <td>3</td>
      <td>0</td>
      <td>786.4</td>
      <td>938.3</td>
      <td>3</td>
      <td>0</td>
      <td>3</td>
      <td>98244.44</td>
      <td>True</td>
    </tr>
    <tr>
      <th>49790</th>
      <td>210</td>
      <td>2024-04-01</td>
      <td>30</td>
      <td>0</td>
      <td>0</td>
      <td>9.0</td>
      <td>29</td>
      <td>3</td>
      <td>26</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>786.4</td>
      <td>938.3</td>
      <td>0</td>
      <td>2</td>
      <td>2</td>
      <td>98244.44</td>
      <td>True</td>
    </tr>
    <tr>
      <th>53147</th>
      <td>210</td>
      <td>2024-05-01</td>
      <td>30</td>
      <td>22</td>
      <td>0</td>
      <td>25.0</td>
      <td>28</td>
      <td>2</td>
      <td>31</td>
      <td>0</td>
      <td>...</td>
      <td>4</td>
      <td>4</td>
      <td>0</td>
      <td>786.4</td>
      <td>938.3</td>
      <td>3</td>
      <td>3</td>
      <td>6</td>
      <td>102732.80</td>
      <td>True</td>
    </tr>
    <tr>
      <th>56584</th>
      <td>210</td>
      <td>2024-06-01</td>
      <td>20</td>
      <td>20</td>
      <td>0</td>
      <td>26.0</td>
      <td>12</td>
      <td>2</td>
      <td>15</td>
      <td>0</td>
      <td>...</td>
      <td>4</td>
      <td>5</td>
      <td>0</td>
      <td>580.0</td>
      <td>693.3</td>
      <td>3</td>
      <td>3</td>
      <td>6</td>
      <td>101949.33</td>
      <td>True</td>
    </tr>
    <tr>
      <th>60112</th>
      <td>210</td>
      <td>2024-07-01</td>
      <td>20</td>
      <td>19</td>
      <td>0</td>
      <td>20.0</td>
      <td>12</td>
      <td>2</td>
      <td>15</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>5</td>
      <td>0</td>
      <td>580.0</td>
      <td>693.3</td>
      <td>4</td>
      <td>0</td>
      <td>4</td>
      <td>101949.33</td>
      <td>True</td>
    </tr>
    <tr>
      <th>63498</th>
      <td>210</td>
      <td>2024-08-01</td>
      <td>20</td>
      <td>18</td>
      <td>0</td>
      <td>33.0</td>
      <td>12</td>
      <td>2</td>
      <td>15</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>580.0</td>
      <td>693.3</td>
      <td>0</td>
      <td>3</td>
      <td>3</td>
      <td>101599.39</td>
      <td>True</td>
    </tr>
    <tr>
      <th>66819</th>
      <td>210</td>
      <td>2024-09-01</td>
      <td>20</td>
      <td>15</td>
      <td>0</td>
      <td>23.0</td>
      <td>12</td>
      <td>2</td>
      <td>15</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>580.0</td>
      <td>693.3</td>
      <td>1</td>
      <td>4</td>
      <td>5</td>
      <td>96296.58</td>
      <td>True</td>
    </tr>
    <tr>
      <th>70168</th>
      <td>210</td>
      <td>2024-10-01</td>
      <td>20</td>
      <td>10</td>
      <td>0</td>
      <td>15.0</td>
      <td>11</td>
      <td>2</td>
      <td>5</td>
      <td>0</td>
      <td>...</td>
      <td>4</td>
      <td>4</td>
      <td>0</td>
      <td>580.0</td>
      <td>693.3</td>
      <td>3</td>
      <td>3</td>
      <td>6</td>
      <td>95788.25</td>
      <td>True</td>
    </tr>
    <tr>
      <th>73476</th>
      <td>210</td>
      <td>2024-11-01</td>
      <td>20</td>
      <td>15</td>
      <td>0</td>
      <td>30.0</td>
      <td>11</td>
      <td>2</td>
      <td>5</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>580.0</td>
      <td>693.3</td>
      <td>2</td>
      <td>2</td>
      <td>4</td>
      <td>95704.53</td>
      <td>True</td>
    </tr>
    <tr>
      <th>76804</th>
      <td>210</td>
      <td>2024-12-01</td>
      <td>20</td>
      <td>16</td>
      <td>0</td>
      <td>22.0</td>
      <td>11</td>
      <td>2</td>
      <td>5</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>4</td>
      <td>0</td>
      <td>580.0</td>
      <td>693.3</td>
      <td>3</td>
      <td>1</td>
      <td>4</td>
      <td>103843.68</td>
      <td>True</td>
    </tr>
    <tr>
      <th>80256</th>
      <td>210</td>
      <td>2025-01-01</td>
      <td>20</td>
      <td>16</td>
      <td>0</td>
      <td>20.0</td>
      <td>11</td>
      <td>2</td>
      <td>5</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>665.0</td>
      <td>693.3</td>
      <td>0</td>
      <td>3</td>
      <td>3</td>
      <td>110776.44</td>
      <td>True</td>
    </tr>
    <tr>
      <th>83660</th>
      <td>210</td>
      <td>2025-02-01</td>
      <td>20</td>
      <td>16</td>
      <td>0</td>
      <td>33.0</td>
      <td>11</td>
      <td>2</td>
      <td>5</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>665.0</td>
      <td>693.3</td>
      <td>1</td>
      <td>5</td>
      <td>6</td>
      <td>113672.36</td>
      <td>True</td>
    </tr>
    <tr>
      <th>87074</th>
      <td>210</td>
      <td>2025-03-01</td>
      <td>20</td>
      <td>16</td>
      <td>0</td>
      <td>25.0</td>
      <td>11</td>
      <td>2</td>
      <td>5</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>665.0</td>
      <td>693.3</td>
      <td>2</td>
      <td>0</td>
      <td>2</td>
      <td>131995.21</td>
      <td>True</td>
    </tr>
    <tr>
      <th>90418</th>
      <td>210</td>
      <td>2025-04-01</td>
      <td>20</td>
      <td>20</td>
      <td>0</td>
      <td>27.0</td>
      <td>11</td>
      <td>2</td>
      <td>5</td>
      <td>0</td>
      <td>...</td>
      <td>2</td>
      <td>3</td>
      <td>0</td>
      <td>665.0</td>
      <td>693.3</td>
      <td>3</td>
      <td>2</td>
      <td>5</td>
      <td>131255.28</td>
      <td>True</td>
    </tr>
    <tr>
      <th>93706</th>
      <td>210</td>
      <td>2025-05-01</td>
      <td>20</td>
      <td>20</td>
      <td>0</td>
      <td>23.0</td>
      <td>11</td>
      <td>2</td>
      <td>5</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>665.0</td>
      <td>693.3</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>131965.43</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
<p>29 rows × 23 columns</p>
</div>




```python
df_fct[df_fct["monthly_published_ads"] != df_fct["monthly_unique_published_ads"]]
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
      <th>157</th>
      <td>210</td>
      <td>2023-01-01</td>
      <td>20</td>
      <td>4</td>
      <td>0</td>
      <td>4.0</td>
      <td>3</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>3</td>
      <td>2</td>
      <td>0</td>
      <td>757.6</td>
      <td>938.3</td>
      <td>1</td>
      <td>2</td>
      <td>3</td>
      <td>90925.00</td>
      <td>True</td>
    </tr>
    <tr>
      <th>792</th>
      <td>1117</td>
      <td>2023-01-01</td>
      <td>700</td>
      <td>700</td>
      <td>0</td>
      <td>1198.0</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>0</td>
      <td>...</td>
      <td>80</td>
      <td>16</td>
      <td>19</td>
      <td>1000.0</td>
      <td>3281.7</td>
      <td>10</td>
      <td>27</td>
      <td>37</td>
      <td>16611.55</td>
      <td>True</td>
    </tr>
    <tr>
      <th>865</th>
      <td>1221</td>
      <td>2023-01-01</td>
      <td>700</td>
      <td>700</td>
      <td>0</td>
      <td>1198.0</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>0</td>
      <td>...</td>
      <td>75</td>
      <td>8</td>
      <td>22</td>
      <td>1000.0</td>
      <td>3281.7</td>
      <td>8</td>
      <td>30</td>
      <td>38</td>
      <td>16618.18</td>
      <td>True</td>
    </tr>
    <tr>
      <th>1173</th>
      <td>1644</td>
      <td>2023-01-01</td>
      <td>700</td>
      <td>700</td>
      <td>0</td>
      <td>1198.0</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>0</td>
      <td>...</td>
      <td>77</td>
      <td>40</td>
      <td>18</td>
      <td>1000.0</td>
      <td>3281.7</td>
      <td>30</td>
      <td>40</td>
      <td>70</td>
      <td>16946.75</td>
      <td>True</td>
    </tr>
    <tr>
      <th>1222</th>
      <td>1716</td>
      <td>2023-01-01</td>
      <td>300</td>
      <td>300</td>
      <td>0</td>
      <td>386.0</td>
      <td>52</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>34</td>
      <td>11</td>
      <td>5</td>
      <td>427.2</td>
      <td>1935.0</td>
      <td>8</td>
      <td>14</td>
      <td>22</td>
      <td>24825.37</td>
      <td>True</td>
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
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
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
      <th>96659</th>
      <td>6873</td>
      <td>2025-05-01</td>
      <td>75</td>
      <td>33</td>
      <td>0</td>
      <td>60.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>4</td>
      <td>5</td>
      <td>1</td>
      <td>718.3</td>
      <td>718.3</td>
      <td>2</td>
      <td>2</td>
      <td>4</td>
      <td>16492.05</td>
      <td>True</td>
    </tr>
    <tr>
      <th>96661</th>
      <td>6875</td>
      <td>2025-05-01</td>
      <td>1000</td>
      <td>980</td>
      <td>0</td>
      <td>1530.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>8</td>
      <td>6</td>
      <td>0</td>
      <td>1666.7</td>
      <td>4558.3</td>
      <td>5</td>
      <td>9</td>
      <td>14</td>
      <td>23426.60</td>
      <td>True</td>
    </tr>
    <tr>
      <th>96704</th>
      <td>6923</td>
      <td>2025-05-01</td>
      <td>100</td>
      <td>100</td>
      <td>0</td>
      <td>179.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>791.7</td>
      <td>791.7</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>23137.47</td>
      <td>True</td>
    </tr>
    <tr>
      <th>96705</th>
      <td>6924</td>
      <td>2025-05-01</td>
      <td>1000</td>
      <td>990</td>
      <td>0</td>
      <td>1454.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>15</td>
      <td>17</td>
      <td>3</td>
      <td>1666.7</td>
      <td>4558.3</td>
      <td>14</td>
      <td>13</td>
      <td>27</td>
      <td>21765.83</td>
      <td>True</td>
    </tr>
    <tr>
      <th>96715</th>
      <td>6934</td>
      <td>2025-05-01</td>
      <td>1000</td>
      <td>984</td>
      <td>0</td>
      <td>1426.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>9</td>
      <td>25</td>
      <td>1</td>
      <td>1666.7</td>
      <td>4558.3</td>
      <td>14</td>
      <td>15</td>
      <td>29</td>
      <td>20868.21</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
<p>3379 rows × 23 columns</p>
</div>




```python
df_fct[df_fct["monthly_total_invoice"]<0]
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
  </tbody>
</table>
<p>0 rows × 23 columns</p>
</div>




```python
advertiser_id_2 = df_fct[df_fct["advertiser_zrive_id"] == 2]
advertiser_id_2["monthly_leads"].mean()
```




    np.float64(8.931034482758621)




```python
advertiser_id_2["monthly_oro_ads"].median()
```




    np.float64(10.0)




```python
churn_3m_1 = (
    df_fct.groupby("advertiser_zrive_id")["period_int"].nunique() <=3
)

```


```python
(
    df_fct.groupby("advertiser_zrive_id")["period_int"]
    .nunique()
    .loc[lambda s: s>=3]
)

```




    advertiser_zrive_id
    1        5
    2       29
    3        4
    4       29
    5       11
            ..
    7026    29
    7027    10
    7036    26
    7046    23
    7066     5
    Name: period_int, Length: 6397, dtype: int64




```python
def plot_advertiser_timeseries(
    df,
    advertiser_id,
    cols,
    date_col="period_int",
    id_col="advertiser_zrive_id",
    figsize=(10, 4),
    title=None
):
    df_adv = (
        df[df[id_col] == advertiser_id]
        .sort_values(date_col)
        .set_index(date_col)
    )

    if df_adv.empty:
        raise ValueError(f"No hay datos para {id_col} = {advertiser_id}")

    ax = df_adv[cols].plot(figsize=figsize)

    ax.set_xlabel("Periodo")
    ax.set_ylabel("Valor")

    if title is None:
        title = f"Evolución temporal - Advertiser {advertiser_id}"
    ax.set_title(title)

    plt.tight_layout()
    plt.show()
```


```python
cols = [
    "monthly_published_ads",
    "monthly_unique_published_ads",
    "monthly_distinct_ads",
]

plot_advertiser_timeseries(df_fct, 316, cols)
```


    
![png](EDA_jtc_files/EDA_jtc_235_0.png)
    



```python
cols = [
    "monthly_shows",
    "monthly_visits",
    "monthly_leads"
]
plot_advertiser_timeseries(df_fct, 316, cols)
```


    
![png](EDA_jtc_files/EDA_jtc_236_0.png)
    



```python
cols = [
    "monthly_total_invoice",
    "monthly_total_reference_price"
]
plot_advertiser_timeseries(df_fct, 2, cols)
```


    
![png](EDA_jtc_files/EDA_jtc_237_0.png)
    



```python
cols = [
    "monthly_contracted_ads",
    "monthly_published_ads",
]
plot_advertiser_timeseries(df_fct, 2, cols)
```


    
![png](EDA_jtc_files/EDA_jtc_238_0.png)
    



```python
mask = (
    df_fct.groupby("advertiser_zrive_id")["period_int"].nunique() == 2
)

advertisers = mask[mask].index
df_fct[df_fct["advertiser_zrive_id"].isin(advertisers)]
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
      <th>136</th>
      <td>178</td>
      <td>2023-01-01</td>
      <td>20</td>
      <td>11</td>
      <td>11</td>
      <td>17.0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>234.0</td>
      <td>515.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>16197.06</td>
      <td>True</td>
    </tr>
    <tr>
      <th>217</th>
      <td>294</td>
      <td>2023-01-01</td>
      <td>100</td>
      <td>74</td>
      <td>74</td>
      <td>105.0</td>
      <td>8</td>
      <td>8</td>
      <td>3</td>
      <td>0</td>
      <td>...</td>
      <td>10</td>
      <td>7</td>
      <td>4</td>
      <td>166.7</td>
      <td>2210.0</td>
      <td>7</td>
      <td>14</td>
      <td>21</td>
      <td>34313.58</td>
      <td>True</td>
    </tr>
    <tr>
      <th>219</th>
      <td>299</td>
      <td>2023-01-01</td>
      <td>14</td>
      <td>14</td>
      <td>14</td>
      <td>28.0</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>5</td>
      <td>6</td>
      <td>2</td>
      <td>175.0</td>
      <td>730.0</td>
      <td>6</td>
      <td>6</td>
      <td>12</td>
      <td>26982.14</td>
      <td>True</td>
    </tr>
    <tr>
      <th>236</th>
      <td>323</td>
      <td>2023-01-01</td>
      <td>75</td>
      <td>42</td>
      <td>42</td>
      <td>47.0</td>
      <td>4</td>
      <td>4</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>5</td>
      <td>3</td>
      <td>1</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>3</td>
      <td>3</td>
      <td>6</td>
      <td>34337.50</td>
      <td>True</td>
    </tr>
    <tr>
      <th>266</th>
      <td>363</td>
      <td>2023-01-01</td>
      <td>100</td>
      <td>89</td>
      <td>89</td>
      <td>109.0</td>
      <td>5</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>8</td>
      <td>7</td>
      <td>1</td>
      <td>1463.0</td>
      <td>2596.7</td>
      <td>4</td>
      <td>5</td>
      <td>9</td>
      <td>24165.09</td>
      <td>True</td>
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
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
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
      <th>96778</th>
      <td>6997</td>
      <td>2025-05-01</td>
      <td>3</td>
      <td>2</td>
      <td>2</td>
      <td>2.0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>43.3</td>
      <td>175.0</td>
      <td>0</td>
      <td>3</td>
      <td>3</td>
      <td>8500.00</td>
      <td>True</td>
    </tr>
    <tr>
      <th>96779</th>
      <td>6998</td>
      <td>2025-05-01</td>
      <td>35</td>
      <td>35</td>
      <td>35</td>
      <td>33.0</td>
      <td>0</td>
      <td>1</td>
      <td>6</td>
      <td>0</td>
      <td>...</td>
      <td>19</td>
      <td>31</td>
      <td>11</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>21</td>
      <td>38</td>
      <td>59</td>
      <td>34735.45</td>
      <td>True</td>
    </tr>
    <tr>
      <th>96780</th>
      <td>6999</td>
      <td>2025-05-01</td>
      <td>20</td>
      <td>9</td>
      <td>9</td>
      <td>23.0</td>
      <td>2</td>
      <td>2</td>
      <td>5</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>5</td>
      <td>4</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>5</td>
      <td>12</td>
      <td>17</td>
      <td>42137.83</td>
      <td>True</td>
    </tr>
    <tr>
      <th>96781</th>
      <td>7000</td>
      <td>2025-05-01</td>
      <td>10</td>
      <td>7</td>
      <td>7</td>
      <td>9.0</td>
      <td>1</td>
      <td>1</td>
      <td>4</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>95.0</td>
      <td>381.7</td>
      <td>2</td>
      <td>2</td>
      <td>4</td>
      <td>14688.89</td>
      <td>True</td>
    </tr>
    <tr>
      <th>96828</th>
      <td>7069</td>
      <td>2025-05-01</td>
      <td>10</td>
      <td>5</td>
      <td>5</td>
      <td>6.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>4</td>
      <td>4</td>
      <td>3</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>3</td>
      <td>11</td>
      <td>14</td>
      <td>7495.00</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
<p>490 rows × 23 columns</p>
</div>




```python
cols = [
    "monthly_total_invoice"
]
plot_advertiser_timeseries(df_fct, 2, cols)
```


    
![png](EDA_jtc_files/EDA_jtc_240_0.png)
    



```python
df_fct[df_fct["advertiser_zrive_id"] == 2]["monthly_total_invoice"].plot(kind='box')
```




    <Axes: >




    
![png](EDA_jtc_files/EDA_jtc_241_1.png)
    



```python

```


```python

```


```python

```


```python

```
