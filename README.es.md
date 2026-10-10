# ecobici-red-cdmx

*[English version](README.md)*

**Diseño de la red de Ecobici con demanda censurada y un registro público de predicciones para la expansión de 2026.**

En agosto de 2026 SEMOVI anunció que Ecobici pasará de 687 a 1,111 estaciones (+424), de 9,308 a 15,000
bicis, y llegará a Iztapalapa, Iztacalco y Tlalpan. Este proyecto pregunta, sólo con datos públicos:
¿dónde poner esas estaciones y cuántos anclajes y bicis darle a cada una? Y lo más importante: **registra
las predicciones antes de que abran las estaciones**, para compararlas después con los viajes reales.

## Registro de predicciones · 4 de octubre de 2026

| Alcaldía | Sitios | Anclajes | Bicis | Viajes/día · modelo de sitio | Viajes/día · ancla Encuesta OD 2017 (bajo – medio – alto) |
| --- | --- | --- | --- | --- | --- |
| Iztapalapa | 248 | 6,728 | 3,345 | 18,754 | 17,031 – 33,962 – 36,326 |
| Iztacalco | 60 | 1,686 | 835 | 4,692 | 4,693 – 9,358 – 10,009 |
| Tlalpan | 116 | 3,038 | 1,508 | 8,470 | 2,159 – 4,305 – 4,605 |

- Sitio por sitio, con intervalo de 80%: [`registro/registro_predicciones_2026-10-04.csv`](registro/registro_predicciones_2026-10-04.csv)
- Supuestos y protocolo de validación: [`registro/registro_predicciones_2026-10-04.json`](registro/registro_predicciones_2026-10-04.json)
- Huella SHA-256 del CSV: `46140e98aea343dce4ad729e7b517843dca48f7a35e7dadbf6aaa5fde217f45b` ([`registro/SHA256SUMS`](registro/SHA256SUMS))

Verificar que el archivo no cambió: `python scripts/verificar_registro.py`

**Hipótesis principal:** el modelo de sitio y la encuesta discrepan (Tlalpan: el modelo duplica a la
encuesta; Iztapalapa: la encuesta casi duplica al modelo). Los datos reales dirán qué ancla sirve más para
planear en zonas sin historia.

**Cómo se validará:** cuando las estaciones aparezcan en el feed y en los datos abiertos mensuales,
(1) por alcaldía, viajes/día reales contra ambas anclas; (2) por estación real, la predicción del sitio
registrado más cercano (<300 m): cobertura del intervalo 80% y captura de demanda del top-20%;
(3) comparación contra una regla simple (cercanía a Metro/Metrobús + ciclovía). La demanda real de las
estaciones nuevas se medirá corrigiendo por censura (tiempo vacías o llenas) con la captura del estado
de este repositorio.

![Sitios registrados](registro/mapa_registro.png)

## Captura del estado de las estaciones

[`.github/workflows/captura_gbfs.yml`](.github/workflows/captura_gbfs.yml) guarda cada ~5 minutos el estado
de todas las estaciones (bicis disponibles, dañadas, anclajes libres) en la rama `gbfs-archive`, y registra
en `info/cambios.csv` cada alta, baja o cambio de capacidad: así se detecta cuándo abre cada estación nueva.
GitHub no garantiza el horario exacto; se esperan retrasos y huecos ocasionales.

## Resultados hasta ahora

| Etapa | Hallazgo |
| --- | --- |
| Reconstrucción (ene–sep 2026) | 13.9 M viajes; ~3,900 bicis rebalanceadas en camión por día, reconstruidas por cadena de cada bici. |
| Demanda censurada | Retiros observados 50,832/día; demanda estimada 53,793/día (+5.8%); retiros perdidos 370–4,141/día. |
| Demanda censurada, método de la tesis (`run_bayes.py`) | Aprendizaje bayesiano en malla con demanda binomial negativa y verosimilitud de cola en ventanas con agotamiento. Fuera de muestra (entrena ene–jun, prueba jul–sep): mejor puntaje logarítmico en retiros (−0.8129 vs. −0.8148 ingenuo y −0.8221 descartando ventanas censuradas); intervalos 80% cubren 81.6%. Retiros latentes 55,409/día vs. 50,887 implícitos en los viajes (+8.9%). |
| Ranking de sitios (validación espacial) | Un modelo de gradient boosting captura 28.8% de la demanda con el top-20%, igual que una regla simple (29.0%); oráculo 38.7%. |
| Ola de expansión 2023→2024 (120 estaciones) | Modelo 28.9% vs. regla simple 27.6% (diferencia no significativa). Los modelos subestimaron 12–21% el nivel de demanda de estaciones nuevas → factor de corrección 1.198. |

Lectura honesta: rankear esquina por esquina no mejora sobre una regla simple. El aporte está en el
dimensionamiento (demanda censurada, corrección de nivel, intervalos) y en el reparto entre zonas.

## Método

1. **Cadenas por bici** (`src/chains.py`): si un viaje termina en A y el siguiente de esa bici empieza en
   B ≠ A, hubo una reubicación. Con eso se reconstruyen cotas del inventario por estación cada 10 min.
2. **Demanda censurada** (`run_censored.py`): sólo se cuentan retiros cuando seguro había bici y devoluciones
   cuando seguro había anclaje; tasa por estación × hora con Gamma-Poisson jerárquico. Adapta el tratamiento
   de censura endógena de la tesis *Diseño y evaluación de una política de reposición multiproducto sensible a
   liquidez para nanostores* (Gamón Guerrero y Macías Herrera, UNAM, 2026).
3. **Contexto** (`src/context.py`): Censo 2020 por AGEB, DENUE, GTFS, infraestructura ciclista.
4. **Prueba retrospectiva** (`run_wave.py`) y **asignación** (`run_allocation.py`): selección voraz de 424
   sitios con separación ≥300 m; anclajes y bicis proporcionales a la demanda corregida.

## Reproducir

Python 3.11+ con pandas, numpy, scipy, scikit-learn y matplotlib. Los datos no se incluyen; colócalos en
`data/` (o define `ECOBICI_DATA`):

- `data/*.csv` y `data/viajes_2026/*.csv`: viajes mensuales de [Ecobici datos abiertos](https://ecobici.cdmx.gob.mx/datos-abiertos/)
- `data/station_information.json`: [GBFS de Ecobici](https://gbfs.mex.lyftbikes.com/gbfs/gbfs.json)
- `data/contexto/`: GTFS, estaciones del sistema anterior, cicloestaciones, infraestructura ciclista y Censo 2020 por AGEB de [Datos Abiertos CDMX](https://datos.cdmx.gob.mx/); [DENUE](https://www.inegi.org.mx/app/descarga/) y [EOD 2017](https://www.inegi.org.mx/programas/eod/2017/) de INEGI

```bash
python tests/test_chains.py
python run_phase0.py --raw data/viajes_2026 --stations data/station_information.json --out out/phase0_jan_sep
python run_context.py && python run_wave.py && python run_censored.py && python run_allocation.py
```

## Créditos y trabajo relacionado

- [Jero110/movilidad-cdmx](https://github.com/Jero110/movilidad-cdmx): rebalanceo operativo de Ecobici con un asignador MILP; este proyecto aborda el diseño de la red, no la operación diaria.
- [MaxHalford/bike-sharing-history](https://github.com/MaxHalford/bike-sharing-history): archivo histórico de estado de estaciones.
- Datos de Ecobici, Gobierno de la Ciudad de México, INEGI. Código bajo licencia MIT.
