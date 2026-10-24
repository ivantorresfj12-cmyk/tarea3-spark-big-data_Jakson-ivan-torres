# -*- coding: utf-8 -*-
"""
Tarea 3 - Procesamiento en batch con Apache Spark (PySpark)
Curso Big Data (202016911) - UNAD
Autor: Jakson Iván Torres Fuentes

Problema: caracterizar la contratación pública registrada en SECOP II en cuanto a
la concentración del gasto, las modalidades de selección y el nivel de competencia,
a partir de una muestra de 50.000 registros del dataset
"SECOP II - Procesos de Contratación" (datos.gov.co, id p6dx-8zbt).

Preguntas que responde el análisis:
  P1. ¿Qué departamentos y entidades concentran el mayor valor adjudicado?
  P2. ¿Qué modalidades de contratación predominan, en número de procesos y en valor?
  P3. ¿Qué proporción de procesos resulta adjudicada y cuál termina desierta o cancelada?
  P4. ¿Cuántas ofertas recibe en promedio un proceso según su modalidad?
  P5. ¿Qué diferencia porcentual existe entre el precio base y el valor adjudicado?
  P6. ¿Cómo evoluciona mensualmente la publicación de procesos?
  P7. ¿Qué proveedores reciben la mayor proporción del valor adjudicado?

Ejecución:
    spark-submit tarea3_batch.py            # análisis completo
    spark-submit tarea3_batch.py --pausa    # deja la Spark UI (puerto 4040) abierta al final
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")  # la VM no tiene entorno gráfico: los gráficos se guardan como PNG
import matplotlib.pyplot as plt
import pandas as pd

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------
RUTA_HDFS = "hdfs://localhost:9000/Tarea3/secop.csv"
DIR_SALIDA = os.path.expanduser("~/tarea3_resultados")
os.makedirs(DIR_SALIDA, exist_ok=True)

spark = (SparkSession.builder
         .appName("Tarea3_SECOP_Batch")
         # En un solo nodo, 200 particiones de shuffle generan tareas casi vacías
         .config("spark.sql.shuffle.partitions", "8")
         .config("spark.sql.legacy.timeParserPolicy", "CORRECTED")
         .getOrCreate())
spark.sparkContext.setLogLevel("WARN")


def titulo(texto):
    print("\n" + "=" * 78)
    print(texto)
    print("=" * 78)


def guardar(sdf, nombre):
    """Convierte un resultado agregado (pequeño) a pandas y lo guarda en CSV."""
    pdf = sdf.toPandas()
    # Formato regional de Excel en Colombia: separador ";", decimal "," y UTF-8 con BOM
    pdf.to_csv(os.path.join(DIR_SALIDA, f"{nombre}.csv"), index=False,
               sep=";", decimal=",", encoding="utf-8-sig")
    return pdf


# ---------------------------------------------------------------------------
# 1. Carga del dataset desde HDFS
# ---------------------------------------------------------------------------
titulo("1. CARGA DEL DATASET DESDE HDFS")
df_raw = (spark.read
          .option("header", True)
          .option("multiLine", True)   # algunas descripciones traen saltos de línea
          .option("escape", '"')
          .csv(RUTA_HDFS))

total_raw = df_raw.count()
print(f"Archivo: {RUTA_HDFS}")
print(f"Registros leídos: {total_raw:,} | Columnas: {len(df_raw.columns)}")


def normalizar(nombre):
    return nombre.lower().strip().replace(" ", "_")


def ubicar(*candidatas):
    """Devuelve el nombre real de la primera columna que coincida (exacta y luego parcial)."""
    reales = {normalizar(c): c for c in df_raw.columns}
    for cand in candidatas:
        if normalizar(cand) in reales:
            return reales[normalizar(cand)]
    for cand in candidatas:
        for real in df_raw.columns:
            if normalizar(cand) in normalizar(real):
                return real
    return None


MAPA = {
    "id_proceso":   ubicar("id_del_proceso", "id_proceso", "referencia_del_proceso"),
    "entidad":      ubicar("entidad", "nombre_entidad"),
    "departamento": ubicar("departamento_entidad", "departamento"),
    "modalidad":    ubicar("modalidad_de_contratacion", "modalidad"),
    "estado":       ubicar("estado_del_procedimiento", "estado_resumen", "estado"),
    "fecha":        ubicar("fecha_de_publicacion_del", "fecha_de_publicacion", "fecha_publicacion"),
    "precio_base":  ubicar("precio_base"),
    "valor_adj":    ubicar("valor_total_adjudicacion", "valor_adjudicado"),
    "adjudicado":   ubicar("adjudicado"),
    "proveedor":    ubicar("nombre_del_proveedor", "proveedor_adjudicado"),
    "ofertas":      ubicar("respuestas_al_procedimiento", "conteo_de_respuestas_a_ofertas"),
}

print("\nColumnas seleccionadas para el análisis (de las disponibles):")
for clave, real in MAPA.items():
    print(f"  {clave:<13} -> {real}")

faltantes = [c for c in ("entidad", "modalidad", "precio_base", "fecha") if MAPA[c] is None]
if faltantes:
    print(f"\nERROR: no se encontraron las columnas esenciales {faltantes}. "
          "Ajuste los nombres en el diccionario MAPA.")
    spark.stop()
    sys.exit(1)


def col_o_nula(clave):
    real = MAPA[clave]
    return F.col(f"`{real}`") if real else F.lit(None).cast("string")


df = df_raw.select([col_o_nula(k).alias(k) for k in MAPA])

# ---------------------------------------------------------------------------
# 2. Limpieza
# ---------------------------------------------------------------------------
titulo("2. LIMPIEZA DE DATOS")

NO_INFORMATIVOS = ["", "NO DEFINIDO", "NO APLICA", "N/A", "NULL", "NONE"]
for c in ("entidad", "departamento", "modalidad", "estado", "adjudicado", "proveedor"):
    df = df.withColumn(c, F.upper(F.trim(F.col(c))))
    df = df.withColumn(c, F.when(F.col(c).isin(NO_INFORMATIVOS), None).otherwise(F.col(c)))
df = df.withColumn("proveedor", F.when(F.col("proveedor").isin("NO ADJUDICADO"), None)
                                 .otherwise(F.col("proveedor")))


def a_numero(c):
    """Elimina símbolos de moneda y separadores y convierte a número."""
    return F.regexp_replace(F.col(c), r"[^0-9.\-]", "").cast("double")


df = (df.withColumn("precio_base", a_numero("precio_base"))
        .withColumn("valor_adj", a_numero("valor_adj"))
        .withColumn("ofertas", a_numero("ofertas")))

f10 = F.substring(F.col("fecha"), 1, 10)
df = df.withColumn("fecha", F.coalesce(F.to_date(f10, "yyyy-MM-dd"),
                                       F.to_date(f10, "MM/dd/yyyy"),
                                       F.to_date(f10, "dd/MM/yyyy")))

print("Valores nulos por columna (después de normalizar tipos):")
df.select([F.sum(F.col(c).isNull().cast("int")).alias(c) for c in df.columns]).show(truncate=False)

antes = df.count()
df = df.dropDuplicates(["id_proceso"]) if MAPA["id_proceso"] else df.dropDuplicates()
duplicados = antes - df.count()

df = df.filter(F.col("entidad").isNotNull() & F.col("modalidad").isNotNull())

df = (df.withColumn("precio_base", F.when(F.col("precio_base") > 0, F.col("precio_base")))
        .withColumn("valor_adj", F.when(F.col("valor_adj") > 0, F.col("valor_adj")))
        .withColumn("ofertas", F.when(F.col("ofertas") >= 0, F.col("ofertas"))))

total_limpio = df.count()
print(f"Registros originales:            {total_raw:,}")
print(f"Duplicados eliminados:           {duplicados:,}")
print(f"Descartados sin entidad/modal.:  {antes - duplicados - total_limpio:,}")
print(f"Registros limpios para análisis: {total_limpio:,}")

# ---------------------------------------------------------------------------
# 3. Transformación
# ---------------------------------------------------------------------------
titulo("3. TRANSFORMACIÓN (COLUMNAS DERIVADAS)")
df = (df.withColumn("anio", F.year("fecha"))
        .withColumn("mes", F.date_format("fecha", "yyyy-MM"))
        .withColumn("es_adjudicado",
                    F.when(F.col("adjudicado").isin("SI", "SÍ", "TRUE", "1")
                           | (F.col("valor_adj") > 0), 1).otherwise(0))
        # Diferencia % entre lo adjudicado y el precio base; se excluyen registros
        # con valor adjudicado mayor a 3 veces el precio base (errores de captura)
        .withColumn("dif_pct",
                    F.when((F.col("es_adjudicado") == 1)
                           & F.col("precio_base").isNotNull()
                           & F.col("valor_adj").isNotNull()
                           & (F.col("valor_adj") <= F.col("precio_base") * 3),
                           F.round((F.col("valor_adj") - F.col("precio_base"))
                                   / F.col("precio_base") * 100, 2))))
df.cache()
df.printSchema()
df.select("entidad", "modalidad", "precio_base", "valor_adj", "dif_pct", "mes",
          "es_adjudicado").show(5, truncate=30)

# ---------------------------------------------------------------------------
# 4. Análisis exploratorio (EDA)
# ---------------------------------------------------------------------------
titulo("4.1 ESTADÍSTICAS DESCRIPTIVAS")
df.select(F.col("precio_base").alias("precio_base_cop"),
          F.col("valor_adj").alias("valor_adjudicado_cop"),
          F.col("ofertas").alias("ofertas_recibidas"),
          F.col("dif_pct").alias("diferencia_pct")) \
  .summary("count", "mean", "stddev", "min", "25%", "50%", "75%", "max") \
  .show(truncate=False)

total_adj = df.agg(F.sum("valor_adj")).first()[0] or 0.0

# --- P2: modalidades -------------------------------------------------------
titulo("4.2 [P2] MODALIDADES: NÚMERO DE PROCESOS Y VALOR")
por_modalidad = (df.groupBy("modalidad")
                 .agg(F.count("*").alias("procesos"),
                      F.round(F.sum("precio_base") / 1e9, 2).alias("precio_base_mm"),
                      F.round(F.sum("valor_adj") / 1e9, 2).alias("valor_adjudicado_mm"))
                 .withColumn("pct_procesos",
                             F.round(F.col("procesos") / F.lit(total_limpio) * 100, 1))
                 .orderBy(F.desc("procesos")))
por_modalidad.show(truncate=False)
pdf_modalidad = guardar(por_modalidad, "p2_modalidades")

# --- P1: departamentos y entidades ----------------------------------------
titulo("4.3 [P1] TOP 10 DEPARTAMENTOS POR VALOR ADJUDICADO")
por_depto = (df.filter(F.col("departamento").isNotNull())
             .groupBy("departamento")
             .agg(F.count("*").alias("procesos"), F.sum("valor_adj").alias("valor"))
             .orderBy(F.desc("valor")).limit(10)
             .withColumn("valor_adjudicado_mm", F.round(F.col("valor") / 1e9, 2))
             .withColumn("participacion_pct",
                         F.round(F.col("valor") / F.lit(total_adj) * 100, 2))
             .drop("valor"))
por_depto.show(truncate=False)
pdf_depto = guardar(por_depto, "p1_top10_departamentos")

titulo("4.4 [P1] TOP 10 ENTIDADES POR VALOR ADJUDICADO")
por_entidad = (df.groupBy("entidad")
               .agg(F.count("*").alias("procesos"), F.sum("valor_adj").alias("valor"))
               .orderBy(F.desc("valor")).limit(10)
               .withColumn("valor_adjudicado_mm", F.round(F.col("valor") / 1e9, 2))
               .withColumn("participacion_pct",
                           F.round(F.col("valor") / F.lit(total_adj) * 100, 2))
               .drop("valor"))
por_entidad.show(truncate=50)
pdf_entidad = guardar(por_entidad, "p1_top10_entidades")
conc_entidades = float(pdf_entidad["participacion_pct"].fillna(0).sum())
print(f"Las 10 entidades principales concentran el {conc_entidades:.2f} % del valor adjudicado.")

# --- P3: estados (RDD) -----------------------------------------------------
titulo("4.5 [P3] ESTADO DEL PROCEDIMIENTO (RDD: map + reduceByKey)")
estados = (df.select("estado").rdd
           .map(lambda fila: (fila["estado"] or "SIN ESTADO", 1))
           .reduceByKey(lambda a, b: a + b)
           .sortBy(lambda par: -par[1])
           .collect())
print(f"  {'ESTADO':<45} {'PROCESOS':>9} {'%':>7}")
for estado, n in estados[:12]:
    print(f"  {estado:<45} {n:>9,} {n / total_limpio * 100:>6.1f}%")
with open(os.path.join(DIR_SALIDA, "p3_estados.csv"), "w", encoding="utf-8-sig") as f:
    f.write("estado;procesos;pct\n")
    for estado, n in estados:
        pct = f"{n / total_limpio * 100:.2f}".replace(".", ",")
        f.write(f"\"{estado}\";{n};{pct}\n")


def pct_estado(*palabras):
    n = sum(c for e, c in estados if any(p in e for p in palabras))
    return n / total_limpio * 100


pct_adjudicados = df.agg(F.avg("es_adjudicado")).first()[0] * 100
pct_desiertos = pct_estado("DESIERT")
pct_cancelados = pct_estado("CANCELAD", "REVOCAD", "TERMINAD ANORMAL")
print(f"\nAdjudicados: {pct_adjudicados:.1f} % | Desiertos: {pct_desiertos:.1f} % | "
      f"Cancelados/revocados: {pct_cancelados:.1f} %")

# --- P4: competencia -------------------------------------------------------
titulo("4.6 [P4] OFERTAS PROMEDIO POR MODALIDAD (NIVEL DE COMPETENCIA)")
ofertas_mod = (df.filter(F.col("ofertas").isNotNull())
               .groupBy("modalidad")
               .agg(F.count("*").alias("procesos"),
                    F.round(F.avg("ofertas"), 2).alias("ofertas_promedio"),
                    F.round(F.avg(F.when(F.col("ofertas") <= 1, 1).otherwise(0)) * 100, 1)
                     .alias("pct_con_1_o_0_ofertas"))
               .orderBy(F.desc("ofertas_promedio")))
ofertas_mod.show(truncate=False)
pdf_ofertas = guardar(ofertas_mod, "p4_ofertas_por_modalidad")

# --- P5: diferencia precio base vs adjudicado (Spark SQL) ------------------
titulo("4.7 [P5] DIFERENCIA % ENTRE PRECIO BASE Y VALOR ADJUDICADO (SPARK SQL)")
df.createOrReplaceTempView("secop")
dif_mod = spark.sql("""
    SELECT modalidad,
           COUNT(dif_pct)                           AS procesos_adjudicados,
           ROUND(AVG(dif_pct), 2)                   AS dif_promedio_pct,
           ROUND(PERCENTILE_APPROX(dif_pct, 0.5), 2) AS dif_mediana_pct
    FROM secop
    WHERE dif_pct IS NOT NULL
    GROUP BY modalidad
    ORDER BY procesos_adjudicados DESC
""")
dif_mod.show(truncate=False)
pdf_dif = guardar(dif_mod, "p5_diferencia_precio_adjudicado")
dif_global = df.agg(F.avg("dif_pct")).first()[0]

# --- P6: evolución mensual -------------------------------------------------
titulo("4.8 [P6] EVOLUCIÓN MENSUAL DE LA PUBLICACIÓN DE PROCESOS")
por_mes = (df.filter(F.col("mes").isNotNull())
           .groupBy("mes")
           .agg(F.count("*").alias("procesos"),
                F.round(F.sum("precio_base") / 1e9, 2).alias("precio_base_mm"))
           .orderBy("mes"))
por_mes.show(36, truncate=False)
pdf_mes = guardar(por_mes, "p6_evolucion_mensual")

# --- P7: proveedores -------------------------------------------------------
titulo("4.9 [P7] TOP 10 PROVEEDORES POR VALOR ADJUDICADO")
por_proveedor = (df.filter(F.col("proveedor").isNotNull() & F.col("valor_adj").isNotNull())
                 .groupBy("proveedor")
                 .agg(F.count("*").alias("procesos_ganados"), F.sum("valor_adj").alias("valor"))
                 .orderBy(F.desc("valor")).limit(10)
                 .withColumn("valor_adjudicado_mm", F.round(F.col("valor") / 1e9, 2))
                 .withColumn("participacion_pct",
                             F.round(F.col("valor") / F.lit(total_adj) * 100, 2))
                 .drop("valor"))
por_proveedor.show(truncate=50)
pdf_prov = guardar(por_proveedor, "p7_top10_proveedores")
conc_prov = float(pdf_prov["participacion_pct"].fillna(0).sum()) if not pdf_prov.empty else 0.0

# ---------------------------------------------------------------------------
# 5. Visualización
# ---------------------------------------------------------------------------
titulo("5. GENERACIÓN DE GRÁFICOS")


def barras_h(pdf, etiqueta, valor, titulo_g, xlabel, archivo, color="#1f5f99"):
    if pdf.empty:
        return
    datos = pdf.iloc[::-1]  # el mayor queda arriba
    plt.figure(figsize=(11, 6))
    plt.barh([str(e)[:45] for e in datos[etiqueta]], datos[valor].fillna(0), color=color)
    plt.title(titulo_g, fontsize=13)
    plt.xlabel(xlabel)
    plt.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(DIR_SALIDA, archivo), dpi=120)
    plt.close()
    print(f"  Gráfico generado: {archivo}")


barras_h(pdf_depto, "departamento", "valor_adjudicado_mm",
         "P1. Top 10 departamentos por valor adjudicado", "Miles de millones de pesos",
         "01_departamentos_valor_adjudicado.png")
barras_h(pdf_entidad, "entidad", "valor_adjudicado_mm",
         "P1. Top 10 entidades por valor adjudicado", "Miles de millones de pesos",
         "02_entidades_valor_adjudicado.png")
barras_h(pdf_modalidad, "modalidad", "procesos",
         "P2. Procesos por modalidad de contratación", "Número de procesos",
         "03_modalidades.png")

pdf_estados = pd.DataFrame(estados[:8], columns=["estado", "procesos"])
barras_h(pdf_estados, "estado", "procesos",
         "P3. Procesos por estado del procedimiento", "Número de procesos",
         "04_estados.png", color="#7a4ea3")
barras_h(pdf_ofertas, "modalidad", "ofertas_promedio",
         "P4. Ofertas promedio por modalidad", "Ofertas por proceso",
         "05_ofertas_por_modalidad.png", color="#2e8b57")
barras_h(pdf_dif, "modalidad", "dif_mediana_pct",
         "P5. Mediana de la diferencia % (adjudicado vs. precio base)",
         "Diferencia porcentual", "06_diferencia_precio_adjudicado.png", color="#c0504d")

if not pdf_mes.empty:
    plt.figure(figsize=(12, 5))
    plt.plot(pdf_mes["mes"], pdf_mes["procesos"], marker="o", color="#1f5f99")
    paso = max(1, len(pdf_mes) // 12)
    plt.xticks(range(0, len(pdf_mes), paso), pdf_mes["mes"][::paso], rotation=45)
    plt.title("P6. Evolución mensual de procesos publicados", fontsize=13)
    plt.ylabel("Número de procesos")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(DIR_SALIDA, "07_evolucion_mensual.png"), dpi=120)
    plt.close()
    print("  Gráfico generado: 07_evolucion_mensual.png")

barras_h(pdf_prov, "proveedor", "valor_adjudicado_mm",
         "P7. Top 10 proveedores por valor adjudicado", "Miles de millones de pesos",
         "08_proveedores_valor_adjudicado.png", color="#e08a1e")

# ---------------------------------------------------------------------------
# 6. Respuesta a las preguntas
# ---------------------------------------------------------------------------
titulo("6. RESPUESTA A LAS PREGUNTAS DEL PROBLEMA")
if not pdf_depto.empty:
    d = pdf_depto.iloc[0]
    print(f"P1. {d['departamento']} es el departamento con mayor valor adjudicado "
          f"({d['valor_adjudicado_mm']:,.2f} mil millones, {d['participacion_pct']:.1f} %). "
          f"El top 10 de entidades concentra el {conc_entidades:.1f} % del valor adjudicado.")
if not pdf_modalidad.empty:
    m = pdf_modalidad.iloc[0]
    mv = pdf_modalidad.sort_values("valor_adjudicado_mm", ascending=False).iloc[0]
    print(f"P2. Por número predomina {m['modalidad']} ({m['pct_procesos']:.1f} % de los procesos); "
          f"por valor adjudicado, {mv['modalidad']}.")
print(f"P3. Adjudicados: {pct_adjudicados:.1f} % | desiertos: {pct_desiertos:.1f} % | "
      f"cancelados o revocados: {pct_cancelados:.1f} %.")
if not pdf_ofertas.empty:
    o_max, o_min = pdf_ofertas.iloc[0], pdf_ofertas.iloc[-1]
    print(f"P4. Mayor competencia: {o_max['modalidad']} ({o_max['ofertas_promedio']} ofertas); "
          f"menor: {o_min['modalidad']} ({o_min['ofertas_promedio']} ofertas).")
if dif_global is not None:
    print(f"P5. En promedio, el valor adjudicado difiere {dif_global:.2f} % del precio base "
          "(valor negativo = ahorro frente al presupuesto).")
if not pdf_mes.empty:
    pico = pdf_mes.sort_values("procesos", ascending=False).iloc[0]
    print(f"P6. El mes con más procesos publicados es {pico['mes']} ({int(pico['procesos']):,}).")
if not pdf_prov.empty:
    p = pdf_prov.iloc[0]
    print(f"P7. {p['proveedor']} recibe la mayor proporción ({p['participacion_pct']:.1f} %); "
          f"el top 10 de proveedores suma el {conc_prov:.1f} % del valor adjudicado.")
print(f"\nResultados (CSV y PNG) guardados en: {DIR_SALIDA}")

if "--pausa" in sys.argv:
    input("\nSpark UI disponible en http://<IP-de-la-VM>:4040 - presione Enter para finalizar...")

df.unpersist()
spark.stop()
