# Databricks notebook source
from pyspark.sql.functions import col, explode, when
from pyspark.sql import functions as F
import spacy
from pyspark.sql.functions import col, pandas_udf
from pyspark.sql.types import StringType
import pandas as pd
from pyspark.sql.functions import year
from pyspark.sql.functions import col, monotonically_increasing_id
from pyspark.sql.window import Window
from pyspark.sql.functions import row_number, col




# COMMAND ----------

!python -m spacy download en_core_web_sm

# COMMAND ----------

# MAGIC %md
# MAGIC #Lectura de nuestros datos almacenados en el storage account

# COMMAND ----------

abfs_path = f"abfss://articles@inetumau.dfs.core.windows.net/archivo.json"
df = spark.read.format("json").option("inferSchema", "true").load(abfs_path)


# COMMAND ----------

# MAGIC %md
# MAGIC #Extraccion del campo author de la data leida

# COMMAND ----------

df = df.withColumn("author", explode(col("authors")))
df = df.withColumn("author_name", col("author")["name"])


# COMMAND ----------

# MAGIC %md
# MAGIC #Instancia de  clase spacy para el proceso NLP

# COMMAND ----------


nlp = spacy.load("en_core_web_sm")

# COMMAND ----------

categories = {
    "science": ["NASA", "astronomy", "telescope", "research", "space exploration", "planets", "galaxy", "physicists", "theory", "study", "astrophysicist", "space laboratory"],
    "missions": ["launch", "rocket", "mission", "crew", "space station", "rovers", "moon", "mars", "jupiter", "satellites", "orbiter", "lunar exploration", "space travel"],
    "technology": ["technology", "innovation", "reusable rockets", "artificial intelligence", "automation", "sensors", "navigation", "space systems", "advanced materials"],
    "politics": ["government", "regulation", "space policy", "funding", "legislation", "congress", "alliances", "international agreements", "space competition", "space industry", "corporations"],
    "commercial": ["company", "SpaceX", "Blue Origin", "private", "commercial launches", "private industry", "private rocket", "investors", "business development", "space market"],
    "security": ["defense", "military satellites", "national security", "cybersecurity", "military technology", "space warfare", "orbital surveillance", "space systems", "weapon systems"],
    "astronomy": ["star", "galaxy", "telescope", "observation", "exoplanets", "stellar clusters", "black holes", "nebula", "radio telescope", "spectroscopy", "search for life"],
    "climate and environment": ["climate change", "space weather", "weather satellites", "earth observation", "thermal waters", "global temperature", "solar cycle", "space climate", "solar energy"],
    "mars exploration": ["Mars", "rovers", "landing", "explorer", "martian surface", "water on Mars", "Martian atmosphere", "mission to Mars", "Curiosity", "Perseverance"],
    "astronautics": ["astronaut", "International Space Station", "spacelab", "crewed mission", "life support system", "space module", "microgravity experiments"],
    "space future": ["colonization", "terraforming", "space life", "lunar station", "nuclear rockets", "interplanetary exploration", "space transportation", "advanced propulsion systems"]
}

# COMMAND ----------

# MAGIC %md
# MAGIC #Definicion de nuestras funciones que van a permitir extraer palabras claves, identificacion entidades y clasificacion de articulos

# COMMAND ----------

def extract_entities(text, entity_type):
    if not text:
        return ""
    doc = nlp(text)
    return ", ".join(set(ent.text for ent in doc.ents if ent.label_ == entity_type))

# COMMAND ----------

def extract_keywords(text):
    if not text:
        return ""
    doc = nlp(text)
    return ", ".join(set(token.text for token in doc if token.pos_ in ['NOUN', 'ADJ']))

# COMMAND ----------

def classif_text(text):
    if not text:
        return ""
    doc = nlp(text)
    category_scores = {category: 0 for category in categories}
    
    for token in doc:
        token_text = token.text.strip().lower()
        for category, keywords in categories.items():
            for keyword in keywords:
                if keyword.lower() in token_text:
                    category_scores[category] += 1
    category = max(category_scores, key=category_scores.get)
    
    return category


# COMMAND ----------

@pandas_udf(StringType())
def extract_org_udf(texts: pd.Series) -> pd.Series:
    return texts.apply(lambda x: extract_entities(x, "ORG"))

@pandas_udf(StringType())
def extract_person_udf(texts: pd.Series) -> pd.Series:
    return texts.apply(lambda x: extract_entities(x, "PERSON"))

@pandas_udf(StringType())
def extract_place_udf(texts: pd.Series) -> pd.Series:
    return texts.apply(lambda x: extract_entities(x, "LOC"))

@pandas_udf(StringType())
def extract_keywords_udf(texts: pd.Series) -> pd.Series:
    return texts.apply(lambda x: extract_keywords(x))

@pandas_udf(StringType())
def classify(texts: pd.Series) -> pd.Series:
    return texts.apply(lambda x: classif_text(x))

# COMMAND ----------

df = df.withColumn("entity_company", extract_org_udf(col("summary")))
df = df.withColumn("entity_people", extract_person_udf(col("summary")))
df = df.withColumn("entity_place", extract_place_udf(col("summary")))
df = df.withColumn("key_words", extract_keywords_udf(col("summary")))
df = df.withColumn("category", classify(col("summary")))

# COMMAND ----------

# MAGIC %md
# MAGIC #Filtrado y limpieza de nuestro Dataframe final 

# COMMAND ----------

df_final = df.select("id",
                     "summary",
                     "news_site",
                     "published_at",
                     "updated_at",
                     "title",
                     "type",
                     "author_name",                     
                     when(col("entity_company").isNull() | (col("entity_company") == ""), "None").otherwise(col("entity_company")).alias("entity_company"),                     
                     when(col("entity_people").isNull() | (col("entity_people") == ""), "None").otherwise(col("entity_people")).alias("entity_people"),                     
                     when(col("entity_place").isNull() | (col("entity_place") == ""), "None").otherwise(col("entity_place")).alias("entity_place"),                     
                     when(col("key_words").isNull() | (col("key_words") == ""), "None").otherwise(col("key_words")).alias("key_words"),
                     "category")

# COMMAND ----------

display(df_final.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC # Tendencias de Temas por tiempo

# COMMAND ----------

df_time = df_final.withColumn("published_at", F.to_timestamp("published_at"))

# COMMAND ----------

df_tend = df_time.groupBy(F.month("published_at").alias("month"), "category").count()
df_tend = df_tend.orderBy("month", F.desc("count"))

display(df_tend)

# COMMAND ----------

# MAGIC %md
# MAGIC #Analisis de fuentes mas activas

# COMMAND ----------

#Analisis de fuentes mas activas
df_sources = df_final.groupBy("news_site").count()
df_sources=df_sources.orderBy(F.desc("count"))
display(df_sources)

# COMMAND ----------

# MAGIC %md
# MAGIC #Particion y almacenamiento de datos historicos en formato parquet en el data lake

# COMMAND ----------

#Particion de datos historicos en formato parquet
abfs_path = "abfss://articles@inetumau.dfs.core.windows.net/historical_data/"
df_time = df_time.withColumn("year", year("published_at"))
df_time.write.mode("overwrite").partitionBy("year").parquet(abfs_path)

# COMMAND ----------

# MAGIC %md
# MAGIC #Cacheo de datos

# COMMAND ----------

#Cacheo de datos
df_final.cache()

# COMMAND ----------

# MAGIC %md
# MAGIC #Creacion de las dimensiones y tabla de hechos en Dtabricks SQL

# COMMAND ----------

"""%sql
create table autor.autors.autor(
  id_autor int PRIMARY KEY,
  author_name string  
)
PARTITIONED BY (id_autor);

create table autor.autors.site(
  id_site int PRIMARY KEY,
  news_site string  
)
PARTITIONED BY (id_site);

create table autor.autors.category(
  id_category int PRIMARY KEY,
  category string  
)
PARTITIONED BY (id_category);"""

# COMMAND ----------

"""%sql
create table autor.autors.hechos(
  id_autor int foreign key references autor.autors.autor(id_autor),
  id_site int foreign key references autor.autors.site(id_site),
  id_category int foreign key references autor.autors.category(id_category),
  cantidad_noticias_autor int,
  cantidad_noticias_categoria int,
  cantidad_noticias_site int)
  PARTITIONED BY (id_autor,id_site,id_category);"""

# COMMAND ----------

# MAGIC %md
# MAGIC #Poblado de la dimension autor

# COMMAND ----------

df_authors = df_final.select("author_name").distinct()
window_spec = Window.orderBy("author_name")
df_authors = df_authors.withColumn("id_autor", row_number().over(window_spec))





# COMMAND ----------

df_authors.write.mode("overwrite").saveAsTable("autor.autors.autor")

# COMMAND ----------

# MAGIC %md
# MAGIC #Poblado de la dimension category

# COMMAND ----------

df_categories = df_final.select("category").distinct()
window_spec = Window.orderBy("category")
df_categories = df_categories.withColumn("id_category", row_number().over(window_spec))


# COMMAND ----------

display(df_categories)

# COMMAND ----------

df_categories.write.mode("overwrite").saveAsTable("autor.autors.category")

# COMMAND ----------

# MAGIC %md
# MAGIC #Poblado de la dimension site

# COMMAND ----------

df_site = df_final.select("news_site").distinct()
window_spec = Window.orderBy("news_site")
df_site = df_site.withColumn("id_site", row_number().over(window_spec))

# COMMAND ----------

df_site.write.mode("overwrite").saveAsTable("autor.autors.site")

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from autor.autors.autor;
# MAGIC

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from autor.autors.category;
# MAGIC

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from autor.autors.site

# COMMAND ----------

# MAGIC %md
# MAGIC #Poblado de la dimension Hechos.

# COMMAND ----------

df_final.createOrReplaceTempView("df_final")

# COMMAND ----------

"""%sql
INSERT INTO autor.autors.hechos(id_autor,id_site,id_category,cantidad_noticias_autor,cantidad_noticias_categoria,cantidad_noticias_site)
SELECT 
    a.id_autor,
    s.id_site,
    c.id_category,
    COUNT(df.author_name) AS cantidad_noticias_autor,
    COUNT(*) OVER(PARTITION BY c.id_category) AS cantidad_noticias_categoria,
    COUNT(*) OVER(PARTITION BY s.id_site) AS cantidad_noticias_site
FROM df_final df
JOIN autor.autors.autor a ON (df.author_name = a.author_name)
JOIN autor.autors.site s ON (df.news_site = s.news_site)
JOIN autor.autors.category c ON (df.category = c.category)
GROUP BY a.id_autor, s.id_site, c.id_category;"""

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from autor.autors.hechos;

# COMMAND ----------

# MAGIC %md
# MAGIC #Analisis SQL

# COMMAND ----------

# MAGIC %md
# MAGIC ##Tendencia de temas por mes

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT 
# MAGIC     DATE_TRUNC('month', df.published_at) AS month,
# MAGIC     c.category,
# MAGIC     COUNT(*) AS tendencia_count
# MAGIC FROM df_final df
# MAGIC JOIN autor.autors.category c ON (df.category = c.category)
# MAGIC GROUP BY month, c.category
# MAGIC ORDER BY month, tendencia_count DESC;
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ##Fuentes mas influyentes

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT 
# MAGIC     s.news_site,
# MAGIC     COUNT(*) AS count_sources
# MAGIC FROM df_final df
# MAGIC JOIN autor.autors.site s ON (df.news_site = s.news_site)
# MAGIC GROUP BY s.news_site
# MAGIC ORDER BY count_sources DESC;
# MAGIC

# COMMAND ----------

