#!/usr/bin/env python3
"""
build_graph.py
==============
Construye el grafo de conocimiento de la ETSI Informática en Neo4j.

Esquema de nodos:
  (:ETSI)               — la escuela en sí
  (:Titulacion)         — cada título oficial (grado, máster, doctorado)
  (:PlanEstudio)        — versión concreta de un plan (Plan2010, Plan2023...)
  (:Estado)             — valor de vigencia: Vigente | EnExtincion | EnImplantacion | Extinto
  (:DistribucionCreditos) — desglose de ECTS de un plan
  (:Mencion)            — especialidad dentro de un grado
  (:Universidad)        — institución que co-imparte un título
  (:Normativa)          — reglamento o normativa, con ámbito de aplicación

Relaciones principales:
  (:ETSI)-[:IMPARTE]->(:Titulacion)
  (:Titulacion)-[:TIENE_PLAN]->(:PlanEstudio)
  (:PlanEstudio)-[:TIENE_ESTADO]->(:Estado)
  (:PlanEstudio)-[:TIENE_DISTRIBUCION]->(:DistribucionCreditos)
  (:Titulacion)-[:OFERTA_MENCION]->(:Mencion)
  (:Titulacion)-[:IMPARTE_CON]->(:Universidad)
  (:PlanEstudio)-[:REGULADO_POR]->(:Normativa)
  (:Normativa)-[:APLICA_A]->(:PlanEstudio)

Uso:
    python -m src.ingestion.build_graph              # Poblar el grafo
    python -m src.ingestion.build_graph --reset      # Borrar todo y repoblar
    python -m src.ingestion.build_graph --verify     # Solo listar nodos existentes
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from neo4j import GraphDatabase, Driver

load_dotenv()

# ── Conexión a Neo4j (variables de entorno) ────────────────────────────────────

NEO4J_URI      = os.getenv("NEO4J_URI",      "bolt://localhost:7687")
NEO4J_USER     = os.getenv("NEO4J_USER",     "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")

# ── Datos del grafo ────────────────────────────────────────────────────────────
#
# Toda esta información está verificada contra las fuentes oficiales:
#   - https://www.uma.es/etsi-informatica/
#   - Memorias de Verificación de cada título
#   - datos_maestros.txt del proyecto
#

# ---------------------------------------------------------------------------
# NODOS ESTADO (singleton por valor)
# ---------------------------------------------------------------------------
ESTADOS = [
    {"valor": "Vigente",          "descripcion": "Título activo. Admite nuevos alumnos."},
    {"valor": "EnImplantacion",   "descripcion": "Título nuevo en despliegue progresivo. Admite nuevos alumnos."},
    {"valor": "EnExtincion",      "descripcion": "Título en proceso de extinción. No admite nuevos alumnos de 1er curso. Los alumnos actuales pueden continuar."},
    {"valor": "Extinto",          "descripcion": "Título completamente extinto. No se imparte. Pertenece a planes anteriores al EEES (Plan Bolonia)."},
]

# ---------------------------------------------------------------------------
# NODOS UNIVERSIDAD (co-impartidores externos)
# ---------------------------------------------------------------------------
UNIVERSIDADES = [
    {"codigo": "UMA",  "nombre": "Universidad de Málaga",  "ciudad": "Málaga"},
    {"codigo": "US",   "nombre": "Universidad de Sevilla", "ciudad": "Sevilla"},
    {"codigo": "UNIPI","nombre": "Università di Pisa",     "ciudad": "Pisa"},
]

# ---------------------------------------------------------------------------
# TITULACIONES
# Cada entrada define la titulación y sus planes de estudio anidados.
# ---------------------------------------------------------------------------
TITULACIONES = [

    # ── GRADOS PLAN 2010 + 2023 ──────────────────────────────────────────────

    {
        "codigo":      "GIS",
        "nombre":      "Grado en Ingeniería del Software",
        "tipo":        "Grado",
        "rama":        "Ingeniería y Arquitectura",
        "url":         "https://www.uma.es/grado-en-ingenieria-del-software/",
        "profesion_regulada": False,
        "planes": [
            {
                "codigo":       "GIS-2010",
                "nombre":       "Plan 2010",
                "anio_inicio":  2010,
                "boe_fecha":    "2011-10-20",
                "plazas":       120,
                "ects_total":   240,
                "anios_duracion": 4,
                "estado":       "Vigente",
                "distribucion": {
                    "formacion_basica":  60,
                    "obligatorios":     126,
                    "optativos":         36,
                    "practicas_externas": 6,
                    "tfg":               12,
                    "total":            240,
                },
                "normativa": ["REG-TFG-UMA", "REG-PE-ETSI"],
            },
            {
                "codigo":       "GIS-2023",
                "nombre":       "Plan 2023",
                "anio_inicio":  2023,
                "boe_fecha":    None,
                "plazas":       120,
                "ects_total":   240,
                "anios_duracion": 4,
                "estado":       "Vigente",
                "distribucion": None,   # pendiente de publicar memoria completa
                "normativa": ["REG-TFG-UMA", "REG-PE-ETSI"],
            },
        ],
        "menciones":    [],
        "co_universidades": [],
    },

    {
        "codigo":      "GII",
        "nombre":      "Grado en Ingeniería Informática",
        "tipo":        "Grado",
        "rama":        "Ingeniería y Arquitectura",
        "url":         "https://www.uma.es/grado-en-ingenieria-informatica/",
        "profesion_regulada": False,
        "planes": [
            {
                "codigo":       "GII-2010",
                "nombre":       "Plan 2010",
                "anio_inicio":  2010,
                "boe_fecha":    "2011-10-20",
                "plazas":       120,
                "ects_total":   240,
                "anios_duracion": 4,
                "estado":       "Vigente",
                "distribucion": {
                    "formacion_basica":  60,
                    "obligatorios":     132,
                    "optativos":         30,
                    "practicas_externas": 6,
                    "tfg":               12,
                    "total":            240,
                },
                "normativa": ["REG-TFG-UMA", "REG-PE-ETSI"],
            },
            {
                "codigo":       "GII-2023",
                "nombre":       "Plan 2023",
                "anio_inicio":  2023,
                "boe_fecha":    None,
                "plazas":       120,
                "ects_total":   240,
                "anios_duracion": 4,
                "estado":       "Vigente",
                "distribucion": None,
                "normativa": ["REG-TFG-UMA", "REG-PE-ETSI"],
            },
        ],
        "menciones":    [],
        "co_universidades": [],
    },

    {
        "codigo":      "GIC",
        "nombre":      "Grado en Ingeniería de Computadores",
        "tipo":        "Grado",
        "rama":        "Ingeniería y Arquitectura",
        "url":         "https://www.uma.es/grado-en-ingenieria-de-computadores/",
        "profesion_regulada": False,
        "planes": [
            {
                "codigo":       "GIC-2010",
                "nombre":       "Plan 2010",
                "anio_inicio":  2010,
                "boe_fecha":    "2011-10-20",
                "plazas":       65,
                "ects_total":   240,
                "anios_duracion": 4,
                # Plan en extinción; no admite nuevos alumnos.
                "estado":       "EnExtincion",
                "distribucion": {
                    "formacion_basica":  60,
                    "obligatorios":     132,
                    "optativos":         30,
                    "practicas_externas": 6,
                    "tfg":               12,
                    "total":            240,
                },
                "normativa": ["REG-TFG-UMA", "REG-PE-ETSI"],
            },
        ],
        "menciones":    [],
        "co_universidades": [],
    },

    {
        "codigo":      "GISalud",
        "nombre":      "Grado en Ingeniería de la Salud",
        "tipo":        "Grado",
        "rama":        "Ingeniería y Arquitectura",
        "url":         "https://www.uma.es/grado-en-ingenieria-de-la-salud/",
        "profesion_regulada": False,
        "planes": [
            {
                "codigo":       "GISalud-2011",
                "nombre":       "Plan 2011",
                "anio_inicio":  2011,
                "boe_fecha":    "2013-03-21",
                "plazas":       65,
                "ects_total":   240,
                "anios_duracion": 4,
                "estado":       "Vigente",
                "distribucion": None,
                "normativa": ["REG-TFG-UMA", "REG-PE-ETSI"],
            },
            {
                "codigo":       "GISalud-2023",
                "nombre":       "Plan 2023",
                "anio_inicio":  2023,
                "boe_fecha":    None,
                "plazas":       65,
                "ects_total":   240,
                "anios_duracion": 4,
                "estado":       "Vigente",
                "distribucion": None,
                "normativa": ["REG-TFG-UMA", "REG-PE-ETSI"],
            },
        ],
        "menciones": [
            {"nombre": "Ingeniería Biomédica",  "activa_en_uma": True},
            {"nombre": "Bioinformática",         "activa_en_uma": True},
            {"nombre": "Informática Clínica",    "activa_en_uma": False},  # solo en Sevilla
        ],
        "co_universidades": ["US"],
    },

    {
        "codigo":      "GCIA",
        "nombre":      "Grado en Ciberseguridad e Inteligencia Artificial",
        "tipo":        "Grado",
        "rama":        "Ingeniería y Arquitectura",
        "url":         "https://www.uma.es/grado-en-ciberseguridad-e-inteligencia-artificial/",
        "profesion_regulada": False,
        "planes": [
            {
                "codigo":       "GCIA-2023",
                "nombre":       "Plan 2023",
                "anio_inicio":  2023,
                "boe_fecha":    "2023-12-30",
                "plazas":       65,
                "ects_total":   240,
                "anios_duracion": 4,
                "estado":       "EnImplantacion",
                "distribucion": None,
                "normativa": [],
            },
        ],
        "menciones":    [],
        "co_universidades": [],
    },

    # ── DOBLE GRADO ───────────────────────────────────────────────────────────

    {
        "codigo":      "DGIIM",
        "nombre":      "Doble Grado en Ingeniería Informática y Matemáticas",
        "tipo":        "DobleGrado",
        "rama":        "Ingeniería y Arquitectura / Ciencias",
        "url":         "https://www.uma.es/informatica-matematicas/",
        "profesion_regulada": False,
        "planes": [
            {
                "codigo":       "DGIIM-2019",
                "nombre":       "Plan 2019",
                "anio_inicio":  2019,
                "boe_fecha":    None,
                "plazas":       40,
                "ects_total":   360,   # doble grado: aprox. 360 ECTS en 5.5 años
                "anios_duracion": 5,   # más de 4 años por ser doble titulación
                "estado":       "Vigente",
                "distribucion": None,
                "normativa": [],
            },
            {
                "codigo":       "DGIIM-2023",
                "nombre":       "Plan 2023",
                "anio_inicio":  2023,
                "boe_fecha":    None,
                "plazas":       40,
                "ects_total":   360,
                "anios_duracion": 5,
                "estado":       "Vigente",
                "distribucion": None,
                "normativa": [],
            },
        ],
        "menciones":    [],
        "co_universidades": [],   # impartido internamente por ETSI + Fac. Ciencias UMA
    },

    # ── MÁSTERES ──────────────────────────────────────────────────────────────

    {
        "codigo":      "MUII",
        "nombre":      "Máster Universitario en Ingeniería Informática",
        "tipo":        "Master",
        "rama":        "Ingeniería y Arquitectura",
        "url":         "https://www.uma.es/master-en-ingenieria-informatica/cms/menu/informacion-general/",
        "profesion_regulada": False,
        "planes": [
            {
                "codigo":       "MUII-2018",
                "nombre":       "Plan 2018",
                "anio_inicio":  2018,
                "boe_fecha":    None,
                "plazas":       None,
                "ects_total":   60,
                "anios_duracion": 1,
                "estado":       "Vigente",
                "distribucion": None,
                "normativa": ["REG-TFM-ETSI"],
            },
        ],
        "menciones":    [],
        "co_universidades": ["UNIPI"],  # doble titulación con Univ. de Pisa
    },

    {
        "codigo":      "MUISIA",
        "nombre":      "Máster Universitario en Ingeniería del Software e Inteligencia Artificial",
        "tipo":        "Master",
        "rama":        "Ingeniería y Arquitectura",
        "url":         "https://www.uma.es/master-en-ingenieria-del-software-e-inteligencia-artificial/",
        "profesion_regulada": False,
        "planes": [
            {
                "codigo":       "MUISIA-2021",
                "nombre":       "Plan 2021",
                "anio_inicio":  2021,
                "boe_fecha":    None,
                "plazas":       None,
                "ects_total":   60,
                "anios_duracion": 1,
                "estado":       "Vigente",
                "distribucion": None,
                "normativa": ["REG-TFM-ETSI"],
            },
        ],
        "menciones":    [],
        "co_universidades": [],
    },

    {
        "codigo":      "MUCiber",
        "nombre":      "Máster Universitario en Ciberseguridad",
        "tipo":        "Master",
        "rama":        "Ingeniería y Arquitectura",
        "url":         "https://www.uma.es/master-en-ciberseguridad/info/152732/master-universitario-en-ciberseguridad/",
        "profesion_regulada": False,
        "planes": [
            {
                "codigo":       "MUCiber-2021",
                "nombre":       "Plan 2021",
                "anio_inicio":  2021,
                "boe_fecha":    None,
                "plazas":       None,
                "ects_total":   60,
                "anios_duracion": 1,
                "estado":       "Vigente",
                "distribucion": None,
                "normativa": ["REG-TFM-ETSI"],
            },
        ],
        "menciones":    [],
        "co_universidades": [],
    },

    # ── DOCTORADO ─────────────────────────────────────────────────────────────

    {
        "codigo":      "DocTI",
        "nombre":      "Programa de Doctorado en Tecnologías Informáticas",
        "tipo":        "Doctorado",
        "rama":        "Ingeniería y Arquitectura",
        "url":         "https://www.uma.es/doctorado-informatica/",
        "profesion_regulada": False,
        "planes": [
            {
                "codigo":       "DocTI-actual",
                "nombre":       "Plan Vigente",
                "anio_inicio":  2013,
                "boe_fecha":    None,
                "plazas":       None,
                "ects_total":   None,
                "anios_duracion": 3,
                "estado":       "Vigente",
                "distribucion": None,
                "normativa": [],
            },
        ],
        "menciones":    [],
        "co_universidades": [],
    },

    # Planes antiguos extintos

    {
        "codigo":      "ITIG",
        "nombre":      "Ingeniería Técnica en Informática de Gestión",
        "tipo":        "TituloAntiguo",
        "rama":        "Ingeniería",
        "url":         None,
        "profesion_regulada": False,
        "planes": [
            {
                "codigo":       "ITIG-pre2010",
                "nombre":       "Plan anterior al EEES",
                "anio_inicio":  None,
                "boe_fecha":    None,
                "plazas":       None,
                "ects_total":   None,
                "anios_duracion": 3,
                "estado":       "Extinto",
                "distribucion": None,
                "normativa":    ["REG-PLANES-ANTIGUOS"],
            },
        ],
        "menciones":    [],
        "co_universidades": [],
    },

    {
        "codigo":      "ITIS",
        "nombre":      "Ingeniería Técnica en Informática de Sistemas",
        "tipo":        "TituloAntiguo",
        "rama":        "Ingeniería",
        "url":         None,
        "profesion_regulada": False,
        "planes": [
            {
                "codigo":       "ITIS-pre2010",
                "nombre":       "Plan anterior al EEES",
                "anio_inicio":  None,
                "boe_fecha":    None,
                "plazas":       None,
                "ects_total":   None,   # sistema antiguo: ~180 créditos del plan viejo
                "anios_duracion": 3,
                "estado":       "Extinto",
                "distribucion": None,
                "normativa":    ["REG-PLANES-ANTIGUOS"],
            },
        ],
        "menciones":    [],
        "co_universidades": [],
    },

    {
        "codigo":      "LicInf",
        "nombre":      "Ingeniería en Informática (Licenciatura)",
        "tipo":        "TituloAntiguo",
        "rama":        "Ingeniería",
        "url":         None,
        "profesion_regulada": False,
        "planes": [
            {
                "codigo":       "LicInf-pre2010",
                "nombre":       "Plan anterior al EEES",
                "anio_inicio":  None,
                "boe_fecha":    None,
                "plazas":       None,
                "ects_total":   None,
                "anios_duracion": 5,
                "estado":       "Extinto",
                "distribucion": None,
                "normativa":    ["REG-PLANES-ANTIGUOS"],
            },
        ],
        "menciones":    [],
        "co_universidades": [],
    },
]

# ---------------------------------------------------------------------------
# NORMATIVAS (singleton por codigo)
# ---------------------------------------------------------------------------
NORMATIVAS = [
    {
        "codigo":       "REG-TFG-UMA",
        "nombre":       "Reglamento de Trabajo de Fin de Grado (UMA)",
        "tipo":         "Reglamento",
        "ambito":       "Grado",
        "aplica_planes_vigentes": True,
        "aplica_planes_extintos": False,
        "url":          "https://www.uma.es/media/files/Reglamento_TFG_Mayo_2025.pdf",
        "nota":         (
            "Art. 18.1: matrícula requiere 70% créditos superados (168/240). "
            "Defensa requiere 82.5% (198/240). "
            "Máximo 25% similitud en memoria (herramienta antiplagio)."
        ),
    },
    {
        "codigo":       "REG-PE-ETSI",
        "nombre":       "Reglamento de Prácticas Externas (ETSI Informática)",
        "tipo":         "Reglamento",
        "ambito":       "Grado",
        "aplica_planes_vigentes": True,
        "aplica_planes_extintos": False,
        "url":          None,
        "nota":         "Regula las prácticas externas de 6 ECTS en los grados actuales (Plan 2010 y Plan 2023).",
    },
    {
        "codigo":       "REG-TFM-ETSI",
        "nombre":       "Reglamento de Trabajo de Fin de Máster (ETSI Informática)",
        "tipo":         "Reglamento",
        "ambito":       "Master",
        "aplica_planes_vigentes": True,
        "aplica_planes_extintos": False,
        "url":          None,
        "nota":         "Regula el TFM para los másteres universitarios de la ETSI Informática.",
    },
    {
        "codigo":       "REG-PLANES-ANTIGUOS",
        "nombre":       "Normativa de Planes Anteriores al EEES",
        "tipo":         "NormativaGeneral",
        "ambito":       "PlanAntiguo",
        "aplica_planes_vigentes": False,
        "aplica_planes_extintos": True,
        "url":          None,
        "nota":         (
            "Incluye: Reglamento del Proyecto de Fin de Carrera (PFC), "
            "Normativa del Prácticum, cálculo de nota media expediente plan antiguo, "
            "y reconocimiento de créditos de libre configuración por equivalencia. "
            "IMPORTANTE: El Prácticum y el PFC son figuras exclusivas de planes extintos. "
            "En los grados actuales su equivalente son las Prácticas Externas y el TFG."
        ),
    },
    {
        "codigo":       "NORM-RECONOC-LABORAL",
        "nombre":       "Normativa de Reconocimiento de Créditos por Experiencia Laboral",
        "tipo":         "Normativa",
        "ambito":       "Grado",
        "aplica_planes_vigentes": True,
        "aplica_planes_extintos": False,
        "url":          None,
        "nota":         "Máximo 12 créditos reconocibles, a razón de 1 crédito por cada 60 horas de experiencia laboral acreditada.",
    },
    {
        "codigo":       "NORM-CAM-GRUPO",
        "nombre":       "Normativa de Asignación y Cambio de Grupo",
        "tipo":         "Normativa",
        "ambito":       "Grado",
        "aplica_planes_vigentes": True,
        "aplica_planes_extintos": False,
        "url":          None,
        "nota":         "Regula el procedimiento y criterios para solicitar cambio de grupo (2024).",
    },
]

# ── Funciones de construcción del grafo ───────────────────────────────────────

def _create_indexes(driver: Driver) -> None:
    """Crea índices para búsquedas eficientes."""
    with driver.session() as session:
        indexes = [
            "CREATE INDEX titulacion_codigo IF NOT EXISTS FOR (t:Titulacion)  ON (t.codigo)",
            "CREATE INDEX plan_codigo       IF NOT EXISTS FOR (p:PlanEstudio) ON (p.codigo)",
            "CREATE INDEX estado_valor      IF NOT EXISTS FOR (e:Estado)      ON (e.valor)",
            "CREATE INDEX normativa_codigo  IF NOT EXISTS FOR (n:Normativa)   ON (n.codigo)",
            "CREATE INDEX uni_codigo        IF NOT EXISTS FOR (u:Universidad) ON (u.codigo)",
        ]
        for cypher in indexes:
            session.run(cypher)
    print("[Graph] Índices creados / verificados.")


def _reset_graph(driver: Driver) -> None:
    """Elimina todos los nodos y relaciones (útil para repoblar desde cero)."""
    with driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")
    print("[Graph] Grafo vaciado.")


def _create_etsi(driver: Driver) -> None:
    with driver.session() as session:
        session.run("""
            MERGE (e:ETSI {codigo: 'ETSI-UMA'})
            SET e.nombre      = 'Escuela Técnica Superior de Ingeniería Informática',
                e.universidad = 'Universidad de Málaga',
                e.campus      = 'Campus Universitario de Teatinos',
                e.ciudad      = 'Málaga',
                e.telefono    = '952 13 27 00',
                e.fax         = '952 13 26 73',
                e.email_dir   = 'director@informatica.uma.es',
                e.url         = 'https://www.uma.es/etsi-informatica/'
        """)
    print("[Graph] Nodo ETSI creado.")


def _create_estados(driver: Driver) -> None:
    with driver.session() as session:
        for e in ESTADOS:
            session.run("""
                MERGE (s:Estado {valor: $valor})
                SET s.descripcion = $descripcion
            """, **e)
    print(f"[Graph] {len(ESTADOS)} nodos Estado creados.")


def _create_universidades(driver: Driver) -> None:
    with driver.session() as session:
        for u in UNIVERSIDADES:
            session.run("""
                MERGE (u:Universidad {codigo: $codigo})
                SET u.nombre = $nombre,
                    u.ciudad = $ciudad
            """, **u)
    print(f"[Graph] {len(UNIVERSIDADES)} nodos Universidad creados.")


def _create_normativas(driver: Driver) -> None:
    with driver.session() as session:
        for n in NORMATIVAS:
            session.run("""
                MERGE (n:Normativa {codigo: $codigo})
                SET n.nombre                  = $nombre,
                    n.tipo                    = $tipo,
                    n.ambito                  = $ambito,
                    n.aplica_planes_vigentes  = $aplica_planes_vigentes,
                    n.aplica_planes_extintos  = $aplica_planes_extintos,
                    n.url                     = $url,
                    n.nota                    = $nota
            """, **n)
    print(f"[Graph] {len(NORMATIVAS)} nodos Normativa creados.")


def _create_titulaciones(driver: Driver) -> None:
    """Crea titulaciones, planes, distribuciones, menciones y todas las relaciones."""

    with driver.session() as session:
        for tit in TITULACIONES:
            # 1. Nodo Titulacion
            session.run("""
                MERGE (t:Titulacion {codigo: $codigo})
                SET t.nombre              = $nombre,
                    t.tipo                = $tipo,
                    t.rama                = $rama,
                    t.url                 = $url,
                    t.profesion_regulada  = $profesion_regulada
            """,
                codigo=tit["codigo"],
                nombre=tit["nombre"],
                tipo=tit["tipo"],
                rama=tit["rama"],
                url=tit.get("url"),
                profesion_regulada=tit.get("profesion_regulada", False),
            )

            # 2. Relación ETSI -[:IMPARTE]-> Titulacion
            session.run("""
                MATCH (e:ETSI {codigo: 'ETSI-UMA'})
                MATCH (t:Titulacion {codigo: $codigo})
                MERGE (e)-[:IMPARTE]->(t)
            """, codigo=tit["codigo"])

            # 3. Planes de estudio
            for plan in tit["planes"]:
                session.run("""
                    MERGE (p:PlanEstudio {codigo: $codigo})
                    SET p.nombre          = $nombre,
                        p.anio_inicio     = $anio_inicio,
                        p.boe_fecha       = $boe_fecha,
                        p.plazas          = $plazas,
                        p.ects_total      = $ects_total,
                        p.anios_duracion  = $anios_duracion
                """,
                    codigo=plan["codigo"],
                    nombre=plan["nombre"],
                    anio_inicio=plan.get("anio_inicio"),
                    boe_fecha=plan.get("boe_fecha"),
                    plazas=plan.get("plazas"),
                    ects_total=plan.get("ects_total"),
                    anios_duracion=plan.get("anios_duracion"),
                )

                # Titulacion -[:TIENE_PLAN]-> PlanEstudio
                session.run("""
                    MATCH (t:Titulacion  {codigo: $t_codigo})
                    MATCH (p:PlanEstudio {codigo: $p_codigo})
                    MERGE (t)-[:TIENE_PLAN]->(p)
                """, t_codigo=tit["codigo"], p_codigo=plan["codigo"])

                # PlanEstudio -[:TIENE_ESTADO]-> Estado
                session.run("""
                    MATCH (p:PlanEstudio {codigo: $p_codigo})
                    MATCH (s:Estado      {valor:  $estado})
                    MERGE (p)-[:TIENE_ESTADO]->(s)
                """, p_codigo=plan["codigo"], estado=plan["estado"])

                # DistribucionCreditos (solo si hay datos)
                if plan.get("distribucion"):
                    d = plan["distribucion"]
                    dist_codigo = f"DIST-{plan['codigo']}"
                    session.run("""
                        MERGE (d:DistribucionCreditos {codigo: $codigo})
                        SET d.formacion_basica    = $formacion_basica,
                            d.obligatorios        = $obligatorios,
                            d.optativos           = $optativos,
                            d.practicas_externas  = $practicas_externas,
                            d.tfg                 = $tfg,
                            d.total               = $total
                    """,
                        codigo=dist_codigo,
                        formacion_basica=d["formacion_basica"],
                        obligatorios=d["obligatorios"],
                        optativos=d["optativos"],
                        practicas_externas=d["practicas_externas"],
                        tfg=d["tfg"],
                        total=d["total"],
                    )
                    session.run("""
                        MATCH (p:PlanEstudio        {codigo: $p_codigo})
                        MATCH (d:DistribucionCreditos{codigo: $d_codigo})
                        MERGE (p)-[:TIENE_DISTRIBUCION]->(d)
                    """, p_codigo=plan["codigo"], d_codigo=dist_codigo)

                # Relaciones con Normativa
                for norm_codigo in plan.get("normativa", []):
                    session.run("""
                        MATCH (p:PlanEstudio {codigo: $p_codigo})
                        MATCH (n:Normativa   {codigo: $n_codigo})
                        MERGE (p)-[:REGULADO_POR]->(n)
                        MERGE (n)-[:APLICA_A]->(p)
                    """, p_codigo=plan["codigo"], n_codigo=norm_codigo)

            # 4. Menciones
            for mencion in tit.get("menciones", []):
                men_codigo = f"MEN-{tit['codigo']}-{mencion['nombre'].replace(' ', '_')}"
                session.run("""
                    MERGE (m:Mencion {codigo: $codigo})
                    SET m.nombre         = $nombre,
                        m.activa_en_uma  = $activa_en_uma
                """,
                    codigo=men_codigo,
                    nombre=mencion["nombre"],
                    activa_en_uma=mencion.get("activa_en_uma", True),
                )
                session.run("""
                    MATCH (t:Titulacion {codigo: $t_codigo})
                    MATCH (m:Mencion    {codigo: $m_codigo})
                    MERGE (t)-[:OFERTA_MENCION]->(m)
                """, t_codigo=tit["codigo"], m_codigo=men_codigo)

            # 5. Co-universidades
            for uni_codigo in tit.get("co_universidades", []):
                session.run("""
                    MATCH (t:Titulacion  {codigo: $t_codigo})
                    MATCH (u:Universidad {codigo: $u_codigo})
                    MERGE (t)-[:IMPARTE_CON]->(u)
                """, t_codigo=tit["codigo"], u_codigo=uni_codigo)

    n_titulaciones = len(TITULACIONES)
    n_planes       = sum(len(t["planes"]) for t in TITULACIONES)
    print(f"[Graph] {n_titulaciones} titulaciones y {n_planes} planes de estudio creados.")


def _verify(driver: Driver) -> None:
    """Imprime un resumen del contenido actual del grafo."""
    with driver.session() as session:
        labels = ["ETSI", "Titulacion", "PlanEstudio", "Estado",
                  "DistribucionCreditos", "Mencion", "Universidad", "Normativa"]
        print("\n[Graph] Contenido actual del grafo:")
        print(f"  {'Etiqueta':<25} {'Nodos':>8}")
        print(f"  {'-'*35}")
        for label in labels:
            result = session.run(f"MATCH (n:{label}) RETURN count(n) AS c")
            count  = result.single()["c"]
            print(f"  {label:<25} {count:>8}")

        # Relaciones
        rels = session.run("MATCH ()-[r]->() RETURN type(r) AS t, count(r) AS c ORDER BY c DESC")
        print(f"\n  {'Relación':<30} {'Aristas':>8}")
        print(f"  {'-'*40}")
        for r in rels:
            print(f"  {r['t']:<30} {r['c']:>8}")

        # Titulaciones por estado
        print("\n  Estado de cada titulación:")
        result = session.run("""
            MATCH (t:Titulacion)-[:TIENE_PLAN]->(p:PlanEstudio)-[:TIENE_ESTADO]->(s:Estado)
            RETURN t.nombre AS titulo, p.nombre AS plan, s.valor AS estado
            ORDER BY t.tipo, t.nombre, p.anio_inicio
        """)
        for r in result:
            print(f"  [{r['estado']:>16}] {r['titulo']} ({r['plan']})")


# ── Punto de entrada ───────────────────────────────────────────────────────────

def build(reset: bool = False) -> None:
    if not NEO4J_PASSWORD:
        print("[ERROR] Variable de entorno NEO4J_PASSWORD no configurada.")
        print("        Añade NEO4J_PASSWORD=<tu_contraseña> al archivo .env")
        sys.exit(1)

    print(f"\n[Graph] Conectando a Neo4j en {NEO4J_URI} como '{NEO4J_USER}'...")
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    try:
        driver.verify_connectivity()
        print("[Graph] Conexión establecida correctamente.\n")
    except Exception as exc:
        print(f"[ERROR] No se pudo conectar a Neo4j: {exc}")
        print("        ¿Está Neo4j en ejecución? Comprueba NEO4J_URI, NEO4J_USER y NEO4J_PASSWORD en .env")
        driver.close()
        sys.exit(1)

    if reset:
        print("[Graph] Modo --reset: eliminando todos los nodos existentes...")
        _reset_graph(driver)

    _create_indexes(driver)
    _create_etsi(driver)
    _create_estados(driver)
    _create_universidades(driver)
    _create_normativas(driver)
    _create_titulaciones(driver)
    _verify(driver)

    driver.close()
    print("\n[Graph] Grafo construido con éxito.")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Construye el grafo de conocimiento de la ETSI Informática en Neo4j."
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Eliminar todos los nodos existentes antes de repoblar (idempotente).",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Solo mostrar el contenido actual del grafo, sin modificar nada.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    if not NEO4J_PASSWORD:
        print("[ERROR] Variable de entorno NEO4J_PASSWORD no configurada.")
        sys.exit(1)

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    try:
        driver.verify_connectivity()
    except Exception as exc:
        print(f"[ERROR] No se pudo conectar a Neo4j: {exc}")
        sys.exit(1)

    if args.verify:
        _verify(driver)
        driver.close()
        sys.exit(0)

    driver.close()
    build(reset=args.reset)
