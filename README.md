# Peace territories 

In this repository you could find the analysis related with Mexico City Peace territories strategy performed by the SGIRPC.

The analysis is based on data coming from the Participatory Assessment prepared by the Undersecretariat for Territories of Peace and submitted to the Ministry of Comprehensive Risk Management and Civil Protection (SGIRPC) on May 13, 2026 and enriched with data produced by the SGIRPC regarding hazards. 

The original database (DB) was transformed into two DB, one corresponding to Individual Surveys records and second referring to homes. Several new vectors were created through reclassification and spatial analysis using information from the Mexico City Risk Atlas shape files corresponding to the Single Registry of Emergency Situations, updated in December 2025.

Responsible researchers: Removed for anonymized review.

## Preamble of the strategy

Between March 17 and April 19, 2026, the Undersecretariat for Territories of Peace and Equality conducted an assessment of the peace situation in Mexico City’s, twenty three Territories of Peace and Equality, covering several neighborhoods, corresponding to 125,963 households comprising 157,483 individuals across 16 county districts.

Given the confidential nature of certain data, protected by the Personal Data Protection Act (reference to the law), a public version was prepared in accordance with Mexican law. The resulting dataset includes 153,676 respondents and 120,685 households following the data cleaning process referred in this document.

##  Processing of Personal Information

Based on Mexican law (Article 3, Section X of the Law on the Protection of Personal Data in Possession of Obligated Parties of Mexico City) and, in some fields, sensitive personal data (Section XI: health, sexual preference, ethnic origin). Therefore, the public version removes such information, respecting the principles of purpose, loyalty, minimization, and proportionality (Article 9).

| Column type | Examples | Public version |

|---|---|---|

| Direct identifiers | name (nombre), respondent (encuestado), interviewer (encuestador), respondent\_signature (firma\_encuestado) | Suppress |

| Address | street (calle), external\_number (numero\_ext), internal\_number (numero\_int) | Suppress |

| Geolocation | coordinates (coordenadas) | Suppress; keep polygon |

| Free text | observations (observaciones), detail (\*\_detalle), other (\*\_otro) | Suppress |

| Sensitive thematic data | lgbtttq, indigenous\_comunity (comunidad\_originaria), disability (discapacidad), diagnosed\_condition (padecimiento\_diag), pregnant\_women (embarazadas), substances (sustancias) | Keep only as polygon-level aggregates with k ≥ 5 |

| Non-sensitive demographics | age (edad), sex (sexo), education\_level (nivel\_estudios), ocupation\_category (ocupacion\_categoria), marital\_status (estado\_civil), member (integrante) | Keep; evaluate 5-year age bands if k < 5 |

| Household variables | dwelling (vivienda), health (salud), service\_equipment (equipamiento\_de\_servicios), security (seguridad), annexes (anexos) | Aggregate by polygon when detail would expose unique cells |

| Internal household identifier | general\_id (id\_general) | Replace with a different, non-derivable pseudonym |



