# Mapping Digital Mental Health Sentiment Across U.S. Cities

This repository contains the code and data pipeline for investigating mental health sentiment across 11 major U.S. cities. By combining **Natural Language Processing (NLP)** with **Socioeconomic Data**, this project explores how city-level emotional distress relates to local economic conditions and the availability of mental health services.

The final comprehensive project report is available under `document` folder.

**Course:** Data Science I  
**Students:** Zhiyang Cheng, Jiayi Li, Cheng Gao  

---

## 📌 Project Overview
Traditional mental health monitoring often lacks geographic granularity and suffers from data lag. This project leverages over 1.5 million Reddit posts/comments (2009–2024) as a proxy for real-time emotional distress.

### Key Findings
* **Pandemic Impact:** A universal spike in negative sentiment was observed across nearly all cities starting in 2020.
* **Geographic Persistence:** Cities like **Seattle** and **Portland** consistently exhibit higher negative sentiment rates compared to cities like **Atlanta** or **Houston**, suggesting that mental health is shaped by enduring local environments.
* **Thematic Drivers:** Using **Structural Topic Modeling (STM)**, we identified that negative sentiment is driven by three primary clusters: 
    1. **High-Risk Mental Health** (Suicide-related)
    2. **Socio-Economic Pressure** (Cost of living, housing)
    3. **Environmental Factors** (Seasonal Affective Disorder).

---

## 📊 Methodology
* **Sentiment Classification:** A **RoBERTa-base** model fine-tuned on 8,000 GPT-labeled observations (accuracy >95%).
* **Topic Modeling:** **Structural Topic Modeling (STM)** used to identify 15 distinct thematic topics.
* **Regression Analysis:** OLS regression modeling the relationship between **Negative Affect Rate** and **Mental Health Provider Rates** (from CountyHealthRankings), controlling for ACS economic variables.

---

## 💾 Data and Model Access
Due to the size of the datasets, part of `data/` and `model/`folder is not included in this repository. 

**[Download the Data and Model Folder Here](https://drive.google.com/drive/folders/1VhU4UsSmluL5evieSLUu4Je0r1odKZ_q?usp=drive_link)**

**Installation:** Download the folder and place it in the **root path** of this project directory.
The data folder contains:
* `/cleaned`: Data processed and ready for analysis.

---

## ⚙️ Replication Instructions
To replicate this research, please run the scripts in the `codes/` folder according to their numerical prefix:

1.  **Main Scripts (e.g., `01_...`, `02_...`):** Run these in numerical order to produce the final results.
2.  **Assistance Scripts (e.g., `01a_...`, `01b_...`):** These scripts handle supplementary tasks or data cleaning. They are helpful for understanding the workflow but are not the primary drivers of the final output..

## Other things to notice
1.  ** the sample data used to train the model to identify unrelated observations from all posts is not sampled from the all_word corpus(observation that contain direct words and non-direct words). For 8000 observations, 4000 observations are sampled from observations that contain depression and suicide related words, while another 4000 observations are sampled from direct-word corpus. The final sample dataset has been deduplicated.
