library(tidyverse)
library(stm)
library(lubridate)
library(quanteda)

dta <- read_csv("data/cleaned/direct_corpus_predicted.csv")

dta <- dta %>%
  mutate(date_numeric = as.numeric(as.Date(date)))%>%
  filter(is_negative == 1)

corp <- corpus(dta, text_field = "text")

docvars(corp, "text") <- dta$text

my_stopwords <- c(
  stopwords("en"),
  stopwords("es"),
  "it’s", "i’m", "don’t", "can’t", "won’t",
  "just", "like",
  "go", "went", "going", "gone",
  "make", "made",
  "get", "got"
)

toks <- tokens(corp, 
               remove_punct = TRUE, 
               remove_symbols = FALSE, 
               remove_numbers = TRUE, 
               remove_url = TRUE) %>% 
  tokens_tolower() %>%
  tokens_remove(my_stopwords)

dfm_counts <- dfm(toks)

dfm_trimmed <- dfm_trim(dfm_counts, min_docfreq = 4)

out <- convert(dfm_trimmed, to = "stm")

topic_model <- stm(
  documents = out$documents, 
  vocab = out$vocab, 
  K = 20, 
  prevalence = ~ source_type + city + s(date_numeric),
  data = out$meta,
  init.type = "Spectral",
  seed = 114514,
  verbose = TRUE
)

print("Estimating effects for K=20...")
estimate_model <- estimateEffect(
  formula = ~ source_type + city + s(date_numeric), 
  stmobj = topic_model, 
  meta = out$meta, 
  uncertainty = "Global"
)

dir.create("models/R_topic_model", recursive = TRUE, showWarnings = FALSE)
saveRDS(topic_model, "models/R_topic_model/stm_topic_model_20.rds")
saveRDS(estimate_model, "models/R_topic_model/stm_topic_model_estimated_20.rds")

dir.create("models/inputs", recursive = TRUE, showWarnings = FALSE)
saveRDS(out, "models/inputs/out.rds")
saveRDS(dta, "models/inputs/dta.rds")

