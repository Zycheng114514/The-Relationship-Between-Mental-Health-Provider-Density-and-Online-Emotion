library(tidyverse)
library(stm)
library(lubridate)
library(quanteda)

dta <- read_csv("data/cleaned/the_dataset.csv")

dta <- dta %>%
  mutate(date_numeric = as.numeric(as.Date(datetime)))

corp <- corpus(dta, text_field = "content")

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

dfm_trimmed <- dfm_trim(dfm_counts, min_docfreq = 5)

out <- convert(dfm_trimmed, to = "stm")

topic_model <- stm(
  documents = out$documents, 
  vocab = out$vocab, 
  K = 20, 
  prevalence = ~ predicted_bws_score + s(date_numeric),
  data = out$meta,
  init.type = "Spectral",
  seed = 114514,
  verbose = TRUE
)

dir.create("data/models/R_topic_model", recursive = TRUE, showWarnings = FALSE)
saveRDS(topic_model, "data/models/R_topic_model/stm_topic_model.rds")