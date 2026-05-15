#!/usr/bin/env Rscript
# Минимальная R-обёртка над statcheck.
# Читает текст из stdin или из файла (--file PATH), возвращает JSON-массив
# найденных тестов в схеме, совместимой с нашим пайплайном.
#
# Маппинг полей (statcheck -> наш формат). В современной версии statcheck:
#   test_type        -> test_type ("t","F","chi","z","r","Q")
#   test_value       -> statistic_value
#   df1, df2         -> df1, df2 (нормализуется на Python-стороне,
#                                  т.к. для t/z/chi/r statcheck кладёт df в df2)
#   p_comp           -> p_equality ("<", "=", ">")
#   reported_p       -> reported_p
#   computed_p       -> computed_p
#   error            -> p_error  (TRUE = reported не сходится с computed)
#   decision_error   -> decision_error (TRUE = знак значимости перевёрнут)
#   raw              -> raw_text

suppressPackageStartupMessages({
  library(statcheck)
  library(jsonlite)
})

args <- commandArgs(trailingOnly = TRUE)

read_text <- function() {
  if (length(args) >= 2 && args[1] == "--file") {
    con <- file(args[2], "r", encoding = "UTF-8")
    on.exit(close(con))
    paste(readLines(con, warn = FALSE), collapse = "\n")
  } else {
    paste(readLines("stdin", warn = FALSE), collapse = "\n")
  }
}

text <- read_text()

if (nchar(trimws(text)) == 0) {
  cat("[]")
  quit(status = 0)
}

# глушим всё что statcheck печатает в stdout/stderr, кроме нашего JSON.
# в частности на пустой результат statcheck печатает "statcheck did not find
# any results" в stdout, что ломает JSON-парсинг на стороне Python.
res <- NULL
sink_path <- tempfile()
sink_con <- file(sink_path, open = "wt")
sink(sink_con, type = "output")
sink(sink_con, type = "message")
tryCatch({
  res <<- tryCatch(
    suppressMessages(suppressWarnings(statcheck(text, messages = FALSE))),
    error = function(e) NULL
  )
}, finally = {
  sink(type = "message")
  sink(type = "output")
  close(sink_con)
  unlink(sink_path)
})

if (is.null(res) || nrow(res) == 0) {
  cat("[]")
  quit(status = 0)
}

safe <- function(x) if (is.null(x) || length(x) == 0) NA else x
col   <- function(name) if (name %in% colnames(res)) res[[name]] else rep(NA, nrow(res))

# fallback по разным именам столбцов между версиями statcheck
col_any <- function(...) {
  names <- c(...)
  for (n in names) {
    if (n %in% colnames(res)) return(res[[n]])
  }
  rep(NA, nrow(res))
}

rows <- lapply(seq_len(nrow(res)), function(i) {
  list(
    test_type        = as.character(safe(col_any("test_type", "Statistic")[i])),
    statistic_value  = as.numeric(safe(col_any("test_value", "Value")[i])),
    df1              = as.numeric(safe(col_any("df1")[i])),
    df2              = as.numeric(safe(col_any("df2")[i])),
    p_equality       = as.character(safe(col_any("p_comp", "Reported.Comparison")[i])),
    reported_p       = as.numeric(safe(col_any("reported_p", "Reported.P.Value")[i])),
    computed_p       = as.numeric(safe(col_any("computed_p", "Computed")[i])),
    p_error          = as.logical(safe(col_any("error", "Error")[i])),
    decision_error   = as.logical(safe(col_any("decision_error", "DecisionError")[i])),
    raw_text         = as.character(safe(col_any("raw", "Raw")[i]))
  )
})

cat(toJSON(rows, auto_unbox = TRUE, na = "null", null = "null"))
