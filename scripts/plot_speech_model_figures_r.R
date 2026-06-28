local_lib <- normalizePath(file.path("image_output", "wenben", "Rlib"), winslash = "/", mustWork = FALSE)
.libPaths(c(local_lib, .libPaths()))

suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(tidyr)
  library(jsonlite)
  library(readr)
  library(stringr)
  library(scales)
})

out_dir <- file.path("image_output", "\u56fe\u50cf")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

z <- function(x) x
label_cn <- c(
  anger = "\u6124\u6012",
  disgust = "\u538c\u6076",
  fear = "\u6050\u60e7",
  joy = "\u9ad8\u5174",
  sadness = "\u60b2\u4f24",
  surprise = "\u60ca\u8bb6",
  neutral = "\u4e2d\u6027"
)
metric_cn <- c(accuracy = "\u51c6\u786e\u7387", precision = "\u7cbe\u786e\u7387", recall = "\u53ec\u56de\u7387", f1 = "F1")
cn <- list(
  acc = "\u51c6\u786e\u7387", prec = "\u7cbe\u786e\u7387", rec = "\u53ec\u56de\u7387", f1 = "F1",
  train_loss = "\u8bad\u7ec3\u96c6Loss", val_loss = "\u9a8c\u8bc1\u96c6Loss",
  train_acc = "\u8bad\u7ec3\u96c6Accuracy", val_acc = "\u9a8c\u8bc1\u96c6Accuracy",
  metric = "\u6307\u6807", score = "\u5206\u6570", exp = "\u5b9e\u9a8c\u7248\u672c", epoch = "\u8bad\u7ec3\u8f6e\u6b21",
  true = "\u771f\u5b9e\u60c5\u7eea", pred = "\u9884\u6d4b\u60c5\u7eea", count = "\u9519\u8bef\u6837\u4f8b\u6570",
  confusion = "\u6df7\u6dc6\u77e9\u9635", error = "\u9519\u8bef\u6837\u4f8b\u5206\u6790",
  classbar = "\u5404\u7c7b\u522b_Precision_Recall_F1_\u67f1\u72b6\u56fe",
  source_prefix = "source_data_\u8bed\u97f3\u6a21\u578b"
)
palette <- c("\u51c6\u786e\u7387" = "#4C78A8", "\u7cbe\u786e\u7387" = "#72B7B2", "\u53ec\u56de\u7387" = "#F58518", "F1" = "#54A24B",
             "\u8bad\u7ec3\u96c6Loss" = "#4C78A8", "\u9a8c\u8bc1\u96c6Loss" = "#E45756",
             "\u8bad\u7ec3\u96c6Accuracy" = "#4C78A8", "\u9a8c\u8bc1\u96c6Accuracy" = "#E45756")

theme_set(theme_classic(base_size = 10, base_family = "sans") + theme(
  axis.line = element_line(linewidth = 0.35, colour = "black"),
  axis.ticks = element_line(linewidth = 0.35, colour = "black"),
  plot.title = element_text(size = 12, face = "bold"),
  plot.subtitle = element_text(size = 9, colour = "#555555"),
  plot.caption = element_text(size = 7.5, colour = "#666666"),
  legend.title = element_text(size = 9), legend.text = element_text(size = 8.5),
  panel.grid.major.y = element_line(linewidth = 0.25, colour = "#E8E8E8"),
  panel.grid.major.x = element_blank(), panel.grid.minor = element_blank()
))

`%||%` <- function(a, b) if (is.null(a) || length(a) == 0) b else a
num <- function(x) as.numeric(x %||% NA)
safe_name <- function(x) str_replace_all(str_replace_all(x, "[\\\\/:*?\"<>|]", "_"), "\\s+", "_")
save_png <- function(plot, stem, width = 8.6, height = 5.2, dpi = 320) {
  ggsave(file.path(out_dir, paste0(safe_name(stem), ".png")), plot = plot, width = width, height = height, dpi = dpi, bg = "white", limitsize = FALSE)
}

labels_from_report <- function(report, cm = NULL) {
  labs <- intersect(names(label_cn), names(report))
  if (length(labs) == 0 && !is.null(cm)) labs <- names(label_cn)[seq_len(length(cm))]
  labs
}
class_from_report <- function(report, meta, labels) {
  bind_rows(lapply(labels, function(lbl) {
    item <- report[[lbl]]; if (is.null(item)) return(NULL)
    tibble(family=meta$family, experiment=meta$experiment, class=lbl, class_cn=unname(label_cn[lbl]),
           precision=num(item$precision), recall=num(item$recall), f1=num(item[["f1-score"]]), support=num(item$support), source=meta$source)
  }))
}
conf_from_matrix <- function(cm, labels, meta) {
  if (is.null(cm)) return(tibble())
  mat <- do.call(rbind, lapply(cm, as.numeric)); labels <- labels[seq_len(nrow(mat))]
  as.data.frame(as.table(mat)) |> as_tibble() |>
    transmute(family=meta$family, experiment=meta$experiment,
              true_class=labels[as.integer(Var1)], pred_class=labels[as.integer(Var2)],
              true_cn=unname(label_cn[true_class]), pred_cn=unname(label_cn[pred_class]), count=as.numeric(Freq), source=meta$source) |>
    group_by(family, experiment, true_class) |>
    mutate(row_total=sum(count), row_rate=if_else(row_total > 0, count / row_total, 0)) |> ungroup()
}
history_from_list <- function(hist, meta) {
  bind_rows(lapply(hist %||% list(), function(e) tibble(
    family=meta$family, experiment=meta$experiment, epoch=as.integer(e$epoch %||% NA),
    train_loss=num(e$train_loss), val_loss=num(e$val_loss %||% e$validation_loss),
    train_accuracy=num(e$train_accuracy), val_accuracy=num(e$val_accuracy %||% e$validation_accuracy), source=meta$source)))
}
parse_single <- function(path, family, experiment, report, accuracy, cm, history=list()) {
  labels <- labels_from_report(report, cm); meta <- list(family=family, experiment=experiment, source=path); macro <- report[["macro avg"]] %||% list()
  list(summary=tibble(family=family, experiment=experiment, accuracy=num(accuracy), precision=num(macro$precision), recall=num(macro$recall), f1=num(macro[["f1-score"]]), source=path),
       class=class_from_report(report, meta, labels), conf=conf_from_matrix(cm, labels, meta), history=history_from_list(history, meta))
}
parse_metrics_json <- function(path) {
  x <- fromJSON(path, simplifyVector=FALSE)
  if (!is.null(x$test)) return(parse_single(path, "MFCC-RBF-SVM", "MFCC\u57fa\u7ebf\u6d4b\u8bd5\u96c6", x$test$classification_report, x$test$accuracy, x$test$confusion_matrix, x$history))
  if (!is.null(x$validation)) return(parse_single(path, "SIMSAN", tools::file_path_sans_ext(basename(path)), x$validation$classification_report, x$validation$accuracy, x$validation$confusion_matrix, x$history))
  parse_single(path, "\u8bed\u97f3\u6a21\u578b", tools::file_path_sans_ext(basename(path)), x$classification_report, x$accuracy %||% x$test_accuracy, x$confusion_matrix, x$history)
}
parse_final_test_json <- function(path) {
  x <- fromJSON(path, simplifyVector=FALSE); exp <- tools::file_path_sans_ext(basename(path))
  family <- if (str_detect(exp, "wavlm_clean")) "WavLM" else if (str_detect(exp, regex("simsan", ignore_case=TRUE))) "SIMSAN" else "\u8bed\u97f3\u6a21\u578b"
  report <- x$classification_report %||% NULL
  if (is.null(report) && !is.null(x$test_per_class_precision)) {
    labs <- names(x$test_per_class_precision)
    report <- setNames(lapply(labs, function(lbl) list(precision=x$test_per_class_precision[[lbl]], recall=x$test_per_class_recall[[lbl]], `f1-score`=x$test_per_class_f1[[lbl]], support=x$test_per_class_support[[lbl]] %||% NA)), labs)
    report[["macro avg"]] <- list(precision=mean(as.numeric(unlist(x$test_per_class_precision)), na.rm=TRUE), recall=mean(as.numeric(unlist(x$test_per_class_recall)), na.rm=TRUE), `f1-score`=x$test_macro_f1, support=x$test_sample_count %||% x$samples %||% NA)
  }
  parse_single(path, family, exp, report, x$test_accuracy %||% x$accuracy, x$test_confusion_matrix %||% x$confusion_matrix, list())
}
parse_wavlm_v2 <- function(path) {
  x <- fromJSON(path, simplifyVector=FALSE)
  parts <- lapply(x$models, function(m) parse_single(path, "WavLM", paste0("wavlm_clean_v2_", m$id), m$classification_report, m$test_accuracy, m$confusion_matrix, list()))
  list(summary=bind_rows(lapply(parts, `[[`, "summary")), class=bind_rows(lapply(parts, `[[`, "class")), conf=bind_rows(lapply(parts, `[[`, "conf")), history=tibble())
}
read_jsonl_sweeps <- function(path) {
  lines <- readLines(path, warn=FALSE, encoding="UTF-8")
  bind_rows(lapply(lines, function(line) { x <- fromJSON(line, simplifyVector=FALSE); tibble(protocol=x$protocol_name %||% tools::file_path_sans_ext(basename(path)), run_id=x$run_id, family="WavLM\u9a8c\u8bc1\u641c\u7d22", experiment=paste0(x$protocol_name %||% tools::file_path_sans_ext(basename(path)), "_", x$run_id), accuracy=num(x$val_accuracy), precision=mean(as.numeric(unlist(x$val_per_class_precision)), na.rm=TRUE), recall=mean(as.numeric(unlist(x$val_per_class_recall)), na.rm=TRUE), f1=num(x$val_macro_f1), source=path) }))
}

json_parts <- list(
  parse_metrics_json(file.path("models", "speech", "metrics.json")),
  parse_metrics_json(file.path("models", "speech", "simsan_metrics.json")),
  parse_metrics_json(file.path("models", "speech", "simsan_mild_metrics.json")),
  parse_final_test_json(file.path("models", "speech", "simsan_final_test_metrics.json")),
  parse_final_test_json(file.path("models", "speech", "wavlm_simsan_fixed_test_metrics.json")),
  parse_wavlm_v2(file.path("models", "speech", "wavlm_clean_v2_final_results.json")),
  parse_final_test_json(file.path("models", "speech", "wavlm_clean_v3_final_results.json")),
  parse_final_test_json(file.path("models", "speech", "wavlm_clean_v4_final_results.json")),
  parse_final_test_json(file.path("models", "speech", "wavlm_clean_v5_final_results.json"))
)
summary_df <- bind_rows(lapply(json_parts, `[[`, "summary")); class_df <- bind_rows(lapply(json_parts, `[[`, "class")); conf_df <- bind_rows(lapply(json_parts, `[[`, "conf")); history_df <- bind_rows(lapply(json_parts, `[[`, "history"))
sweep_paths <- list.files(file.path("outputs", "speech"), pattern="validation_sweep\\.jsonl$", full.names=TRUE)
sweep_summary <- bind_rows(lapply(sweep_paths, read_jsonl_sweeps))
summary_all <- bind_rows(summary_df, sweep_summary)
write_csv(summary_all, file.path(out_dir, paste0(cn$source_prefix, "\u603b\u4f53\u6307\u6807.csv")))
write_csv(class_df, file.path(out_dir, paste0(cn$source_prefix, "\u5404\u7c7b\u522b\u6307\u6807.csv")))
write_csv(conf_df, file.path(out_dir, paste0(cn$source_prefix, "\u6df7\u6dc6\u77e9\u9635.csv")))
write_csv(history_df, file.path(out_dir, paste0(cn$source_prefix, "\u8bad\u7ec3\u5386\u53f2.csv")))

for (fam in unique(summary_df$family)) {
  fam_df <- summary_df |> filter(family == fam)
  p <- fam_df |> select(experiment, accuracy, precision, recall, f1) |> pivot_longer(c(accuracy, precision, recall, f1), names_to="metric", values_to="value") |>
    mutate(metric_cn=recode(metric, !!!metric_cn), experiment=factor(experiment, levels=fam_df$experiment[order(fam_df$f1, fam_df$accuracy)])) |>
    ggplot(aes(experiment, value, fill=metric_cn)) + geom_col(position=position_dodge(width=0.72), width=0.62) + coord_flip() +
    scale_y_continuous(labels=percent_format(accuracy=1), limits=c(0, NA), expand=expansion(mult=c(0,0.05))) + scale_fill_manual(values=palette, name=cn$metric) +
    labs(title=paste0(fam, " \u6240\u6709\u6700\u7ec8\u5b9e\u9a8c\u603b\u4f53\u6027\u80fd"), subtitle="\u6d4b\u8bd5/\u9a8c\u8bc1\u7ed3\u679c\u7684\u51c6\u786e\u7387\u4e0e\u5b8f\u5e73\u5747 Precision / Recall / F1", x=cn$exp, y=cn$score, caption="\u6570\u636e\u6765\u6e90\uff1a\u8bed\u97f3\u6a21\u578b\u7ed3\u679c JSON")
  save_png(p, paste0(fam, "_\u51c6\u786e\u7387_Precision_Recall_F1_\u6307\u6807\u56fe"), width=10.5, height=max(5, 0.38*nrow(fam_df)+2.2))
}
if (nrow(sweep_summary) > 0) for (proto in unique(sweep_summary$protocol)) {
  sw <- sweep_summary |> filter(protocol == proto) |> arrange(desc(f1)) |> mutate(rank=row_number())
  p <- ggplot(sw, aes(rank, f1)) + geom_point(aes(colour=accuracy), alpha=0.72, size=1.8) +
    scale_colour_gradient(low="#BFD7EA", high="#2F6F9F", labels=percent_format(accuracy=1), name="\u9a8c\u8bc1\u51c6\u786e\u7387") + scale_y_continuous(labels=percent_format(accuracy=1)) +
    labs(title=paste0(proto, " \u5168\u90e8\u9a8c\u8bc1\u641c\u7d22\u5b9e\u9a8c\u6307\u6807\u56fe"), subtitle=paste0("\u6bcf\u4e2a\u70b9\u4ee3\u8868\u4e00\u4e2a\u9a8c\u8bc1\u5b9e\u9a8c\uff1b\u5171 ", nrow(sw), " \u4e2a\u5019\u9009\uff0c\u6309\u5b8f\u5e73\u5747F1\u6392\u5e8f"), x="\u5019\u9009\u6392\u540d", y="\u9a8c\u8bc1\u5b8f\u5e73\u5747F1", caption="\u6570\u636e\u6765\u6e90\uff1avalidation_sweep.jsonl")
  save_png(p, paste0("WavLM_", proto, "_\u5168\u90e8\u9a8c\u8bc1\u641c\u7d22\u5b9e\u9a8c_\u51c6\u786e\u7387_Precision_Recall_F1_\u6307\u6807\u56fe"), width=8.8, height=5.2)
}

plot_one <- function(row) {
  fam <- row$family; exp <- row$experiment; stem <- paste0(fam, "_", exp)
  cls <- class_df |> filter(family == fam, experiment == exp)
  if (nrow(cls) > 0) {
    p <- cls |> select(class_cn, precision, recall, f1) |> pivot_longer(c(precision, recall, f1), names_to="metric", values_to="value") |>
      mutate(metric_cn=recode(metric, !!!metric_cn), class_cn=factor(class_cn, levels=unname(label_cn[unique(cls$class)]))) |>
      ggplot(aes(class_cn, value, fill=metric_cn)) + geom_col(position=position_dodge(width=0.72), width=0.64) +
      scale_y_continuous(labels=percent_format(accuracy=1), limits=c(0,1), expand=expansion(mult=c(0,0.02))) + scale_fill_manual(values=palette, name=cn$metric) +
      labs(title=paste0(fam, "\uff5c", exp, " \u5404\u7c7b\u522b\u7cbe\u786e\u7387/\u53ec\u56de\u7387/F1"), subtitle="\u8bed\u97f3\u60c5\u7eea\u8bc6\u522b\u5206\u7c7b\u62a5\u544a\uff1b\u6309\u7ed3\u679c\u6587\u4ef6\u4e2d\u7684\u771f\u5b9e\u6807\u7b7e\u96c6\u5408\u7ed8\u5236", x=cn$true, y=cn$score, caption="\u6570\u636e\u6765\u6e90\uff1aclassification_report")
    save_png(p, paste0(stem, "_", cn$classbar), width=8.4, height=5.2)
  }
  one_conf <- conf_df |> filter(family == fam, experiment == exp)
  if (nrow(one_conf) > 0) {
    order_cn <- unname(label_cn[unique(one_conf$true_class)])
    heat <- one_conf |> mutate(true_cn=factor(true_cn, levels=order_cn), pred_cn=factor(pred_cn, levels=order_cn), label=if_else(count>0, paste0(count, "\n", percent(row_rate, accuracy=1)), ""))
    p_heat <- ggplot(heat, aes(pred_cn, true_cn, fill=row_rate)) + geom_tile(colour="white", linewidth=0.5) + geom_text(aes(label=label), size=3.05, lineheight=0.88) +
      scale_fill_gradient(low="#F3F7FA", high="#2F6F9F", labels=percent_format(accuracy=1), name="\u884c\u5f52\u4e00\u5316\u6bd4\u4f8b") + coord_fixed() +
      labs(title=paste0(fam, "\uff5c", exp, " \u60c5\u7eea\u6df7\u6dc6\u77e9\u9635"), subtitle="\u884c\u8868\u793a\u771f\u5b9e\u7c7b\u522b\uff0c\u5217\u8868\u793a\u9884\u6d4b\u7c7b\u522b\uff1b\u5355\u5143\u683c\u4e3a\u6570\u91cf\u4e0e\u884c\u5185\u6bd4\u4f8b", x=cn$pred, y=cn$true, caption="\u6570\u636e\u6765\u6e90\uff1aconfusion_matrix") + theme(axis.text.x=element_text(angle=30, hjust=1), panel.grid=element_blank())
    save_png(p_heat, paste0(stem, "_", cn$confusion), width=7.2, height=6.0)
    err <- one_conf |> filter(true_class != pred_class, count > 0) |> transmute(pair=paste0(true_cn, " \u2192 ", pred_cn), n=count) |> arrange(desc(n)) |> slice_head(n=12) |> mutate(pair=factor(pair, levels=rev(pair)))
    p_err <- ggplot(err, aes(pair, n)) + geom_col(width=0.7, fill="#D27B53") + coord_flip() +
      labs(title=paste0(fam, "\uff5c", exp, " ", cn$error), subtitle="\u6309\u6df7\u6dc6\u77e9\u9635\u7edf\u8ba1\u7684\u9ad8\u9891\u8bef\u5224\u8def\u5f84", x="\u771f\u5b9e\u7c7b\u522b \u2192 \u9884\u6d4b\u7c7b\u522b", y=cn$count, caption="\u8bf4\u660e\uff1a\u73b0\u6709\u8bed\u97f3\u7ed3\u679c\u672a\u4fdd\u5b58\u9010\u6761\u9519\u8bef\u97f3\u9891\uff0c\u672c\u56fe\u4f7f\u7528\u6df7\u6dc6\u77e9\u9635\u8ba1\u6570")
    save_png(p_err, paste0(stem, "_\u9519\u8bef\u6837\u4f8b\u5206\u6790\u56fe"), width=8.2, height=5.3)
  }
  hist <- history_df |> filter(family == fam, experiment == exp)
  if (nrow(hist) > 0) {
    loss_long <- hist |> select(epoch, train_loss, val_loss) |> pivot_longer(c(train_loss, val_loss), names_to="metric", values_to="value") |> filter(!is.na(value)) |> mutate(metric_cn=recode(metric, train_loss=cn$train_loss, val_loss=cn$val_loss))
    if (nrow(loss_long)>0) save_png(ggplot(loss_long, aes(epoch, value, colour=metric_cn, group=metric_cn)) + geom_line(linewidth=0.85) + geom_point(size=1.7) + scale_colour_manual(values=palette, name="\u66f2\u7ebf") + labs(title=paste0(fam,"\uff5c",exp," \u8bad\u7ec3\u96c6/\u9a8c\u8bc1\u96c6 Loss \u66f2\u7ebf"), subtitle="\u7528\u4e8e\u89c2\u5bdf\u795e\u7ecf\u8bed\u97f3\u6a21\u578b\u6536\u655b\u8d8b\u52bf", x=cn$epoch, y="Loss", caption="\u6570\u636e\u6765\u6e90\uff1ahistory"), paste0(stem,"_\u8bad\u7ec3\u96c6_\u9a8c\u8bc1\u96c6_Loss_\u66f2\u7ebf"), width=7.5, height=4.8)
    acc_long <- hist |> select(epoch, train_accuracy, val_accuracy) |> pivot_longer(c(train_accuracy, val_accuracy), names_to="metric", values_to="value") |> filter(!is.na(value)) |> mutate(metric_cn=recode(metric, train_accuracy=cn$train_acc, val_accuracy=cn$val_acc))
    if (nrow(acc_long)>0) save_png(ggplot(acc_long, aes(epoch, value, colour=metric_cn, group=metric_cn)) + geom_line(linewidth=0.85) + geom_point(size=1.7) + scale_colour_manual(values=palette, name="\u66f2\u7ebf") + scale_y_continuous(labels=percent_format(accuracy=1), limits=c(0,1), expand=expansion(mult=c(0,0.02))) + labs(title=paste0(fam,"\uff5c",exp," \u8bad\u7ec3\u96c6/\u9a8c\u8bc1\u96c6 Accuracy \u66f2\u7ebf"), subtitle="\u8bad\u7ec3\u96c6\u4e0e\u9a8c\u8bc1\u96c6\u51c6\u786e\u7387\u968f epoch \u53d8\u5316", x=cn$epoch, y="Accuracy", caption="\u6570\u636e\u6765\u6e90\uff1ahistory"), paste0(stem,"_\u8bad\u7ec3\u96c6_\u9a8c\u8bc1\u96c6_Accuracy_\u66f2\u7ebf"), width=7.5, height=4.8)
  }
}
invisible(lapply(seq_len(nrow(summary_df)), function(i) plot_one(summary_df[i, ])))
message("done final=", nrow(summary_df), " sweep=", nrow(sweep_summary))
