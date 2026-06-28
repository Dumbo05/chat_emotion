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

out_dir <- file.path("image_output", "wenben")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

labels_en <- c("anger", "disgust", "fear", "joy", "sadness", "surprise", "neutral")
labels_cn <- c(
  anger = "愤怒",
  disgust = "厌恶",
  fear = "恐惧",
  joy = "高兴",
  sadness = "悲伤",
  surprise = "惊讶",
  neutral = "中性"
)
metric_cn <- c(
  accuracy = "准确率",
  precision = "精确率",
  recall = "召回率",
  f1 = "F1",
  macro_f1 = "宏平均F1",
  train_loss = "训练集Loss",
  val_loss = "验证集Loss",
  val_accuracy = "验证集Accuracy"
)

palette <- c(
  "准确率" = "#4C78A8",
  "精确率" = "#72B7B2",
  "召回率" = "#F58518",
  "F1" = "#54A24B",
  "训练集Loss" = "#4C78A8",
  "验证集Loss" = "#E45756",
  "训练集Accuracy" = "#4C78A8",
  "验证集Accuracy" = "#E45756"
)

theme_set(
  theme_classic(base_size = 10, base_family = "sans") +
    theme(
      axis.line = element_line(linewidth = 0.35, colour = "black"),
      axis.ticks = element_line(linewidth = 0.35, colour = "black"),
      legend.title = element_text(size = 9),
      legend.text = element_text(size = 8.5),
      strip.text = element_text(size = 9, face = "bold"),
      plot.title = element_text(size = 12, face = "bold"),
      plot.subtitle = element_text(size = 9, colour = "#555555"),
      plot.caption = element_text(size = 7.5, colour = "#666666"),
      panel.grid.major.y = element_line(linewidth = 0.25, colour = "#E8E8E8"),
      panel.grid.major.x = element_blank(),
      panel.grid.minor = element_blank()
    )
)

safe_name <- function(x) {
  x |>
    str_replace_all("[\\\\/:*?\"<>|]", "_") |>
    str_replace_all("\\s+", "_")
}

save_png <- function(plot, stem, width = 9, height = 5.6, dpi = 320) {
  ggsave(
    filename = file.path(out_dir, paste0(safe_name(stem), ".png")),
    plot = plot,
    width = width,
    height = height,
    dpi = dpi,
    bg = "white",
    limitsize = FALSE
  )
}

`%||%` <- function(a, b) {
  if (is.null(a) || length(a) == 0) b else a
}

detect_family <- function(path, x) {
  model_text <- paste(x$model %||% "", x$model_name %||% "", x$experiment_id %||% "", x$experiment_name %||% "", path)
  model_text_l <- tolower(model_text)
  if (str_detect(model_text_l, "xlm.*large")) return("XLM-RoBERTa-large")
  if (str_detect(model_text_l, "xlm|roberta")) return("XLM-RoBERTa-base")
  if (str_detect(model_text_l, "mbert|multilingual-cased|bert-base-multilingual")) return("mBERT")
  "文本模型"
}

detect_experiment <- function(path, x) {
  id <- x$experiment_id %||% x$experiment_name %||% basename(dirname(path))
  if (identical(id, ".") || is.na(id) || id == "") basename(path) else id
}

extract_group <- function(exp_id, x) {
  g <- x$group %||% NA_character_
  if (!is.na(g)) return(as.character(g))
  if (str_detect(exp_id, "baseline|BASELINE")) return("baseline")
  if (str_detect(exp_id, "LEN")) return(str_extract(exp_id, "LEN-[0-9]+"))
  if (str_detect(exp_id, "len[0-9]+")) return(str_extract(exp_id, "len[0-9]+"))
  if (str_detect(exp_id, "LR-[0-9]+")) return(str_extract(exp_id, "LR-[0-9]+"))
  if (str_detect(exp_id, "lr[0-9.eE-]+")) return(str_extract(exp_id, "lr[0-9.eE-]+"))
  "single"
}

report_to_df <- function(report, meta) {
  rows <- lapply(labels_en, function(lbl) {
    item <- report[[lbl]]
    if (is.null(item)) return(NULL)
    tibble(
      family = meta$family,
      experiment = meta$experiment,
      group = meta$group,
      seed = meta$seed,
      class = lbl,
      class_cn = unname(labels_cn[lbl]),
      precision = as.numeric(item$precision %||% NA),
      recall = as.numeric(item$recall %||% NA),
      f1 = as.numeric(item[["f1-score"]] %||% NA),
      support = as.numeric(item$support %||% NA)
    )
  })
  bind_rows(rows)
}

read_metrics_json <- function(path) {
  x <- fromJSON(path, simplifyVector = FALSE)
  exp_id <- detect_experiment(path, x)
  family <- detect_family(path, x)
  meta <- list(
    source = path,
    family = family,
    experiment = exp_id,
    group = extract_group(exp_id, x),
    seed = as.integer(x$seed %||% str_extract(exp_id, "(?<=seed)[0-9]+") %||% NA),
    lr = as.numeric(x$learning_rate %||% x$hyperparams$learning_rate %||% NA),
    max_length = as.integer(x$max_length %||% x$hyperparams$max_length %||% str_extract(exp_id, "(?<=len)[0-9]+") %||% NA),
    batch_size = as.integer(x$batch_size %||% x$hyperparams$batch_size %||% NA),
    best_epoch = as.integer(x$best_epoch %||% NA)
  )

  test_node <- x$test %||% NULL
  if (is.null(test_node)) {
    report <- x$test_classification_report %||% NULL
    accuracy <- as.numeric(x$test_accuracy %||% if (!is.null(report)) report$accuracy else NA)
    macro_f1 <- as.numeric(x$test_macro_f1 %||% if (!is.null(report)) report[["macro avg"]][["f1-score"]] else NA)
    loss <- NA_real_
  } else {
    report <- test_node$classification_report %||% NULL
    accuracy <- as.numeric(test_node$accuracy %||% if (!is.null(report)) report$accuracy else NA)
    macro_f1 <- as.numeric(test_node$macro_f1 %||% if (!is.null(report)) report[["macro avg"]][["f1-score"]] else NA)
    loss <- as.numeric(test_node$loss %||% NA)
  }

  macro_precision <- if (!is.null(report)) as.numeric(report[["macro avg"]]$precision %||% NA) else NA_real_
  macro_recall <- if (!is.null(report)) as.numeric(report[["macro avg"]]$recall %||% NA) else NA_real_

  history_raw <- x$history %||% x$epoch_logs %||% list()
  history <- bind_rows(lapply(history_raw, function(e) {
    tibble(
      family = meta$family,
      experiment = meta$experiment,
      group = meta$group,
      seed = meta$seed,
      epoch = as.integer(e$epoch %||% NA),
      train_loss = as.numeric(e$train_loss %||% NA),
      val_loss = as.numeric(e$validation_loss %||% e$val_loss %||% NA),
      val_accuracy = as.numeric(e$validation_accuracy %||% e$val_accuracy %||% NA),
      val_macro_f1 = as.numeric(e$validation_macro_f1 %||% e$val_macro_f1 %||% NA)
    )
  }))

  summary <- tibble(
    family = meta$family,
    experiment = meta$experiment,
    group = meta$group,
    seed = meta$seed,
    lr = meta$lr,
    max_length = meta$max_length,
    best_epoch = meta$best_epoch,
    test_loss = loss,
    accuracy = accuracy,
    precision = macro_precision,
    recall = macro_recall,
    f1 = macro_f1,
    source = path
  )

  class_metrics <- if (!is.null(report)) report_to_df(report, meta) else tibble()
  list(summary = summary, history = history, class_metrics = class_metrics)
}

metric_paths <- c(
  list.files(file.path("mbert_b", "logs"), pattern = "\\.json$", full.names = TRUE),
  list.files("xlm-roberta-experiments", pattern = "metrics\\.json$", recursive = TRUE, full.names = TRUE),
  file.path("member_b", "metrics.json"),
  file.path("A1", "xlm-roberta", "metrics.json"),
  list.files(file.path("server-results", "text_final_0730_light", "outputs", "text"), pattern = "metrics\\.json$", recursive = TRUE, full.names = TRUE)
)
metric_paths <- metric_paths[file.exists(metric_paths)]

parsed <- lapply(metric_paths, read_metrics_json)
summary_df <- bind_rows(lapply(parsed, `[[`, "summary")) |>
  distinct(family, experiment, source, .keep_all = TRUE)
history_df <- bind_rows(lapply(parsed, `[[`, "history")) |>
  semi_join(summary_df |> select(family, experiment, source), by = c("family", "experiment"))
class_df <- bind_rows(lapply(parsed, `[[`, "class_metrics")) |>
  semi_join(summary_df |> select(family, experiment), by = c("family", "experiment"))

write_csv(summary_df, file.path(out_dir, "source_data_文本模型总体指标.csv"))
write_csv(history_df, file.path(out_dir, "source_data_文本模型训练历史.csv"))
write_csv(class_df, file.path(out_dir, "source_data_文本模型各类别指标.csv"))

read_confusion <- function(path) {
  mat <- read.csv(path, check.names = FALSE, row.names = 1)
  exp_id <- basename(dirname(path))
  metrics_path <- file.path(dirname(path), "metrics.json")
  if (file.exists(metrics_path)) {
    x <- fromJSON(metrics_path, simplifyVector = FALSE)
    exp_id <- detect_experiment(metrics_path, x)
    family <- detect_family(metrics_path, x)
    group <- extract_group(exp_id, x)
    seed <- as.integer(x$seed %||% str_extract(exp_id, "(?<=seed)[0-9]+") %||% NA)
  } else if (str_detect(path, "member_b")) {
    family <- "mBERT"
    group <- "single"
    seed <- 42L
    exp_id <- "member_b"
  } else if (str_detect(path, "A1")) {
    family <- "XLM-RoBERTa-base"
    group <- "single"
    seed <- 42L
    exp_id <- "A1_xlm-roberta"
  } else {
    family <- "文本模型"
    group <- "single"
    seed <- NA_integer_
  }
  as.data.frame(as.table(as.matrix(mat))) |>
    as_tibble() |>
    transmute(
      family = family,
      experiment = exp_id,
      group = group,
      seed = seed,
      true_class = as.character(Var1),
      pred_class = as.character(Var2),
      true_cn = unname(labels_cn[true_class]),
      pred_cn = unname(labels_cn[pred_class]),
      count = as.numeric(Freq)
    ) |>
    group_by(family, experiment, true_class) |>
    mutate(row_total = sum(count), row_rate = if_else(row_total > 0, count / row_total, 0)) |>
    ungroup()
}

conf_paths <- c(
  list.files("xlm-roberta-experiments", pattern = "confusion_matrix\\.csv$", recursive = TRUE, full.names = TRUE),
  file.path("member_b", "confusion_matrix.csv"),
  file.path("A1", "xlm-roberta", "confusion_matrix.csv"),
  list.files(file.path("server-results", "text_final_0730_light", "outputs", "text"), pattern = "confusion_matrix\\.csv$", recursive = TRUE, full.names = TRUE)
)
conf_paths <- conf_paths[file.exists(conf_paths)]
conf_df <- bind_rows(lapply(conf_paths, read_confusion))
write_csv(conf_df, file.path(out_dir, "source_data_文本模型混淆矩阵.csv"))

for (fam in unique(summary_df$family)) {
  fam_df <- summary_df |> filter(family == fam)
  overall <- fam_df |>
    select(experiment, group, accuracy, precision, recall, f1) |>
    pivot_longer(c(accuracy, precision, recall, f1), names_to = "metric", values_to = "value") |>
    mutate(metric_cn = recode(metric, !!!metric_cn),
           experiment = factor(experiment, levels = fam_df$experiment[order(fam_df$f1, fam_df$accuracy)]))
  p <- ggplot(overall, aes(x = experiment, y = value, fill = metric_cn)) +
    geom_col(position = position_dodge(width = 0.72), width = 0.62) +
    coord_flip() +
    scale_y_continuous(labels = percent_format(accuracy = 1), limits = c(0, NA), expand = expansion(mult = c(0, 0.05))) +
    scale_fill_manual(values = palette, name = "指标") +
    labs(
      title = paste0(fam, " 所有实验总体性能"),
      subtitle = "测试集宏平均指标；精确率、召回率、F1 为七类情绪宏平均",
      x = "实验版本",
      y = "分数",
      caption = "数据来源：现有 metrics.json / summary.csv"
    )
  save_png(p, paste0(fam, "_准确率_Precision_Recall_F1_总体指标"), width = 10.5, height = max(5, 0.34 * nrow(fam_df) + 2.2))
}

for (i in seq_len(nrow(summary_df))) {
  row <- summary_df[i, ]
  exp <- row$experiment
  fam <- row$family
  stem <- paste0(fam, "_", exp)

  cls <- class_df |> filter(family == fam, experiment == exp)
  if (nrow(cls) > 0) {
    cls_long <- cls |>
      select(class_cn, precision, recall, f1) |>
      pivot_longer(c(precision, recall, f1), names_to = "metric", values_to = "value") |>
      mutate(metric_cn = recode(metric, !!!metric_cn),
             class_cn = factor(class_cn, levels = unname(labels_cn[labels_en])))
    p_cls <- ggplot(cls_long, aes(x = class_cn, y = value, fill = metric_cn)) +
      geom_col(position = position_dodge(width = 0.72), width = 0.64) +
      scale_y_continuous(labels = percent_format(accuracy = 1), limits = c(0, 1), expand = expansion(mult = c(0, 0.02))) +
      scale_fill_manual(values = palette, name = "指标") +
      labs(
        title = paste0(fam, "｜", exp, " 各类别精确率/召回率/F1"),
        subtitle = "七类情绪测试集分类报告",
        x = "真实情绪类别",
        y = "分数",
        caption = "数据来源：classification_report"
      )
    save_png(p_cls, paste0(stem, "_各类别_Precision_Recall_F1_柱状图"), width = 8.4, height = 5.2)
  }

  hist <- history_df |> filter(family == fam, experiment == exp)
  if (nrow(hist) > 0) {
    loss_long <- hist |>
      select(epoch, train_loss, val_loss) |>
      pivot_longer(c(train_loss, val_loss), names_to = "metric", values_to = "value") |>
      filter(!is.na(value)) |>
      mutate(metric_cn = recode(metric, !!!metric_cn))
    if (nrow(loss_long) > 0) {
      p_loss <- ggplot(loss_long, aes(x = epoch, y = value, colour = metric_cn, group = metric_cn)) +
        geom_line(linewidth = 0.9) +
        geom_point(size = 2.1) +
        scale_colour_manual(values = palette, name = "曲线") +
        scale_x_continuous(breaks = sort(unique(loss_long$epoch))) +
        labs(
          title = paste0(fam, "｜", exp, " 训练集/验证集 Loss 曲线"),
          subtitle = "用于观察收敛与过拟合趋势",
          x = "训练轮次",
          y = "Loss",
          caption = "数据来源：训练历史日志"
        )
      save_png(p_loss, paste0(stem, "_训练集_验证集_Loss_曲线"), width = 7.2, height = 4.8)
    }

    acc_long <- hist |>
      select(epoch, val_accuracy) |>
      pivot_longer(c(val_accuracy), names_to = "metric", values_to = "value") |>
      filter(!is.na(value)) |>
      mutate(metric_cn = recode(metric, !!!metric_cn))
    if (nrow(acc_long) > 0) {
      p_acc <- ggplot(acc_long, aes(x = epoch, y = value, colour = metric_cn, group = metric_cn)) +
        geom_line(linewidth = 0.9) +
        geom_point(size = 2.1) +
        scale_colour_manual(values = c("验证集Accuracy" = "#4C78A8"), name = "曲线") +
        scale_x_continuous(breaks = sort(unique(acc_long$epoch))) +
        scale_y_continuous(labels = percent_format(accuracy = 1), limits = c(0, 1), expand = expansion(mult = c(0, 0.02))) +
        labs(
          title = paste0(fam, "｜", exp, " 训练集/验证集 Accuracy 曲线"),
          subtitle = "当前日志仅记录验证集 Accuracy，训练集 Accuracy 未保存",
          x = "训练轮次",
          y = "Accuracy",
          caption = "数据来源：训练历史日志"
        )
      save_png(p_acc, paste0(stem, "_训练集_验证集_Accuracy_曲线"), width = 7.2, height = 4.8)
    }
  }
}

if (nrow(conf_df) > 0) {
  for (key in unique(paste(conf_df$family, conf_df$experiment, sep = "\t"))) {
    parts <- str_split(key, "\t", simplify = TRUE)
    fam <- parts[1]
    exp <- parts[2]
    one <- conf_df |> filter(family == fam, experiment == exp)
    stem <- paste0(fam, "_", exp)
    heat <- one |>
      mutate(
        true_cn = factor(true_cn, levels = unname(labels_cn[labels_en])),
        pred_cn = factor(pred_cn, levels = unname(labels_cn[labels_en])),
        label = if_else(count > 0, paste0(count, "\n", percent(row_rate, accuracy = 1)), "")
      )
    p_heat <- ggplot(heat, aes(x = pred_cn, y = true_cn, fill = row_rate)) +
      geom_tile(colour = "white", linewidth = 0.5) +
      geom_text(aes(label = label), size = 3.1, lineheight = 0.88) +
      scale_fill_gradient(low = "#F3F7FA", high = "#2F6F9F", labels = percent_format(accuracy = 1), name = "行归一化比例") +
      coord_fixed() +
      labs(
        title = paste0(fam, "｜", exp, " 七类情绪混淆矩阵"),
        subtitle = "行表示真实类别，列表示预测类别；单元格为数量与行内比例",
        x = "预测情绪",
        y = "真实情绪",
        caption = "数据来源：confusion_matrix.csv"
      ) +
      theme(axis.text.x = element_text(angle = 30, hjust = 1), panel.grid = element_blank())
    save_png(p_heat, paste0(stem, "_混淆矩阵"), width = 7.4, height = 6.2)

    errors <- one |>
      filter(true_class != pred_class, count > 0) |>
      arrange(desc(count)) |>
      slice_head(n = 12) |>
      mutate(pair = paste0(true_cn, " → ", pred_cn),
             pair = factor(pair, levels = rev(pair)))
    if (nrow(errors) > 0) {
      p_err <- ggplot(errors, aes(x = pair, y = count, fill = true_cn)) +
        geom_col(width = 0.7, show.legend = FALSE) +
        coord_flip() +
        scale_fill_manual(values = c("#8FB9D4", "#E6A57E", "#9AC7A7", "#D6A6B8", "#BDB2D8", "#E7C66B", "#A8A8A8")) +
        labs(
          title = paste0(fam, "｜", exp, " 错误样例分析"),
          subtitle = "按混淆矩阵统计的高频误判路径，反映最常见错误样例类型",
          x = "真实类别 → 预测类别",
          y = "错误样例数",
          caption = "说明：当前结果未保存逐条错误文本，本图使用可复核的混淆矩阵计数"
        )
      save_png(p_err, paste0(stem, "_错误样例分析图"), width = 8.2, height = 5.3)
    }
  }
}

message("完成：", nrow(summary_df), " 个文本模型实验；", length(list.files(out_dir, pattern = "\\.png$")), " 张 PNG 图片。")
