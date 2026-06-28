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

out_dir <- file.path("image_output", "图像")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

labels_en <- c("surprise", "fear", "disgust", "joy", "sadness", "anger", "neutral")
labels_cn <- c(
  surprise = "惊讶",
  fear = "恐惧",
  disgust = "厌恶",
  joy = "高兴",
  sadness = "悲伤",
  anger = "愤怒",
  neutral = "中性"
)
metric_cn <- c(accuracy = "准确率", precision = "精确率", recall = "召回率", f1 = "F1")
palette <- c("准确率" = "#4C78A8", "精确率" = "#72B7B2", "召回率" = "#F58518", "F1" = "#54A24B",
             "训练集Loss" = "#4C78A8", "验证集Loss" = "#E45756",
             "训练集Accuracy" = "#4C78A8", "验证集Accuracy" = "#E45756")

theme_set(
  theme_classic(base_size = 10, base_family = "sans") +
    theme(
      axis.line = element_line(linewidth = 0.35, colour = "black"),
      axis.ticks = element_line(linewidth = 0.35, colour = "black"),
      plot.title = element_text(size = 12, face = "bold"),
      plot.subtitle = element_text(size = 9, colour = "#555555"),
      plot.caption = element_text(size = 7.5, colour = "#666666"),
      legend.title = element_text(size = 9),
      legend.text = element_text(size = 8.5),
      panel.grid.major.y = element_line(linewidth = 0.25, colour = "#E8E8E8"),
      panel.grid.major.x = element_blank(),
      panel.grid.minor = element_blank()
    )
)

`%||%` <- function(a, b) if (is.null(a) || length(a) == 0) b else a

safe_name <- function(x) {
  x |>
    str_replace_all("[\\\\/:*?\"<>|]", "_") |>
    str_replace_all("\\s+", "_")
}

save_png <- function(plot, stem, width = 8.6, height = 5.2, dpi = 320) {
  ggsave(
    file.path(out_dir, paste0(safe_name(stem), ".png")),
    plot = plot,
    width = width,
    height = height,
    dpi = dpi,
    bg = "white",
    limitsize = FALSE
  )
}

family_from_path <- function(path, x = NULL) {
  text <- tolower(paste(path, x$model %||% "", x$checkpoint_used %||% ""))
  if (str_detect(text, "fer_pretrain|fer-pretrain")) return("SE-ResNet18-FER预训练")
  if (str_detect(text, "se_resnet18|se-resnet18")) return("SE-ResNet18")
  if (str_detect(text, "rafemotionnet|rafdb_emotion")) return("RafEmotionNet")
  "图像模型"
}

experiment_from_path <- function(path) {
  parent <- basename(dirname(path))
  grand <- basename(dirname(dirname(path)))
  if (parent == "image") return(tools::file_path_sans_ext(basename(path)))
  if (parent %in% c("rafdb_emotion", "rafdb_se_resnet18", "rafdb_smoke")) return(parent)
  if (str_detect(path, "[/\\\\]models[/\\\\]image[/\\\\]")) return(parent)
  if (basename(path) == "test_results.json") return(parent)
  paste(grand, parent, sep = "_")
}

report_rows <- function(report, meta) {
  bind_rows(lapply(labels_en, function(lbl) {
    item <- report[[lbl]]
    if (is.null(item)) return(NULL)
    tibble(
      family = meta$family,
      experiment = meta$experiment,
      class = lbl,
      class_cn = unname(labels_cn[lbl]),
      precision = as.numeric(item$precision %||% NA),
      recall = as.numeric(item$recall %||% NA),
      f1 = as.numeric(item[["f1-score"]] %||% NA),
      support = as.numeric(item$support %||% NA),
      source = meta$source
    )
  }))
}

test_result_rows <- function(path) {
  x <- fromJSON(path, simplifyVector = FALSE)
  exp <- experiment_from_path(path)
  fam <- family_from_path(path, x)
  use_tta <- !is.null(x$test_accuracy_flip_tta)
  suffix <- if (use_tta) "flip_tta" else "no_tta"
  p <- x[[paste0("test_per_class_precision_", suffix)]] %||% list()
  r <- x[[paste0("test_per_class_recall_", suffix)]] %||% list()
  f <- x[[paste0("test_per_class_f1_", suffix)]] %||% list()
  cm <- x[[paste0("test_confusion_matrix_", suffix)]] %||% NULL
  summary <- tibble(
    family = fam,
    experiment = exp,
    evaluation = if (use_tta) "水平翻转TTA" else "无TTA",
    accuracy = as.numeric(x[[paste0("test_accuracy_", suffix)]] %||% NA),
    precision = mean(as.numeric(unlist(p[labels_en])), na.rm = TRUE),
    recall = mean(as.numeric(unlist(r[labels_en])), na.rm = TRUE),
    f1 = as.numeric(x[[paste0("test_macro_f1_", suffix)]] %||% mean(as.numeric(unlist(f[labels_en])), na.rm = TRUE)),
    source = path
  )
  class_metrics <- bind_rows(lapply(labels_en, function(lbl) {
    tibble(
      family = fam,
      experiment = exp,
      class = lbl,
      class_cn = unname(labels_cn[lbl]),
      precision = as.numeric(p[[lbl]] %||% NA),
      recall = as.numeric(r[[lbl]] %||% NA),
      f1 = as.numeric(f[[lbl]] %||% NA),
      support = NA_real_,
      source = path
    )
  }))
  conf <- tibble()
  if (!is.null(cm)) {
    mat <- do.call(rbind, lapply(cm, as.numeric))
    conf <- as.data.frame(as.table(mat)) |>
      as_tibble() |>
      transmute(
        family = fam,
        experiment = exp,
        true_class = labels_en[as.integer(Var1)],
        pred_class = labels_en[as.integer(Var2)],
        true_cn = unname(labels_cn[true_class]),
        pred_cn = unname(labels_cn[pred_class]),
        count = as.numeric(Freq),
        source = path
      ) |>
      group_by(family, experiment, true_class) |>
      mutate(row_total = sum(count), row_rate = if_else(row_total > 0, count / row_total, 0)) |>
      ungroup()
  }
  list(summary = summary, class = class_metrics, conf = conf)
}

metrics_rows <- function(path) {
  x <- fromJSON(path, simplifyVector = FALSE)
  exp <- experiment_from_path(path)
  fam <- family_from_path(path, x)
  report <- x$classification_report %||% list()
  summary <- tibble(
    family = fam,
    experiment = exp,
    evaluation = "官方测试集",
    accuracy = as.numeric(x$official_test_accuracy %||% report$accuracy %||% NA),
    precision = as.numeric(report[["macro avg"]]$precision %||% NA),
    recall = as.numeric(report[["macro avg"]]$recall %||% NA),
    f1 = as.numeric(report[["macro avg"]][["f1-score"]] %||% NA),
    source = path
  )
  class_metrics <- report_rows(report, list(family = fam, experiment = exp, source = path))
  history <- bind_rows(lapply(x$history %||% list(), function(e) {
    tibble(
      family = fam,
      experiment = exp,
      epoch = as.integer(e$epoch %||% NA),
      train_loss = as.numeric(e$train_loss %||% NA),
      val_loss = as.numeric(e$val_loss %||% NA),
      train_accuracy = as.numeric(e$train_accuracy %||% NA),
      val_accuracy = as.numeric(e$val_accuracy %||% NA),
      source = path
    )
  }))
  conf <- tibble()
  if (!is.null(x$confusion_matrix)) {
    mat <- do.call(rbind, lapply(x$confusion_matrix, as.numeric))
    conf <- as.data.frame(as.table(mat)) |>
      as_tibble() |>
      transmute(
        family = fam,
        experiment = exp,
        true_class = labels_en[as.integer(Var1)],
        pred_class = labels_en[as.integer(Var2)],
        true_cn = unname(labels_cn[true_class]),
        pred_cn = unname(labels_cn[pred_class]),
        count = as.numeric(Freq),
        source = path
      ) |>
      group_by(family, experiment, true_class) |>
      mutate(row_total = sum(count), row_rate = if_else(row_total > 0, count / row_total, 0)) |>
      ungroup()
  }
  list(summary = summary, class = class_metrics, history = history, conf = conf)
}

read_history_csv <- function(path) {
  exp <- basename(dirname(path))
  fam <- family_from_path(path)
  read_csv(path, show_col_types = FALSE, progress = FALSE) |>
    transmute(
      family = fam,
      experiment = exp,
      epoch = as.integer(epoch),
      train_loss = as.numeric(train_loss),
      val_loss = as.numeric(val_loss),
      train_accuracy = as.numeric(train_accuracy),
      val_accuracy = as.numeric(val_accuracy),
      source = path
    )
}

test_paths <- c(
  list.files(file.path("outputs", "image"), pattern = "test_results\\.json$", recursive = TRUE, full.names = TRUE),
  list.files(file.path("models", "image"), pattern = "test_results\\.json$", recursive = TRUE, full.names = TRUE)
)
metric_paths <- list.files(file.path("models", "image"), pattern = "metrics\\.json$", recursive = TRUE, full.names = TRUE)
history_paths <- list.files(file.path("outputs", "image"), pattern = "epoch_metrics\\.csv$", recursive = TRUE, full.names = TRUE)
failure_paths <- list.files(file.path("outputs", "image"), pattern = "failure_cases\\.csv$", recursive = TRUE, full.names = TRUE)

test_parsed <- lapply(test_paths, test_result_rows)
metric_parsed <- lapply(metric_paths, metrics_rows)

summary_df <- bind_rows(lapply(test_parsed, `[[`, "summary"), lapply(metric_parsed, `[[`, "summary")) |>
  distinct(family, experiment, source, .keep_all = TRUE)
class_df <- bind_rows(lapply(test_parsed, `[[`, "class"), lapply(metric_parsed, `[[`, "class")) |>
  semi_join(summary_df |> select(family, experiment, source), by = c("family", "experiment", "source"))
conf_df <- bind_rows(lapply(test_parsed, `[[`, "conf"), lapply(metric_parsed, `[[`, "conf")) |>
  semi_join(summary_df |> select(family, experiment, source), by = c("family", "experiment", "source"))
history_df <- bind_rows(lapply(metric_parsed, `[[`, "history"), lapply(history_paths, read_history_csv)) |>
  distinct(family, experiment, epoch, source, .keep_all = TRUE)

failure_df <- bind_rows(lapply(failure_paths, function(path) {
  exp <- basename(dirname(path))
  fam <- family_from_path(path)
  read_csv(path, show_col_types = FALSE, progress = FALSE) |>
    transmute(
      family = fam,
      experiment = exp,
      true_class = true_label,
      pred_class = pred_label,
      true_cn = unname(labels_cn[true_class]),
      pred_cn = unname(labels_cn[pred_class]),
      confidence = as.numeric(top1_prob),
      pair = paste0(true_cn, " → ", pred_cn),
      source = path
    )
}))

write_csv(summary_df, file.path(out_dir, "source_data_图像模型总体指标.csv"))
write_csv(class_df, file.path(out_dir, "source_data_图像模型各类别指标.csv"))
write_csv(conf_df, file.path(out_dir, "source_data_图像模型混淆矩阵.csv"))
write_csv(history_df, file.path(out_dir, "source_data_图像模型训练历史.csv"))
write_csv(failure_df, file.path(out_dir, "source_data_图像模型错误样例.csv"))

for (fam in unique(summary_df$family)) {
  fam_df <- summary_df |> filter(family == fam)
  overall <- fam_df |>
    select(experiment, accuracy, precision, recall, f1) |>
    pivot_longer(c(accuracy, precision, recall, f1), names_to = "metric", values_to = "value") |>
    mutate(metric_cn = recode(metric, !!!metric_cn),
           experiment = factor(experiment, levels = fam_df$experiment[order(fam_df$f1, fam_df$accuracy)]))
  p <- ggplot(overall, aes(experiment, value, fill = metric_cn)) +
    geom_col(position = position_dodge(width = 0.72), width = 0.62) +
    coord_flip() +
    scale_y_continuous(labels = percent_format(accuracy = 1), limits = c(0, NA), expand = expansion(mult = c(0, 0.05))) +
    scale_fill_manual(values = palette, name = "指标") +
    labs(title = paste0(fam, " 所有实验总体性能"),
         subtitle = "测试集准确率及七类情绪宏平均 Precision / Recall / F1",
         x = "实验版本", y = "分数", caption = "数据来源：图像模型测试结果文件")
  save_png(p, paste0(fam, "_准确率_Precision_Recall_F1_指标图"), width = 10.5, height = max(5, 0.36 * nrow(fam_df) + 2.2))
}

for (i in seq_len(nrow(summary_df))) {
  row <- summary_df[i, ]
  fam <- row$family
  exp <- row$experiment
  stem <- paste0(fam, "_", exp)

  cls <- class_df |> filter(family == fam, experiment == exp, source == row$source)
  if (nrow(cls) > 0) {
    p_cls <- cls |>
      select(class_cn, precision, recall, f1) |>
      pivot_longer(c(precision, recall, f1), names_to = "metric", values_to = "value") |>
      mutate(metric_cn = recode(metric, !!!metric_cn),
             class_cn = factor(class_cn, levels = unname(labels_cn[labels_en]))) |>
      ggplot(aes(class_cn, value, fill = metric_cn)) +
      geom_col(position = position_dodge(width = 0.72), width = 0.64) +
      scale_y_continuous(labels = percent_format(accuracy = 1), limits = c(0, 1), expand = expansion(mult = c(0, 0.02))) +
      scale_fill_manual(values = palette, name = "指标") +
      labs(title = paste0(fam, "｜", exp, " 各类别精确率/召回率/F1"),
           subtitle = "七类情绪测试集分类报告",
           x = "真实情绪类别", y = "分数", caption = "数据来源：per-class 测试指标")
    save_png(p_cls, paste0(stem, "_各类别_Precision_Recall_F1_柱状图"), width = 8.4, height = 5.2)
  }

  hist <- history_df |> filter(experiment == exp)
  if (nrow(hist) > 0) {
    loss_long <- hist |>
      select(epoch, train_loss, val_loss) |>
      pivot_longer(c(train_loss, val_loss), names_to = "metric", values_to = "value") |>
      mutate(metric_cn = recode(metric, train_loss = "训练集Loss", val_loss = "验证集Loss"))
    p_loss <- ggplot(loss_long, aes(epoch, value, colour = metric_cn, group = metric_cn)) +
      geom_line(linewidth = 0.85) +
      geom_point(size = 1.7) +
      scale_colour_manual(values = palette, name = "曲线") +
      labs(title = paste0(fam, "｜", exp, " 训练集/验证集 Loss 曲线"),
           subtitle = "用于观察图像模型收敛与过拟合趋势",
           x = "训练轮次", y = "Loss", caption = "数据来源：epoch_metrics / history")
    save_png(p_loss, paste0(stem, "_训练集_验证集_Loss_曲线"), width = 7.5, height = 4.8)

    acc_long <- hist |>
      select(epoch, train_accuracy, val_accuracy) |>
      pivot_longer(c(train_accuracy, val_accuracy), names_to = "metric", values_to = "value") |>
      mutate(metric_cn = recode(metric, train_accuracy = "训练集Accuracy", val_accuracy = "验证集Accuracy"))
    p_acc <- ggplot(acc_long, aes(epoch, value, colour = metric_cn, group = metric_cn)) +
      geom_line(linewidth = 0.85) +
      geom_point(size = 1.7) +
      scale_colour_manual(values = palette, name = "曲线") +
      scale_y_continuous(labels = percent_format(accuracy = 1), limits = c(0, 1), expand = expansion(mult = c(0, 0.02))) +
      labs(title = paste0(fam, "｜", exp, " 训练集/验证集 Accuracy 曲线"),
           subtitle = "训练集与验证集准确率随 epoch 变化",
           x = "训练轮次", y = "Accuracy", caption = "数据来源：epoch_metrics / history")
    save_png(p_acc, paste0(stem, "_训练集_验证集_Accuracy_曲线"), width = 7.5, height = 4.8)
  }

  one_conf <- conf_df |> filter(family == fam, experiment == exp, source == row$source)
  if (nrow(one_conf) > 0) {
    heat <- one_conf |>
      mutate(true_cn = factor(true_cn, levels = unname(labels_cn[labels_en])),
             pred_cn = factor(pred_cn, levels = unname(labels_cn[labels_en])),
             label = if_else(count > 0, paste0(count, "\n", percent(row_rate, accuracy = 1)), ""))
    p_heat <- ggplot(heat, aes(pred_cn, true_cn, fill = row_rate)) +
      geom_tile(colour = "white", linewidth = 0.5) +
      geom_text(aes(label = label), size = 3.05, lineheight = 0.88) +
      scale_fill_gradient(low = "#F3F7FA", high = "#2F6F9F", labels = percent_format(accuracy = 1), name = "行归一化比例") +
      coord_fixed() +
      labs(title = paste0(fam, "｜", exp, " 七类情绪混淆矩阵"),
           subtitle = "行表示真实类别，列表示预测类别；单元格为数量与行内比例",
           x = "预测情绪", y = "真实情绪", caption = "数据来源：测试集混淆矩阵") +
      theme(axis.text.x = element_text(angle = 30, hjust = 1), panel.grid = element_blank())
    save_png(p_heat, paste0(stem, "_混淆矩阵"), width = 7.4, height = 6.2)

    fail <- failure_df |> filter(family == fam, experiment == exp)
    if (nrow(fail) > 0) {
      err <- fail |> count(pair, sort = TRUE) |> slice_head(n = 12) |> mutate(pair = factor(pair, levels = rev(pair)))
      subtitle <- "按真实失败样例统计的高频误判路径"
      caption <- "数据来源：failure_cases.csv"
    } else {
      err <- one_conf |> filter(true_class != pred_class, count > 0) |>
        transmute(pair = paste0(true_cn, " → ", pred_cn), n = count) |>
        arrange(desc(n)) |> slice_head(n = 12) |> mutate(pair = factor(pair, levels = rev(pair)))
      subtitle <- "按混淆矩阵统计的高频误判路径"
      caption <- "说明：该实验未保存逐条失败样例，本图使用混淆矩阵计数"
    }
    if (nrow(err) > 0) {
      p_err <- ggplot(err, aes(pair, n)) +
        geom_col(width = 0.7, fill = "#D27B53") +
        coord_flip() +
        labs(title = paste0(fam, "｜", exp, " 错误样例分析"),
             subtitle = subtitle,
             x = "真实类别 → 预测类别", y = "错误样例数", caption = caption)
      save_png(p_err, paste0(stem, "_错误样例分析图"), width = 8.2, height = 5.3)
    }
  }
}

message("完成：", nrow(summary_df), " 个图像模型结果；输出目录 PNG 总数：", length(list.files(out_dir, pattern = "\\.png$")))
