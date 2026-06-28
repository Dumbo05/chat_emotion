# SIMSAN network architecture schematic
# Backend: R grid/grDevices only. No Python rendering.

out_dir <- file.path('outputs', 'speech', 'figures')
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

library(grid)

fig_w <- 10.8
fig_h <- 6.2

palette <- list(
  bg = '#FFFFFF',
  ink = '#1F2933',
  muted = '#52606D',
  input = '#E8F1FA',
  feature = '#E6F4EA',
  conv = '#FFF4D6',
  encoder = '#F3E8FF',
  temporal = '#E6F7FF',
  pool = '#FFE8E8',
  head = '#EAECEF',
  accent = '#2F80ED',
  danger = '#C0392B'
)

push_page <- function() {
  grid.newpage()
  grid.rect(gp = gpar(fill = palette$bg, col = NA))
  grid.text('SIMSAN speech emotion network', x = unit(0.035, 'npc'), y = unit(0.955, 'npc'),
            just = c('left', 'center'), gp = gpar(fontsize = 15, fontface = 'bold', col = palette$ink, fontfamily = 'Microsoft YaHei'))
  grid.text('Dual normalized Log-Mel views -> multi-scale spectral encoder -> temporal modeling -> emotion embedding with speaker-adversarial training',
            x = unit(0.035, 'npc'), y = unit(0.915, 'npc'), just = c('left', 'center'),
            gp = gpar(fontsize = 8.6, col = palette$muted, fontfamily = 'Microsoft YaHei'))
}

box <- function(x, y, w, h, label, fill, fontsize = 8.2, col = palette$ink, lwd = 0.8, fontface = 'plain') {
  grid.roundrect(x = unit(x, 'npc'), y = unit(y, 'npc'), width = unit(w, 'npc'), height = unit(h, 'npc'),
                 r = unit(0.014, 'npc'), gp = gpar(fill = fill, col = col, lwd = lwd))
  grid.text(label, x = unit(x, 'npc'), y = unit(y, 'npc'),
            gp = gpar(fontsize = fontsize, col = col, fontfamily = 'Microsoft YaHei', fontface = fontface),
            just = 'center')
}

draw_arrow <- function(x1, y1, x2, y2, col = palette$muted, lwd = 1.0, curved = FALSE) {
  if (curved) {
    grid.curve(x1 = unit(x1, 'npc'), y1 = unit(y1, 'npc'), x2 = unit(x2, 'npc'), y2 = unit(y2, 'npc'),
               curvature = 0.12, angle = 90, ncp = 10,
               arrow = grid::arrow(length = unit(0.018, 'npc'), type = 'closed'),
               gp = gpar(col = col, lwd = lwd))
  } else {
    grid.segments(x0 = unit(x1, 'npc'), y0 = unit(y1, 'npc'), x1 = unit(x2, 'npc'), y1 = unit(y2, 'npc'),
                  arrow = grid::arrow(length = unit(0.018, 'npc'), type = 'closed'),
                  gp = gpar(col = col, lwd = lwd))
  }
}

small_label <- function(x, y, text, col = palette$muted, size = 6.8, just = 'center') {
  grid.text(text, x = unit(x, 'npc'), y = unit(y, 'npc'), just = just,
            gp = gpar(fontsize = size, col = col, fontfamily = 'Microsoft YaHei'))
}

make_plot <- function() {
  push_page()

  # Row guides
  y_main <- 0.66
  y_split_top <- 0.79
  y_split_bot <- 0.53
  y_head_top <- 0.34
  y_head_bot <- 0.17

  # Input and preprocessing
  box(0.075, y_main, 0.115, 0.105, 'Raw audio\nvariable length', palette$input, 8.0, fontface = 'bold')
  box(0.205, y_main, 0.125, 0.105, '4 s segment\nloop pad / center crop', palette$input, 7.5)
  box(0.345, y_main, 0.13, 0.105, 'Log-Mel\n64 bins, 512 FFT\n400/160 samples', palette$feature, 7.1)
  draw_arrow(0.132, y_main, 0.142, y_main)
  draw_arrow(0.268, y_main, 0.280, y_main)

  # Dual views
  box(0.505, y_split_top, 0.145, 0.095, 'Global z-score\nkeeps spectral shape', palette$feature, 7.2)
  box(0.505, y_split_bot, 0.145, 0.095, 'Per-band norm\nreduces timbre / channel', palette$feature, 7.2)
  draw_arrow(0.410, y_main + 0.025, 0.432, y_split_top - 0.010, curved = TRUE)
  draw_arrow(0.410, y_main - 0.025, 0.432, y_split_bot + 0.010, curved = TRUE)

  box(0.665, y_main, 0.13, 0.105, 'Stacked\n2-channel\nspectrogram', '#EAF7F0', 7.4)
  draw_arrow(0.578, y_split_top, 0.600, y_main + 0.030, curved = TRUE)
  draw_arrow(0.578, y_split_bot, 0.600, y_main - 0.030, curved = TRUE)

  # Multi-scale convolution branches
  branch_x <- c(0.805, 0.805, 0.805)
  branch_y <- c(0.805, 0.665, 0.525)
  branch_lab <- c('Conv branch\n3 x 3', 'Conv branch\n5 x 3', 'Conv branch\n7 x 3')
  for (i in 1:3) {
    box(branch_x[i], branch_y[i], 0.115, 0.080, branch_lab[i], palette$conv, 7.2)
    draw_arrow(0.730, y_main, 0.748, branch_y[i], curved = TRUE)
  }
  box(0.925, y_main, 0.105, 0.105, 'Concat\n48 channels', palette$conv, 7.5, fontface = 'bold')
  for (i in 1:3) draw_arrow(0.862, branch_y[i], 0.872, y_main, curved = TRUE)

  # Encoder row
  box(0.150, 0.385, 0.190, 0.105, 'Depthwise separable\nSE residual blocks\n64 -> 96 -> 128 -> 160', palette$encoder, 7.2, fontface = 'bold')
  box(0.360, 0.385, 0.125, 0.105, 'Frequency\naggregation', palette$encoder, 7.5)
  box(0.540, 0.385, 0.190, 0.105, 'Dilated temporal\nresidual blocks\nd = 1, 2, 4, 8', palette$temporal, 7.4, fontface = 'bold')
  box(0.745, 0.385, 0.145, 0.105, 'Attentive\nstatistics pooling\nmean + std', palette$pool, 7.2, fontface = 'bold')
  box(0.910, 0.385, 0.105, 0.105, '192-D\nembedding', palette$pool, 7.8)

  draw_arrow(0.925, y_main - 0.055, 0.245, 0.438, curved = TRUE)
  draw_arrow(0.245, 0.385, 0.298, 0.385)
  draw_arrow(0.422, 0.385, 0.445, 0.385)
  draw_arrow(0.635, 0.385, 0.672, 0.385)
  draw_arrow(0.817, 0.385, 0.857, 0.385)

  # Heads and losses
  box(0.710, y_head_top, 0.145, 0.095, 'Emotion head\n7-class logits', palette$head, 7.5, fontface = 'bold')
  box(0.710, y_head_bot, 0.145, 0.095, 'Speaker head\nvia GRL', '#FDEDEC', 7.5, col = palette$danger, fontface = 'bold')
  box(0.885, y_head_top, 0.150, 0.095, 'Emotion loss\nlabel smoothing 0.05', '#EEF2FF', 7.0)
  box(0.885, y_head_bot, 0.150, 0.095, 'Adversarial speaker loss\n0.15 x L_speaker\nGRL <= 0.20', '#FDEDEC', 6.8, col = palette$danger)
  draw_arrow(0.910, 0.333, 0.780, y_head_top + 0.050, curved = TRUE)
  draw_arrow(0.910, 0.333, 0.780, y_head_bot + 0.050, curved = TRUE, col = palette$danger)
  draw_arrow(0.782, y_head_top, 0.810, y_head_top)
  draw_arrow(0.782, y_head_bot, 0.810, y_head_bot, col = palette$danger)

  # Training notes
  box(0.185, 0.145, 0.245, 0.105, 'Training regularization\nWeightedRandomSampler; SpecAugment\ntime/frequency mask; freq shift; noise', '#F8FAFC', 6.8)
  box(0.470, 0.145, 0.205, 0.105, 'Optimization\nAdamW 3e-4; batch 48\ncosine decay; early stop', '#F8FAFC', 6.8)
  box(0.150, 0.060, 0.230, 0.052, 'Final inference: dual-checkpoint ensemble\nsimsan_best 0.76 + simsan_balanced 0.24', '#F8FAFC', 6.5)

  # Panel letter and caption-like note
  grid.text('a', x = unit(0.018, 'npc'), y = unit(0.975, 'npc'),
            gp = gpar(fontsize = 16, fontface = 'bold', col = palette$ink, fontfamily = 'Arial'))
  small_label(0.965, 0.045, 'Abbreviations: SE, squeeze-and-excitation; GRL, gradient reversal layer.', just = c('right', 'center'), size = 6.2)
}

save_all <- function(stem) {
  svg_file <- file.path(out_dir, paste0(stem, '.svg'))
  pdf_file <- file.path(out_dir, paste0(stem, '.pdf'))
  png_file <- file.path(out_dir, paste0(stem, '.png'))

  grDevices::svg(svg_file, width = fig_w, height = fig_h, family = 'Microsoft YaHei')
  make_plot()
  dev.off()

  grDevices::cairo_pdf(pdf_file, width = fig_w, height = fig_h, family = 'Microsoft YaHei')
  make_plot()
  dev.off()

  grDevices::png(png_file, width = fig_w, height = fig_h, units = 'in', res = 300, type = 'cairo')
  make_plot()
  dev.off()

  info <- file.info(c(svg_file, pdf_file, png_file))
  print(data.frame(file = rownames(info), bytes = info$size, row.names = NULL))
}

save_all('simsan_network_architecture')
