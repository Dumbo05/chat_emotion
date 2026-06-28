# SIMSAN 中文网络结构示意图（R grid 版）
# 仅使用 R/grid/grDevices 绘图，不使用 Python 图形后端。

out_dir <- file.path('outputs', 'speech', 'figures')
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

library(grid)

fig_w <- 14.5
fig_h <- 8.2
font_cn <- 'Microsoft YaHei'
font_en <- 'Arial'

pal <- list(
  bg = '#FFFFFF',
  ink = '#1F2933',
  muted = '#52606D',
  line = '#6B7280',
  input = '#EAF2FF',
  spec = '#EAF7F0',
  branch = '#FFF4D6',
  encoder = '#F2EAFE',
  temporal = '#E7F7FF',
  pool = '#FFEDED',
  head = '#EEF2F7',
  adv = '#FDECEC',
  accent = '#2F80ED',
  red = '#B42318'
)

open_device <- function(file, type) {
  if (type == 'svg') grDevices::svg(file, width = fig_w, height = fig_h, family = font_cn)
  if (type == 'pdf') grDevices::cairo_pdf(file, width = fig_w, height = fig_h, family = font_cn)
  if (type == 'png') grDevices::png(file, width = fig_w, height = fig_h, units = 'in', res = 320, type = 'cairo')
}

box <- function(x, y, w, h, label, fill, border = pal$ink, fs = 9.5, face = 'plain', r = 0.012) {
  grid.roundrect(unit(x, 'npc'), unit(y, 'npc'), unit(w, 'npc'), unit(h, 'npc'),
                 r = unit(r, 'npc'), gp = gpar(fill = fill, col = border, lwd = 0.75))
  grid.text(label, unit(x, 'npc'), unit(y, 'npc'),
            gp = gpar(fontsize = fs, fontface = face, col = border, fontfamily = font_cn),
            just = 'center')
}

label <- function(x, y, text, fs = 8.2, col = pal$muted, just = 'center', face = 'plain') {
  grid.text(text, unit(x, 'npc'), unit(y, 'npc'), just = just,
            gp = gpar(fontsize = fs, col = col, fontface = face, fontfamily = font_cn))
}

arr <- function(x1, y1, x2, y2, col = pal$line, lwd = 1.05, curve = 0) {
  if (curve == 0) {
    grid.segments(unit(x1, 'npc'), unit(y1, 'npc'), unit(x2, 'npc'), unit(y2, 'npc'),
                  arrow = grid::arrow(length = unit(0.013, 'npc'), type = 'closed'),
                  gp = gpar(col = col, lwd = lwd))
  } else {
    grid.curve(unit(x1, 'npc'), unit(y1, 'npc'), unit(x2, 'npc'), unit(y2, 'npc'),
               curvature = curve, angle = 90, ncp = 12,
               arrow = grid::arrow(length = unit(0.013, 'npc'), type = 'closed'),
               gp = gpar(col = col, lwd = lwd))
  }
}

region <- function(x, y, w, h, title) {
  grid.roundrect(unit(x, 'npc'), unit(y, 'npc'), unit(w, 'npc'), unit(h, 'npc'),
                 r = unit(0.012, 'npc'), gp = gpar(fill = NA, col = '#CBD5E1', lwd = 0.55, lty = 'dashed'))
  label(x - w/2 + 0.010, y + h/2 - 0.020, title, fs = 8.2, col = pal$muted, just = c('left','center'), face = 'bold')
}

make_plot <- function() {
  grid.newpage()
  grid.rect(gp = gpar(fill = pal$bg, col = NA))

  grid.text('SIMSAN 语音情绪识别网络结构', unit(0.035, 'npc'), unit(0.965, 'npc'),
            just = c('left','center'), gp = gpar(fontsize = 18, fontface = 'bold', col = pal$ink, fontfamily = font_cn))
  grid.text('双归一化声谱视图、多尺度频谱卷积、SE 残差编码、膨胀时序建模与注意力统计池化共同形成说话人鲁棒的情绪嵌入。',
            unit(0.035, 'npc'), unit(0.925, 'npc'), just = c('left','center'),
            gp = gpar(fontsize = 10, col = pal$muted, fontfamily = font_cn))

  # 区域背景
  region(0.155, 0.720, 0.245, 0.275, 'A 输入与声谱')
  region(0.405, 0.720, 0.225, 0.275, 'B 双视图')
  region(0.660, 0.720, 0.300, 0.275, 'C 多尺度频谱编码')
  region(0.365, 0.405, 0.390, 0.245, 'D 时频表征压缩')
  region(0.760, 0.405, 0.320, 0.245, 'E 输出头与训练约束')

  # A 输入
  box(0.075, 0.720, 0.100, 0.092, '原始音频\n变长波形', pal$input, fs = 9.0, face = 'bold')
  box(0.200, 0.720, 0.120, 0.092, '统一 4 s\n短音频循环补齐\n长音频中心截取', pal$input, fs = 7.8)
  box(0.330, 0.720, 0.115, 0.092, 'Log-Mel 声谱\n64 Mel bins\n512 FFT', pal$spec, fs = 7.8, face = 'bold')
  arr(0.125, 0.720, 0.140, 0.720)
  arr(0.260, 0.720, 0.272, 0.720)

  # B 双视图
  box(0.455, 0.790, 0.135, 0.080, '全局均值方差\n归一化视图\n保留整体谱形', pal$spec, fs = 7.6)
  box(0.455, 0.650, 0.135, 0.080, '逐频带归一化\n削弱音色与通道\n稳定差异', pal$spec, fs = 7.6)
  arr(0.385, 0.742, 0.390, 0.790, curve = 0.18)
  arr(0.385, 0.698, 0.390, 0.650, curve = -0.18)
  box(0.580, 0.720, 0.105, 0.086, '双通道\n声谱堆叠', '#EAF7F0', fs = 8.3, face = 'bold')
  arr(0.522, 0.790, 0.528, 0.744, curve = -0.15)
  arr(0.522, 0.650, 0.528, 0.696, curve = 0.15)

  # C 多尺度分支：上下分开，避免堆叠
  box(0.710, 0.810, 0.112, 0.068, '3×3 卷积分支\n窄频率邻域', pal$branch, fs = 7.4)
  box(0.710, 0.720, 0.112, 0.068, '5×3 卷积分支\n中等邻域', pal$branch, fs = 7.4)
  box(0.710, 0.630, 0.112, 0.068, '7×3 卷积分支\n宽频率邻域', pal$branch, fs = 7.4)
  box(0.850, 0.720, 0.112, 0.086, '拼接输出\n48 通道', pal$branch, fs = 8.3, face = 'bold')
  arr(0.632, 0.720, 0.654, 0.810, curve = 0.20)
  arr(0.632, 0.720, 0.654, 0.720)
  arr(0.632, 0.720, 0.654, 0.630, curve = -0.20)
  arr(0.766, 0.810, 0.794, 0.750, curve = -0.12)
  arr(0.766, 0.720, 0.794, 0.720)
  arr(0.766, 0.630, 0.794, 0.690, curve = 0.12)

  # D 编码主干
  box(0.165, 0.405, 0.155, 0.088, '深度可分离\nSE 残差块\n64→96→128→160', pal$encoder, fs = 7.7, face = 'bold')
  box(0.345, 0.405, 0.105, 0.088, '频率维\n聚合', pal$encoder, fs = 8.2)
  box(0.510, 0.405, 0.150, 0.088, '膨胀时序\n残差块\nd=1,2,4,8', pal$temporal, fs = 7.9, face = 'bold')
  box(0.690, 0.405, 0.145, 0.088, '注意力统计池化\n加权均值 + 标准差', pal$pool, fs = 7.8, face = 'bold')
  box(0.850, 0.405, 0.100, 0.088, '192 维\n情绪嵌入', pal$pool, fs = 8.4)
  arr(0.850, 0.670, 0.240, 0.450, curve = -0.12)
  arr(0.243, 0.405, 0.292, 0.405)
  arr(0.398, 0.405, 0.435, 0.405)
  arr(0.585, 0.405, 0.618, 0.405)
  arr(0.762, 0.405, 0.800, 0.405)

  # E 输出头
  box(0.690, 0.230, 0.135, 0.080, '情绪分类头\n7 类 logits', pal$head, fs = 8.4, face = 'bold')
  box(0.690, 0.115, 0.135, 0.080, '说话人分类头\n梯度反转层 GRL', pal$adv, border = pal$red, fs = 8.1, face = 'bold')
  box(0.865, 0.230, 0.145, 0.080, '情绪损失\n标签平滑 0.05', '#EEF2FF', fs = 8.0)
  box(0.865, 0.115, 0.145, 0.080, '说话人对抗损失\n0.15 × L_speaker\nGRL ≤ 0.20', pal$adv, border = pal$red, fs = 7.4)
  arr(0.850, 0.360, 0.690, 0.270, curve = 0.12)
  arr(0.850, 0.360, 0.690, 0.155, curve = 0.20, col = pal$red)
  arr(0.758, 0.230, 0.792, 0.230)
  arr(0.758, 0.115, 0.792, 0.115, col = pal$red)

  # 训练策略，放在底部单独说明，避免压主流程
  box(0.215, 0.110, 0.230, 0.095, '训练策略\nWeightedRandomSampler\n时间/频率遮挡、频移、噪声', '#F8FAFC', fs = 7.5)
  box(0.460, 0.110, 0.200, 0.095, '优化设置\nAdamW 3×10⁻⁴\nbatch 48，早停', '#F8FAFC', fs = 7.5)
  box(0.330, 0.035, 0.330, 0.052, '最终推理：双检查点集成  simsan_best ×0.76 + simsan_balanced ×0.24', '#F8FAFC', fs = 7.4, face = 'bold')

  grid.text('图注：SIMSAN 先把变长语音转为 4 s Log-Mel 声谱，再通过双归一化视图削弱说话人和通道差异；多尺度卷积与 SE 残差块提取频谱线索，膨胀时序块和注意力统计池化形成句级嵌入；GRL 说话人头用于降低说话人泄漏。',
            unit(0.035, 'npc'), unit(0.010, 'npc'), just = c('left','bottom'),
            gp = gpar(fontsize = 7.3, col = pal$muted, fontfamily = font_cn))
}

save_all <- function(stem) {
  files <- list(
    svg = file.path(out_dir, paste0(stem, '.svg')),
    pdf = file.path(out_dir, paste0(stem, '.pdf')),
    png = file.path(out_dir, paste0(stem, '.png'))
  )
  for (type in names(files)) {
    open_device(files[[type]], type)
    make_plot()
    dev.off()
  }
  info <- file.info(unlist(files))
  print(data.frame(file = rownames(info), bytes = info$size, row.names = NULL))
}

save_all('simsan_network_architecture_cn_v2')
