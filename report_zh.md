# Wildvideo 问题排查与实验归档（2026-04-18）

## 目标回顾
- 目标流程: 先 ProPainter 再仅对难生成区域做 Diffusion。
- 重点问题: hard mask 区域过大导致 diffusion 过度使用；关键帧出现描边/人形残影。

## 关键结论（当前）
- 已确认并修复“diffusion 过度使用”问题：加入 target 占比上限与难点像素优先。
- 默认建议配置为 sparse ratio=0.35（run 20260418_133713）。
- 边缘卷积（E1/E2）与边缘颜色匹配（E3）在关键帧稳定性上未优于默认。
- 代码层保留了可开关实验能力，但默认应关闭 edgeconv/edgecolor。

## 主要实验时间线
- 20260418_112108: wild_v1_initial_person_camera | note=初版wild配置（删人+相机） | outside=0.014214 | seam=0.530114 | leakage=0.011320 | temporal=1.113855 | score=-0.6581825961585189
- 20260418_113711: wild_v2_aggressive | note=更激进参数（质量回退） | outside=0.032831 | seam=0.896476 | leakage=0.024093 | temporal=1.137309 | score=-1.0666159171838303
- 20260418_121206: wild_new_video_base | note=更换视频后基线 | outside=0.036065 | seam=1.000265 | leakage=0.025131 | temporal=5.877465 | score=-5.913795837002089
- 20260418_122740: anti_halo_phaseA | note=A: 关confidence gating + 提高融合 | outside=0.051481 | seam=0.961423 | leakage=0.033291 | temporal=5.597608 | score=-5.610512010544271
- 20260418_123709: anti_halo_phaseB | note=B: 在A基础收缩final_dilation | outside=0.051909 | seam=1.025029 | leakage=0.034405 | temporal=5.209005 | score=-5.28594266365721
- 20260418_125329: anti_halo_phaseC | note=C: 恢复gating的保守版 | outside=0.040944 | seam=0.921691 | leakage=0.030538 | temporal=5.508830 | score=-5.471465306460484
- 20260418_130347: code_fix_core_alpha | note=代码修复: 核心全替换+边缘羽化 | outside=0.019157 | seam=0.964973 | leakage=0.014305 | temporal=5.362205 | score=-5.346334552052159
- 20260418_133713: sparse_ratio_035 | note=稀疏diffusion 0.35（当前默认） | outside=0.009117 | seam=0.650852 | leakage=0.009180 | temporal=4.593767 | score=-4.253735822361952
- 20260418_135006: sparse_ratio_025 | note=更克制0.25对照 | outside=0.008631 | seam=0.629816 | leakage=0.008897 | temporal=4.448138 | score=-4.086585333441361
- 20260418_141152: edgeconv_e1 | note=边缘卷积E1 | outside=0.009164 | seam=0.661020 | leakage=0.009047 | temporal=4.823720 | score=-4.493904345698183
- 20260418_142630: edgeconv_e2 | note=边缘卷积E2轻量 | outside=0.008991 | seam=0.649229 | leakage=0.009124 | temporal=4.431668 | score=-4.089887511666694
- 20260418_143539: edgecolor_e3 | note=边缘颜色匹配E3 | outside=0.008652 | seam=0.626917 | leakage=0.008858 | temporal=4.709039 | score=-4.344608061102325

## 针对“过度使用 diffusion”的证据
- 修复前(130347): target/hard 均值=0.915, p90=0.994
- sparse 0.35(133713): 均值=0.350, p90=0.350
- sparse 0.25(135006): 均值=0.250, p90=0.250

## 遗留问题
- 在关键帧 000011/000023 附近，仍可能出现轻微边缘轮廓/亮边。
- 时序指标仍高（temporal_instability_ratio_mean 显著大于 1），整体 gate 仍未通过。
- edgeconv/edgecolor 方案存在帧间不稳定和局部雾化风险。

## 当前建议（冻结）
- 默认运行配置使用 sparse=0.35。
- 关闭 enable_edge_convolution_balance。
- 关闭 enable_edge_color_match。
- 若后续继续优化，优先做掩码几何边界优化（非像素后处理）。

## 关联文件
- 指标汇总: analysis_20260418/attempt_metrics.json
- 指标表格: analysis_20260418/attempt_metrics.csv
- 本报告: analysis_20260418/report_zh.md
- 关键代码: src/cv_project/inpainting/diffusion_target.py, src/cv_project/inpainting/freeinpaint_refiner.py
- 默认配置: configs/part3_wildvideo_remove_person_camera.yaml
