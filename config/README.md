# config

完整说明见 [doc/readme/config_readme.md](../doc/readme/config_readme.md)。

```text
config/
├── data/          rail_data、rail_seg（旧）、clop_seg（现役）、clop_seg_bbox、rail_roi
├── model/         yolo26m-seg、yolo26m-seg-p2
├── train/         train_rail_seg、rail_online_augmentation、gas_rail、gas_rail_dice
├── predict/       rail_predict、rail_predict_nmsfree、sam3_predict
├── eval/          seg_postprocess、postprocess_ablation、postprocess_matching、rail_match_gt、experiment_report
└── experiments/   批量清单与完整实验配置
```

代码里用 `config_path("文件名")` 解析配置，兼容旧的平铺路径。
