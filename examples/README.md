# 示例工作流

每个文件都能**独立运行**，用来演示本节点集的接线方式。
全部已实测跑通（2026-10）。

---

## 01_最简生图_角色与LoRA.json

**演示**：角色/动作选择器 → LoRA 堆栈 → 提示词 → 采样 → 保存

```
CheckpointLoader ─┬─ model ──> LoRA堆栈 ── model ──> KSampler
                  ├─ clip  ──> LoRA堆栈 ── clip  ──> CLIPTextEncode(正/负)
                  └─ vae ────────────────────────> VAEDecode
角色/动作选择器 ─┬─ prompt    ──> CLIPTextEncode.text
                 └─ lora_spec ──> LoRA堆栈.lora_spec
```

**关键点**：选择器输出两个字符串，`prompt` 要接文本节点，`lora_spec` 要接 LoRA 堆栈。
LoRA 堆栈的 model 和 clip **两个输出都要往下接**，漏接 clip 会导致提示词用不到 LoRA。

**依赖**：主模型。需要至少一个角色表里的 LoRA 文件存在。

---

## 02_场景标签提取.json

**演示**：参考图 → WD tagger 反推 → 场景标签过滤 → 双预览对比

```
LoadImage ──> WDTimmTagger ──┬──> IllustriousSceneTags
                             └──> PreviewAny（原始全部标签）
                                   IllustriousSceneTags ──> PreviewAny（过滤后）
```

**关键点**：白名单按**类别**过滤，不维护黑名单。
内置强制排除角色名 / 动物耳尾 / 身体部件 / 头发 / 脸 / 服装 / 配饰。

两个预览一起看能定位问题：
- 原始有、过滤没有 → 类别不在白名单，改 `include_categories`
- 原始也没有 → tagger 没认出来，降 `general_threshold`

**依赖**：`ComfyUI-WD-Timm-Tagger` +
`models/wd_taggers/wd-eva02-large-tagger-v3/`（model.safetensors + selected_tags.csv）

---

## 03_场景条件拼接.json

**演示**：把一段场景提示词编码后拼进 conditioning，带真正的开关

```
CheckpointLoader ─┬─ clip ──> CLIPTextEncode(正) ── conditioning ──┐
                  │                                                ├─> SceneConditioning ──> KSampler.positive
                  └─ clip ─────────────────────────────────────────┘
```

**关键点**：`enable=False` 时**原样透传**。
不要用 `ConditioningConcat` 代替——它即使拼空字符串也会塞进 BOS/EOS 两个 token，
关闭后出图会和「完全没接这条线」不一样。

**依赖**：无额外依赖。

---

## 04_手部检测双重约束.json

**演示**：bbox 出框 + segm 验证，覆盖率不足直接否决

```
LoadImage ──┬──> DetectGuard.image
            ├──> BboxDetectorSEGS.image
hand_yolov8n(BBOX) ──> DetectGuard.bbox_detector
PitHandDetailer(SEGM) ──> DetectGuard.segm_detector
DetectGuard ──> BboxDetectorSEGS ──> MaskBatch ──> MaskToImage ──> Preview
```

**关键点**：必须用**两个独立模型**。
同一个 `-seg` 模型的框和掩码来自同一个预测头，重叠率恒为 ~1，校验等于没做。

判定看框内被分割掩码覆盖的比例：真手通常 40~70%，误检接近 0%。
输出仍是 `BBOX_DETECTOR`，所以重绘区域保持 bbox 矩形。

日志会逐个打印：
```
[Illustrious-GUARD] bbox(543,992,645,1221) 置信 0.80  分割覆盖 54.1%  PASS
[Illustrious-GUARD] 候选 1 个 -> 通过 1 个（阈值 覆盖≥15%）
```

**依赖**：`comfyui-impact-pack` + `bbox/hand_yolov8n.pt` + `segm/PitHandDetailer-v2-Test-v9c.pt`

---

## 05_眉眼带V10检测器.json

**演示**：FV 人脸分割 + 眼点 → 眉眼带蒙版

```
LoadImage ──┬──> V10检测器.image
            └──> ForbiddenVision 分割 ── face_mask ──> V10检测器.face_mask
face_yolov8s(BBOX) ──> V10检测器.bbox_detector
V10检测器 ──┬── preview    ──> PreviewImage
            └── brow_mask  ──> MaskToImage ──> PreviewImage
```

**关键点**：**FV 只提供尺寸，不提供角度**。
FV 蒙版近似正圆（实测宽高比 0.82~1.20），二阶矩主轴没有解剖意义，
所以角度一律来自眼点（Eyeful → openpose 眼点），绝不从蒙版主轴取。

`face_mask` 是可选输入：不接就退回 SAM，FV 失败也会自动回退。

**依赖**：`ComfyUI-Forbidden-Vision` + `comfyui-impact-pack` + `bbox/face_yolov8s.pt`

---

## 06_差分拼版与保存.json

**演示**：多张差分 → 拼版 → 分别保存「选中的」和「全部」

```
LoadImage ×2 ──> ImageBatch ──┬──> ExpressionSheet.vars
                              └──> SaveImageEx(all)
LoadImage(主图) ──> ExpressionSheet.main_image
ExpressionSheet ── selected ──> SaveImageEx(selected)
```

**关键点**：
- `main_image` 是必填，一般接修好的成品图
- 跑完后**在画布上点选**想要的那一格，再运行一次，只有下游会重跑
- 拼版内部预览图最大边压到 1280，所以喂 7MP 超分图不会更清楚，只会更慢
  想超分请放在保存分支，不要接在拼版前面

### 关于「切换工作流之后还在不在」

| 需求 | 能不能 | 说明 |
|---|---|---|
| 切回来还能看到拼版 | ✅ | 素材存到 `temp` 目录，`properties` 只留文件名清单，加载时按 URL 取回 |
| 切回来还能点、点了立刻换图 | ✅ | 存的是拆开的素材（主图 / 各差分 / 小图 / 布局） |
| 点了**不运行**就直接落盘 | ❌ | ComfyUI 的输出必须来自一次执行 |

前两项已实现：主图和差分都按 1536 长边（约 1.4 MP）存，切换差分时清晰度和主图一致；
`properties` 里只有约 0.6 KB 的清单，工作流体积几乎不变。
`temp` 被清理后（通常重启 ComfyUI）快照失效，跑一次即可恢复。
第三项是执行模型决定的，详见根目录 README 的「表情差分拼版」一节。

首次拖入节点、还没跑过时会显示**示意布局**（左主图 / 右一列 5 格、一格黄框），
不是空白，也不是一行「等待运行」。

**依赖**：无额外依赖。

---

## 试用建议

1. 先跑 **03**（最快，无依赖）确认节点能正常加载
2. 想看提示词节点怎么接 → **01**
3. 想做手部修复防误检 → **04**
4. 想做表情差分 → **06**，再参考主工作流的完整链路
5. 想复刻参考图的场景/色调 → **02** 起步

所有示例里的 `LoadImage` 需要换成你自己的图（ComfyUI 从 `input/` 目录读）。
