# ComfyUI-Illustrious-Selector

> ComfyUI 自定义节点集，为 Illustrious / SDXL 系二次元动画风格的工作流
> 提供 **表情差分 + 手脸修复** 的配套工具。
>
> 核心追求一句话：**切换表情差分时，脸以外的地方一个像素都不许动。**
虽然最后没实现完全的不动……但是也很接近了。
---

---
## 这套节点解决的三个具体问题

### 1. 表情差分不该让头发跟着抖

用 bbox 做差分蒙版时，「框」的范围必然会碰到头发、耳朵以及之外的背景板。
改动范围一旦越界，几张差分放到一起就能看出头发、背景、发饰，
甚至是脖子和部分衣服在细微移动。

这里的做法是**分割驱动 + 眼点定向**：

- 蒙版 = 脸区 ∪ 眉眼带 ∪ 眼圆，上边界被超椭圆「天花板」裁掉（防止顺着刘海往上爬）
- 眉眼带的方向来自**眼点连线**，跟着头部倾斜转
- 人脸分割只提供尺寸，不提供角度（分割蒙版近似正圆，二阶矩主轴没有解剖意义）

没有眉毛的表情很生硬，所以眉毛是必须的，刘海也只能被带着一起动了

### 2. 手部 bbox 会框到膝盖

bbox检测器和segm边缘检测器都偶尔会把脚趾、头发、或者身体旁边的其它东西识别成手。
而修手是 denoise 很高的重绘，框错了就是直接毁画面。

**检测双重约束**节点：bbox 出框 + segm 分割验证，框内被分割掩码覆盖不足的直接否决。
判定看覆盖率（真手通常 40~70%，误检接近 0%），输出仍是 bbox，重绘区域不变。

> ⚠️ 注意：segm 只用来**验证**，不要拿它当重绘蒙版。
> 分割把手指边界绑得太紧时，重绘会没有足够空间修正手指形状，效果反而大打折扣。

### 3. 核心只有 PNG 保存，但 7MP 图一张 10MB

**保存 JPG/WebP** 节点补上 webp/jpg 保存。
7MP 图 webp q93 只要 1.2 MB，并且把工作流写进 EXIF —— 出图拖回 ComfyUI 画布能还原节点图。解决了表情差分多的情况下很占空间的问题

---

## 节点一览（23 个）

### 提示词

| 节点 | 说明 |
|---|---|
| **角色 / 动作选择器** | 角色表 + 动作表，输出 `prompt` 与 `lora_spec`，自动驱动 LoRA 加载 |
| **提示词显示** | 只读展示当前提示词 |
| **LoRA 堆栈** | 按 `lora_spec` 批量加载 LoRA，输出 model / clip |
| **场景标签过滤** | 把 Danbooru tag 按类别白名单过滤，默认只留场景 / 物品 / 姿势 |
| **场景提示词拼接** | 场景词编码后拼进 conditioning，带开关（关闭时原样透传） |

<details>
<summary><b>角色 / 动作选择器</b> —— 添加角色、收藏、覆写</summary>

**添加一个角色**

1. 填名字、角色提示词，并关联对应的 LoRA
2. 点保存
3. **刷新页面**

新角色会在刷新后出现在候选列表里。
刷新前搜索栏可能会卡顿一会儿，这是正常的 —— 数据正在重建。

**收藏**

点收藏时收藏的是「角色选择栏上的角色」，不是当前提示词。

**覆写（不影响原记录）**

- `character_prompt_override`：临时改这个角色的提示词
- `prefix` / `suffix`：在前后追加提示词

这两项都不写回角色表，下次选同一角色还是原始内容。
</details>

<details>
<summary><b>场景标签过滤</b> —— 怎么调白名单</summary>

节点参数分四层，从宽到严：

| 参数 | 作用 |
|---|---|
| `include_categories` | **白名单**。只保留命中了这些类别的标签 |
| `exclude_categories` | 在路由到白名单之前，先按类别剔除 |
| `extra_exclude` | 直接按标签名剔除（逗号分隔） |
| `max_tags` | 最终最多输出几个 |

类别名来自 `tag_category_v2.json`（11099 条标签 → 类别的映射）。
可用的类别有 `background` / `nature` / `location` / `setting` / `object` /
`props` / `furniture` / `plant` / `weather` / `pose` / `action` / `gesture` / `camera` 等，
默认白名单就是这些。

节点**内置了一套强制排除**（不写在参数里，也改不掉）：
角色名、动物耳尾、身体部件、头发、脸、服装、配饰、头饰、鞋、图案、颜色。
这些是「角色自身特征」，不该混进场景提示词。

**怎么判断调得对不对**

配合两个 `PreviewAny` 一起看：

| 现象 | 原因 | 怎么办 |
|---|---|---|
| 原始有、过滤没有 | 类别不在白名单 | 把那个类别加进 `include_categories` |
| 原始也没有 | tagger 没认出来 | 降低 `WDTimmTagger` 的 `general_threshold` |
| 过滤结果是空的 | `enable` 关着，或 `max_tags` 太小 | 打开 enable / 调大上限 |
</details>

### 检测

| 节点 | 说明 |
|---|---|
| **SAM/FV+眉眼带 V10** | 主检测器。人脸分割定边界、眼点定方向，输出差分用蒙版 |
| **检测双重约束** | bbox 出框 + segm 验证，覆盖率不足则否决，通常用在手部修复 |
| **SAM 眉眼带检测器** | 原始版，仅接 SAM。效果一般但可用，大角度侧脸易失效 |
| **Pose 可视化** / **Pose 可视化 V10** | 把 POSE_KEYPOINT 画到图上，用来核对检测质量 |
| **Mask 分色调试 V10** | 把脸 / 眉 / 眼三部分蒙版分色显示 |

<details>
<summary><b>SAM/FV+眉眼带 V10</b> —— 适用场景</summary>

对差分这类「主体一致性要求很高」的工作，效果我自己是比较满意的。

关键原则是**分割只提供尺寸，不提供角度**：

- 宽度取自人脸分割蒙版的展布
- 角度一律来自眼点（Eyeful 框 → openpose 眼点）
- 分割蒙版的二阶矩主轴**不参与角度计算**

`face_mask` 是可选输入：不接就退回 SAM，FV 失败也会自动回退。
</details>

### 图像

| 节点 | 说明 |
|---|---|
| **表情差分拼版** | 把 N 张差分拼成选图面板，输出选中项 |
| **差分开关** | 关闭时输出空白图，让下游跳过整条差分链 |
| **差分路由** | 图 + 表情 CLIP 成对输出（实验件） |
| **脸部裁切** | 按 FaceDetailer 的 mask 精确裁出脸部特写 |
| **脸部微调** | 交互调脸部框 |
| **保存 JPG/WebP** | 补上 webp/jpg 保存，可把工作流写进 EXIF |
| **种子枢纽** | 一处改 seed 同时控制任意多个采样器 |

<details>
<summary><b>表情差分拼版</b> —— 一些困难？</summary>


拼版节点最大的遗憾是没法单独输出选中的差分图

| 需求 | 能不能 | 做法 / 原因 |
|---|---|---|
| 切回来还能**看到**拼版 |  ✅  | 把素材压成小图存进节点 `properties`，加载时重建画面 |
| 切回来还能**点**，点了立刻换图 | ✅ | 存的不只是整张截图，而是**拆开的素材**：主图 + 每张差分 + 每张小图 + 布局矩形 |
| 点了**不运行**就直接写出文件 | ❌ | ComfyUI 的输出必须来自一次执行，节点拿不到「写文件」的能力 |

前两项已经实现。代价是工作流 JSON 里每个拼版节点涨 **120~200 KB**
（主图 640px、差分 384px、小图 192px，JPEG q0.72），不占显存、不影响出图速度。

全新拖进画布的节点还没跑过时，会显示一张**示意布局**（左主图 / 右一列 5 格、
其中一格黄框表示选中），而不是一行「等待运行」。

### 第三项为什么做不到

这不是本节点的实现问题，是 ComfyUI 的执行模型：

```
点选 只是改了一个参数（selected）
输出 selected_image、以及保存出来的文件，都是「那次执行」的产物
```

工作流必须从上游传信号到下游，节点才会工作 —— 没有执行就没有输出。
要让「点击」直接落盘，只有一条路：自建服务端路由，
把选中那张的 base64 POST 上去、后端直接写盘。

但那个代价不小：

- 绕过用户的**保存节点**，文件名 / 格式 / 保存目录都不受用户控制
- 画布上看得见的保存节点和这个隐式写盘会形成**两套真相**，别人接手会困惑
- 需要在服务端缓存本次运行的图，或让浏览器把几百 KB 的图再传回去

本包**不做**这件事。所以定位是：

> **切回来能看、能点、点了立刻看到那张图；落盘仍需一次运行。**

你可以选择……



**A. 提前设好再跑**

直接在 `selected` 参数里填 1~6，然后一次跑完。

**B. 干脆不用「选中」（最稳）**

把差分整图批次直接接到一个**保存 JPG/WebP** 节点上，
全部差分都存下来，自己在文件管理器里挑。
拼版只用来「看」，不参与输出逻辑。
</details>

<details>
<summary><b>差分开关</b> —— 只是个开关</summary>

是的，它就只有这一个作用：`enable=False` 时输出一张空白图，
让下游的 FaceDetailer 检测不到脸，从而整条差分链直接空转。

它不控制数量、不控制强度，也不参与蒙版计算。
放在差分支路的最前面，是为了「不想跑差分时一键关掉」，省掉差分采样。
</details>

### 工具

| 节点 | 一句话说明 |
|---|---|
| **归一化缩放** | 按像素预算缩放，保持长宽比 |
| **边缘选择器** | canny / lineart / softedge 三选一出边缘图 |
| **潜空间比例** | 按预设或参考图比例算 latent 尺寸比例，可以自定义像素量 |
| **Mask 叠加预览** | 蒙版叠在图上 |
| **差分蒙版钩子** | 差分链的 detailer hook |

<details>
<summary><b>边缘选择器</b> —— 切换模式后要换 ControlNet 模型</summary>

这个节点只负责**出边缘图**，不管 ControlNet 加载哪个模型。
模式切换后，下游 `ControlNetLoader` 里的模型也要跟着换，否则线条会发灰、发糊：

| mode | 边缘特征 | 建议的 ControlNet |
|---|---|---|
| `canny` | 硬边、粗细均匀 | canny 系，或 lineart-anime 也能吃 |
| `lineart` | 细线、保留更多内部结构 | **lineart 系**（不要用 canny 模型） |
| `softedge` | 柔和边缘、容错高 | softedge 系，或 lineart-anime |

节点会通过第二个输出 `controlnet_hint`（STRING）给出当前模式建议的模型名，
也会在服务端日志里打一行提示。可以把它接到 `PreviewAny` 上看。

`low_threshold` / `high_threshold` 只对 canny 模式有效。
</details>

<details>
<summary><b>潜空间比例</b> —— multiple of 要权衡</summary>

按预设或参考图比例算 latent 尺寸。
`multiple_of` 调小可以得到更精确的比例，但 SDXL 的「桶」设计对某些像素尺寸支持不好，
会出现明显的画质下降或构图偏移 —— 需要自己权衡。
一般 32 是保守值，64 更安全。
</details>

---

## 示例工作流

`examples/` 下有 6 个**能独立跑通**的最小工作流（我这里都运行过），
用来演示接线方式：

| 文件 | 演示 |
|---|---|
| `01_最简生图_角色与LoRA.json` | 选择器 → LoRA 堆栈 → 提示词 → 采样 → 保存 |
| `02_场景标签提取.json` | 参考图反推 → 类别白名单过滤 → 双预览对比 |
| `03_场景条件拼接.json` | 场景词编码后拼进 conditioning（开关能真正关闭） |
| `04_手部检测双重约束.json` | bbox 出框 + segm 验证 → 蒙版预览 |
| `05_眉眼带V10检测器.json` | FV 人脸分割 + 眼点 → 眉眼带蒙版 |
| `06_差分拼版与保存.json` | 多图拼版点选 → 分别保存选中 / 全部 |

建议先跑 **03**（无额外依赖）确认节点加载正常。
详见 [`examples/README.md`](examples/README.md)。

---

## 安装

```bash
cd ComfyUI/custom_nodes
git clone <this-repo> ComfyUI-Illustrious-Selector
```

**无额外 Python 依赖**（只用 ComfyUI 自带的 torch / numpy / Pillow）。

### 可选的外部插件

节点本身不强依赖任何东西，但示例会用到：

| 插件 | 用在哪 |
|---|---|
| `comfyui-impact-pack` | FaceDetailer / SAM / SEGS（示例 04、05） |
| `ComfyUI-Forbidden-Vision` | V10 检测器的人脸分割输入（示例 05） |
| `ComfyUI-WD-Timm-Tagger` | 场景标签来源（示例 02） |
| `comfyui_controlnet_aux` | 深度图预处理、lineart / softedge 预处理 |
| `Nvidia_RTX_Nodes_ComfyUI` | RTX VSR 超分（需 N 卡） |
| `rgthree-comfy` | 图像前后对比预览 |

---

## 数据文件

| 文件 | 随仓库分发 | 说明 |
|---|---|---|
| `characters.json` | ✅ | 内置角色表 |
| `characters_i18n.json` | ✅ | 中文名 → 英文 tag 映射 |
| `action.json` | ✅ | 动作 → tag |
| `character_loras.json` | ✅ | 角色 → LoRA 文件名 |
| `tag_category_v2.json` | ✅ | 11099 条 Danbooru 标签 → 类别映射（场景过滤用） |
| `char_thumbs.json` | ❌ | 角色缩略图缓存，34 MB；缺失时不显示缩略图，功能正常 |
| `favorites.json` | ❌ | 个人收藏 |
| `characters_custom.json` | ❌ | 自建角色 |
| `custom_action.json` | ❌ | 自定义动作 |
| `custom_character.json` | ❌ | 自定义角色 |

所有数据文件缺失时都会**返回空字典**，不会报错。

---

## 旧版检测器（legacy）

V7 / V7.1 / V8 / V8.2 / V9 的检测器在 `legacy/` 下，**默认不注册**。
需要打开旧工作流时，启动 ComfyUI 前设置：

```bash
set ILLUSTRIOUS_ENABLE_LEGACY=1      # Windows
export ILLUSTRIOUS_ENABLE_LEGACY=1   # Linux / macOS
```

会额外注册 7 个旧节点（共 30 个）。

---

## 版本演进：多代检测器

眉眼带定位这件事踩了十几版的坑，每一代解决一个具体失效：

| 版本 | 解决的问题 | 引入的失效 |
|---|---|---|
| V7 | 倾斜头部时带子不跟着转 | 平行四边形硬套 |
| V8 | 平行四边形几何 | 刘海造成「假凹」被误杀 |
| V9 | 跨度 / 面积判据 + Eyeful 交叉校验 | SAM 跟随边缘，被刘海带偏 |
| **V10** | **人脸分割（Forbidden Vision）替代 SAM** | —— |

V10 的关键原则：**分割只提供尺寸，不提供角度**。

顺带修掉一个从 V9 就潜伏的 bug：`_lm_plausible` 在文件里定义了两次，
打补丁打到的是第一份（死代码），判据 A/B/C 实际上从未运行过。
V10 删掉了那 109 行。

---

## 已知限制

- 配套工作流依赖**精确的 widget 顺序**，插件大版本更新后可能错位
- `char_thumbs.json` 没随仓库分发，选择器只显示角色名
- legacy 节点的行为可能与当前版本不一致，仅作存档
- 拼版的工作流快照只是「占位画面」，点选仍需重新运行

---

## 致谢

- [Forbidden Vision](https://github.com/numz/ComfyUI-Forbidden-Vision) —— 人脸分割，替代了原先的 SAM 方案
- [comfyui_tag_filter](https://github.com/sugarkwork/comfyui_tag_filter) —— 标签类别表 `tag_category_v2.json`
- [Impact Pack](https://github.com/ltdrdata/ComfyUI-Impact-Pack)

---

## 许可

MIT（LICENSE 待补）
`tag_category_v2.json` 遵循原项目 MIT 许可。
内置角色/动作表为自建数据，使用前请自行确认内容。
