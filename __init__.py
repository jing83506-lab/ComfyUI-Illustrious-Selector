"""ComfyUI-Illustrious-Selector

Illustrious / SDXL 系「高清修复 + 手脸修复 + 表情差分」工作流的配套节点。

节点分组：
  提示词    CharacterActionSelector / PromptDisplay / LoRAStack / SceneTags / SceneConditioning
  检测      SamBrowDetectorV10 / DetectGuard / PosePreview / MaskDebuggerV10
  图像      ExpressionSheet / FaceCrop / SaveImageEx / SeedHub / VariationGate / VariationRouter
  工具      NormalizedResize / EdgeSelector / LatentSize / MaskOverlay / FaceTuner /
            ExpressionMaskHook / SamBrowDetector

旧版检测器（V7/V71/V8/V82/V9）在 legacy/ 下，默认不注册。
若需打开历史工作流（V8.x / V9），启动前设置环境变量：
    ILLUSTRIOUS_ENABLE_LEGACY=1
"""

import os

from .nodes import (
    IllustriousCharacterActionSelector,
    IllustriousPromptDisplay,
    IllustriousEdgeSelector,
    IllustriousNormalizedResize,
    IllustriousLoraStack,
    IllustriousVariationGate,
    IllustriousExpressionSheet,
    IllustriousVariationRouter,
    IllustriousSeedHub,
    IllustriousSaveImageEx,
    IllustriousSceneTags,
    IllustriousSceneConditioning,
    IllustriousDetectGuard,
    IllustriousFaceCrop,
    IllustriousLatentSize,
    IllustriousExpressionMaskHook,
    IllustriousFaceTuner,
    IllustriousMaskOverlay,
    IllustriousSamBrowDetector,
    save_custom,
    save_char_lora,
    toggle_favorite,
    character_thumb,
    char_keys,
)

# 当前检测器：V10（FV 人脸分割 + 眼点驱动眉眼带）
try:
    from .nodes_v10 import (
        IllustriousSamBrowDetectorV10,
        IllustriousMaskDebuggerV10,
        IllustriousPosePreviewV10,
    )
    _V10_OK = True
except Exception as _e:
    print(f"[Illustrious-Selector] nodes_v10 加载失败: {type(_e).__name__}: {_e}")
    _V10_OK = False

# PosePreview 的实现长期放在 nodes_v7.py 里（V10.1 工作流在用），单独取出来
try:
    from .legacy.nodes_v7 import IllustriousPosePreview
    _POSE_OK = True
except Exception as _e:
    print(f"[Illustrious-Selector] IllustriousPosePreview 加载失败: {type(_e).__name__}: {_e}")
    _POSE_OK = False

NODE_CLASS_MAPPINGS = {
    # --- 提示词 ---
    "IllustriousCharacterActionSelector": IllustriousCharacterActionSelector,
    "IllustriousPromptDisplay": IllustriousPromptDisplay,
    "IllustriousLoraStack": IllustriousLoraStack,
    "IllustriousSceneTags": IllustriousSceneTags,
    "IllustriousSceneConditioning": IllustriousSceneConditioning,
    # --- 检测 ---
    "IllustriousSamBrowDetector": IllustriousSamBrowDetector,
    "IllustriousDetectGuard": IllustriousDetectGuard,
    "IllustriousMaskOverlay": IllustriousMaskOverlay,
    # --- 图像 ---
    "IllustriousExpressionSheet": IllustriousExpressionSheet,
    "IllustriousExpressionMaskHook": IllustriousExpressionMaskHook,
    "IllustriousFaceCrop": IllustriousFaceCrop,
    "IllustriousFaceTuner": IllustriousFaceTuner,
    "IllustriousSaveImageEx": IllustriousSaveImageEx,
    "IllustriousSeedHub": IllustriousSeedHub,
    "IllustriousVariationGate": IllustriousVariationGate,
    "IllustriousVariationRouter": IllustriousVariationRouter,
    # --- 工具 ---
    "IllustriousNormalizedResize": IllustriousNormalizedResize,
    "IllustriousEdgeSelector": IllustriousEdgeSelector,
    "IllustriousLatentSize": IllustriousLatentSize,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "IllustriousCharacterActionSelector": "角色 / 动作选择器",
    "IllustriousPromptDisplay": "提示词显示",
    "IllustriousLoraStack": "LoRA 堆栈",
    "IllustriousSceneTags": "场景标签过滤（白名单分类）",
    "IllustriousSceneConditioning": "场景提示词拼接（带开关）",
    "IllustriousSamBrowDetector": "SAM 眉眼带检测器（原始版）",
    "IllustriousDetectGuard": "检测双重约束（bbox×segm）",
    "IllustriousMaskOverlay": "Mask 叠加预览",
    "IllustriousExpressionSheet": "表情差分拼版",
    "IllustriousExpressionMaskHook": "差分蒙版钩子",
    "IllustriousFaceCrop": "脸部裁切",
    "IllustriousFaceTuner": "脸部微调",
    "IllustriousSaveImageEx": "保存 JPG/WebP（可存工作流）",
    "IllustriousSeedHub": "种子枢纽（一处改全链 re-roll）",
    "IllustriousVariationGate": "差分开关",
    "IllustriousVariationRouter": "差分路由（图 + 表情 CLIP）",
    "IllustriousNormalizedResize": "归一化缩放",
    "IllustriousEdgeSelector": "边缘选择器（canny/lineart）",
    "IllustriousLatentSize": "潜空间尺寸",
}

if _POSE_OK:
    NODE_CLASS_MAPPINGS["IllustriousPosePreview"] = IllustriousPosePreview
    NODE_DISPLAY_NAME_MAPPINGS["IllustriousPosePreview"] = "Pose 可视化"

if _V10_OK:
    NODE_CLASS_MAPPINGS["IllustriousSamBrowDetectorV10"] = IllustriousSamBrowDetectorV10
    NODE_CLASS_MAPPINGS["IllustriousMaskDebuggerV10"] = IllustriousMaskDebuggerV10
    NODE_CLASS_MAPPINGS["IllustriousPosePreviewV10"] = IllustriousPosePreviewV10
    NODE_DISPLAY_NAME_MAPPINGS["IllustriousSamBrowDetectorV10"] = "SAM/FV+眉眼带 V10（人脸分割优先）"
    NODE_DISPLAY_NAME_MAPPINGS["IllustriousMaskDebuggerV10"] = "Mask 分色调试 V10"
    NODE_DISPLAY_NAME_MAPPINGS["IllustriousPosePreviewV10"] = "Pose 可视化 V10"

# ------------------------------------------------------------------ 可选的旧版节点
if os.environ.get("ILLUSTRIOUS_ENABLE_LEGACY", "").strip() in ("1", "true", "True"):
    try:
        from .legacy.nodes_v7 import IllustriousSamBrowDetectorV7
        from .legacy.nodes_v71 import (IllustriousSamBrowDetectorV71,
                                       IllustriousMaskDebuggerV71)
        from .legacy.nodes_v8 import IllustriousSamBrowDetectorV8
        from .legacy.nodes_v82 import IllustriousSamBrowDetectorV82
        from .legacy.nodes_v9 import (IllustriousSamBrowDetectorV9,
                                      IllustriousMaskDebuggerV9)
        _L = {
            "IllustriousSamBrowDetectorV7": (IllustriousSamBrowDetectorV7, "SAM 眉眼带 V7"),
            "IllustriousSamBrowDetectorV71": (IllustriousSamBrowDetectorV71, "SAM 眉眼带 V7.1"),
            "IllustriousMaskDebuggerV71": (IllustriousMaskDebuggerV71, "Mask 分色调试 V7.1"),
            "IllustriousSamBrowDetectorV8": (IllustriousSamBrowDetectorV8, "SAM 眉眼带 V8"),
            "IllustriousSamBrowDetectorV82": (IllustriousSamBrowDetectorV82, "SAM 眉眼带 V8.2"),
            "IllustriousSamBrowDetectorV9": (IllustriousSamBrowDetectorV9, "SAM 眉眼带 V9"),
            "IllustriousMaskDebuggerV9": (IllustriousMaskDebuggerV9, "Mask 分色调试 V9"),
        }
        for k, (c, dn) in _L.items():
            NODE_CLASS_MAPPINGS[k] = c
            NODE_DISPLAY_NAME_MAPPINGS[k] = dn
        print("[Illustrious-Selector] 已启用 legacy 旧版检测器 %d 个" % len(_L))
    except Exception as _e:
        print(f"[Illustrious-Selector] legacy 加载失败: {type(_e).__name__}: {_e}")

WEB_DIRECTORY = "./web"
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]

print(f"[Illustrious-Selector] 已注册 {len(NODE_CLASS_MAPPINGS)} 个节点")
