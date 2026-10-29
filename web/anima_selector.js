import { app } from "../../scripts/app.js";

const NODE_CLASS = "AnimaCharacterActionSelector";

function findWidget(node, name) {
  return node?.widgets?.find((w) => w && w.name === name);
}

// 直接把新列表写进 combo 控件的 options，并尽力触发前端重绘
function applyCharacterList(node, characters) {
  const w = findWidget(node, "character");
  if (w && Array.isArray(characters)) {
    w.options = w.options || {};
    w.options.values = characters;
  }
  // 官方刷新入口（正确名字是复数、且不需要参数）
  try {
    if (typeof app.refreshComboInNodes === "function") {
      app.refreshComboInNodes();
    }
  } catch (e) {
    console.warn("[Illustrious-Selector] refreshComboInNodes failed", e);
  }
  try {
    node.setDirtyCanvas?.(true, true);
  } catch (e) {
    /* ignore */
  }
}

async function saveAndRefresh(node) {
  const nameW = findWidget(node, "new_character_name");
  const promptW = findWidget(node, "new_character_prompt");
  const charW = findWidget(node, "character");
  const loraW = findWidget(node, "character_lora");
  const loraWeightW = findWidget(node, "character_lora_weight");

  const name = String(nameW?.value ?? "").trim();
  const prompt = String(promptW?.value ?? "").trim();
  const character = String(charW?.value ?? "").trim();
  const lora = String(loraW?.value ?? "").trim();
  let lora_weight = Number(loraWeightW?.value ?? 1.0);
  if (!Number.isFinite(lora_weight)) lora_weight = 1.0;

  if (!(name && prompt) && !(lora && character)) {
    alert("需要至少一项：\n• 填 new_character_name + new_character_prompt 新增角色\n• 或给当前选中角色填 character_lora（绑定 LoRA）");
    return;
  }
  try {
    const r = await fetch("/anima_selector/save", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name, prompt, character, lora, lora_weight }),
    });
    const j = await r.json().catch(() => null);
    if (!r.ok || !j) {
      alert(`保存路由不可用（HTTP ${r.status}）。请重启 ComfyUI 后再试。`);
      return;
    }
    if (!j.ok) {
      alert("保存失败: " + (j.error || r.status));
      return;
    }
    applyCharacterList(node, j.characters);
    if (name && prompt && charW) charW.value = name;
    const target = j.lora_target || ((name && prompt) ? name : character);
    const msgs = [];
    if (name && prompt) msgs.push(`新角色 ${name}`);
    if (lora && target) msgs.push(`角色「${target}」绑定 LoRA「${lora}」权重 ${lora_weight}`);
    alert("已保存：\n" + msgs.join("\n"));
  } catch (e) {
    alert("请求失败: " + e);
  }
}

async function toggleFavorite(node) {
  const cw = findWidget(node, "character");
  const name = String(cw?.value ?? "").trim();
  if (!name || name === "None" || name === "random") {
    alert("请先在下拉里选中一个具体角色再收藏");
    return;
  }
  try {
    const r = await fetch("/anima_selector/favorite", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name }),
    });
    const j = await r.json().catch(() => null);
    if (!r.ok || !j) {
      alert(
        `收藏路由不可用（HTTP ${r.status}）。\n` +
        `这通常是 ComfyUI 后端还没加载新代码：请【重启 ComfyUI】后按 Ctrl+F5 刷新页面再试。`
      );
      return;
    }
    if (!j.ok) {
      alert("操作失败: " + (j.error || r.status));
      return;
    }
    applyCharacterList(node, j.characters);
    if (cw) cw.value = name;
    const isFav = Array.isArray(j.favorites) && j.favorites.includes(name);
    alert((isFav ? "⭐ 已收藏（置顶显示）：" : "已取消收藏：") + name);
  } catch (e) {
    alert("请求失败: " + e);
  }
}

const PREVIEW_MAX_H = 200;

function previewScale(widget, width) {
  if (!widget.image || !widget.image.naturalHeight) return { w: 0, h: 0 };
  const maxW = Math.max(32, width - 16);
  const s = Math.min(maxW / widget.image.naturalWidth, PREVIEW_MAX_H / widget.image.naturalHeight, 1);
  return { w: widget.image.naturalWidth * s, h: widget.image.naturalHeight * s };
}

function resizeNodeToFit(node) {
  try {
    const minSize = node.computeSize();
    // 只增不减：保留用户拖拽后的高度，仅在比内容最小高度还矮时撑高
    node.setSize([
      Math.max(node.size[0], minSize[0]),
      Math.max(node.size[1], minSize[1]),
    ]);
    node.setDirtyCanvas?.(true, true);
  } catch (e) {
    /* ignore */
  }
}

async function updatePreview(node) {
  const w = node?._charImageWidget;
  if (!w) return;
  const cw = findWidget(node, "character");
  const name = String(cw?.value ?? "").trim();
  if (!name || name === "None" || name === "random") {
    w.image = null;
    resizeNodeToFit(node);
    return;
  }
  try {
    const r = await fetch("/anima_selector/char_image?name=" + encodeURIComponent(name));
    const j = await r.json();
    if (!j || !j.ok || !j.image) {
      w.image = null;
      resizeNodeToFit(node);
      return;
    }
    const img = new Image();
    img.onload = () => {
      w.image = img;
      resizeNodeToFit(node);
    };
    img.onerror = () => {
      w.image = null;
      resizeNodeToFit(node);
    };
    img.src = j.image;
  } catch (e) {
    w.image = null;
    resizeNodeToFit(node);
  }
}

app.registerExtension({
  name: "AnimaCharacterSelector.Save",
  async nodeCreated(node) {
    if (node?.comfyClass !== NODE_CLASS && node?.type !== NODE_CLASS) return;

    // 角色缩略图预览（自定义 canvas widget：computeSize 动态撑高，不会压住下面的按钮）
    let imageWidget = null;
    try {
      imageWidget = node.addCustomWidget({
        name: "char_image",
        type: "custom_image",
        y: 0,
        image: null,
        draw: function (ctx, n, width, y) {
          if (!this.image || !this.image.complete) return;
          const s = previewScale(this, width);
          if (s.w <= 0) return;
          const x = (width - s.w) / 2;
          ctx.drawImage(this.image, x, y, s.w, s.h);
        },
        computeSize: function (width) {
          const s = previewScale(this, width);
          return [width, s.h > 0 ? s.h + 10 : 0];
        },
      });
    } catch (e) {
      console.warn("[Illustrious-Selector] addCustomWidget failed", e);
    }
    node._charImageWidget = imageWidget;

    // 只在 configure 时恢复一次保存的高度（不做持续跟踪，避免正反馈）
    const origConfigure = node.onConfigure;
    node.onConfigure = function (info) {
      if (typeof origConfigure === "function") origConfigure.apply(this, arguments);
      const savedH = info?.size?.[1];
      if (savedH) {
        const self = this;
        requestAnimationFrame(() => {
          try {
            self.setSize([self.size[0], savedH]);
          } catch (e) {
            /* ignore */
          }
        });
      }
    };

    const charW = findWidget(node, "character");
    if (charW) {
      const origCb = charW.callback;
      charW.callback = function () {
        const r = typeof origCb === "function" ? origCb.apply(this, arguments) : undefined;
        updatePreview(node);
        return r;
      };
    }
    setTimeout(() => updatePreview(node), 400);

    // 底部按钮
    node.addWidget("button", "💾 保存并刷新角色", null, () => saveAndRefresh(node));
    node.addWidget("button", "⭐ 收藏 / 取消收藏当前角色", null, () => toggleFavorite(node));

    // 右键菜单兜底
    const orig = node.getExtraMenuOptions;
    node.getExtraMenuOptions = function (canvas, options) {
      if (typeof orig === "function") orig.apply(this, arguments);
      options.push({
        content: "💾 保存并刷新角色（写入角色库）",
        callback: () => saveAndRefresh(node),
      });
      options.push({
        content: "⭐ 收藏 / 取消收藏当前角色（置顶）",
        callback: () => toggleFavorite(node),
      });
    };
  },
});
