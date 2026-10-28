import { app } from "../../scripts/app.js";

const TUNER_CLASS = "IllustriousFaceTuner";
const TARGET_CLASS = "FaceDetailer";

// FaceDetailer 的 widgets_values 索引映射（Impact Pack 的固定顺序）
const INDEX_MAP = {
  guide_size: 0,
  max_size: 2,
  denoise: 9,
  feather: 10,
  bbox_threshold: 13,
  bbox_dilation: 14,
  bbox_crop_factor: 15,
  sam_detection_hint: 16,
  sam_dilation: 17,
  sam_bbox_expansion: 19,
  noise_mask_feather: 26,
};

app.registerExtension({
  name: "Illustrious.FaceTuner",
  async nodeCreated(node) {
    if (node?.comfyClass !== TUNER_CLASS && node?.type !== TUNER_CLASS) return;

    const collect = () => {
      const out = {};
      for (const w of node.widgets || []) {
        if (!w || w.type === "button") continue;
        if (w.name in INDEX_MAP) out[w.name] = w.value;
      }
      return out;
    };

    const applyTo = (targets) => {
      const settings = collect();
      let changed = 0;
      let nodeCnt = 0;
      for (const t of targets) {
        const wv = Array.isArray(t.widgets_values) ? t.widgets_values : null;
        let touched = false;
        for (const [name, idx] of Object.entries(INDEX_MAP)) {
          if (!(name in settings)) continue;
          const val = settings[name];

          // 【关键】ComfyUI 执行时读的是 widget.value，不是 widgets_values。
          // 只改 widgets_values 的话，界面和实际送进后端的值都不会变。
          const w = (t.widgets || []).find((x) => x && x.name === name);
          if (w && w.value !== val) {
            w.value = val;
            try {
              w.callback?.(val);
            } catch (e) {
              /* ignore */
            }
            touched = true;
            changed++;
          }

          // 同时同步序列化数组（保存工作流时用）
          if (wv && idx < wv.length && wv[idx] !== val) {
            wv[idx] = val;
            if (!w) {
              touched = true;
              changed++;
            }
          }
        }
        if (touched) {
          nodeCnt++;
          try {
            t.setDirtyCanvas?.(true, true);
          } catch (e) {
            /* ignore */
          }
        }
      }
      try {
        app.graph.setDirtyCanvas(true, true);
      } catch (e) {
        /* ignore */
      }
      console.log(`[FaceTuner] 已同步 ${changed} 项到 ${nodeCnt} 个 FaceDetailer`);
      return { changed, nodeCnt };
    };

    if (typeof node.addWidget === "function") {
      node.addWidget("button", "↻ 应用到所有 FaceDetailer", null, () => {
        const all = (app.graph?._nodes || []).filter((n) => n.type === TARGET_CLASS);
        const r = applyTo(all);
        node.title = `${node.title || "Face 参数总控"} (${r.nodeCnt}个已更新)`;
        node.setDirtyCanvas?.(true, true);
      });

      node.addWidget("button", "↻ 仅应用到差分 FD (132~136)", null, () => {
        const ids = [132, 133, 134, 135, 136];
        const all = (app.graph?._nodes || []).filter(
          (n) => n.type === TARGET_CLASS && ids.includes(n.id)
        );
        const r = applyTo(all);
        node.title = `${node.title || "Face 参数总控"} (${r.nodeCnt}个已更新)`;
        node.setDirtyCanvas?.(true, true);
      });

      node.addWidget("button", "⬅ 从 FaceDetailer 读回当前值", null, () => {
        const all = (app.graph?._nodes || []).filter((n) => n.type === TARGET_CLASS);
        if (!all.length) {
          console.warn("[FaceTuner] 画布上没有 FaceDetailer");
          return;
        }
        const src = all[0];
        const wv = src.widgets_values;
        if (!Array.isArray(wv)) return;
        for (const [name, idx] of Object.entries(INDEX_MAP)) {
          const w = (node.widgets || []).find((x) => x && x.name === name);
          const val = (Array.isArray(wv) && idx < wv.length) ? wv[idx] : undefined;
          if (w && val !== undefined) w.value = val;
          else if (Array.isArray(wv) && idx < wv.length) {
            const w2 = (node.widgets || []).find((x) => x && x.name === name);
            if (!w2) continue;
          }
        }
        node.setDirtyCanvas?.(true, true);
        console.log(`[FaceTuner] 已从 FaceDetailer ${src.id} 读回参数`);
      });
    }
  },
});
