import { app } from "../../scripts/app.js";

const NODE_CLASS = "IllustriousExpressionSheet";

function findWidget(node, name) {
  return node?.widgets?.find((w) => w && w.name === name);
}

app.registerExtension({
  name: "IllustriousExpressionSheet.Live",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData?.name !== NODE_CLASS) return;

    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      if (onNodeCreated) onNodeCreated.apply(this, arguments);
      const node = this;

      // 前端合成状态
      const st = {
        mainImg: null,
        varImgs: [],
        thumbImgs: [],
        canvasW: 0,
        canvasH: 0,
        rects: { main: { x: 0, y: 0, w: 0, h: 0 }, thumbs: [] },
        border: 0,
        bg: "#202020",
        selected: 0,
        _draw: null,
        _fallbackImg: null,
      };
      node._exprSheetState = st;

      // 1. DOM 容器 + 内部画布（兼容 Nodes 2.0 与传统渲染器）
      const container = document.createElement("div");
      container.style.cssText =
        "width:100%;height:100%;min-height:220px;overflow:hidden;position:relative;" +
        "display:flex;align-items:center;justify-content:center;background:#151820;" +
        "border:1px solid #262c3a;border-radius:6px;box-sizing:border-box;user-select:none;";

      const cvs = document.createElement("canvas");
      cvs.style.cssText = "display:block;width:100%;height:100%;pointer-events:auto;";
      container.appendChild(cvs);

      // 2. 渲染（按容器宽高 contain，绝不溢出）
      //
      // 空状态：还没跑过时，画一张「示意布局」——左边主图、右边一列差分小图，
      // 其中一格高亮表示选中。目的是让人第一次把节点拖上画布就知道它是干什么的，
      // 而不是只看到一行「等待运行」。
      const drawSchematic = (ctx, clientW, clientH) => {
        // 虚拟画布：主图 600x900 + gap + 右侧一列 5 个 130x162 的小格
        const VW = 600, VH = 900, GAP = 8, BW = 130, BH = 162, N = 5, COLX = 600 + 8;
        const CWv = COLX + BW;
        const CHv = VH;
        const TEXT_H = 54;

        const pad = 8;
        const availW = Math.max(40, clientW - pad * 2);
        const availH = Math.max(40, clientH - pad * 2 - TEXT_H);
        const s = Math.min(availW / CWv, availH / CHv);
        const dw = CWv * s, dh = CHv * s;
        const dx = (clientW - dw) / 2;
        const dy = (clientH - TEXT_H - dh) / 2;

        // 背板
        ctx.fillStyle = "#202020";
        ctx.fillRect(dx, dy, dw, dh);

        // 主图占位
        ctx.fillStyle = "#2b3040";
        ctx.fillRect(dx, dy, 600 * s, VH * s);
        ctx.strokeStyle = "#3c4457";
        ctx.lineWidth = 1;
        ctx.strokeRect(dx + 0.5, dy + 0.5, 600 * s - 1, VH * s - 1);

        ctx.fillStyle = "#8b96ad";
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        const fs = Math.max(9, Math.min(16, 22 * s * 2));
        ctx.font = fs + "px sans-serif";
        ctx.fillText("主图", dx + 300 * s, dy + VH * s / 2 - fs * 0.9);
        ctx.font = Math.max(8, fs * 0.78) + "px sans-serif";
        ctx.fillStyle = "#68718a";
        ctx.fillText("selected = 0", dx + 300 * s, dy + VH * s / 2 + fs * 0.9);

        // 右侧一列小图（第 2 格高亮，示意「选中」）
        const stripH = N * BH + (N - 1) * GAP;
        let y = Math.max(0, (VH - stripH) / 2);
        for (let i = 0; i < N; i++) {
          const rx = dx + COLX * s;
          const ry = dy + y * s;
          const rw = BW * s;
          const rh = BH * s;
          const hl = i === 1;

          ctx.fillStyle = hl ? "#ffcc00" : "#FFFFFF";
          ctx.fillRect(rx, ry, rw, rh);
          const b = Math.max(1, 3 * s);
          ctx.fillStyle = "#2b3040";
          ctx.fillRect(rx + b, ry + b, rw - 2 * b, rh - 2 * b);

          ctx.fillStyle = hl ? "#ffcc00" : "#7e879c";
          ctx.font = Math.max(8, Math.min(13, rw * 0.16)) + "px sans-serif";
          ctx.textAlign = "center";
          ctx.textBaseline = "middle";
          ctx.fillText(String(i + 1), rx + rw / 2, ry + rh / 2);

          y += BH + GAP;
        }

        // 底部说明
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillStyle = "#9aa4bb";
        ctx.font = Math.max(10, Math.min(14, clientW * 0.032)) + "px sans-serif";
        ctx.fillText("运行后在此显示真实差分（左主图 / 右小图）", clientW / 2, clientH - TEXT_H / 2 - 12);
        ctx.fillStyle = "#6c768a";
        ctx.font = Math.max(9, Math.min(12, clientW * 0.026)) + "px sans-serif";
        ctx.fillText("点小图切换主图 · 黄色框 = 当前选中", clientW / 2, clientH - TEXT_H / 2 + 10);
      };

      const renderSheet = () => {
        const rect = container.getBoundingClientRect();
        const clientW = container.clientWidth || Math.round(rect.width) || 300;
        const clientH = container.clientHeight || Math.round(rect.height) || 260;
        const dpr = window.devicePixelRatio || 1;

        if (cvs.width !== Math.round(clientW * dpr) || cvs.height !== Math.round(clientH * dpr)) {
          cvs.width = Math.round(clientW * dpr);
          cvs.height = Math.round(clientH * dpr);
        }

        const ctx = cvs.getContext("2d");
        ctx.save();
        ctx.scale(dpr, dpr);
        ctx.clearRect(0, 0, clientW, clientH);

        const pad = 8;
        const availW = Math.max(32, clientW - pad * 2);
        const availH = Math.max(32, clientH - pad * 2);

        if (!st.mainImg || !st.mainImg.complete || st.canvasW <= 0 || st.canvasH <= 0) {
          const fb = st._fallbackImg;
          if (fb && fb.complete && fb.naturalWidth > 0) {
            const sc = Math.min(availW / fb.naturalWidth, availH / fb.naturalHeight);
            const dw = fb.naturalWidth * sc;
            const dh = fb.naturalHeight * sc;
            const dx = (clientW - dw) / 2;
            const dy = (clientH - dh) / 2;
            ctx.drawImage(fb, dx, dy, dw, dh);
            st._draw = { x: dx, y: dy, s: sc, w: dw, h: dh, live: false };
          } else {
            // 没跑过、也没有历史快照：画示意布局，让人一眼看懂这不是普通预览
            drawSchematic(ctx, clientW, clientH);
          }
          ctx.restore();
          return;
        }

        // 同时受宽高约束（contain），彻底杜绝底部溢出
        const s = Math.min(availW / st.canvasW, availH / st.canvasH);
        const dw = st.canvasW * s;
        const dh = st.canvasH * s;
        const dx = (clientW - dw) / 2;
        const dy = (clientH - dh) / 2;

        ctx.fillStyle = st.bg || "#202020";
        ctx.fillRect(dx, dy, dw, dh);

        const sel = Number(findWidget(node, "selected")?.value ?? st.selected ?? 0);

        // 主图（selected=0 原图，1~N 对应差分整图）
        const mr = st.rects.main;
        if (mr && mr.w > 0) {
          const src =
            sel > 0 && st.varImgs[sel - 1] && st.varImgs[sel - 1].complete
              ? st.varImgs[sel - 1]
              : st.mainImg;
          if (src && src.complete) {
            ctx.drawImage(src, dx + mr.x * s, dy + mr.y * s, mr.w * s, mr.h * s);
          }
        }

        // 缩略图 + 边框
        for (let i = 0; i < st.rects.thumbs.length; i++) {
          const r = st.rects.thumbs[i];
          const t = st.thumbImgs[i];
          if (!r || !t || !t.complete) continue;

          const isSelected = sel === i + 1;
          ctx.fillStyle = isSelected ? "#ffcc00" : "#FFFFFF";
          ctx.fillRect(dx + r.x * s, dy + r.y * s, r.w * s, r.h * s);

          ctx.drawImage(
            t,
            dx + (r.x + st.border) * s,
            dy + (r.y + st.border) * s,
            (r.w - 2 * st.border) * s,
            (r.h - 2 * st.border) * s
          );

          if (isSelected) {
            ctx.save();
            ctx.strokeStyle = "#ffcc00";
            ctx.lineWidth = 3;
            ctx.strokeRect(dx + r.x * s - 1, dy + r.y * s - 1, r.w * s + 2, r.h * s + 2);
            ctx.restore();
          }
        }

        st._draw = { x: dx, y: dy, s: s, w: dw, h: dh, live: true };
        ctx.restore();
      };

      // 3. 命中检测（DOM 局部坐标，不受画布缩放影响）
      const getLocalCoords = (e) => {
        const rect = cvs.getBoundingClientRect();
        const scaleX = (cvs.clientWidth || rect.width) / (rect.width || 1);
        const scaleY = (cvs.clientHeight || rect.height) / (rect.height || 1);
        return [(e.clientX - rect.left) * scaleX, (e.clientY - rect.top) * scaleY];
      };

      const hitTest = (localX, localY) => {
        const d = st._draw;
        if (!d || !st.rects.thumbs.length) return -1;
        const lx = localX - d.x;
        const ly = localY - d.y;
        if (lx < 0 || ly < 0 || lx > d.w || ly > d.h) return -1;
        const ix = lx / d.s;
        const iy = ly / d.s;

        for (let i = 0; i < st.rects.thumbs.length; i++) {
          const r = st.rects.thumbs[i];
          if (ix >= r.x && ix <= r.x + r.w && iy >= r.y && iy <= r.y + r.h) {
            return i + 1;
          }
        }
        const mr = st.rects.main;
        if (mr && ix >= mr.x && ix <= mr.x + mr.w && iy >= mr.y && iy <= mr.y + mr.h) {
          return 0;
        }
        return -1;
      };

      const applySelection = (idx) => {
        if (idx < 0) return;
        const selW = findWidget(node, "selected");
        if (selW) {
          selW.value = idx;
          selW.callback?.(idx);
        }
        st.selected = idx;
        renderSheet();
        node.setDirtyCanvas?.(true, true);
      };

      // 4. 鼠标交互：区分点击与拖动
      let downPos = null;
      cvs.addEventListener("pointerdown", (e) => {
        if (e.button !== 0) return;
        const [lx, ly] = getLocalCoords(e);
        const idx = hitTest(lx, ly);
        if (idx >= 0) {
          e.stopPropagation();
          downPos = [e.clientX, e.clientY, idx];
        }
      });

      cvs.addEventListener("pointerup", (e) => {
        if (e.button !== 0 || !downPos) return;
        const [sx, sy, idx] = downPos;
        downPos = null;
        if (Math.hypot(e.clientX - sx, e.clientY - sy) < 6) {
          e.stopPropagation();
          applySelection(idx);
        }
      });

      cvs.addEventListener("pointermove", (e) => {
        const [lx, ly] = getLocalCoords(e);
        const idx = hitTest(lx, ly);
        cvs.style.cursor = idx > 0 ? "pointer" : idx === 0 ? "default" : "grab";
      });

      // 5. 容器尺寸变化自动重绘
      const ro = new ResizeObserver(() => renderSheet());
      ro.observe(container);

      // selected 控件被手动修改时同步重绘
      const selW = findWidget(node, "selected");
      if (selW) {
        const origCb = selW.callback;
        selW.callback = function (v) {
          const r = typeof origCb === "function" ? origCb.apply(this, arguments) : undefined;
          st.selected = Number(v ?? 0);
          renderSheet();
          return r;
        };
      }

      // 6. 挂载为 DOM Widget（固定最小高度，绝不动态暴涨）
      const domWidget = node.addDOMWidget("expr_sheet", "sheet", container, {
        serialize: false,
        hideOnZoom: false,
      });
      domWidget.computeSize = function (width) {
        // 默认高度给足，否则拼版图会被压得很小
        return [width, 950];
      };

      // 初始尺寸：只在过小时撑开，不覆盖用户保存的高度
      requestAnimationFrame(() => {
        if (node.size[0] < 340 || node.size[1] < 1000) {
          node.setSize([Math.max(node.size[0], 360), Math.max(node.size[1], 1000)]);
        }
        renderSheet();
      });

      // 7. 接收后端数据（不再用 computeSize 覆盖 node.size[1]）
      const origExecuted = node.onExecuted;
      node.onExecuted = function (message) {
        if (typeof origExecuted === "function") origExecuted.apply(this, arguments);
        if (!message) return;

        st.selected = Number(message?.selected?.[0] ?? 0);

        const rectsRaw = message?.rects?.[0];
        if (rectsRaw) {
          try {
            const parsed = typeof rectsRaw === "string" ? JSON.parse(rectsRaw) : rectsRaw;
            st.rects = { main: parsed.main, thumbs: parsed.thumbs || [] };
            st.border = parsed.border || 0;
            st.bg = parsed.bg ? "#" + String(parsed.bg).replace(/^#/, "") : "#202020";
          } catch (e) {
            st.rects = { main: { x: 0, y: 0, w: 0, h: 0 }, thumbs: [] };
          }
        }
        if (message?.canvas?.[0]) {
          st.canvasW = message.canvas[0][0] || 0;
          st.canvasH = message.canvas[0][1] || 0;
        }

        const load = (b64, cb) => {
          if (!b64) return;
          const im = new Image();
          im.onload = () => {
            cb(im);
            renderSheet();
          };
          im.src = b64;
        };

        load(message?.main_img?.[0], (im) => (st.mainImg = im));
        if (Array.isArray(message?.var_imgs)) {
          st.varImgs = [];
          message.var_imgs.forEach((b64, i) => load(b64, (im) => (st.varImgs[i] = im)));
        }
        if (Array.isArray(message?.thumb_imgs)) {
          st.thumbImgs = [];
          message.thumb_imgs.forEach((b64, i) => load(b64, (im) => (st.thumbImgs[i] = im)));
        }
        load(message?.preview?.[0], (im) => {
          st._fallbackImg = im;
          // 后端把素材写进 temp 目录，这里只存清单（几百字节）
          persistSnap(message?.snap?.[0]);
        });
      };

      // ---------------------------------------------------------------
      // 快照持久化：让拼版在「切换工作流之后」依然有图、依然能点
      //
      // ComfyUI 每次执行只把 ui 数据推给前端，不写进工作流；节点 JSON 里
      // 只有 widgets_values 和 properties 会被保存。而把图片 base64 塞进
      // properties 会把工作流撑爆 —— 实测 640px 主图就 85 KB，1MP 要 180 KB，
      // 再加 5 张同清晰度的差分就超过 1 MB。
      //
      // 所以改成：后端把素材存成 JPEG 放到 temp 目录，properties 里只存
      // 几百字节的清单，重建时用 /api/view 取原图。
      // 分辨率不再受 JSON 体积限制：主图和差分都按 1280 长边存（约 0.9MP）。
      //
      // 代价：temp 目录被清理后（通常重启 ComfyUI）快照失效，
      // 会退回示意布局，跑一次即可恢复。
      // ---------------------------------------------------------------
      const SNAP_KEY = "illustrious_sheet_snap_v2";

      const snapUrl = (entry, dir) => {
        if (!entry) return null;
        const name = String(entry).split("|")[0];
        if (!name) return null;
        const qs = new URLSearchParams({
          filename: name,
          subfolder: dir || "",
          type: "temp",
        });
        return `/api/view?${qs.toString()}`;
      };

      const persistSnap = (raw) => {
        // raw 是后端给的清单 JSON 字符串；只存清单，不存图片数据
        if (!raw) return;
        try {
          const snap = typeof raw === "string" ? JSON.parse(raw) : raw;
          if (!snap || !snap.dir) return;
          const text = JSON.stringify(snap);
          node.properties = node.properties || {};
          if (node.properties[SNAP_KEY] === text) return;
          node.properties[SNAP_KEY] = text;
          app.graph?.setDirtyCanvas?.(true, true);
        } catch (e) {}
      };

      // 工作流加载完成时按清单重建画面
      const origOnConfigure = node.onConfigure;
      node.onConfigure = function (info) {
        const r = typeof origOnConfigure === "function"
          ? origOnConfigure.apply(this, arguments)
          : undefined;
        try {
          const raw = node.properties?.[SNAP_KEY];
          if (!raw) return r;
          const snap = JSON.parse(raw);
          if (!snap || !snap.dir) return r;

          if (snap.canvas) {
            st.canvasW = Number(snap.canvas[0]) || 0;
            st.canvasH = Number(snap.canvas[1]) || 0;
          }
          if (snap.rects) st.rects = { main: snap.rects.main, thumbs: snap.rects.thumbs || [] };
          st.border = snap.border || 0;
          st.bg = snap.bg || "#202020";

          const mk = (entry, cb) => {
            const u = snapUrl(entry, snap.dir);
            if (!u) return;
            const im = new Image();
            im.onload = () => {
              cb(im);
              renderSheet();
            };
            im.src = u;
          };

          mk(snap.main, (im) => { st.mainImg = im; st._fallbackImg = im; });
          st.varImgs = [];
          (snap.vars || []).forEach((e, i) => mk(e, (im) => (st.varImgs[i] = im)));
          st.thumbImgs = [];
          (snap.thumbs || []).forEach((e, i) => mk(e, (im) => (st.thumbImgs[i] = im)));
          st.selected = Number(findWidget(node, "selected")?.value ?? 0);
        } catch (e) {}
        return r;
      };

      // 8. 右键菜单兜底
      const origMenu = node.getExtraMenuOptions;
      node.getExtraMenuOptions = function (canvas, options) {
        if (typeof origMenu === "function") origMenu.apply(this, arguments);
        options.push({
          content: "↩ 原主图 (selected = 0)",
          callback: () => applySelection(0),
        });
        for (let i = 0; i < st.rects.thumbs.length; i++) {
          options.push({
            content: `🎭 差分 ${i + 1}`,
            callback: () => applySelection(i + 1),
          });
        }
      };
    };
  },
});
