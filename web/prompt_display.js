import { app } from "../../scripts/app.js";

const NODE_CLASS = "IllustriousPromptDisplay";

app.registerExtension({
  name: "Illustrious.PromptDisplay",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData?.name !== NODE_CLASS) return;

    // 节点创建时挂一个只读的文本显示区
    const onCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      if (onCreated) onCreated.apply(this, arguments);

      const el = document.createElement("pre");
      el.style.cssText =
        "white-space:pre-wrap;word-break:break-word;font-size:12px;line-height:1.5;" +
        "height:180px;overflow:auto;margin:0;padding:6px;background:#0f1218;" +
        "border:1px solid #262c3a;border-radius:6px;color:#d6dbe8;";
      this.__promptEl = el;
      this.addDOMWidget("prompt_display", "display", el, { serialize: false, hideOnZoom: false });
      try {
        this.setSize([Math.max(this.size[0], 320), Math.max(this.size[1], 230)]);
      } catch (e) {
        /* ignore */
      }
    };

    // 执行后把收到的文本显示出来
    const onExecuted = nodeType.prototype.onExecuted;
    nodeType.prototype.onExecuted = function (message) {
      if (onExecuted) onExecuted.apply(this, arguments);
      const t =
        (message && message.text && message.text[0]) ||
        (message && message.display && message.display[0]) ||
        "";
      if (this.__promptEl) this.__promptEl.textContent = t;
    };
  },
});
