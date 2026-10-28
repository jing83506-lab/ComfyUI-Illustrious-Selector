import os
import re
import json
import random

DIR = os.path.dirname(os.path.abspath(__file__))
CUSTOM_FILE = os.path.join(DIR, 'characters_custom.json')
FAVORITES_FILE = os.path.join(DIR, 'favorites.json')
CHARACTER_LORAS_FILE = os.path.join(DIR, 'character_loras.json')

_WEBUI_TAG = re.compile(r'<(?:lora|lyco|hypernet):[^>]*>', re.IGNORECASE)
_CONTROL = ('None', 'random')

DEFAULT_PREFIX = "masterpiece, best quality, amazing quality, very aesthetic, absurdres, newest, year 2025"


def _strip_webui_tags(text):
    """Remove A1111-only <lora:...>/<lyco:...> syntax that ComfyUI does not parse."""
    text = _WEBUI_TAG.sub('', text)
    tokens = [t.strip() for t in text.split(',')]
    tokens = [t for t in tokens if t]
    return ', '.join(tokens)


def _load_json(name):
    path = os.path.join(DIR, name)
    if os.path.exists(path):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"[Illustrious-Selector] failed to load {name}: {e}")
    return {}


def _merge(base, extra):
    if isinstance(extra, dict):
        base.update(extra)
    return base


# characters_i18n.json: { "<中文名>": "<english character, series>" } (display -> prompt)
CHARACTERS = _merge({}, _load_json('characters_i18n.json'))
if not CHARACTERS:
    CHARACTERS = _load_json('characters.json')
_merge(CHARACTERS, _load_json('custom_character.json'))
# 用户新增/修改的角色（持久化，优先于内置数据）
_merge(CHARACTERS, _load_json('characters_custom.json'))

# action.json: { "<action>": "<tag string>" }
ACTIONS = _merge({}, _load_json('action.json'))
_merge(ACTIONS, _load_json('custom_action.json'))


def _pinyin_key(name):
    """按拼音排序的 key：中文转拼音（如 塔什干 -> tashigan），非中文原样按字母排。"""
    try:
        from pypinyin import lazy_pinyin
        return ("".join(lazy_pinyin(name)).lower(), name)
    except Exception:
        return (name.lower(), name)


_favorites_cache = None


def load_favorites():
    """读取收藏角色列表（缓存，toggle 后更新）。"""
    global _favorites_cache
    if _favorites_cache is None:
        data = _load_json('favorites.json')
        favs = data.get('favorites') if isinstance(data, dict) else None
        _favorites_cache = [str(x) for x in favs] if isinstance(favs, list) else []
    return list(_favorites_cache)


def _save_favorites(favs):
    global _favorites_cache
    _favorites_cache = [str(x) for x in favs]
    try:
        with open(FAVORITES_FILE, 'w', encoding='utf-8') as f:
            json.dump({'favorites': _favorites_cache}, f, ensure_ascii=False, indent=0)
        return True
    except Exception as e:
        print(f"[Illustrious-Selector] failed to save favorites: {e}")
        return False


def toggle_favorite(name):
    """收藏 / 取消收藏；返回 (是否成功, 最新收藏列表)。"""
    name = (name or '').strip()
    if not name or name in _CONTROL:
        return False, load_favorites()
    favs = load_favorites()
    if name in favs:
        favs.remove(name)
    else:
        favs.append(name)
    _save_favorites(favs)
    return True, favs


_char_thumbs = None


# action.json 里引用的 LoRA 大多是 SD1.5 版本，和 Illustrious(SDXL) 不兼容，
# 强行加载会破坏画面。这里按「文件名小写」跳过（与新项目 custom_lora_loader 的黑名单一致）。
_INCOMPATIBLE_LORAS = {
    'cuminass-000015', 'reverse suspended congress_redim', 'RSCongress'.lower(),
    'pov_strangling', 'phpose-000011r', 'hairholdingblowjob', 'add_detail',
    'torogao_v3-000010', 'symmetrical docking orgy', 'chihunhentai',
    'upright_front_above_50', 'upright_front_leg_lock_2',
    'broken-15', 'hhbjbbc', 'vacuum fellatio 2_1', 'vacuumfera2',
}


_LORA_TAG = re.compile(r'<lora:([^:>]+?)(?::(-?\d+(?:\.\d+)?))?\s*>', re.IGNORECASE)


def extract_loras(text):
    """从文本里抽取 <lora:name:weight> 形式的 LoRA，返回 [(name, weight), ...]。"""
    out = []
    for m in _LORA_TAG.finditer(text or ''):
        name = (m.group(1) or '').strip()
        try:
            weight = float(m.group(2)) if m.group(2) is not None else 1.0
        except Exception:
            weight = 1.0
        if name:
            out.append((name, weight))
    return out


def lora_spec_str(loras):
    """[(name, weight)] -> 'name:0.8|name2:0.5'"""
    return '|'.join('%s:%g' % (n, w) for n, w in loras)


_char_loras = None


def load_char_loras():
    """角色 -> 绑定 LoRA 的映射（缓存）。"""
    global _char_loras
    if _char_loras is None:
        data = _load_json('character_loras.json')
        _char_loras = data if isinstance(data, dict) else {}
    return _char_loras


def save_char_lora(character, lora, weight=1.0):
    """给角色绑定 LoRA（lora 为空则解绑）。"""
    global _char_loras
    character = (character or '').strip()
    if not character or character in _CONTROL:
        return False
    lora = (lora or '').strip()
    data = dict(load_char_loras())
    if lora:
        try:
            data[character] = {'lora': lora, 'weight': float(weight)}
        except Exception:
            data[character] = {'lora': lora, 'weight': 1.0}
    else:
        data.pop(character, None)
    _char_loras = data
    try:
        with open(CHARACTER_LORAS_FILE, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=0)
        print(f"[Illustrious-Selector] character lora: {character} -> {lora or '(cleared)'}")
        return True
    except Exception as e:
        print(f"[Illustrious-Selector] failed to save character lora: {e}")
        return False


def _norm_tag(text):
    """把 tag 串归一化成排序后的 token 集合，用来跨来源匹配（容忍顺序/下划线差异）。"""
    tokens = []
    for part in re.split(r'[,;]+', text or ''):
        tokens += re.split(r'[\s_]+', part.strip().lower())
    return ' '.join(sorted(set(t for t in tokens if t)))


def load_char_thumbs():
    """角色缩略图索引（归一化 tag -> data URL），懒加载。"""
    global _char_thumbs
    if _char_thumbs is None:
        raw = _load_json('char_thumbs.json') or {}
        _char_thumbs = {}
        for k, v in raw.items():
            _char_thumbs[_norm_tag(k)] = v
        print(f"[Illustrious-Selector] loaded {len(_char_thumbs)} character thumbnails")
    return _char_thumbs


def character_thumb(name):
    """返回所选角色的缩略图 data URL；没有则返回空字符串。"""
    name = (name or '').strip()
    if not name or name in _CONTROL:
        return ''
    tag = CHARACTERS.get(name)
    if not tag:
        return ''
    return load_char_thumbs().get(_norm_tag(tag), '')


def char_keys():
    """动态生成角色下拉列表：收藏角色置顶，其余跟随；两组都按拼音排序。"""
    names = [k for k in CHARACTERS.keys() if k not in _CONTROL and CHARACTERS[k]]
    fav_set = set(n for n in load_favorites() if n in names)
    head = sorted(fav_set, key=_pinyin_key)
    tail = sorted((n for n in names if n not in fav_set), key=_pinyin_key)
    return ['None', 'random'] + head + tail


def action_keys():
    return ['None', 'random'] + sorted(
        (k for k in ACTIONS.keys() if k not in _CONTROL and ACTIONS[k]),
        key=_pinyin_key,
    )


def save_custom(name, prompt):
    """Persist a user-added / user-edited character and update the in-memory dict."""
    name = (name or '').strip()
    prompt = (prompt or '').strip()
    if not name or not prompt:
        return False
    data = _load_json('characters_custom.json')
    data[name] = prompt
    try:
        with open(CUSTOM_FILE, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=0)
        CHARACTERS[name] = prompt
        print(f"[Illustrious-Selector] saved custom character: {name}")
        return True
    except Exception as e:
        print(f"[Illustrious-Selector] failed to save custom character: {e}")
        return False


class IllustriousCharacterActionSelector:
    @classmethod
    def INPUT_TYPES(cls):
        # 前 5 个保持旧版顺序，新增项追加到末尾，避免已保存工作流的值按位置错位。
        # character / action 列表每次都动态生成，保存后刷新即可见。
        return {
            "required": {
                "character": (char_keys(), {"default": "None"}),
                "action": (action_keys(), {"default": "None"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFF}),
                "prefix": ("STRING", {"multiline": True, "default": DEFAULT_PREFIX}),
                "suffix": ("STRING", {"multiline": True, "default": ""}),
                "character_prompt_override": (
                    "STRING",
                    {"multiline": True, "default": "",
                     "tooltip": "填写则覆盖所选角色的提示词（留空=用角色库原值）。"},
                ),
                "new_character_name": (
                    "STRING",
                    {"multiline": False, "default": "",
                     "tooltip": "新增角色名（会成为下拉选项）。与下面提示词一起填时保存。"},
                ),
                "new_character_prompt": (
                    "STRING",
                    {"multiline": True, "default": "",
                     "tooltip": "新增角色的提示词（英文 tag）。同名会覆盖旧角色。"},
                ),
                "character_lora": (
                    "STRING",
                    {"multiline": False, "default": "",
                     "tooltip": "给【当前下拉选中的角色】绑定 LoRA 名（可不带 .safetensors）。点保存按钮写入；留空=解绑。"},
                ),
                "character_lora_weight": (
                    "STRING",
                    {"multiline": False, "default": "1.0",
                     "tooltip": "角色 LoRA 权重（数字字符串，如 0.8）。"},
                ),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("prompt", "lora_spec")
    FUNCTION = "run"
    CATEGORY = "Illustrious/Prompt"

    def run(self, character, action, seed, prefix, suffix,
            character_prompt_override, new_character_name, new_character_prompt,
            character_lora, character_lora_weight):
        rng = random.Random(int(seed))

        # 新增/修改持久化（按钮也会走同一个后端保存逻辑）
        if (new_character_name or '').strip() and (new_character_prompt or '').strip():
            save_custom(new_character_name, new_character_prompt)

        # 角色提示词：override 优先
        override = (character_prompt_override or '').strip()
        if override:
            char_tag = override
        elif character == 'random':
            valid = [k for k in char_keys() if k not in _CONTROL]
            char_tag = CHARACTERS.get(rng.choice(valid), '') if valid else ''
        elif character == 'None':
            char_tag = ''
        else:
            char_tag = CHARACTERS.get(character, '')

        if action == 'random':
            valid = [k for k in action_keys() if k not in _CONTROL]
            action_tag = ACTIONS.get(rng.choice(valid), '') if valid else ''
        elif action == 'None':
            action_tag = ''
        else:
            action_tag = ACTIONS.get(action, '')

        # LoRA：action 里内嵌的 <lora:...> + 角色绑定的 LoRA
        loras = list(extract_loras(action_tag))
        if character not in _CONTROL:
            bind = load_char_loras().get(character)
            if isinstance(bind, dict) and bind.get('lora'):
                loras.append((bind['lora'], float(bind.get('weight', 1.0))))

        parts = [prefix.strip(), char_tag.strip(), str(action_tag).strip(), suffix.strip()]
        parts = [p.strip(' ,') for p in parts if p and p.strip()]
        return (_strip_webui_tags(", ".join(parts)), lora_spec_str(loras))


class IllustriousLoraStack:
    """根据 lora_spec（如 'name:0.8|name2:0.5'）自动加载一叠 LoRA。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "lora_spec": ("STRING", {"multiline": False, "forceInput": True, "default": "",
                                        "tooltip": "接选择器的 lora_spec 输出，格式 name:weight|name2:weight2。"}),
                "strength_multiplier": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 4.0, "step": 0.05}),
                "silent": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("MODEL", "CLIP")
    RETURN_NAMES = ("model", "clip")
    FUNCTION = "run"
    CATEGORY = "Illustrious/LoRA"

    def run(self, model, clip, lora_spec, strength_multiplier, silent):
        import folder_paths
        import comfy.sd
        import comfy.utils

        spec = (lora_spec or '').strip()
        if not spec:
            return (model, clip)

        available = set(folder_paths.get_filename_list("loras"))
        loaded = []
        for item in spec.split('|'):
            item = item.strip()
            if not item:
                continue
            if ':' in item:
                name, _, w = item.rpartition(':')
                try:
                    weight = float(w)
                except Exception:
                    name, weight = item, 1.0
            else:
                name, weight = item, 1.0
            name = name.strip()
            stem = os.path.splitext(name)[0].lower()
            if stem in _INCOMPATIBLE_LORAS or name.lower() in _INCOMPATIBLE_LORAS:
                if not silent:
                    print(f"[Illustrious-LoraStack] skip incompatible with SDXL (SD1.5 LoRA?): {name}")
                continue
            fname = None
            for cand in (name, name + '.safetensors', name + '.pt', name + '.ckpt'):
                if cand in available:
                    fname = cand
                    break
            if not fname:
                if not silent:
                    print(f"[Illustrious-LoraStack] not found, skip: {name}")
                continue
            try:
                path = folder_paths.get_full_path("loras", fname)
                lora = comfy.utils.load_torch_file(path, safe_load=True)
                s = weight * float(strength_multiplier)
                model, clip = comfy.sd.load_lora_for_models(model, clip, lora, s, s)
                loaded.append((fname, s))
            except Exception as e:
                if not silent:
                    print(f"[Illustrious-LoraStack] failed {fname}: {e}")
        if loaded and not silent:
            print("[Illustrious-LoraStack] loaded: " + ", ".join(f"{n}@{w:g}" for n, w in loaded))
        return (model, clip)


class IllustriousVariationGate:
    """差分分支开关：enable=False 时用 ExecutionBlocker 阻断下游，直接跳过差分生成。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "enable": ("BOOLEAN", {"default": True,
                                       "label_on": "生成差分", "label_off": "只要主图"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"
    CATEGORY = "Illustrious/Image"

    def run(self, image, enable):
        if not enable:
            try:
                from comfy_execution.graph import ExecutionBlocker
                print("[Illustrious-Gate] 差分已关闭，跳过下游分支")
                return (ExecutionBlocker(None),)
            except Exception as e:
                print(f"[Illustrious-Gate] ExecutionBlocker 不可用（{e}），差分分支仍会执行")
        return (image,)


class IllustriousExpressionSheet:
    """表情差分拼版：显示用脸部小图，切换用 FaceDetailer 输出的整图（含环境/身体）。
    在节点里左键点击小图即可把主图切换成对应差分。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "main_image": ("IMAGE",),
                "enable": ("BOOLEAN", {"default": True}),
                "gap": ("INT", {"default": 8, "min": 0, "max": 128, "step": 1}),
                "border": ("INT", {"default": 3, "min": 0, "max": 32, "step": 1}),
                "var_scale": ("FLOAT", {"default": 0.18, "min": 0.05, "max": 0.6, "step": 0.01,
                                        "tooltip": "每个小图长边占主图长边的比例"}),
                "bg_color": ("STRING", {"default": "202020"}),
                "border_color": ("STRING", {"default": "FFFFFF"}),
                "selected": ("INT", {"default": 0, "min": 0, "max": 6, "step": 1,
                                     "tooltip": "0=原主图；1~6=对应差分。在节点里点小图自动切换。"}),
            },
            "optional": {
                "vars": ("IMAGE", {"tooltip": "差分整图（批处理：每张一个差分，用于切换/保存）"}),
                "thumbs": ("IMAGE", {"tooltip": "脸部小图（批处理：每张一个差分，用于显示）"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "INT")
    RETURN_NAMES = ("sheet", "selected_image", "selected_index")
    FUNCTION = "run"
    CATEGORY = "Illustrious/Image"

    @staticmethod
    def _hex(s, torch):
        s = (s or "").strip().lstrip("#")
        if len(s) == 3:
            s = "".join(c * 2 for c in s)
        try:
            return torch.tensor([int(s[i:i + 2], 16) / 255.0 for i in (0, 2, 4)])
        except Exception:
            return torch.tensor([0.125, 0.125, 0.125])

    @staticmethod
    def _preview_b64(tensor, max_side=1280):
        """把 sheet 转成 base64 JPEG（限尺寸），供节点内绘制 + 点击定位。"""
        import base64
        import io

        from PIL import Image

        arr = (tensor[0].cpu().numpy() * 255).clip(0, 255).astype("uint8")
        img = Image.fromarray(arr)
        scale = min(1.0, max_side / max(img.width, img.height))
        if scale < 1.0:
            img = img.resize((max(1, int(img.width * scale)), max(1, int(img.height * scale))), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=85)
        return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode(), scale

    def run(self, main_image, enable, gap, border, var_scale, bg_color, border_color, selected,
            vars=None, thumbs=None):
        import torch
        import torch.nn.functional as F

        def _split(t):
            if t is None:
                return []
            n = int(t.shape[0])
            return [t[i:i + 1] for i in range(n)]

        fulls = _split(vars)
        thumbs = _split(thumbs)
        if not thumbs:
            thumbs = fulls

        # 当前主图：selected=0 用原图，否则用对应差分的【整图】
        idx = int(selected)
        current = main_image
        if 1 <= idx <= len(fulls):
            current = fulls[idx - 1]
        elif idx > len(fulls):
            idx = 0

        if not enable or not thumbs:
            return {"ui": {"selected": [idx]}, "result": (current, current, idx)}

        main = current
        H, W = int(main.shape[1]), int(main.shape[2])
        landscape = W > H
        n = len(thumbs)

        def _resize(img, w, h):
            x = img.permute(0, 3, 1, 2)
            x = F.interpolate(x, size=(h, w), mode="bicubic", align_corners=False).clamp(0, 1)
            return x.permute(0, 2, 3, 1)

        cells = []
        for v in thumbs:
            vh, vw = int(v.shape[1]), int(v.shape[2])
            if landscape:
                cw = max(1, int(round(var_scale * W)))
                ch = max(1, int(round(cw * vh / vw)))
            else:
                ch = max(1, int(round(var_scale * H)))
                cw = max(1, int(round(ch * vw / vh)))
            cells.append(_resize(v, cw, ch))

        def _strip_len():
            if landscape:
                return sum(c.shape[2] for c in cells) + gap * (n - 1) + 2 * border * n
            return sum(c.shape[1] for c in cells) + gap * (n - 1) + 2 * border * n

        limit = W if landscape else H
        if _strip_len() > limit:
            k = limit / _strip_len()
            cells = [_resize(c, max(1, int(c.shape[2] * k)), max(1, int(c.shape[1] * k))) for c in cells]

        bg = self._hex(bg_color, torch)
        bd = self._hex(border_color, torch)
        strip = _strip_len()
        rects = []

        if landscape:
            row_h = max(c.shape[1] for c in cells) + 2 * border
            CW = max(W, strip)
            CH = H + gap + row_h
            x = max(0, (CW - strip) // 2)
            y0 = H + gap
            canvas = torch.zeros((1, CH, CW, 3), dtype=main.dtype)
            canvas[:] = bg
            canvas[0, 0:H, 0:W] = main[0]
            for i, c in enumerate(cells):
                h, w = c.shape[1], c.shape[2]
                canvas[0, y0:y0 + h + 2 * border, x:x + w + 2 * border] = bd
                canvas[0, y0 + border:y0 + border + h, x + border:x + border + w] = c[0]
                rects.append({"i": i + 1, "x": int(x), "y": int(y0), "w": int(w + 2 * border), "h": int(h + 2 * border)})
                x += w + 2 * border + gap
        else:
            col_w = max(c.shape[2] for c in cells) + 2 * border
            CH = max(H, strip)
            CW = W + gap + col_w
            y = max(0, (CH - strip) // 2)
            x0 = W + gap
            canvas = torch.zeros((1, CH, CW, 3), dtype=main.dtype)
            canvas[:] = bg
            canvas[0, 0:H, 0:W] = main[0]
            for i, c in enumerate(cells):
                h, w = c.shape[1], c.shape[2]
                canvas[0, y:y + h + 2 * border, x0:x0 + w + 2 * border] = bd
                canvas[0, y + border:y + border + h, x0 + border:x0 + border + w] = c[0]
                rects.append({"i": i + 1, "x": int(x0), "y": int(y), "w": int(w + 2 * border), "h": int(h + 2 * border)})
                y += h + 2 * border + gap

        ui = {"selected": [idx], "canvas": [[int(canvas.shape[2]), int(canvas.shape[1])]]}
        try:
            # 整张 sheet（兜底用）
            b64, scale = self._preview_b64(canvas)
            ui["preview"] = [b64]
            ui["preview_scale"] = [scale]
            # 前端合成用的素材（原始 sheet 坐标，前端自己缩放）
            ui["rects"] = [json.dumps({
                "main": {"x": 0, "y": 0, "w": int(W), "h": int(H)},
                "thumbs": rects,
                "border": int(border),
                "bg": str(bg_color),
            })]
            try:
                ui["main_img"] = [self._preview_b64(main_image, 2048)[0]]
            except Exception as e:
                print(f"[Illustrious-Sheet] main preview failed: {e}")
            try:
                ui["var_imgs"] = [self._preview_b64(v, 2048)[0] for v in fulls]
            except Exception as e:
                print(f"[Illustrious-Sheet] var previews failed: {e}")
            try:
                ui["thumb_imgs"] = [self._preview_b64(t, 800)[0] for t in thumbs]
            except Exception as e:
                print(f"[Illustrious-Sheet] thumb previews failed: {e}")
        except Exception as e:
            print(f"[Illustrious-Sheet] preview failed: {e}")

        return {"ui": ui, "result": (canvas, current, idx)}


class IllustriousVariationRouter:
    """把「选中的差分图」和「对应的表情 conditioning」成对路由到下游。

    差分的 conditioning 必须跟着图一起走，否则后续高清/分块重绘会把表情改回去。
    index 来自 ExpressionSheet 的 selected_index（0=原图，1~5=各差分）。
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "index": ("INT", {"default": 1, "min": 0, "max": 5, "step": 1,
                                  "tooltip": "0=原图（用 cond_base）；1~5=对应差分"}),
                "cond_base": ("CONDITIONING",),
                "cond_1": ("CONDITIONING",),
                "cond_2": ("CONDITIONING",),
                "cond_3": ("CONDITIONING",),
                "cond_4": ("CONDITIONING",),
                "cond_5": ("CONDITIONING",),
            },
        }

    RETURN_TYPES = ("IMAGE", "CONDITIONING", "STRING")
    RETURN_NAMES = ("image", "conditioning", "which")
    FUNCTION = "run"
    CATEGORY = "Illustrious/Image"

    def run(self, image, index, cond_base, cond_1, cond_2, cond_3, cond_4, cond_5):
        idx = int(index)
        table = [
            ("base", cond_base),
            ("var_1", cond_1),
            ("var_2", cond_2),
            ("var_3", cond_3),
            ("var_4", cond_4),
            ("var_5", cond_5),
        ]
        if idx < 0 or idx > 5:
            idx = 0
        name, cond = table[idx]
        print(f"[Illustrious-Router] 选中 {name} (index={idx})")
        return (image, cond, name)


class IllustriousSeedHub:
    """种子枢纽：全流程只维护一个种子。

    改这一个值 → 所有接进来的采样器一起换种子 → 整条链重roll。
    不动它 → 缓存全部命中，可以只跑下游（选差分/换参数）。
    第 k 个输出 = seed + (k-1)，保证各阶段种子互不相同又完全可复现。
    """

    N_OUT = 10

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFF,
                                 "control_after_generate": True,
                                 "tooltip": "总种子。改这个值=整条链重roll；不动它=只跑下游。"}),
                "offset_style": (["递增(+k)", "全部相同"], {"default": "递增(+k)",
                                  "tooltip": "递增：seed_k = seed+k-1；全部相同：所有输出都等于 seed"}),
            },
        }

    RETURN_TYPES = ("INT",) * 10
    RETURN_NAMES = tuple("seed_%d" % (i + 1) for i in range(10))
    FUNCTION = "run"
    CATEGORY = "Illustrious/Image"

    def run(self, seed, offset_style="递增(+k)"):
        base = int(seed)
        # 兼容前端把 control_after_generate 的值误传进来的情况
        if offset_style not in ("递增(+k)", "全部相同"):
            offset_style = "递增(+k)"
        if offset_style == "全部相同":
            vals = tuple(base for _ in range(self.N_OUT))
        else:
            vals = tuple(base + i for i in range(self.N_OUT))
        print("[Illustrious-SeedHub] seed=%d (%s) -> %s" % (base, offset_style, list(vals[:4])))
        return vals


class IllustriousSaveImageEx:
    """保存为 JPG / WebP / PNG，可把工作流写进 EXIF。

    核心 SaveImage 只能存 PNG，7MP 差分一张就十几 MB；这里存 JPG/WebP 能小 5~10 倍。
    - WebP：体积最小，且 EXIF 里能放完整工作流（ComfyUI 可读回）
    - JPG ：兼容性最好；但 JPEG 的 APP1 段上限 65533 字节，
            工作流 JSON 常超过这个值，所以只嵌 prompt（执行参数）
    """

    def __init__(self):
        import folder_paths
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "filename_prefix": ("STRING", {"default": "ComfyUI"}),
                "format": (["webp", "jpg", "png"], {"default": "webp",
                                                   "tooltip": "jpg=兼容性最好；webp=最小且能存完整工作流"}),
                "quality": ("INT", {"default": 92, "min": 40, "max": 100, "step": 1}),
                "embed_workflow": ("BOOLEAN", {"default": True,
                                               "tooltip": "把工作流/提示词写进 EXIF（WebP 可被 ComfyUI 读回）"}),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    RETURN_TYPES = ()
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = "Illustrious/Image"

    def save(self, images, filename_prefix="ComfyUI", format="jpg", quality=92,
             embed_workflow=True, prompt=None, extra_pnginfo=None):
        import json
        import os

        import folder_paths
        import numpy as np
        from PIL import Image

        full_output_folder, filename, counter, subfolder, _ = folder_paths.get_save_image_path(
            filename_prefix, self.output_dir, images[0].shape[1], images[0].shape[0])

        wf_txt = None
        pr_txt = None
        if extra_pnginfo and isinstance(extra_pnginfo, dict):
            w = extra_pnginfo.get("workflow")
            if w is not None:
                wf_txt = w if isinstance(w, str) else json.dumps(w, ensure_ascii=False)
        if prompt is not None:
            pr_txt = prompt if isinstance(prompt, str) else json.dumps(prompt, ensure_ascii=False)

        results = []
        for image in images:
            arr = np.clip(255.0 * image.cpu().numpy(), 0, 255).astype(np.uint8)
            pim = Image.fromarray(arr)

            ex = Image.Exif()
            if embed_workflow:
                # JPEG 的 APP1 段硬上限 65533 字节；WebP 的 EXIF chunk 是 32 位长度，没有这个限制
                _lim = 60000 if format == "jpg" else 10 ** 9
                if wf_txt and len(wf_txt) < _lim:
                    ex[0x010E] = "Workflow: " + wf_txt
                elif wf_txt:
                    print("[Illustrious-SaveEx] 工作流 %d 字节超过 JPEG EXIF 上限，已跳过（改用 webp 可保存）"
                          % len(wf_txt))
                if pr_txt and len(pr_txt) < _lim:
                    ex[0x013B] = "Prompt: " + pr_txt

            fname = "%s_%05d_.%s" % (filename, counter, format)
            path = os.path.join(full_output_folder, fname)

            if format == "png":
                pim.save(path, "PNG", compress_level=4)
            elif format == "webp":
                # method 只影响压缩耗时/体积，不影响画质：实测 7MP 图
                # method=6 -> 3.07s / 1.15MB ；method=2 -> 0.91s / 1.24MB
                # 取 2，编码快 3 倍、体积只多 8%
                pim.save(path, "WEBP", quality=int(quality), method=2,
                         exif=ex.tobytes())
            else:
                pim.save(path, "JPEG", quality=int(quality), subsampling=0,
                         optimize=True, exif=ex.tobytes())

            results.append({"filename": fname, "subfolder": subfolder, "type": "output"})
            counter += 1

        print("[Illustrious-SaveEx] 保存 %d 张 (%s q=%d) 到 %s"
              % (len(results), format, quality, full_output_folder))
        return {"ui": {"images": results}}


class IllustriousSceneTags:
    """把 WD tagger 的输出过滤成"场景提示词"。

    只保留背景/自然/地点/设定/物品/道具/姿势/动作这累类别，
    发色、瞳色、服装、体型、角色名等全部丢弃（靠白名单，不靠黑名单）。
    """

    _CACHE = None

    DEFAULT_INCLUDE = ("pose, action, gesture, camera, background, nature, "
                       "location, environment, setting, object, props, food, "
                       "furniture, plant, weather")

    # 内置永久排除：这些类别代表「角色自身特征」，无论白名单怎么写都不该进提示词。
    # 用「命中任一即丢弃」而不是「必须全不命中」——因为标签是多标签的，
    # 例如 sitting = [body, pose]，如果按「含 body 就丢」会把姿势也丢掉，所以
    # body 不在排除表里，只排真正的角色外观类。
    BASE_EXCLUDE = frozenset((
        "character", "name", "animal_features", "animal_ears", "ears",
        "tail", "body_parts", "hair", "face", "clothing", "costume",
        "accessories", "accessory", "headwear", "footwear", "legwear",
        "underwear", "pattern", "color",
    ))

    # 内置标签黑名单：分类上属于 camera/camera_subject，但语义是「人数」，
    # 不属于场景描述（本工作流恒为单人）。
    BASE_EXCLUDE_TAGS = frozenset((
        "solo", "1girl", "1boy", "multiple girls", "multiple boys",
        "duo", "group", "2girls", "3girls", "6+girls", "male focus",
    ))

    @classmethod
    def _cat_table(cls):
        if cls._CACHE is None:
            import json
            import os
            p = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "tag_category_v2.json")
            try:
                with open(p, encoding="utf-8") as f:
                    cls._CACHE = json.load(f)
            except Exception as e:
                print("[Illustrious-SceneTags] 分类表读取失败: %s" % e)
                cls._CACHE = {}
        return cls._CACHE

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "default": "",
                                    "tooltip": "接 WD Timm Tagger 的 STRING 输出"}),
                "enable": ("BOOLEAN", {"default": True,
                                       "tooltip": "关掉时输出空字符串，下游不受影响"}),
                "include_categories": ("STRING", {"default": cls.DEFAULT_INCLUDE,
                                                  "tooltip": "保留这些类别（逗号分隔）"}),
                "exclude_categories": ("STRING", {"default": "",
                                                  "tooltip": "再额外剔除这些类别（逗号分隔）\n"
                                                             "内置已永久排除：角色/名字/动物耳尾/身体部件/头发/脸/服装/配饰/头饰/鞋/内衣/图案/颜色"}),
                "extra_exclude": ("STRING", {"default": "",
                                             "tooltip": "额外要剔除的具体标签（逗号分隔）"}),
                "max_tags": ("INT", {"default": 30, "min": 1, "max": 200, "step": 1}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("scene_tags",)
    FUNCTION = "run"
    CATEGORY = "Illustrious/Image"

    def run(self, text, enable=True, include_categories="", exclude_categories="",
            extra_exclude="", max_tags=30):
        if not enable:
            print("[Illustrious-SceneTags] 已关闭，输出空字符串")
            return ("",)
        if not (text or "").strip():
            return ("",)

        table = self._cat_table()
        inc = {c.strip().lower() for c in (include_categories or "").split(",") if c.strip()}
        exc = {c.strip().lower() for c in (exclude_categories or "").split(",") if c.strip()}
        exc |= self.BASE_EXCLUDE
        extra = {t.strip().lower() for t in (extra_exclude or "").split(",") if t.strip()}

        kept = []
        dropped = 0
        seen = set()
        for raw in (text or "").split(","):
            tag = raw.strip()
            if not tag:
                continue
            low = tag.lower()
            if low in seen:
                continue
            seen.add(low)
            if low in extra or low in self.BASE_EXCLUDE_TAGS:
                dropped += 1
                continue
            cats = table.get(low.replace(" ", "_")) or table.get(low)
            if not cats:
                dropped += 1          # 未收录：保守丢弃
                continue
            cs = {c.lower() for c in cats}
            if exc and (cs & exc):
                dropped += 1
                continue
            if not (cs & inc):
                dropped += 1
                continue
            kept.append(tag)

        out = kept[:int(max_tags)]
        print("[Illustrious-SceneTags] 保留 %d 个 / 丢弃 %d 个 -> %s"
              % (len(out), dropped, ", ".join(out[:8]) + ("..." if len(out) > 8 else "")))
        if not out:
            return ("",)
        # 末尾带 ", " ：本节点内部走的是 token 级拼接（不受影响），
        # 但下游若用 StringConcatenate / 手动复制 / 训练集打标，缺逗号会和后一个词粘起来。
        return (", ".join(out) + ", ",)


class IllustriousSceneConditioning:
    """把场景标签编码后拼进 conditioning。

    和直接用 ConditioningConcat 的区别：enable=False 时**原样透传**，
    不会塞进一段空条件，所以关闭后出图与完全没接这条线一致。
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "conditioning": ("CONDITIONING",),
                "text": ("STRING", {"multiline": True, "default": ""}),
                "enable": ("BOOLEAN", {"default": True}),
                "weight": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05,
                                     "tooltip": "拼接前对场景条件做强度缩放（1.0=原样）"}),
            },
        }

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "run"
    CATEGORY = "Illustrious/Image"

    def run(self, clip, conditioning, text, enable=True, weight=1.0):
        import torch
        txt = (text or "").strip()
        if not enable or not txt:
            print("[Illustrious-SceneCond] 透传（enable=%s, 有文本=%s）" % (enable, bool(txt)))
            return (conditioning,)

        from nodes import CLIPTextEncode
        scene_con = CLIPTextEncode().encode(clip, txt)[0]

        w = float(weight)
        if abs(w - 1.0) > 1e-3:
            scene_con = [[t, {**d, "strength": d.get("strength", 1.0) * w}]
                         for (t, d) in scene_con]

        # 与 ComfyUI 的 ConditioningConcat 等价：序列维拼接
        out = []
        src = scene_con[0][0]
        for (t1, d1) in conditioning:
            out.append([torch.cat((t1, src), dim=1), dict(d1)])
        print("[Illustrious-SceneCond] 已拼接 %d 字符的场景提示词（weight=%.2f）" % (len(txt), w))
        return (out,)


class _GuardedBBoxDetector:
    """BBOX_DETECTOR 协议的包装器：内部先跑两个检测器，再用 segm 否决 bbox。

    Impact 的 FaceDetailer 只需要 setAux() 和 detect() 两个方法。
    """

    def __init__(self, bbox_det, segm_det, segm_threshold=0.30, min_overlap=0.15,
                 keep_when_no_segm=True):
        self.bbox_det = bbox_det
        self.segm_det = segm_det
        self.segm_threshold = float(segm_threshold)
        self.min_overlap = float(min_overlap)
        self.keep_when_no_segm = bool(keep_when_no_segm)

    def setAux(self, x):
        try:
            self.bbox_det.setAux(x)
        except Exception:
            pass

    @staticmethod
    def _scalar(v, default=0.0):
        """把 numpy 标量 / 单元素数组 / python 数字统一成 float。

        Impact 的 SEG.confidence 有时是 np.float32，有时是 shape=(1,) 的数组，
        直接 float() 会抛 "only 0-dimensional arrays can be converted"。
        """
        try:
            import numpy as np
            a = np.asarray(v)
            if a.size == 0:
                return default
            return float(a.reshape(-1)[0])
        except Exception:
            return default

    @staticmethod
    def _seg_mask(seg, shape):
        import numpy as np
        m = np.zeros(shape, dtype=bool)
        try:
            # 注意：本方法是 staticmethod，不能写 self
            x1, y1, x2, y2 = [int(_GuardedBBoxDetector._scalar(v)) for v in seg.crop_region]
            cm = np.asarray(seg.cropped_mask)
            x1 = max(0, x1)
            y1 = max(0, y1)
            h = min(cm.shape[0], shape[0] - y1)
            w = min(cm.shape[1], shape[1] - x1)
            if h > 0 and w > 0:
                m[y1:y1 + h, x1:x1 + w] = cm[:h, :w] > 0.5
        except Exception as e:
            # 不要静默失败：这里一旦抛异常，掩码会全 False，所有框都会被误杀
            print("[Illustrious-GUARD] 取分割掩码失败（该 segm 将被忽略）: %s: %s"
                  % (type(e).__name__, str(e)[:120]))
        return m

    def detect(self, image, threshold, dilation, crop_factor, drop_size=1, detailer_hook=None):
        import numpy as np

        bbox_segs = self.bbox_det.detect(image, threshold, dilation, crop_factor,
                                        drop_size, detailer_hook)
        if bbox_segs is None or len(bbox_segs) < 2:
            return bbox_segs

        shape = bbox_segs[0]
        try:
            segm_segs = self.segm_det.detect(image, self.segm_threshold, 0, 1.0, 1, None)
        except Exception as e:
            print("[Illustrious-GUARD] segm 检测失败，直接放行全部 bbox: %s" % e)
            return bbox_segs

        segm_list = (segm_segs[1] if segm_segs and len(segm_segs) > 1 else []) or []
        if not segm_list:
            if self.keep_when_no_segm:
                print("[Illustrious-GUARD] segm 没给出任何掩码（可能是模型未命中），"
                      "按 keep_when_no_segm=True 放行 %d 个框" % len(bbox_segs[1]))
                return bbox_segs
            print("[Illustrious-GUARD] segm 无掩码 -> 全部否决")
            return (shape, [])

        # 把所有 segm 掩码并成一张全图掩码
        segm_all = np.zeros(shape, dtype=bool)
        for s in segm_list:
            segm_all |= self._seg_mask(s, shape)

        kept = []
        for seg in bbox_segs[1]:
            try:
                bx1, by1, bx2, by2 = [int(self._scalar(v)) for v in seg.bbox]
            except Exception:
                kept.append(seg)
                continue
            bx1 = max(0, bx1); by1 = max(0, by1)
            bx2 = min(shape[1], bx2); by2 = min(shape[0], by2)
            if bx2 <= bx1 or by2 <= by1:
                continue
            rect = segm_all[by1:by2, bx1:bx2]
            area = float(rect.size)
            cover = float(rect.sum()) / area if area > 0 else 0.0

            tag = "PASS" if cover >= self.min_overlap else "DROP"
            print("[Illustrious-GUARD] bbox(%d,%d,%d,%d) 置信 %.2f  分割覆盖 %.1f%%  %s"
                  % (bx1, by1, bx2, by2, self._scalar(getattr(seg, "confidence", 0.0)),
                     cover * 100.0, tag))
            if cover >= self.min_overlap:
                kept.append(seg)

        print("[Illustrious-GUARD] 候选 %d 个 -> 通过 %d 个（阈值 覆盖≥%.0f%%）"
              % (len(bbox_segs[1]), len(kept), self.min_overlap * 100))
        return (shape, kept)

    def detect_combined(self, image, threshold, dilation):
        from impact.core import segs_to_combined_mask
        return segs_to_combined_mask(self.detect(image, threshold, dilation, 1))


class IllustriousDetectGuard:
    """手部检测双重约束（bbox 出框 + segm 验证）。

    输出仍是 BBOX_DETECTOR，直接接 FaceDetailer 的 bbox_detector 输入，
    所以重绘区域保持为 bbox 矩形，不会被换成分割掩码。
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "bbox_detector": ("BBOX_DETECTOR",),
                "segm_detector": ("SEGM_DETECTOR",),
                "segm_threshold": ("FLOAT", {"default": 0.30, "min": 0.0, "max": 1.0, "step": 0.01,
                                             "tooltip": "分割检测器自己的置信度阈值（不是 bbox 的）"}),
                "min_overlap": ("FLOAT", {"default": 0.15, "min": 0.0, "max": 1.0, "step": 0.01,
                                          "tooltip": "框内被分割掩码覆盖的最小比例；低于它视为误检\n"
                                                     "真手通常 40~70%，误检接近 0%"}),
                "keep_when_no_segm": ("BOOLEAN", {"default": True,
                                                  "tooltip": "分割一个都没给时是否放行全部框\n"
                                                             "开=退回旧行为（防分割失手）"}),
            },
        }

    RETURN_TYPES = ("BBOX_DETECTOR",)
    RETURN_NAMES = ("bbox_detector",)
    FUNCTION = "run"
    CATEGORY = "Illustrious/Detect"

    def run(self, image, bbox_detector, segm_detector, segm_threshold=0.30,
            min_overlap=0.15, keep_when_no_segm=True):
        det = _GuardedBBoxDetector(bbox_detector, segm_detector, segm_threshold,
                                   min_overlap, keep_when_no_segm)
        print("[Illustrious-GUARD] 已启用：bbox × segm 交叉验证（覆盖阈值 %.0f%%）"
              % (float(min_overlap) * 100))
        return (det,)


class IllustriousFaceCrop:
    """用 FaceDetailer 输出的 mask 算出精确人脸 bbox，把图裁成贴脸特写。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "padding": ("FLOAT", {"default": 0.10, "min": 0.0, "max": 0.6, "step": 0.01,
                                      "tooltip": "在 bbox 外再扩出去的比例"}),
                "square": ("BOOLEAN", {"default": True,
                                       "tooltip": "裁成正方形，避免不同人脸框比例差异"}),
                "min_size": ("INT", {"default": 64, "min": 16, "max": 2048, "step": 8}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"
    CATEGORY = "Illustrious/Image"

    def run(self, image, mask, padding, square, min_size):
        import torch

        H, W = int(image.shape[1]), int(image.shape[2])
        m = mask
        if m.dim() == 3:
            m = m.max(dim=0).values
        m = m.float()
        ys, xs = torch.where(m > 0.05)
        if ys.numel() == 0:
            return (image,)

        y0, y1 = int(ys.min()), int(ys.max()) + 1
        x0, x1 = int(xs.min()), int(xs.max()) + 1

        # mask 与 image 尺寸可能不同（guide_size 缩放），按比例换算
        mh, mw = int(m.shape[0]), int(m.shape[1])
        if (mh, mw) != (H, W):
            sx, sy = W / float(mw), H / float(mh)
            x0, x1 = int(round(x0 * sx)), int(round(x1 * sx))
            y0, y1 = int(round(y0 * sy)), int(round(y1 * sy))

        # 外扩
        bw, bh = max(1, x1 - x0), max(1, y1 - y0)
        px, py = int(round(bw * padding)), int(round(bh * padding))
        x0 = max(0, x0 - px); x1 = min(W, x1 + px)
        y0 = max(0, y0 - py); y1 = min(H, y1 + py)

        if square:
            side = max(x1 - x0, y1 - y0, min_size)
            side = min(side, W, H)
            cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
            x0 = max(0, min(W - side, cx - side // 2))
            y0 = max(0, min(H - side, cy - side // 2))
            x1, y1 = x0 + side, y0 + side

        # 太小的直接跳过裁切（避免放大成糊图）
        if (x1 - x0) < min_size or (y1 - y0) < min_size:
            return (image,)

        return (image[:, y0:y1, x0:x1, :],)


class IllustriousPromptDisplay:
    """接收来自角色选择器（或任意节点）的提示词，并在节点内显示出来。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "forceInput": True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "run"
    CATEGORY = "Illustrious/Prompt"
    OUTPUT_NODE = True

    def run(self, text):
        # 返回 ui.text 供前端显示；同时把文本原样透传出去
        return {"ui": {"text": (text,)}, "result": (text,)}


def _annotate_tensor(model, tensor_image, resolution=1024, **kwargs):
    """等价于 comfyui_controlnet_aux 的 common_annotator_call。"""
    import numpy as np
    import torch

    batch = tensor_image.shape[0]
    out_tensor = None
    for i, image in enumerate(tensor_image):
        np_image = np.asarray(image.cpu() * 255.0, dtype=np.uint8)
        np_result = model(np_image, output_type="np", detect_resolution=resolution, **kwargs)
        out = torch.from_numpy(np.asarray(np_result).astype(np.float32) / 255.0)
        if out_tensor is None:
            out_tensor = torch.zeros(batch, *out.shape, dtype=torch.float32)
        out_tensor[i] = out
    return out_tensor


class IllustriousEdgeSelector:
    """边缘控制选择器：canny / lineart / softedge，输出边缘图给 ControlNet 用。

    注意：本节点只负责「出边缘图」，下游 ControlNet 的模型要按 mode 手动换。
    用 canny 的 ControlNet 去接 lineart 的边缘图，线条会发灰、发糊。
    """

    MODES = ["canny", "lineart", "softedge"]

    # mode -> 建议的 ControlNet 模型（仅供参考，用户自己在 ControlNetLoader 里选）
    HINTS = {
        "canny": "controlnet-canny-sdxl-1.0-v2 / controlnet-lineart-anime-sdxl-fp16",
        "lineart": "controlnet-lineart-anime-sdxl-fp16（lineart，不是 canny）",
        "softedge": "controlnet-softedge-sdxl / controlnet-lineart-anime-sdxl-fp16",
    }

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mode": (cls.MODES, {"default": "canny",
                                     "tooltip": "canny=硬边线稿；lineart=动漫线稿（更细）；softedge=柔和边缘\n"
                                                "切换后记得同步换下游 ControlNet 模型，否则线条会糊"}),
                "resolution": ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 64}),
                "low_threshold": ("FLOAT", {"default": 0.4, "min": 0.0, "max": 1.0, "step": 0.01,
                                            "tooltip": "仅 canny 模式有效"}),
                "high_threshold": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.01,
                                             "tooltip": "仅 canny 模式有效"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("edge", "controlnet_hint")
    FUNCTION = "run"
    CATEGORY = "Illustrious/Edges"

    def run(self, image, mode, resolution, low_threshold, high_threshold):
        import numpy as np
        import torch
        import comfy.model_management as mm

        mode = (mode or "canny").lower()
        hint = self.HINTS.get(mode, "")

        if mode == "canny":
            import cv2
            arr = (image.cpu().numpy() * 255.0).astype(np.uint8)
            outs = []
            for img in arr:
                gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                edges = cv2.Canny(gray, int(low_threshold * 255), int(high_threshold * 255))
                outs.append(cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB))
            print("[Illustrious-Edge] canny，下游 ControlNet 建议: %s" % hint)
            return (torch.from_numpy(np.stack(outs)).float() / 255.0, hint)

        device = mm.get_torch_device()
        if mode == "lineart":
            from custom_controlnet_aux.lineart import LineartDetector
            model = LineartDetector.from_pretrained().to(device)
            try:
                out = _annotate_tensor(model, image, resolution=resolution, coarse=False)
            finally:
                del model
                mm.soft_empty_cache()
            print("[Illustrious-Edge] lineart，下游 ControlNet 建议: %s" % hint)
            return (out, hint)

        if mode == "softedge":
            from custom_controlnet_aux.hed import HEDdetector
            model = HEDdetector.from_pretrained().to(device)
            try:
                out = _annotate_tensor(model, image, resolution=resolution, safe=False)
            finally:
                del model
                mm.soft_empty_cache()
            print("[Illustrious-Edge] softedge，下游 ControlNet 建议: %s" % hint)
            return (out, hint)

        raise ValueError("Unknown edge mode: %r" % (mode,))


class IllustriousNormalizedResize:
    """按目标总像素量等比缩放：保持长宽比，不拉伸、不裁切。

    以 1024x1024（1.0 MP ≈ 1048576 像素）为基准，让任意长宽比的图片缩放后
    总像素量接近目标值。尺寸会对齐到 multiple_of 的倍数（默认 64）。
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "megapixels": ("FLOAT", {"default": 1.0, "min": 0.05, "max": 16.0, "step": 0.05}),
                "multiple_of": ("INT", {"default": 64, "min": 8, "max": 256, "step": 8}),
                "upscale_small": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "run"
    CATEGORY = "Illustrious/Utils"

    def run(self, image, megapixels, multiple_of, upscale_small):
        import math

        # ComfyUI IMAGE: (B, H, W, C)
        _b, h, w, _c = image.shape
        target = float(megapixels) * 1024.0 * 1024.0
        scale = math.sqrt(target / float(w * h))
        if not upscale_small:
            scale = min(1.0, scale)

        m = max(8, int(multiple_of))
        nw = max(m, int(round(w * scale / m)) * m)
        nh = max(m, int(round(h * scale / m)) * m)
        if nw == w and nh == h:
            return (image,)

        import torch.nn.functional as F

        x = image.permute(0, 3, 1, 2)  # BCHW
        x = F.interpolate(x, size=(nh, nw), mode="bicubic", align_corners=False, antialias=True)
        x = x.clamp(0.0, 1.0).permute(0, 2, 3, 1)  # BHWC
        return (x,)


class IllustriousLatentSize:
    """常用比例/分辨率切换器：下拉选预设，或选 custom 用下面的宽高。"""

    PRESETS = [
        "1024x1024 (1:1)",
        "896x1152 (7:9)", "1152x896 (9:7)",
        "832x1216 (13:19)", "1216x832 (19:13)",
        "768x1344 (4:7)", "1344x768 (7:4)",
        "720x960 (3:4)", "960x720 (4:3)",
        "640x1536 (5:12)", "1536x640 (12:5)",
        "640x640 (1:1 小)", "custom",
    ]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "preset": (cls.PRESETS, {"default": "1024x1024 (1:1)"}),
                "width": ("INT", {"default": 1024, "min": 64, "max": 4096, "step": 8}),
                "height": ("INT", {"default": 1024, "min": 64, "max": 4096, "step": 8}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 64}),
                "aspect_from_image": ("BOOLEAN", {
                    "default": True,
                    "label_on": "跟随参考图比例",
                    "label_off": "用预设宽高",
                    "tooltip": "接入 image 后，按参考图的宽高比重新计算 width/height；"
                               "总像素量保持为当前预设的面积（只换比例，不换分辨率档位）",
                }),
                "multiple_of": ("INT", {"default": 64, "min": 8, "max": 256, "step": 8,
                                        "tooltip": "派生尺寸对齐到该倍数（SDXL 建议 64）"}),
            },
            "optional": {
                "image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("LATENT", "INT", "INT")
    RETURN_NAMES = ("latent", "width", "height")
    FUNCTION = "run"
    CATEGORY = "Illustrious/Latent"

    def run(self, preset, width, height, batch_size,
            aspect_from_image=True, multiple_of=64, image=None):
        import math

        import torch

        if preset != "custom":
            try:
                wh = preset.split(" ")[0]
                w, h = wh.split("x")
                width, height = int(w), int(h)
            except Exception:
                pass

        src_note = ""
        warn = ""
        if image is not None and aspect_from_image:
            try:
                ih, iw = int(image.shape[1]), int(image.shape[2])
            except Exception:
                ih = iw = 0
            if iw > 0 and ih > 0:
                area = float(width) * float(height)     # 保留预设的"总像素预算"
                ar = float(iw) / float(ih)
                m = max(8, int(multiple_of))
                MAXS, MINS = 8192, 64

                # 1) 先按面积求理想宽高
                h = math.sqrt(area / ar)
                w = ar * h

                # 2) 长边超限时按比例整体缩放（保持比例，牺牲面积）
                longest = max(w, h)
                if longest > MAXS:
                    k = MAXS / longest
                    w, h = w * k, h * k
                    warn = " 长边超 %d，已等比缩小" % MAXS
                # 3) 短边过小时固定短边、按面积回推长边（此处比例必然失真）
                if h < MINS:
                    h = MINS
                    w = area / h
                    warn = " 比例过于极端，短边受限于 %dpx" % MINS
                if w < MINS:
                    w = MINS
                    h = area / w
                    warn = " 比例过于极端，短边受限于 %dpx" % MINS

                # 4) 吸附到倍数：在附近几个候选里挑"比例误差最小"的组合
                base_h = h
                best = None
                for k in range(-8, 9):
                    cand_h = base_h + k * m
                    if cand_h < m:
                        continue
                    cand_h = int(round(cand_h / m)) * m
                    if cand_h < m:
                        continue
                    cand_w = int(round((area / cand_h) / m)) * m
                    if cand_w < m or cand_w > MAXS or cand_h > MAXS:
                        continue
                    r_err = abs((cand_w / float(cand_h)) - ar) / ar
                    a_err = abs(cand_w * cand_h - area) / area
                    score = r_err + 0.25 * a_err      # 比例优先，面积次之
                    if best is None or score < best[0]:
                        best = (score, cand_w, cand_h)

                if best is None:
                    new_w = max(m, int(round((area / h) / m)) * m)
                    new_h = max(m, int(round(h / m)) * m)
                else:
                    _, new_w, new_h = best

                width, height = int(new_w), int(new_h)
                real_ar = width / float(height)
                drift = abs(real_ar - ar) / ar
                src_note = " <- 参考图 %dx%d (%.3f:1) 实得 %.3f:1%s" % (
                    iw, ih, ar, real_ar, warn if drift > 0.02 else "")

        width = max(64, int(width) // 8 * 8)
        height = max(64, int(height) // 8 * 8)
        latent = torch.zeros([int(batch_size), 4, height // 8, width // 8])
        print(f"[Illustrious-LatentSize] {width}x{height} (batch {batch_size}){src_note}")
        return ({"samples": latent}, width, height)


class _ExpressionMaskDetailerHook:
    """Impact Pack DetailerHook。

    重要：Impact Pack 的 post_detection 在 SAM 之前被调用，之后会用
    `segs_bitwise_and_mask(segs, sam_mask)` 做 AND —— 所以本 Hook 的改动
    会被 SAM 覆盖。使用本 Hook 时请【不要连接 sam_model_opt】，
    由 Hook 从 bbox 直接构造脸型 mask。
    """

    def __init__(self, extend_up, extend_lr, extend_down, trim_bottom_ratio,
                 eyebrow_guarantee, feather, build_face_mask=False,
                 face_width_ratio=0.95, face_height_ratio=1.02, chin_taper=0.30,
                 top_shrink=0.0):
        self.extend_up = int(extend_up)
        self.extend_lr = int(extend_lr)
        self.extend_down = int(extend_down)
        self.trim_bottom_ratio = float(trim_bottom_ratio)
        self.eyebrow_guarantee = bool(eyebrow_guarantee)
        self.feather = int(feather)
        self.build_face_mask = bool(build_face_mask)
        self.face_width_ratio = float(face_width_ratio)
        self.face_height_ratio = float(face_height_ratio)
        self.chin_taper = float(chin_taper)
        self.top_shrink = float(top_shrink)
        self.cur_step = 0
        self.total_step = 0

    def _taper(self, ny):
        """下半脸水平半径收窄系数：ny>0（下巴方向）时按比例收窄，形成二次元鹅蛋脸。"""
        if self.chin_taper <= 0.001:
            return 1.0
        return 1.0 - self.chin_taper * ny.clamp(0.0, 1.0)

    def post_detection(self, segs):
        import numpy as np
        import torch
        import torch.nn.functional as F

        if not segs or not segs[1]:
            return segs

        shape = segs[0]
        new_seg_list = []

        for seg in segs[1]:
            cmask = seg.cropped_mask
            cx1, cy1, cx2, cy2 = seg.crop_region
            bx1, by1, bx2, by2 = seg.bbox

            is_numpy = isinstance(cmask, np.ndarray)
            m = torch.from_numpy(cmask).float() if is_numpy else cmask.detach().clone().float()

            orig_ndim = m.ndim
            if m.ndim == 2:
                m = m.unsqueeze(0).unsqueeze(0)
            elif m.ndim == 3:
                m = m.unsqueeze(0)

            ch, cw = int(m.shape[2]), int(m.shape[3])
            bw = max(1, bx2 - bx1)
            bh = max(1, by2 - by1)

            lbx1 = max(0, min(cw, bx1 - cx1))
            lby1 = max(0, min(ch, by1 - cy1))
            lbx2 = max(0, min(cw, bx2 - cx1))
            lby2 = max(0, min(ch, by2 - cy1))

            yy = torch.arange(ch, device=m.device, dtype=torch.float32).view(1, 1, ch, 1)
            xx = torch.arange(cw, device=m.device, dtype=torch.float32).view(1, 1, 1, cw)

            # 0. 判断输入是否为"bbox 实心矩形"（未接 SAM）。是则用 bbox 构造脸型椭圆。
            #    注意：bbox 检测器的 cropped_mask 只在 bbox 内为 1、裁切边距为 0，
            #    所以不能用 m.mean() 判断，要按 bbox 区域内的填充率判断。
            if self.build_face_mask and (lbx2 - lbx1) > 8 and (lby2 - lby1) > 8:
                rect = ((xx >= lbx1) & (xx <= lbx2) & (yy >= lby1) & (yy <= lby2)).float()
                rect_area = float(rect.sum())
                if rect_area > 0:
                    cover = float((m * rect).sum()) / rect_area
                    if cover > 0.85:
                        fcx = (lbx1 + lbx2) * 0.5
                        fcy = (lby1 + lby2) * 0.5
                        frx = max(1.0, 0.5 * (lbx2 - lbx1) * self.face_width_ratio)
                        fry = max(1.0, 0.5 * (lby2 - lby1) * self.face_height_ratio)
                        ny = (yy - fcy) / fry
                        # 上半脸半径单独缩放（top_shrink>0 时压低上沿，避免盖到刘海）
                        if self.top_shrink > 0.001:
                            ny = torch.where(ny < 0, ny / max(1e-3, 1.0 - self.top_shrink), ny)
                        nx = (xx - fcx) / (frx * self._taper(ny))
                        m = (nx ** 2 + ny ** 2 <= 1.0).float()

            # 1. 定向膨胀：向上 extend_up、向下 extend_down、左右 extend_lr
            u, d, lr = self.extend_up, self.extend_down, self.extend_lr
            if u > 0 or d > 0 or lr > 0:
                kh = u + d + 1
                kw = lr * 2 + 1
                m_padded = F.pad(m, (lr, lr, d, u), mode="constant", value=0.0)
                m = F.max_pool2d(m_padded, kernel_size=(kh, kw), stride=1)

            # 2. 眉弓/眼部椭圆保底（SAM 漏判时强制并入）
            if self.eyebrow_guarantee and (lbx2 > lbx1) and (lby2 > lby1):
                ecx = lbx1 + 0.50 * bw
                ecy = lby1 + 0.36 * bh
                erx = max(1.0, 0.38 * bw)
                ery = max(1.0, 0.24 * bh)
                ellipse = (((xx - ecx) / erx) ** 2 + ((yy - ecy) / ery) ** 2 <= 1.0).float()
                m = torch.maximum(m, ellipse)

            # 3. 脖子/下巴底边截断
            if self.trim_bottom_ratio > -0.5:
                cutoff_y = int(round(lby2 - self.trim_bottom_ratio * bh))
                cutoff_y = max(0, min(ch, cutoff_y))
                if cutoff_y < ch:
                    m[:, :, cutoff_y:, :] = 0.0

            # 顶部限制（允许略超 bbox 上沿，容纳高挑眉）
            top_limit = max(0, int(round(lby1 - 0.10 * bh)))
            if top_limit > 0:
                m[:, :, :top_limit, :] = 0.0

            # 4. 羽化
            if self.feather > 0:
                k = self.feather * 2 + 1
                sigma = max(1.0, self.feather / 2.0)
                coords = torch.arange(k, dtype=torch.float32, device=m.device) - self.feather
                g1d = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
                g1d = g1d / g1d.sum()
                g2d = (g1d[:, None] @ g1d[None, :]).view(1, 1, k, k)
                m = F.conv2d(F.pad(m, (self.feather,) * 4, mode="reflect"), g2d)
                m = m.clamp(0.0, 1.0)

            if orig_ndim == 2:
                m_out = m.squeeze(0).squeeze(0)
            elif orig_ndim == 3:
                m_out = m.squeeze(0)
            else:
                m_out = m

            if is_numpy:
                m_out = m_out.cpu().numpy()

            SEG = type(seg)
            new_seg = SEG(
                seg.cropped_image,
                m_out,
                seg.confidence,
                seg.crop_region,
                seg.bbox,
                seg.label,
                seg.control_net_wrapper,
            )
            new_seg_list.append(new_seg)

        return (shape, new_seg_list)

    # 其余钩子透传，避免 Impact Pack 调用报错
    def set_steps(self, info):
        # 基类行为：info 是 (cur_step, total_step)
        try:
            self.cur_step, self.total_step = info
        except Exception:
            pass

    def post_crop_region(self, w, h, item_bbox, crop_region):
        return crop_region

    def touch_scaled_size(self, w, h):
        return w, h

    def post_upscale(self, img, mask=None):
        return img

    def post_encode(self, samples):
        return samples

    def pre_decode(self, samples):
        return samples

    def post_decode(self, pixels):
        return pixels

    def post_paste(self, image):
        return image

    def cycle_latent(self, latent):
        # 必须原样返回！Impact Pack 会用返回值替换 refined_latent
        return latent

    def get_custom_noise(self, seed, noise, is_touched):
        return noise, is_touched

    def get_custom_sampler(self):
        return None

    def get_skip_sampling(self):
        return False

    def should_retry_patch(self, patch):
        return False

    def pre_ksample(self, model, seed, steps, cfg, sampler_name, scheduler, positive, negative,
                    upscaled_latent, denoise):
        return model, seed, steps, cfg, sampler_name, scheduler, positive, negative, upscaled_latent, denoise


class IllustriousExpressionMaskHook:
    """给 FaceDetailer 的 detailer_hook：解决二次元表情差分漏眉毛 / 吃脖子问题。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "extend_up": ("INT", {"default": 24, "min": 0, "max": 128, "step": 2,
                                      "tooltip": "Mask 向上膨胀像素（专门往上覆盖眉毛）"}),
                "extend_lr": ("INT", {"default": 6, "min": 0, "max": 64, "step": 1,
                                      "tooltip": "Mask 向左右微扩像素"}),
                "extend_down": ("INT", {"default": 0, "min": 0, "max": 64, "step": 1,
                                        "tooltip": "Mask 向下膨胀像素（设 0 保护脖子）"}),
                "trim_bottom_ratio": ("FLOAT", {"default": 0.04, "min": -0.20, "max": 0.30, "step": 0.01,
                                                "tooltip": "从人脸 BBox 底边往上裁切的比例"}),
                "eyebrow_guarantee": ("BOOLEAN", {"default": True,
                                                  "tooltip": "强制把眉眼椭圆区并入 Mask，确保眉毛必动"}),
                "build_face_mask": ("BOOLEAN", {"default": False,
                                                "tooltip": "开启后：输入是整块矩形（未接 SAM）时，用 bbox 构造脸型椭圆"}),
                "face_width_ratio": ("FLOAT", {"default": 0.95, "min": 0.3, "max": 1.5, "step": 0.01,
                                               "tooltip": "脸型椭圆宽度相对 bbox 的比例"}),
                "face_height_ratio": ("FLOAT", {"default": 1.02, "min": 0.3, "max": 1.5, "step": 0.01,
                                                "tooltip": "脸型椭圆高度相对 bbox 的比例"}),
                "chin_taper": ("FLOAT", {"default": 0.30, "min": 0.0, "max": 0.8, "step": 0.01,
                                         "tooltip": "下半脸收窄比例（二次元尖下巴），0=标准椭圆"}),
                "feather": ("INT", {"default": 8, "min": 0, "max": 32, "step": 1,
                                    "tooltip": "Mask 边缘柔化像素"}),
                "top_shrink": ("FLOAT", {"default": 0.30, "min": 0.0, "max": 0.8, "step": 0.01,
                                         "tooltip": "上半脸半径缩小比例（压低上沿，0=不压）"}),
            }
        }

    RETURN_TYPES = ("DETAILER_HOOK",)
    RETURN_NAMES = ("detailer_hook",)
    FUNCTION = "create_hook"
    CATEGORY = "Illustrious/Image"

    def create_hook(self, extend_up, extend_lr, extend_down, trim_bottom_ratio,
                    eyebrow_guarantee, feather, build_face_mask=False,
                    face_width_ratio=0.95, face_height_ratio=1.02, chin_taper=0.30,
                    top_shrink=0.30):
        return (_ExpressionMaskDetailerHook(
            extend_up=extend_up,
            extend_lr=extend_lr,
            extend_down=extend_down,
            trim_bottom_ratio=trim_bottom_ratio,
            eyebrow_guarantee=eyebrow_guarantee,
            feather=feather,
            build_face_mask=build_face_mask,
            face_width_ratio=face_width_ratio,
            face_height_ratio=face_height_ratio,
            chin_taper=chin_taper,
            top_shrink=top_shrink,
        ),)


class IllustriousFaceTuner:
    """集中调节所有 FaceDetailer 的关键参数（前端按钮一键同步到画布上的所有 FaceDetailer）。"""

    HINTS = ["center-1", "mask-point", "mask-point-bbox"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "guide_size": ("INT", {"default": 1024, "min": 64, "max": 4096, "step": 64,
                                       "tooltip": "FaceDetailer 的目标尺寸"}),
                "max_size": ("INT", {"default": 1400, "min": 64, "max": 4096, "step": 64,
                                     "tooltip": "重绘裁切图的边长上限（越大越清晰越吃显存）"}),
                "denoise": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01,
                                      "tooltip": "重绘幅度（越高改动越大）"}),
                "feather": ("INT", {"default": 5, "min": 0, "max": 64, "step": 1,
                                    "tooltip": "回贴羽化"}),
                "noise_mask_feather": ("INT", {"default": 20, "min": 0, "max": 128, "step": 1,
                                               "tooltip": "噪声掩码羽化"}),
                "bbox_threshold": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.01,
                                             "tooltip": "检测置信度阈值（越高越严格）"}),
                "bbox_dilation": ("INT", {"default": 0, "min": -512, "max": 512, "step": 1,
                                          "tooltip": "检测框膨胀像素（用 Mask Hook 时建议 0）"}),
                "bbox_crop_factor": ("FLOAT", {"default": 2.5, "min": 1.0, "max": 10.0, "step": 0.1,
                                               "tooltip": "裁切范围倍数（越大上下文越多、脸越小）"}),
                "sam_detection_hint": (cls.HINTS, {"default": "center-1",
                                                   "tooltip": "SAM 提示模式（center-1 最稳定）"}),
                "sam_dilation": ("INT", {"default": 0, "min": -512, "max": 512, "step": 1,
                                         "tooltip": "SAM mask 四周均匀膨胀（像素）"}),
                "sam_bbox_expansion": ("INT", {"default": 0, "min": 0, "max": 1000, "step": 1,
                                               "tooltip": "bbox 外扩像素（注意是整数，不是比例）"}),
            }
        }

    RETURN_TYPES = ("FACE_SETTINGS",)
    RETURN_NAMES = ("settings",)
    FUNCTION = "run"
    CATEGORY = "Illustrious/Image"

    def run(self, **kwargs):
        print(f"[Illustrious-FaceTuner] {kwargs}")
        return (kwargs,)


class IllustriousMaskOverlay:
    """把 MASK 以半透明色块叠加到 IMAGE 上，用于肉眼核对重绘范围。
    可选裁切到判定区（mask 的 bbox）并补边，方便放大查看。"""

    COLORS = ["红", "绿", "蓝", "黄", "青", "品红", "白"]
    MODES = ["裁切到判定区", "叠加整图"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "mode": (cls.MODES, {"default": "裁切到判定区"}),
                "color": (cls.COLORS, {"default": "红"}),
                "opacity": ("FLOAT", {"default": 0.55, "min": 0.0, "max": 1.0, "step": 0.05,
                                      "tooltip": "叠加不透明度"}),
                "padding": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 2.0, "step": 0.05,
                                      "tooltip": "裁切时在 mask 外扩的比例"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("overlay",)
    FUNCTION = "run"
    CATEGORY = "Illustrious/Utils"

    _RGB = {
        "红": (1.0, 0.2, 0.2), "绿": (0.2, 1.0, 0.2), "蓝": (0.2, 0.4, 1.0),
        "黄": (1.0, 0.9, 0.2), "青": (0.2, 1.0, 1.0), "品红": (1.0, 0.2, 1.0),
        "白": (1.0, 1.0, 1.0),
    }

    def run(self, image, mask, mode, color, opacity, padding):
        import torch
        import torch.nn.functional as F

        img = image
        m = mask
        if m.ndim == 2:
            m = m.unsqueeze(0)
        if m.shape[0] == 1 and img.shape[0] > 1:
            m = m.repeat(img.shape[0], 1, 1)

        h, w = int(img.shape[1]), int(img.shape[2])
        if int(m.shape[1]) != h or int(m.shape[2]) != w:
            m = F.interpolate(m.unsqueeze(1).float(), size=(h, w),
                              mode="bilinear", align_corners=False).squeeze(1)

        rgb = self._RGB.get(color, (1.0, 0.2, 0.2))
        col = torch.tensor(rgb, dtype=img.dtype, device=img.device).view(1, 1, 1, 3)
        a = (m.unsqueeze(-1) * float(opacity)).to(img.dtype)
        out = (img * (1.0 - a) + col * a).clamp(0.0, 1.0)

        if mode == self.MODES[0]:
            nz = (m > 0.5).nonzero()
            if nz.numel() > 0:
                y1 = int(nz[:, 1].min()); y2 = int(nz[:, 1].max()) + 1
                x1 = int(nz[:, 2].min()); x2 = int(nz[:, 2].max()) + 1
                bw = max(1, x2 - x1); bh = max(1, y2 - y1)
                pad_x = int(bw * float(padding)); pad_y = int(bh * float(padding))
                x1 = max(0, x1 - pad_x); x2 = min(w, x2 + pad_x)
                y1 = max(0, y1 - pad_y); y2 = min(h, y2 + pad_y)
                side = max(x2 - x1, y2 - y1)
                cx = (x1 + x2) // 2; cy = (y1 + y2) // 2
                x1 = max(0, cx - side // 2); x2 = min(w, x1 + side); x1 = max(0, x2 - side)
                y1 = max(0, cy - side // 2); y2 = min(h, y1 + side); y1 = max(0, y2 - side)
                out = out[:, y1:y2, x1:x2, :]
                print(f"[Illustrious-MaskOverlay] crop -> ({x2 - x1}x{y2 - y1})")

        return (out,)


class _HybridSamBrowDetector:
    """包装 BBOX_DETECTOR + SAM_MODEL。

    在 detect() 内部自己调用 SAM 得到精确皮肤遮罩（避开手指/鬓角/脖子），
    再用【并集 OR】叠加 bbox 上半部分的眉眼椭圆。
    这样彻底绕开 FaceDetailer 内部对 SAM 的 bitwise AND 限制。
    """

    def __init__(self, bbox_detector, sam_model=None,
                 extend_up=16, extend_lr=2,
                 brow_width_ratio=0.72, brow_top_ratio=0.12, brow_bottom_ratio=0.56,
                 trim_bottom_ratio=0.02, feather=6,
                 face_width_ratio=1.0, chin_taper=0.30,
                 sam_threshold=0.93, sam_bbox_expansion=0, sam_dilation=0):
        self.bbox_detector = bbox_detector
        self.sam_model = sam_model
        self.extend_up = int(extend_up)
        self.extend_lr = int(extend_lr)
        self.brow_width_ratio = float(brow_width_ratio)
        self.brow_top_ratio = float(brow_top_ratio)
        self.brow_bottom_ratio = float(brow_bottom_ratio)
        self.trim_bottom_ratio = float(trim_bottom_ratio)
        self.feather = int(feather)
        self.face_width_ratio = float(face_width_ratio)
        self.chin_taper = float(chin_taper)
        self.sam_threshold = float(sam_threshold)
        self.sam_bbox_expansion = int(sam_bbox_expansion)
        self.sam_dilation = int(sam_dilation)
        self.last_mode = None

    # ---- 内部工具 ----
    def _blur(self, m, torch, F):
        if self.feather <= 0:
            return m
        k = self.feather * 2 + 1
        sigma = max(1.0, self.feather / 2.0)
        coords = torch.arange(k, dtype=torch.float32, device=m.device) - self.feather
        g1d = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g1d = g1d / g1d.sum()
        g2d = (g1d[:, None] @ g1d[None, :]).view(1, 1, k, k)
        return F.conv2d(F.pad(m, (self.feather,) * 4, mode="reflect"), g2d).clamp(0.0, 1.0)

    def _sam_mask_full(self, image, segs):
        """调用 Impact Pack 的 SAM，返回全图 mask（torch，1xHxW）或 None。"""
        if self.sam_model is None:
            return None
        try:
            import impact.core as impact_core
            sam_mask = impact_core.make_sam_mask(
                self.sam_model, segs, image, "center-1",
                self.sam_dilation, self.sam_threshold,
                self.sam_bbox_expansion, 0.7, "False",
            )
            if sam_mask is None:
                return None
            return sam_mask
        except Exception as e:
            print(f"[Illustrious-HybridMask] SAM 失败，回退纯几何: {type(e).__name__} {str(e)[:120]}")
            return None

    def detect(self, image, threshold, dilation, crop_factor, drop_size=1, detailer_hook=None):
        import numpy as np
        import torch
        import torch.nn.functional as F

        # 1) 原版 YOLO 拿 bbox（dilation 传 0）
        segs = self.bbox_detector.detect(image, threshold, 0, crop_factor, drop_size, detailer_hook=None)
        if not segs or not segs[1]:
            return segs

        # 2) SAM 精确皮肤遮罩
        sam_mask = self._sam_mask_full(image, segs)

        mode = "sam+brow" if sam_mask is not None else "geo+brow"
        if mode != self.last_mode:
            print(f"[Illustrious-HybridMask] 模式切换 -> {mode}")
            self.last_mode = mode

        items = []
        for seg in segs[1]:
            cx1, cy1, cx2, cy2 = seg.crop_region
            bx1, by1, bx2, by2 = seg.bbox

            if sam_mask is not None:
                base = sam_mask
                if isinstance(base, (list, tuple)):
                    base = base[0] if len(base) > 0 else None
                if base is not None and isinstance(base, np.ndarray):
                    base = torch.from_numpy(np.ascontiguousarray(base))
                if base is not None:
                    if base.ndim == 3:
                        base = base[0]
                    elif base.ndim == 4:
                        base = base[0][0]
                    base = base[cy1:cy2, cx1:cx2].float()
                    if base.numel() == 0:
                        base = None
            else:
                base = None

            if base is None:
                # 回退：bbox 矩形 → 由下面的椭圆逻辑处理
                base = torch.ones(int(cy2 - cy1), int(cx2 - cx1))

            m = base.unsqueeze(0).unsqueeze(0)
            ch, cw = int(m.shape[2]), int(m.shape[3])

            bw = max(1.0, float(bx2 - bx1))
            bh = max(1.0, float(by2 - by1))
            lbx1 = float(bx1 - cx1); lby1 = float(by1 - cy1)
            lbx2 = float(bx2 - cx1); lby2 = float(by2 - cy1)

            yy = torch.arange(ch, dtype=torch.float32, device=m.device).view(1, 1, ch, 1)
            xx = torch.arange(cw, dtype=torch.float32, device=m.device).view(1, 1, 1, cw)

            # 3) 纯几何回退：用 bbox 造脸型椭圆
            if sam_mask is None and (lbx2 - lbx1) > 8 and (lby2 - lby1) > 8:
                fcx = (lbx1 + lbx2) * 0.5
                fcy = (lby1 + lby2) * 0.5
                frx = max(1.0, 0.5 * (lbx2 - lbx1) * self.face_width_ratio)
                fry = max(1.0, 0.5 * (lby2 - lby1))
                ny = (yy - fcy) / fry
                taper = 1.0 - self.chin_taper * ny.clamp(0.0, 1.0)
                nx = (xx - fcx) / (frx * taper)
                m = (nx ** 2 + ny ** 2 <= 1.0).float()

            # 4) 仅向上 + 左右微扩（向下为 0，绝不往手指/脖子方向扩）
            u, lr = self.extend_up, self.extend_lr
            if u > 0 or lr > 0:
                kh = u + 1
                kw = lr * 2 + 1
                m = F.max_pool2d(F.pad(m, (lr, lr, 0, u), mode="constant", value=0.0),
                                 kernel_size=(kh, kw), stride=1)

            # 5) 上半脸眉眼椭圆（窄，只盖眉眼），与 SAM 取并集
            ecx = lbx1 + 0.50 * bw
            ecy = lby1 + 0.50 * (self.brow_top_ratio + self.brow_bottom_ratio) * bh
            erx = max(1.0, 0.50 * self.brow_width_ratio * bw)
            ery = max(1.0, 0.50 * max(0.02, self.brow_bottom_ratio - self.brow_top_ratio) * bh)
            brow = (((xx - ecx) / erx) ** 2 + ((yy - ecy) / ery) ** 2 <= 1.0).float()
            m = torch.maximum(m, brow)

            # 6) 边界裁剪
            cut_bottom = int(round(lby2 - self.trim_bottom_ratio * bh))
            cut_bottom = max(0, min(ch, cut_bottom))
            if cut_bottom < ch:
                m[:, :, cut_bottom:, :] = 0.0
            cut_top = max(0, int(round(lby1 - 0.05 * bh)))
            if cut_top > 0:
                m[:, :, :cut_top, :] = 0.0
            cut_left = max(0, int(round(lbx1)))
            if cut_left > 0:
                m[:, :, :, :cut_left] = 0.0
            cut_right = max(0, min(cw, int(round(lbx2))))
            if cut_right < cw:
                m[:, :, :, cut_right:] = 0.0

            # 7) 羽化
            m = self._blur(m, torch, F)

            m_out = m.squeeze(0).squeeze(0).cpu().numpy().astype(np.float32)

            SEG = type(seg)
            items.append(SEG(seg.cropped_image, m_out, seg.confidence,
                             seg.crop_region, seg.bbox, seg.label, seg.control_net_wrapper))

        out_segs = (segs[0], items)
        if detailer_hook is not None and hasattr(detailer_hook, "post_detection"):
            out_segs = detailer_hook.post_detection(out_segs)
        return out_segs

    def setAux(self, x):
        if hasattr(self.bbox_detector, "setAux"):
            self.bbox_detector.setAux(x)

    def __getattr__(self, name):
        return getattr(self.bbox_detector, name)


class IllustriousSamBrowDetector:
    """SAM 精确脸颊/下巴边缘 + bbox 眉眼椭圆并集，输出给 FaceDetailer 的 bbox_detector。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "bbox_detector": ("BBOX_DETECTOR",),
                "extend_up": ("INT", {"default": 16, "min": 0, "max": 96, "step": 2,
                                      "tooltip": "SAM 遮罩向上膨胀像素"}),
                "extend_lr": ("INT", {"default": 2, "min": 0, "max": 32, "step": 1,
                                      "tooltip": "SAM 遮罩左右微扩像素（建议 0~4）"}),
                "brow_width_ratio": ("FLOAT", {"default": 0.72, "min": 0.40, "max": 0.98, "step": 0.02,
                                               "tooltip": "眉眼椭圆占脸宽比例"}),
                "brow_top_ratio": ("FLOAT", {"default": 0.12, "min": 0.0, "max": 0.40, "step": 0.02,
                                             "tooltip": "眉眼椭圆上沿位置（占脸高）"}),
                "brow_bottom_ratio": ("FLOAT", {"default": 0.56, "min": 0.30, "max": 0.85, "step": 0.02,
                                                "tooltip": "眉眼椭圆下沿位置（占脸高）"}),
                "trim_bottom_ratio": ("FLOAT", {"default": 0.02, "min": -0.10, "max": 0.25, "step": 0.01,
                                                "tooltip": "底部脖子保护截断比例"}),
                "feather": ("INT", {"default": 6, "min": 0, "max": 24, "step": 1,
                                    "tooltip": "合成遮罩边缘羽化像素"}),
                "face_width_ratio": ("FLOAT", {"default": 1.00, "min": 0.4, "max": 1.5, "step": 0.02,
                                               "tooltip": "SAM 失败时的回退椭圆宽度比例"}),
                "chin_taper": ("FLOAT", {"default": 0.30, "min": 0.0, "max": 0.8, "step": 0.02,
                                         "tooltip": "回退椭圆的下半脸收窄"}),
                "sam_threshold": ("FLOAT", {"default": 0.93, "min": 0.5, "max": 1.0, "step": 0.01,
                                            "tooltip": "SAM 阈值"}),
                "sam_bbox_expansion": ("INT", {"default": 0, "min": 0, "max": 200, "step": 2,
                                               "tooltip": "SAM 提示框外扩像素"}),
                "sam_dilation": ("INT", {"default": 0, "min": -64, "max": 64, "step": 1,
                                         "tooltip": "SAM mask 均匀膨胀像素"}),
            },
            "optional": {
                "sam_model": ("SAM_MODEL",),
            },
        }

    RETURN_TYPES = ("BBOX_DETECTOR",)
    RETURN_NAMES = ("bbox_detector",)
    FUNCTION = "build"
    CATEGORY = "Illustrious/Image"

    def build(self, bbox_detector, extend_up, extend_lr, brow_width_ratio,
              brow_top_ratio, brow_bottom_ratio, trim_bottom_ratio, feather,
              face_width_ratio, chin_taper, sam_threshold, sam_bbox_expansion,
              sam_dilation, sam_model=None):
        return (_HybridSamBrowDetector(
            bbox_detector=bbox_detector,
            sam_model=sam_model,
            extend_up=extend_up,
            extend_lr=extend_lr,
            brow_width_ratio=brow_width_ratio,
            brow_top_ratio=brow_top_ratio,
            brow_bottom_ratio=brow_bottom_ratio,
            trim_bottom_ratio=trim_bottom_ratio,
            feather=feather,
            face_width_ratio=face_width_ratio,
            chin_taper=chin_taper,
            sam_threshold=sam_threshold,
            sam_bbox_expansion=sam_bbox_expansion,
            sam_dilation=sam_dilation,
        ),)
