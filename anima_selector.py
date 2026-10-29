"""Anima 版角色 / 动作选择器。

与 Illustrious 版的区别只有一个：**数据源独立**。
    anima_characters.json         角色名 -> 提示词（触发词 + 外观 tag）
    anima_character_loras.json    角色名 -> {lora, weight}
    anima_characters_custom.json  用户自己新增的角色

为什么必须独立：Anima 的角色 LoRA 与 Illustrious 的完全不同，
触发词也另有一套（常见是「角色名 + 一整套服饰 tag」），
两者混用会把错误的 LoRA 和触发词塞进提示词。

动作库（ACTIONS）直接复用 Illustrious 版 —— 动作都是通用 Danbooru tag，
Anima 同样认识，没必要复制一份。
"""

import json
import os
import random

from . import nodes as _IL
from .nodes import (_CONTROL, DEFAULT_PREFIX as _IL_PREFIX, _pinyin_key,
                    _strip_webui_tags, action_keys, extract_loras, lora_spec_str)

DIR = os.path.dirname(os.path.abspath(__file__))
ANIMA_CHARACTERS_FILE = os.path.join(DIR, 'anima_characters.json')
ANIMA_CUSTOM_FILE = os.path.join(DIR, 'anima_characters_custom.json')
ANIMA_LORAS_FILE = os.path.join(DIR, 'anima_character_loras.json')
ANIMA_FAVORITES_FILE = os.path.join(DIR, 'anima_favorites.json')

# Anima 官方推荐的正面前缀（Danbooru 质量词 + 年代词）
DEFAULT_PREFIX = "masterpiece, best quality, score_7, safe, newest, year 2025"

ACTIONS = _IL.ACTIONS


def _load(path):
    if os.path.exists(path):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"[Anima-Selector] 读取失败 {os.path.basename(path)}: {e}")
    return {}


# 角色库 = Illustrious 的完整库（2173 个） + Anima 专用触发词覆盖
#
# 为什么这样：Anima 是在 Danbooru tag 上训练的，本身就认识几千个角色，
# 不需要 LoRA 也能出图。所以角色列表不该只有「有 Anima LoRA 的少数几个」；
# 真正需要独立的是「哪些角色有 Anima LoRA、触发词是什么」这一层
# （anima_character_loras.json）以及少数字符串不同的提示词（本文件）。
CHARACTERS = dict(_IL.CHARACTERS)                        # 完整角色库，大量兜底
CHARACTERS.update(_load(ANIMA_CHARACTERS_FILE))          # Anima 专用触发词（覆盖同名）
CHARACTERS.update(_load(ANIMA_CUSTOM_FILE))

_char_loras = None


def load_char_loras():
    global _char_loras
    if _char_loras is None:
        _char_loras = _load(ANIMA_LORAS_FILE)
    return _char_loras


def load_favorites():
    d = _load(ANIMA_FAVORITES_FILE)
    return d if isinstance(d, list) else []


def _save_favorites(favs):
    try:
        with open(ANIMA_FAVORITES_FILE, 'w', encoding='utf-8') as f:
            json.dump(favs, f, ensure_ascii=False, indent=0)
    except Exception as e:
        print(f"[Anima-Selector] 收藏保存失败: {e}")


def char_keys():
    """下拉列表：① 有 Anima LoRA 的角色 ② 收藏 ③ 其余，各自按拼音排序"""
    names = [k for k in CHARACTERS.keys() if k not in _CONTROL and CHARACTERS[k]]
    bound = set(load_char_loras().keys())
    have_lora = [n for n in names if n in bound]
    fav = [n for n in load_favorites() if n in names and n not in bound]
    rest = [n for n in names if n not in bound and n not in fav]
    have_lora.sort(key=_pinyin_key)
    fav.sort(key=_pinyin_key)
    rest.sort(key=_pinyin_key)
    return list(_CONTROL) + have_lora + fav + rest      # _CONTROL 是 tuple，必须转 list 才能拼


def save_custom(name, prompt):
    name = (name or '').strip()
    prompt = (prompt or '').strip()
    if not name or not prompt:
        return False
    data = _load(ANIMA_CUSTOM_FILE)
    data[name] = prompt
    try:
        with open(ANIMA_CUSTOM_FILE, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=0)
        CHARACTERS[name] = prompt
        return True
    except Exception as e:
        print(f"[Anima-Selector] 保存角色失败: {e}")
        return False


def save_char_lora(character, lora, weight=1.0):
    global _char_loras
    character = (character or '').strip()
    if not character or character in _CONTROL:
        return False
    lora = (lora or '').strip()
    data = dict(load_char_loras())
    if lora:
        try:
            w = float(weight)
        except Exception:
            w = 1.0
        data[character] = {'lora': lora, 'weight': w}
    else:
        data.pop(character, None)
    try:
        with open(ANIMA_LORAS_FILE, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=0)
        _char_loras = data
        return True
    except Exception as e:
        print(f"[Anima-Selector] 保存 LoRA 绑定失败: {e}")
        return False


def toggle_favorite(name):
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


def character_thumb(name):
    """缩略图共用 Illustrious 的图库。

    内置角色库里的大量角色两边提示词相同，缩略图直接能命中；
    那 8 个 Anima 专用触发词因为 tag 变了，查不到图 —— 显示为空即可。
    """
    try:
        return _IL.character_thumb(name) or ''
    except Exception:
        return ''


class AnimaCharacterActionSelector:
    """Anima 版：输出 (prompt, lora_spec)。"""

    @classmethod
    def INPUT_TYPES(cls):
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
                     "tooltip": "给【当前下拉选中的角色】绑定 Anima LoRA 名。"},
                ),
                "character_lora_weight": (
                    "STRING",
                    {"multiline": False, "default": "1.0",
                     "tooltip": "角色 LoRA 权重（数字字符串，如 0.9）。"},
                ),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("prompt", "lora_spec")
    FUNCTION = "run"
    CATEGORY = "Anima/Prompt"

    def run(self, character, action, seed, prefix, suffix,
            character_prompt_override, new_character_name, new_character_prompt,
            character_lora, character_lora_weight):
        rng = random.Random(int(seed))

        if (new_character_name or '').strip() and (new_character_prompt or '').strip():
            save_custom(new_character_name, new_character_prompt)

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

        loras = list(extract_loras(action_tag))
        if character not in _CONTROL:
            bind = load_char_loras().get(character)
            if isinstance(bind, dict) and bind.get('lora'):
                loras.append((bind['lora'], float(bind.get('weight', 1.0))))

        parts = [prefix.strip(), char_tag.strip(), str(action_tag).strip(), suffix.strip()]
        parts = [p.strip(' ,') for p in parts if p and p.strip()]
        return (_strip_webui_tags(", ".join(parts)), lora_spec_str(loras))


# ---------------------------------------------------------------- HTTP 路由
_ROUTES_DONE = False


def register_routes():
    global _ROUTES_DONE
    if _ROUTES_DONE:
        return
    try:
        from aiohttp import web
        from server import PromptServer
        routes = PromptServer.instance.routes
    except Exception as e:
        print(f"[Anima-Selector] 路由注册跳过: {type(e).__name__}: {e}")
        return

    @routes.post('/anima_selector/save')
    async def _save(request):
        try:
            d = await request.json()
        except Exception:
            d = {}
        name = str(d.get('name') or '').strip()
        prompt = str(d.get('prompt') or '').strip()
        character = str(d.get('character') or '').strip()
        lora = str(d.get('lora') or '').strip()
        try:
            weight = float(d.get('lora_weight', 1.0))
        except Exception:
            weight = 1.0
        ok, err, target = False, '', character
        try:
            if name and prompt:
                ok = bool(save_custom(name, prompt)) or ok
                target = name
            if lora and (character or name):
                ok = bool(save_char_lora(character or name, lora, weight)) or ok
        except Exception as e:
            err = f'{type(e).__name__}: {e}'
        if not ok and not err:
            err = '没有可保存的内容'
        return web.json_response({'ok': ok, 'characters': char_keys(),
                                  'lora_target': target, 'error': err})

    @routes.post('/anima_selector/favorite')
    async def _fav(request):
        try:
            d = await request.json()
        except Exception:
            d = {}
        name = str(d.get('name') or '').strip()
        ok, favs, err = False, load_favorites(), ''
        try:
            ok, favs = toggle_favorite(name)
        except Exception as e:
            err = f'{type(e).__name__}: {e}'
        return web.json_response({'ok': bool(ok), 'characters': char_keys(),
                                  'favorites': favs, 'error': err})

    @routes.get('/anima_selector/char_image')
    async def _img(request):
        return web.json_response({'ok': False, 'image': ''})

    _ROUTES_DONE = True
    print("[Anima-Selector] 已注册 3 个 HTTP 路由")


register_routes()
