"""V8：从 V7.1 复制出来的独立版本。

与 V7.1 的唯一区别：SAM 遮罩的几何估计改成【行剖面 + 中位数拟合】，
不再用二阶矩求等效椭圆 —— 异色瞳时一只眼被识别成洞、另一只不是，
遮罩会形成不对称的"凸"形，二阶矩会被单侧鼓包拖偏。



V8.2：在 V8 基础上加「警告模式」与阈值余量日志。

与 V8 的差异：
  1) pose_check 从 BOOLEAN 改成三档：强制 / 仅警告 / 关闭
  2) 判定通过时也打印各指标与余量，便于统计误杀率
  3) eye-outside 改用 bbox 半宽（行剖面中位数系统性偏小约 24%）
  4) 团块交叉校验阈值改为 max(0.18*脸宽, 14px)，避免小脸误杀
  5) 眼距下限 0.22 -> 0.38（真实失败样本 0.37、正常 0.57+）
"""
"""Illustrious Selector - V7 实验节点

不改动 nodes.py 里的既有节点，全部为新类。

核心思路：
  1. YOLO 给 bbox，SAM(center-1) 给精确皮肤遮罩（避开手指/鬓角/脖子）
  2. 对遮罩做二阶矩分析，得到主轴倾角 theta（＝脸在图像平面内的倾斜）
  3. 在 (u, v) 旋转坐标系里构造眉眼带椭圆：
        u = 沿「额头 → 下巴」轴的方向
        v = 垂直该轴的方向
     头歪时眉眼带跟着转，两只眉毛都能覆盖
  4. 向上膨胀限制在眉眼带宽度内，避免顺着刘海发丝爬升
"""
import math

import numpy as np
import torch
import torch.nn.functional as F

try:  # 只在模块导入时尝试一次；实际使用时会用 _get_impact_core() 懒加载
    import impact.core as impact_core
except Exception:  # pragma: no cover
    impact_core = None


def _get_impact_core():
    """懒加载 impact.core。

    ComfyUI 加载自定义节点的顺序不固定：如果本模块先于 Impact Pack 导入，
    模块顶层的 import 会失败（impact_core=None），导致 SAM 静默失效。
    所以这里每次执行时再试一次。
    """
    global impact_core
    if impact_core is not None:
        return impact_core
    try:
        import importlib
        impact_core = importlib.import_module("impact.core")
        print("[V82-Mask] impact.core 懒加载成功")
    except Exception as e:
        print(f"[V82-Mask] impact.core 不可用: {type(e).__name__} {str(e)[:120]}")
        impact_core = None
    return impact_core


SEG_FIELDS = ['cropped_image', 'cropped_mask', 'confidence',
              'crop_region', 'bbox', 'label', 'control_net_wrapper']


def _make_seg(like, cropped_mask):
    SEG = type(like)
    return SEG(like.cropped_image, cropped_mask, like.confidence,
               like.crop_region, like.bbox, like.label, like.control_net_wrapper)


def _principal_angle(m2, ch, cw, device):
    """由遮罩二阶中心矩求主轴角度（相对 +x 轴，弧度）。

    对竖直的脸部椭圆：mu02 > mu20 且 mu11≈0 -> theta ≈ pi/2（即"向下"方向）。
    头倾斜时 theta 会相应偏移。
    """
    total = float(m2.sum().item())
    if total <= 1e-6:
        return None, None, None

    ys = torch.arange(ch, dtype=torch.float32, device=device).view(ch, 1)
    xs = torch.arange(cw, dtype=torch.float32, device=device).view(1, cw)

    cx = float((m2 * xs).sum().item() / total)
    cy = float((m2 * ys).sum().item() / total)

    dx = xs - cx
    dy = ys - cy
    mu20 = float((m2 * dx * dx).sum().item() / total)
    mu02 = float((m2 * dy * dy).sum().item() / total)
    mu11 = float((m2 * dx * dy).sum().item() / total)

    theta = 0.5 * math.atan2(2.0 * mu11, mu20 - mu02)
    return cx, cy, theta


def _mask_extent(m2, theta, cx, cy, device, ch, cw):
    """遮罩在 (u,v) 旋转坐标系下的范围。u 沿主轴（正值指向下巴）。"""
    ys = torch.arange(ch, dtype=torch.float32, device=device).view(ch, 1)
    xs = torch.arange(cw, dtype=torch.float32, device=device).view(1, cw)
    dx = xs - cx
    dy = ys - cy
    u = dx * math.cos(theta) + dy * math.sin(theta)
    v = -dx * math.sin(theta) + dy * math.cos(theta)
    sel = m2 > 0.5
    if not bool(sel.any()):
        return None
    uu = u.expand(ch, cw)[sel]
    vv = v.expand(ch, cw)[sel]
    return (float(uu.min().item()), float(uu.max().item()),
            float(vv.min().item()), float(vv.max().item()))


def _eye_pair_from_holes(m2, device):
    """从 SAM 皮肤遮罩的"洞"里找眼睛。

    center-1 模式下 SAM 分出的是皮肤连通域，眼睛（尤其二次元大眼睛）会被挖成洞。
    取 bbox 内部、不贴边的最大两个洞，其质心即左右眼中心。

    返回 (x1, y1, x2, y2) 或 None。
    """
    try:
        import scipy.ndimage as ndi
    except Exception:
        return None

    arr = (m2.detach().cpu().numpy() > 0.5)
    if not arr.any():
        return None
    ys, xs = np.where(arr)
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    x1, x2 = int(xs.min()), int(xs.max()) + 1
    H, W = y2 - y1, x2 - x1
    if H < 24 or W < 24:
        return None

    box = arr[y1:y2, x1:x2]
    lab, n = ndi.label(~box)
    if n == 0:
        return None

    cands = []
    for i in range(1, n + 1):
        comp = (lab == i)
        area = int(comp.sum())
        if area < 0.001 * H * W or area > 0.20 * H * W:
            continue
        # 贴边的是背景，不是眼睛
        if comp[0, :].any() or comp[-1, :].any() or comp[:, 0].any() or comp[:, -1].any():
            continue
        cy, cx = ndi.center_of_mass(comp)
        if cy > 0.72 * H:
            continue
        cands.append((area, float(cx) + x1, float(cy) + y1))

    if len(cands) < 2:
        return None
    cands.sort(key=lambda t: -t[0])
    a1, cx1, cy1 = cands[0]
    a2, cx2, cy2 = cands[1]
    if min(a1, a2) / max(a1, a2) < 0.28:
        return None
    if abs(cx1 - cx2) < 0.15 * W:
        return None
    if cx1 > cx2:
        cx1, cy1, cx2, cy2 = cx2, cy2, cx1, cy1
    return (cx1, cy1, cx2, cy2)


_INSIGHT = {"app": None, "failed": False}


def _face_ellipse(m2, device):
    """由 SAM 脸遮罩求等效椭圆。

    返回 (cx, cy, theta, half_a, half_b, long_is_vertical) 或 None。
    均匀填充椭圆的二阶中心矩 = 半轴^2 / 4，所以半轴 = 2*sqrt(特征值)。
    """
    try:
        import torch
    except Exception:
        return None
    if m2 is None:
        return None
    # 兼容 (H,W) / (1,H,W) / (1,1,H,W)
    while m2.dim() > 2:
        m2 = m2[0]
    m = (m2 > 0.5).float()
    total = float(m.sum())
    if total < 200:
        return None
    ch, cw = int(m.shape[0]), int(m.shape[1])
    ys = torch.arange(ch, device=m.device, dtype=torch.float32).view(ch, 1)
    xs = torch.arange(cw, device=m.device, dtype=torch.float32).view(1, cw)
    cx = float((m * xs).sum() / total)
    cy = float((m * ys).sum() / total)
    dx = xs - cx
    dy = ys - cy
    mu20 = float((m * dx * dx).sum() / total)
    mu02 = float((m * dy * dy).sum() / total)
    mu11 = float((m * dx * dy).sum() / total)
    theta = 0.5 * math.atan2(2.0 * mu11, mu20 - mu02)
    common = math.sqrt(max(0.0, ((mu20 - mu02) * 0.5) ** 2 + mu11 * mu11))
    l1 = 0.5 * (mu20 + mu02) + common          # 较大特征值
    l2 = 0.5 * (mu20 + mu02) - common
    half_a = 2.0 * math.sqrt(max(1e-6, l1))    # 沿 theta 方向
    half_b = 2.0 * math.sqrt(max(1e-6, l2))
    # theta 是"较大特征值"的主轴方向；判断它更接近竖直还是水平
    deg = math.degrees(theta) % 180.0
    long_is_vertical = 45.0 <= deg < 135.0
    return (cx, cy, theta, half_a, half_b, long_is_vertical)


def _landmarks_from_mask_geometry(m2, device, ox=0.0, oy=0.0):
    """完全不依赖姿势：只用 SAM 脸遮罩的几何推导出一整套五官点。

    SAM 在异色瞳 / 换表情时依然稳定，所以这条链是最后的可靠兜底。
    返回的坐标是【全图坐标】（与 _landmarks_from_pose 一致）。
    """
    el = _face_ellipse(m2, device)
    if el is None:
        return None
    cx, cy, theta, half_a, half_b, long_vert = el
    if long_vert:
        eye_ang = theta + math.pi * 0.5     # 眼线沿"脸宽"方向
        half_w, half_h = half_b, half_a
    else:
        eye_ang = theta
        half_w, half_h = half_a, half_b
    if half_w < 6.0 or half_h < 6.0:
        return None

    evx, evy = math.cos(eye_ang), math.sin(eye_ang)
    upx, upy = evy, -evx                    # 眼线法线，朝上
    if upy > 0:
        upx, upy = -upx, -upy

    # 眼心：脸中心略偏上（约 6% 半高）
    midx = cx + upx * 0.06 * half_h
    midy = cy + upy * 0.06 * half_h
    # 眼距 ≈ 0.55 倍脸宽 → 半距 0.55*half_w
    # 眼距/(2*half_w)：实测二次元大眼脸约 0.80~0.83，
    # 取 0.80 让带子天然覆盖两只眼；想更窄/更宽用 landmark_pad 微调
    hd = 0.80 * half_w
    e1 = (midx - evx * hd, midy - evy * hd)
    e2 = (midx + evx * hd, midy + evy * hd)
    eye_w = 2.0 * hd
    # 眉毛原始位置：眼线上方 0.28 倍眼距（后续 brow_offset 会再抬）
    bd = 0.28 * eye_w

    def make_brow(center_x, center_y, sgn):
        out = []
        for k in (-0.75, -0.40, 0.0, 0.40, 0.75):
            out.append((
                center_x + evx * (sgn * k * hd) + upx * bd,
                center_y + evy * (sgn * k * hd) + upy * bd,
            ))
        return out

    r_brow = make_brow(e1[0], e1[1], 1.0)
    l_brow = make_brow(e2[0], e2[1], 1.0)
    # 鼻子：眼心下方 0.55 倍眼距；脖子再往下 1.6 倍
    nose = (midx - upx * 0.55 * eye_w, midy - upy * 0.55 * eye_w)
    neck = (midx - upx * 1.60 * eye_w, midy - upy * 1.60 * eye_w)

    pts = [e1, e2] + r_brow + l_brow + [nose, neck]
    pts_full = [(p[0] + ox, p[1] + oy) for p in pts]

    return {
        "eyes": (e1[0] + ox, e1[1] + oy, e2[0] + ox, e2[1] + oy),
        "eye_w": eye_w,
        "pts": pts_full,
        "r_brow": [(p[0] + ox, p[1] + oy) for p in r_brow],
        "l_brow": [(p[0] + ox, p[1] + oy) for p in l_brow],
        "nose": (nose[0] + ox, nose[1] + oy),
        "neck": (neck[0] + ox, neck[1] + oy),
        "has_brow": True,
        "src": "mask-geo",
    }


def _lm_plausible(lm, el, ox=0.0, oy=0.0):
    """用 SAM 脸遮罩的等效椭圆校验姿势五官点是否可信。

    异色瞳 / 闭眼 / 换表情时 SDPose 的眼点经常漂移，
    典型症状：只有一只眼、两眼挤在一起、眼点跑到脸外。
    返回 (ok, reason)。
    """
    if lm is None:
        return (False, "no-lm")
    if el is None:
        return (True, "no-mask")          # 没有遮罩就没法判断，放行
    cx, cy, theta, half_a, half_b, long_vert = el
    half_w = half_b if long_vert else half_a
    half_h = half_a if long_vert else half_b
    if half_w < 6.0 or half_h < 6.0:
        return (True, "tiny-mask")

    e = lm.get("eyes")
    if not e:
        return (False, "no-eyes")
    e1 = (e[0] - ox, e[1] - oy)
    e2 = (e[2] - ox, e[3] - oy)
    d = math.hypot(e2[0] - e1[0], e2[1] - e1[1])
    face_w = 2.0 * half_w
    face_h = 2.0 * half_h

    # 1) 眼距占脸宽的比例（二次元正脸约 0.45~0.65，侧脸会更小）
    ratio = d / max(1e-3, face_w)
    if ratio < 0.38 or ratio > 0.98:
        return (False, "eye-dist %.2f" % ratio,
                "眼距/脸宽=%.2f (允许 0.38~0.98, 余量 %+.2f)"
                % (ratio, min(ratio - 0.38, 0.98 - ratio)))

    # 2) 两眼中点相对脸中心的位置
    mx = 0.5 * (e1[0] + e2[0])
    my = 0.5 * (e1[1] + e2[1])
    off_x = abs(mx - cx) / max(1e-3, face_w)
    off_y = abs(my - cy) / max(1e-3, face_h)
    if off_x > 0.30 or off_y > 0.40:
        return (False, "eye-mid off %.2f/%.2f" % (off_x, off_y),
                "眼中点偏移 x=%.2f/0.30 y=%.2f/0.40 (超 %+.2f / %+.2f)"
                % (off_x, off_y, off_x - 0.30, off_y - 0.40))

    # 3) 眼线方向应与"脸宽方向"大致平行
    if long_vert:
        wang = theta + math.pi * 0.5
    else:
        wang = theta
    lj = math.atan2(e2[1] - e1[1], e2[0] - e1[0])
    # 直线没有方向：差值按 180° 取模，180° 与 0° 等价
    dd = abs((lj - wang) % math.pi)
    dd = min(dd, math.pi - dd)
    if math.degrees(dd) > 40.0:
        return (False, "eye-line %.0fdeg" % math.degrees(dd),
                "眼线与脸主轴差 %.1f°/40° (超 %+.1f°)"
                % (math.degrees(dd), math.degrees(dd) - 40.0))

    # 4) 眼点必须落在脸遮罩内部（容差 12%）
    for (px, py) in (e1, e2):
        nx = (px - cx) / max(1e-3, half_w)
        ny = (py - cy) / max(1e-3, half_h)
        if nx * nx + ny * ny > 1.32:      # 1.15^2 ≈ 1.32
            return (False, "eye-outside")

    m1 = min(ratio - 0.38, 0.98 - ratio)
    m2 = min(0.30 - off_x, 0.40 - off_y)
    m3 = 40.0 - math.degrees(dd)
    return (True, "ok",
            "眼距=%.2f(余%+.2f) 中点=%.2f/%.2f(余%+.2f) 眼线差=%.0f°(余%+.0f°)"
            % (ratio, m1, off_x, off_y, m2, math.degrees(dd), m3))



def _eye_pair_from_insightface(image, bbox, device):
    """用 insightface buffalo_l 取 5 点关键点（真人训练，二次元可能失效）。"""
    if _INSIGHT["failed"]:
        return None
    try:
        if _INSIGHT["app"] is None:
            from insightface.app import FaceAnalysis
            app_ = FaceAnalysis(
                name="buffalo_l",
                root=_legacy_if_root(),
                allowed_modules=["detection"],
                providers=["CPUExecutionProvider"],
            )
            app_.prepare(ctx_id=-1, det_size=(640, 640))
            _INSIGHT["app"] = app_
            print("[V82-Mask] insightface 已加载")
        app_ = _INSIGHT["app"]

        img = image
        if img.ndim == 4:
            img = img[0]
        arr = (img.detach().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
        bgr = arr[:, :, ::-1]
        faces = app_.get(bgr)
        if not faces:
            return None

        bx1, by1, bx2, by2 = bbox
        best = None
        for f in faces:
            fx1, fy1, fx2, fy2 = f.bbox
            iw = max(1.0, min(fx2, bx2) - max(fx1, bx1))
            ih = max(1.0, min(fy2, by2) - max(fy1, by1))
            inter = iw * ih
            area_f = max(1.0, (fx2 - fx1) * (fy2 - fy1))
            area_b = max(1.0, (bx2 - bx1) * (by2 - by1))
            iou = inter / (area_f + area_b - inter)
            if best is None or iou > best[0]:
                best = (iou, f)
        if best is None or best[0] < 0.2:
            return None

        kps = best[1].kps
        if kps is None or len(kps) < 2:
            return None
        (ex1, ey1), (ex2, ey2) = float(kps[0][0]), float(kps[0][1]), float(kps[1][0]), float(kps[1][1])
        if ex1 > ex2:
            ex1, ey1, ex2, ey2 = ex2, ey2, ex1, ey1
        return (ex1, ey1, ex2, ey2)
    except Exception as e:
        _INSIGHT["failed"] = True
        print(f"[V82-Mask] insightface 不可用，后续跳过: {type(e).__name__} {str(e)[:100]}")
        return None


def _eye_pair_from_pose(pose_data, img_w, img_h, bbox, expand=0.25):
    """从 POSE_KEYPOINT 里挑出与人脸 bbox 匹配的那对眼睛。

    兼容 body25（15=REye,16=LEye）与 coco18（14=REye,15=LEye）两种索引；
    坐标会按 canvas_width/height -> 图像尺寸 归一化缩放。
    """
    if not pose_data:
        return None
    try:
        items = pose_data if isinstance(pose_data, (list, tuple)) else [pose_data]
        flat_items = []
        for it in items:
            if isinstance(it, dict):
                flat_items.append(it)
            elif isinstance(it, (list, tuple)):
                flat_items.extend([x for x in it if isinstance(x, dict)])
        items = flat_items
        bx1, by1, bx2, by2 = [float(v) for v in bbox]
        bw = max(1.0, bx2 - bx1)
        bh = max(1.0, by2 - by1)
        ex_min = bx1 - expand * bw
        ex_max = bx2 + expand * bw
        ey_min = by1 - expand * bh
        ey_max = by2 + expand * bh

        best = None
        for item in items:
            if not isinstance(item, dict):
                continue
            cw = float(item.get("canvas_width") or img_w)
            chh = float(item.get("canvas_height") or img_h)
            sx = float(img_w) / max(1.0, cw)
            sy = float(img_h) / max(1.0, chh)
            for person in item.get("people") or []:
                flat = person.get("pose_keypoints_2d") or []
                if len(flat) < 3 * 16:
                    continue
                for ia, ib in ((15, 16), (14, 15), (16, 17)):
                    if len(flat) < 3 * (max(ia, ib) + 1):
                        continue
                    def pt(i):
                        x = flat[3 * i] * sx
                        y = flat[3 * i + 1] * sy
                        c = flat[3 * i + 2]
                        return x, y, c
                    ax, ay, ac = pt(ia)
                    bx_, by_, bc = pt(ib)
                    if ac <= 0.05 or bc <= 0.05:
                        continue
                    mx = (ax + bx_) * 0.5
                    my = (ay + by_) * 0.5
                    if not (ex_min <= mx <= ex_max and ey_min <= my <= ey_max):
                        continue
                    dist = math.hypot(ax - bx_, ay - by_)
                    if dist < 0.15 * bw or dist > 1.4 * bw:
                        continue
                    if abs(ay - by_) > 0.6 * bh:
                        continue
                    # 越接近 bbox 中心越好
                    score = abs(mx - (bx1 + bx2) * 0.5) / bw + abs(my - (by1 + by2) * 0.5) / bh
                    score -= 0.001 * (ac + bc)
                    if best is None or score < best[0]:
                        pair = (ax, ay, bx_, by_) if ax <= bx_ else (bx_, by_, ax, ay)
                        best = (score, pair)
        return best[1] if best else None
    except Exception as e:
        print(f"[V82-Mask] POSE_KEYPOINT 解析失败: {type(e).__name__} {str(e)[:100]}")
        return None


def _landmarks_from_pose(pose_data, img_w, img_h, bbox, expand=0.30):
    """从 POSE_KEYPOINT 的 face_keypoints_2d（OpenPose 70 点）提取眉眼几何。

    索引（OpenPose face 70）：
        17-21 右眉  22-26 左眉
        36-41 右眼  42-47 左眼
        68/69 右/左瞳孔（可选）

    返回 dict(crop 无关，全图坐标)：
        eyes=(x1,y1,x2,y2), eye_w, pts=[(x,y)...]（眉毛+眼睛点）
    或 None。
    """
    if not pose_data:
        return None
    try:
        items = pose_data if isinstance(pose_data, (list, tuple)) else [pose_data]
        flat_items = []
        for it in items:
            if isinstance(it, dict):
                flat_items.append(it)
            elif isinstance(it, (list, tuple)):
                flat_items.extend([x for x in it if isinstance(x, dict)])

        bx1, by1, bx2, by2 = [float(v) for v in bbox]
        bw = max(1.0, bx2 - bx1)
        bh = max(1.0, by2 - by1)

        best = None
        for item in flat_items:
            cw = float(item.get("canvas_width") or img_w)
            chh = float(item.get("canvas_height") or img_h)
            sx = float(img_w) / max(1.0, cw)
            sy = float(img_h) / max(1.0, chh)
            for person in item.get("people") or []:
                fk = person.get("face_keypoints_2d") or []
                if len(fk) < 3 * 48:
                    continue

                def gp(i):
                    x = fk[3 * i] * sx
                    y = fk[3 * i + 1] * sy
                    c = fk[3 * i + 2]
                    return x, y, c

                def group(idx):
                    pts = []
                    for i in idx:
                        if 3 * i + 2 >= len(fk):
                            continue
                        x, y, c = gp(i)
                        if c > 0.05 and (x > 0 or y > 0):
                            pts.append((x, y))
                    return pts

                r_brow = group(range(17, 22))
                l_brow = group(range(22, 27))
                r_eye = group(range(36, 42))
                l_eye = group(range(42, 48))
                if len(r_eye) + len(l_eye) < 4:
                    continue

                def mean(p):
                    return (sum(q[0] for q in p) / len(p), sum(q[1] for q in p) / len(p))

                if r_eye and l_eye:
                    e1 = mean(r_eye)
                    e2 = mean(l_eye)
                else:
                    pts = r_eye + l_eye
                    xs = sorted(q[0] for q in pts)
                    ys = [q[1] for q in pts]
                    e1 = (xs[0], sum(ys) / len(ys))
                    e2 = (xs[-1], sum(ys) / len(ys))
                if e1[0] > e2[0]:
                    e1, e2 = e2, e1

                mx = (e1[0] + e2[0]) * 0.5
                my = (e1[1] + e2[1]) * 0.5
                if not (bx1 - expand * bw <= mx <= bx2 + expand * bw):
                    continue
                if not (by1 - expand * bh <= my <= by2 + expand * bh):
                    continue
                eye_w = math.hypot(e2[0] - e1[0], e2[1] - e1[1])
                if eye_w < 0.10 * bw or eye_w > 1.2 * bw:
                    continue
                if abs(e2[1] - e1[1]) > 0.7 * bh:
                    continue

                pts = r_brow + l_brow + r_eye + l_eye
                if len(pts) < 6:
                    pts = r_eye + l_eye

                # 鼻子 / 脖子（用于确定平行四边形短边方向）
                nose = None
                for i in (30, 33, 31, 27):
                    if 3 * i + 2 < len(fk):
                        x, y, c = gp(i)
                        if c > 0.05 and (x > 0 or y > 0):
                            nose = (x, y)
                            break
                neck = None
                bk = person.get("pose_keypoints_2d") or []
                if len(bk) >= 6:
                    for i in (1, 8):
                        if 3 * i + 2 < len(bk):
                            x, y, c = bk[3 * i] * sx, bk[3 * i + 1] * sy, bk[3 * i + 2]
                            if c > 0.05 and (x > 0 or y > 0):
                                neck = (x, y)
                                break
                if nose is None and neck is not None:
                    nose = ((e1[0] + e2[0]) * 0.5, (e1[1] + e2[1]) * 0.5 + 0.25 * eye_w)

                score = abs(mx - (bx1 + bx2) * 0.5) / bw + abs(my - (by1 + by2) * 0.5) / bh
                if best is None or score < best[0]:
                    best = (score, {
                        "eyes": (e1[0], e1[1], e2[0], e2[1]),
                        "eye_w": eye_w,
                        "pts": pts,
                        "r_brow": r_brow,
                        "l_brow": l_brow,
                        "nose": nose,
                        "neck": neck,
                        "has_brow": bool(r_brow and l_brow),
                    })
        return best[1] if best else None
    except Exception as e:
        print(f"[V82-Mask] face_keypoints 解析失败: {type(e).__name__} {str(e)[:100]}")
        return None


def _head_axis_from_pose(pose_data, img_w, img_h, bbox, expand=0.6):
    """用骨骼的 脖子(1) -> 鼻子(0) 连线估计头的纵向轴。

    返回单位向量 (ux, uy)，方向为「脖子 → 鼻子」= 脸朝上的方向。
    校验：鼻子必须落在人脸 bbox 附近，否则丢弃。
    """
    if not pose_data:
        return None
    try:
        items = pose_data if isinstance(pose_data, (list, tuple)) else [pose_data]
        flat = []
        for it in items:
            if isinstance(it, dict):
                flat.append(it)
            elif isinstance(it, (list, tuple)):
                flat.extend([x for x in it if isinstance(x, dict)])

        bx1, by1, bx2, by2 = [float(v) for v in bbox]
        bw = max(1.0, bx2 - bx1)
        bh = max(1.0, by2 - by1)

        best = None
        for item in flat:
            cw = float(item.get("canvas_width") or img_w)
            chh = float(item.get("canvas_height") or img_h)
            sx = float(img_w) / max(1.0, cw)
            sy = float(img_h) / max(1.0, chh)
            for person in item.get("people") or []:
                bk = person.get("pose_keypoints_2d") or []
                if len(bk) < 3 * 2:
                    continue

                def gp(i):
                    if 3 * i + 2 >= len(bk):
                        return None
                    x, y, c = bk[3 * i], bk[3 * i + 1], bk[3 * i + 2]
                    if c < 0.05 or (x == 0 and y == 0):
                        return None
                    return x * sx, y * sy

                nose = gp(0)
                neck = gp(1)
                if nose is None or neck is None:
                    continue
                if not (bx1 - expand * bw <= nose[0] <= bx2 + expand * bw):
                    continue
                if not (by1 - expand * bh <= nose[1] <= by2 + expand * bh):
                    continue
                dx = nose[0] - neck[0]
                dy = nose[1] - neck[1]
                n = math.hypot(dx, dy)
                if n < 1e-3:
                    continue
                score = abs(nose[0] - (bx1 + bx2) * 0.5) / bw + abs(nose[1] - (by1 + by2) * 0.5) / bh
                if best is None or score < best[0]:
                    best = (score, (dx / n, dy / n))
        return best[1] if best else None
    except Exception as e:
        print(f"[V82-Mask] 鼻颈轴解析失败: {type(e).__name__} {str(e)[:100]}")
        return None


def _blur(m, radius):
    if radius <= 0:
        return m
    k = radius * 2 + 1
    sigma = max(1.0, radius / 2.0)
    coords = torch.arange(k, dtype=torch.float32, device=m.device) - radius
    g1d = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g1d = g1d / g1d.sum()
    g2d = (g1d[:, None] @ g1d[None, :]).view(1, 1, k, k)
    return F.conv2d(F.pad(m, (radius,) * 4, mode="reflect"), g2d).clamp(0.0, 1.0)




# ======================= V8 覆盖：鲁棒几何估计 =======================
def _landmarks_from_eye_pair(eye, ox=0.0, oy=0.0):
    """只有一对眼点时，合成眉毛/鼻子/脖子，复用「有眼睛时」的整套逻辑。

    这样 band_top_extend / brow_offset / 超椭圆 / cut_above_band 都照常生效，
    不会因为缺眉毛而降级成"贴着眼睛的椭圆"。
    """
    if eye is None:
        return None
    ex1, ey1, ex2, ey2 = eye
    dx, dy = ex2 - ex1, ey2 - ey1
    d = math.hypot(dx, dy)
    if d < 4.0:
        return None
    evx, evy = dx / d, dy / d
    upx, upy = evy, -evx
    if upy > 0:
        upx, upy = -upx, -upy
    mx, my = 0.5 * (ex1 + ex2), 0.5 * (ey1 + ey2)
    bd = 0.28 * d          # 眉毛原始高度（brow_offset 会再往上抬）

    def mk_brow(cx0, cy0):
        out = []
        for k in (-0.22, -0.11, 0.0, 0.11, 0.22):
            out.append((cx0 + evx * (k * d) + upx * bd,
                        cy0 + evy * (k * d) + upy * bd))
        return out

    r_brow = mk_brow(ex1, ey1)
    l_brow = mk_brow(ex2, ey2)
    nose = (mx - upx * 0.55 * d, my - upy * 0.55 * d)
    neck = (mx - upx * 1.60 * d, my - upy * 1.60 * d)
    pts = [(ex1, ey1), (ex2, ey2)] + r_brow + l_brow + [nose, neck]
    return {
        "eyes": (ex1 + ox, ey1 + oy, ex2 + ox, ey2 + oy),
        "eye_w": d,
        "pts": [(p[0] + ox, p[1] + oy) for p in pts],
        "r_brow": [(p[0] + ox, p[1] + oy) for p in r_brow],
        "l_brow": [(p[0] + ox, p[1] + oy) for p in l_brow],
        "nose": (nose[0] + ox, nose[1] + oy),
        "neck": (neck[0] + ox, neck[1] + oy),
        "has_brow": True,
        "src": "eye-pair",
    }

def _eye_pair_from_blobs(img_np, m2, device, top_frac=0.70, dbg=None):
    """在 SAM 皮肤遮罩内部，用"色无关"的暗/高饱和团块找两只眼睛。

    异色瞳时红瞳/金瞳色相完全不同，任何按色相写的规则都会失效；
    但「虹膜比周围皮肤更暗 或 更饱和」对任意瞳色都成立。
    SAM 遮罩负责限定搜索范围（排除头发），因此不会把发丝当眼睛。

    返回 (x1, y1, x2, y2) 或 None。
    """
    try:
        import numpy as np
        import scipy.ndimage as ndi
    except Exception:
        return None
    if img_np is None or m2 is None:
        return None
    while m2.dim() > 2:
        m2 = m2[0]
    face = (m2 > 0.5).cpu().numpy()
    if face.sum() < 400:
        return None
    h, w = face.shape[:2]
    if img_np.shape[0] != h or img_np.shape[1] != w:
        return None

    rgb = img_np.astype("float32")
    mx = rgb.max(axis=2)
    mn = rgb.min(axis=2)
    lum = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
    sat = (mx - mn) / np.maximum(1e-4, mx)

    skin_lum = float(np.median(lum[face]))
    skin_sat = float(np.median(sat[face]))
    odd = (lum < skin_lum * 0.70) | (sat > skin_sat + 0.16)
    odd &= face

    ys, xs = np.nonzero(face)
    fy0, fy1 = int(ys.min()), int(ys.max()) + 1
    fx0, fx1 = int(xs.min()), int(xs.max()) + 1
    fh = max(1, fy1 - fy0)
    band = np.zeros_like(odd)
    band[fy0:int(fy0 + top_frac * fh), fx0:fx1] = True
    odd &= band

    lab, n = ndi.label(odd)
    if n == 0:
        return None
    farea = float(face.sum())
    cands = []
    for i in range(1, n + 1):
        comp = (lab == i)
        a = int(comp.sum())
        if a < 0.0015 * farea or a > 0.22 * farea:
            continue
        bys, bxs = np.nonzero(comp)
        bw = int(bxs.max() - bxs.min()) + 1
        bh = int(bys.max() - bys.min()) + 1
        if bw < 4 or bh < 3:
            continue
        fill = a / float(bw * bh)
        if fill < 0.35:
            continue
        ar = bw / float(bh)
        if ar < 0.40 or ar > 3.2:
            continue
        cyx = ndi.center_of_mass(comp)
        cands.append((a, float(cyx[1]), float(cyx[0]), fill))

    if len(cands) < 2:
        return None
    cands.sort(key=lambda t: -t[0])
    top = cands[:7]
    best = None
    for i in range(len(top)):
        for j in range(i + 1, len(top)):
            a1, x1, y1, f1 = top[i]
            a2, x2, y2, f2 = top[j]
            ratio = min(a1, a2) / max(a1, a2)
            if ratio < 0.40:
                continue
            sep = abs(x1 - x2)
            if sep < 0.20 * (fx1 - fx0) or sep > 0.95 * (fx1 - fx0):
                continue
            if abs(y1 - y2) > 0.42 * fh:
                continue
            score = ratio * 2.0 + (f1 + f2) * 0.5 - abs(sep / (fx1 - fx0) - 0.62)
            if best is None or score > best[0]:
                best = (score, x1, y1, x2, y2)
    if best is None:
        if dbg is not None:
            dbg["blob_fail"] = "no-pair(%d cands)" % len(cands)
        return None
    _, x1, y1, x2, y2 = best
    if x1 > x2:
        x1, y1, x2, y2 = x2, y2, x1, y1
    return (x1, y1, x2, y2)

def _face_frame_robust(m2, device):
    """用「行剖面 + 中位数」估计人脸框，取代二阶矩等效椭圆。

    为什么不用二阶矩：异色瞳时一只眼被 SAM 识别成"洞"、另一只不是皮肤色时
    不是洞，遮罩会形成单侧"凸"起。二阶矩是平方加权，一个鼓包就能把中心
    和主轴拖偏很远。行剖面法只看每行的左右端点：
      * 只保留宽度足够（>45% 最大宽度）的行 → 天然排除头发丝/下巴尖
      * 中线用最小二乘拟合后按残差二次剔除离群 → 局部鼓包被丢掉
      * 半宽取中位数 → 单侧鼓包只影响少数行，不影响中位数
    """
    try:
        import numpy as np
        import scipy.ndimage as ndi
    except Exception:
        return None
    if m2 is None:
        return None
    while m2.dim() > 2:
        m2 = m2[0]
    arr = (m2 > 0.5).detach().cpu().numpy()
    if arr.sum() < 200:
        return None

    # 只保留最大连通域（去掉飞散的碎片）
    lab, n = ndi.label(arr)
    if n > 1:
        sizes = np.bincount(lab.ravel())
        sizes[0] = 0
        arr = (lab == int(np.argmax(sizes)))
        if arr.sum() < 200:
            return None

    ys, xs = np.where(arr)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    rows = []
    for y in range(y0, y1):
        w = np.where(arr[y])[0]
        if w.size >= 2:
            rows.append((y, float(w[0]), float(w[-1]), float(w.size)))
    if len(rows) < 12:
        return None

    allw = np.array([r[3] for r in rows], dtype=float)
    # 阈值取"中位数的一半"而不是"最大值的一半"：
    # 鼓包只会抬高 max，不会抬高 median，因此阈值本身也抗畸变
    keep = [r for r in rows if r[3] >= 0.5 * float(np.median(allw))]
    if len(keep) < 8:
        return None

    yy = np.array([r[0] for r in keep], dtype=float)
    cc = np.array([0.5 * (r[1] + r[2]) for r in keep], dtype=float)
    ww = np.array([r[3] for r in keep], dtype=float)

    a, b = np.polyfit(yy, cc, 1)
    ok = np.ones(yy.shape, dtype=bool)
    for _ in range(2):
        resid = np.abs(cc - (a * yy + b))
        thr = max(2.0, 2.5 * float(np.median(resid)))
        ok = resid <= thr
        if int(ok.sum()) >= 6:
            a, b = np.polyfit(yy[ok], cc[ok], 1)
    if int(ok.sum()) < 6:
        ok = np.ones(yy.shape, dtype=bool)

    # 半宽：取"竖直方向中间 70% 的行"的宽度中位数。
    # 不看被残差过滤掉的行，也不看顶/底部（头发、下巴尖）
    lo_mid = y0 + 0.15 * (y1 - y0)
    hi_mid = y1 - 0.15 * (y1 - y0)
    midw = [r[3] for r in rows if lo_mid <= r[0] <= hi_mid]
    half_w = 0.5 * float(np.median(midw if midw else ww[ok]))
    xs_f = xs.astype(np.float64)
    ys_f = ys.astype(np.float64)
    half_h = 0.5 * float(y1 - y0)
    cy = 0.5 * (y0 + y1)
    cx = a * cy + b

    # ---- 方位角：改用二阶矩主轴 ----
    # 行剖面被单侧头发/鼓包带偏时会把倾斜方向算反（实测真实样本：行剖面 +14°，
    # 真值 -22°，而二阶矩给 -21°）。所以中心/尺寸用行剖面，方位角仍用二阶矩。
    dxm = xs_f - cx
    dym = ys_f - cy
    mu20 = float((dxm * dxm).mean())
    mu02 = float((dym * dym).mean())
    mu11 = float((dxm * dym).mean())
    theta = 0.5 * math.atan2(2.0 * mu11, mu20 - mu02)
    # theta 指向"较大特征值"那一侧：若长轴接近竖直，则眼线 = 长轴 + 90°
    deg = math.degrees(theta) % 180.0
    long_vert = 45.0 <= deg < 135.0
    ev_ang = theta + math.pi * 0.5 if long_vert else theta
    ev_x, ev_y = math.cos(ev_ang), math.sin(ev_ang)
    up_x, up_y = ev_y, -ev_x               # 朝上
    if up_y > 0:
        up_x, up_y = -up_x, -up_y

    return {
        "cx": cx, "cy": cy,
        "half_w": max(4.0, half_w), "half_h": max(4.0, half_h),
        "ev_ang": ev_ang,
        "up_x": up_x, "up_y": up_y,
        "top": float(y0), "bottom": float(y1 - 1),
        "bbox_half_w": 0.5 * float(xs.max() - xs.min() + 1),
        "bbox_half_h": 0.5 * float(y1 - y0),
        "axis_x": -up_y, "axis_y": up_x,
    }


def _landmarks_from_mask_geometry(m2, device, ox=0.0, oy=0.0):
    """完全不依赖姿势：只用 SAM 脸遮罩的鲁棒几何推导出一整套五官点。"""
    fr = _face_frame_robust(m2, device)
    if fr is None:
        return None
    half_w, half_h = fr["half_w"], fr["half_h"]
    if half_w < 6.0 or half_h < 6.0:
        return None
    ev_ang = fr["ev_ang"]
    evx, evy = math.cos(ev_ang), math.sin(ev_ang)
    upx, upy = fr["up_x"], fr["up_y"]

    midx = fr["cx"] + upx * 0.06 * half_h
    midy = fr["cy"] + upy * 0.06 * half_h
    hd = 0.55 * half_w
    e1 = (midx - evx * hd, midy - evy * hd)
    e2 = (midx + evx * hd, midy + evy * hd)
    eye_w = 2.0 * hd
    bd = 0.28 * eye_w

    def make_brow(cx0, cy0):
        out = []
        for k in (-0.75, -0.40, 0.0, 0.40, 0.75):
            out.append((cx0 + evx * (k * hd) + upx * bd,
                        cy0 + evy * (k * hd) + upy * bd))
        return out

    r_brow = make_brow(e1[0], e1[1])
    l_brow = make_brow(e2[0], e2[1])
    nose = (midx - upx * 0.55 * eye_w, midy - upy * 0.55 * eye_w)
    neck = (midx - upx * 1.60 * eye_w, midy - upy * 1.60 * eye_w)
    pts = [e1, e2] + r_brow + l_brow + [nose, neck]

    return {
        "eyes": (e1[0] + ox, e1[1] + oy, e2[0] + ox, e2[1] + oy),
        "eye_w": eye_w,
        "pts": [(p[0] + ox, p[1] + oy) for p in pts],
        "r_brow": [(p[0] + ox, p[1] + oy) for p in r_brow],
        "l_brow": [(p[0] + ox, p[1] + oy) for p in l_brow],
        "nose": (nose[0] + ox, nose[1] + oy),
        "neck": (neck[0] + ox, neck[1] + oy),
        "has_brow": True,
        "src": "mask-geo",
    }


def _lm_plausible(lm, fr, ox=0.0, oy=0.0):
    """用鲁棒脸框校验姿势五官点。

    返回 (ok, reason, detail)：
      ok  = True 表示通过
      detail 是给日志用的指标串（含余量）
    """
    if lm is None:
        return (False, "no-lm", "无五官点")
    if fr is None:
        return (True, "no-mask", "无SAM遮罩，跳过校验")
    half_w, half_h = fr["half_w"], fr["half_h"]
    if half_w < 6.0 or half_h < 6.0:
        return (True, "tiny-mask", "脸框过小，跳过校验")
    e = lm.get("eyes")
    if not e:
        return (False, "no-eyes", "无眼点")
    e1 = (e[0] - ox, e[1] - oy)
    e2 = (e[2] - ox, e[3] - oy)
    d = math.hypot(e2[0] - e1[0], e2[1] - e1[1])
    face_w = 2.0 * half_w
    face_h = 2.0 * half_h

    ratio = d / max(1e-3, face_w)
    if ratio < 0.38 or ratio > 0.98:
        return (False, "eye-dist %.2f" % ratio,
                "眼距/脸宽=%.2f (允许 0.38~0.98, 余量 %+.2f)"
                % (ratio, min(ratio - 0.38, 0.98 - ratio)))

    mx = 0.5 * (e1[0] + e2[0])
    my = 0.5 * (e1[1] + e2[1])
    off_x = abs(mx - fr["cx"]) / max(1e-3, face_w)
    off_y = abs(my - fr["cy"]) / max(1e-3, face_h)
    if off_x > 0.30 or off_y > 0.40:
        return (False, "eye-mid off %.2f/%.2f" % (off_x, off_y),
                "眼中点偏移 x=%.2f/0.30 y=%.2f/0.40 (超 %+.2f / %+.2f)"
                % (off_x, off_y, off_x - 0.30, off_y - 0.40))

    lj = math.atan2(e2[1] - e1[1], e2[0] - e1[0])
    dd = abs((lj - fr["ev_ang"]) % math.pi)
    dd = min(dd, math.pi - dd)
    if math.degrees(dd) > 40.0:
        return (False, "eye-line %.0fdeg" % math.degrees(dd),
                "眼线与脸主轴差 %.1f°/40° (超 %+.1f°)"
                % (math.degrees(dd), math.degrees(dd) - 40.0))

    # 用 bbox 半宽而不是行剖面中位数：后者实测系统性偏小约 24%
    ew = max(half_w, fr.get("bbox_half_w", half_w))
    eh = max(half_h, fr.get("bbox_half_h", half_h))
    for (px, py) in (e1, e2):
        nx = (px - fr["cx"]) / max(1e-3, ew)
        ny = (py - fr["cy"]) / max(1e-3, eh)
        if nx * nx + ny * ny > 1.45:
            return (False, "eye-outside",
                    "眼点在脸椭圆外 (归一化距离 %.2f > 1.20)" % math.sqrt(nx * nx + ny * ny))

    # 全部通过：把各项余量拼出来，便于统计"擦边通过"
    m1 = min(ratio - 0.38, 0.98 - ratio)
    m2 = min(0.30 - off_x, 0.40 - off_y)
    m3 = 40.0 - math.degrees(dd)
    return (True, "ok",
            "眼距=%.2f(余%+.2f) 中点=%.2f/%.2f(余%+.2f) 眼线差=%.0f°(余%+.0f°)"
            % (ratio, m1, off_x, off_y, m2, math.degrees(dd), m3))


class _TiltAwareBrowDetectorV82:
    """BBOX_DETECTOR 包装器：SAM 精确边缘 + 跟随倾斜的眉眼带（并集）。"""

    def __init__(self, bbox_detector, sam_model=None,
                 extend_up=12, extend_lr=2,
                 brow_width_ratio=0.72, brow_top_ratio=0.12, brow_bottom_ratio=0.56,
                 trim_bottom_ratio=0.02, feather=6,
                 tilt_follow=True, max_tilt_deg=45.0, band_limited_dilate=True,
                 face_width_ratio=1.0, chin_taper=0.30,
                 sam_threshold=0.93, sam_bbox_expansion=0, sam_dilation=0,
                 eye_source="auto", pose_data=None, landmark_pad=1.25,
                 brow_offset=1.7, eye_radius_ratio=0.5,
                 band_top_extend=0.45, band_bottom_ratio=0.85,
                 fill_holes=True, band_p=4.0, brow_lift_floor=0.38,
                 cut_above_band=True, axis_slope=0.01, axis_knee=30.0, axis_max_w=0.50,
                 pose_check="强制"):
        self.bbox_detector = bbox_detector
        self.sam_model = sam_model
        self.extend_up = int(extend_up)
        self.extend_lr = int(extend_lr)
        self.brow_width_ratio = float(brow_width_ratio)
        self.brow_top_ratio = float(brow_top_ratio)
        self.brow_bottom_ratio = float(brow_bottom_ratio)
        self.trim_bottom_ratio = float(trim_bottom_ratio)
        self.feather = int(feather)
        self.tilt_follow = bool(tilt_follow)
        self.max_tilt_deg = float(max_tilt_deg)
        self.max_tilt = math.radians(float(max_tilt_deg))
        self.band_limited_dilate = bool(band_limited_dilate)
        self.face_width_ratio = float(face_width_ratio)
        self.chin_taper = float(chin_taper)
        self.sam_threshold = float(sam_threshold)
        self.sam_bbox_expansion = int(sam_bbox_expansion)
        self.sam_dilation = int(sam_dilation)
        self.eye_source = str(eye_source)
        self.pose_data = pose_data
        self.landmark_pad = float(landmark_pad)
        self.band_top_extend = float(band_top_extend)
        self.band_bottom_ratio = float(band_bottom_ratio)
        self.brow_offset = float(brow_offset)
        self.eye_radius_ratio = float(eye_radius_ratio)
        self.max_axis_dev = 40.0          # 鼻颈轴相对眼线法线的最大偏转（度）
        self.collect_parts = False
        self.last_parts = None
        self.fill_holes = bool(fill_holes)
        self.band_p = float(band_p)
        self.brow_lift_floor = float(brow_lift_floor)
        self.cut_above_band = bool(cut_above_band)
        self.axis_slope = float(axis_slope)
        self.axis_knee = float(axis_knee)
        self.axis_max_w = float(axis_max_w)
        # 三档：强制 / 仅警告 / 关闭
        self.pose_check = str(pose_check)
        self.last_eye_src = '-'
        self._cache = None
        self.last_info = None

    # ---------- 缓存：5 个差分节点共用同一张图，避免重复跑 SAM ----------
    @staticmethod
    def _median_pt(pts):
        """逐分量中位数，比均值抗离群点。"""
        if not pts:
            return (0.0, 0.0)
        xs = sorted(p[0] for p in pts)
        ys = sorted(p[1] for p in pts)
        n = len(xs)
        mx = xs[n // 2] if n % 2 else 0.5 * (xs[n // 2 - 1] + xs[n // 2])
        my = ys[n // 2] if n % 2 else 0.5 * (ys[n // 2 - 1] + ys[n // 2])
        return (mx, my)

    def _fingerprint(self, image, threshold, crop_factor, drop_size):
        try:
            t = image[0] if image.ndim == 4 else image
            h, w = int(t.shape[0]), int(t.shape[1])
            sh = max(1, h // 32)
            sw = max(1, w // 32)
            samp = t[::sh, ::sw].float()
            return (tuple(int(v) for v in image.shape),
                    round(float(samp.mean()), 6),
                    round(float(samp.std()), 6),
                    round(float(threshold), 4),
                    round(float(crop_factor), 4),
                    int(drop_size))
        except Exception:
            return None
        self.last_info = None

    # ---------- 内部 ----------
    def _sam_full_mask(self, image, segs):
        if self.sam_model is None:
            print("[V82-Mask] sam_model 未连接，走纯几何")
            return None
        core = _get_impact_core()
        if core is None:
            return None
        try:
            mask = core.make_sam_mask(
                self.sam_model, segs, image, "center-1",
                self.sam_dilation, self.sam_threshold,
                self.sam_bbox_expansion, 0.7, "False",
            )
        except Exception as e:
            print(f"[V82-Mask] SAM 失败，回退几何: {type(e).__name__} {str(e)[:160]}")
            return None
        if mask is None:
            return None
        if isinstance(mask, (list, tuple)):
            mask = mask[0] if len(mask) > 0 else None
        if mask is None:
            return None
        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(np.ascontiguousarray(mask))
        if mask.ndim == 4:
            mask = mask[0][0]
        elif mask.ndim == 3:
            mask = mask[0]
        return mask.float()

    def _crop_base(self, full, seg, ch, cw):
        if full is None:
            return None
        cx1, cy1, cx2, cy2 = seg.crop_region
        if full.shape[0] < cy2 or full.shape[1] < cx2:
            return None
        base = full[cy1:cy2, cx1:cx2]
        if base.shape[0] != ch or base.shape[1] != cw:
            base = F.interpolate(base.view(1, 1, *base.shape), size=(ch, cw),
                                 mode="bilinear", align_corners=False)[0][0]
        return base.clamp(0.0, 1.0)

    # ---------- 主流程 ----------
    def detect(self, image, threshold, dilation, crop_factor, drop_size=1, detailer_hook=None):
        segs = self.bbox_detector.detect(image, threshold, 0, crop_factor, drop_size,
                                         detailer_hook=None)
        if not segs or not segs[1]:
            return segs

        collect = bool(getattr(self, "collect_parts", False))
        parts = {"face": [], "brow": [], "eye": []} if collect else None

        # 缓存（5 个差分节点 + 本节点的预览共用同一次 SAM 结果）
        _key = self._fingerprint(image, threshold, crop_factor, drop_size)
        if _key is not None and self._cache is not None and self._cache[0] == _key:
            self.last_parts = self._cache[3]
            return (self._cache[1], list(self._cache[2]))

        full = self._sam_full_mask(image, segs)
        device = image.device

        items = []
        for seg in segs[1]:
            cx1, cy1, cx2, cy2 = seg.crop_region
            bx1, by1, bx2, by2 = seg.bbox
            ch = max(1, int(cy2 - cy1))
            cw = max(1, int(cx2 - cx1))

            base = self._crop_base(full, seg, ch, cw)
            mode = "sam" if base is not None else "geo"

            lbx1 = float(bx1 - cx1)
            lby1 = float(by1 - cy1)
            lbx2 = float(bx2 - cx1)
            lby2 = float(by2 - cy1)
            bw = max(1.0, lbx2 - lbx1)
            bh = max(1.0, lby2 - lby1)

            if base is None:
                # 纯几何回退：bbox 内实心矩形 → 后面按椭圆处理
                base = torch.zeros(ch, cw, device=device)
                x1i = max(0, int(round(lbx1)))
                x2i = min(cw, int(round(lbx2)))
                y1i = max(0, int(round(lby1)))
                y2i = min(ch, int(round(lby2)))
                if x2i > x1i and y2i > y1i:
                    base[y1i:y2i, x1i:x2i] = 1.0

            m = base.view(1, 1, ch, cw)
            m_raw = m.clone()          # 填洞前的原始 SAM 遮罩（眼孔/嘴孔还在）

            # 【V7.1】填平 SAM 皮肤遮罩内部的孔洞（眼睛/嘴巴），
            # 只填"被皮肤完全包围"的洞，不会吞掉外部头发
            if self.fill_holes and mode == "sam":
                try:
                    import scipy.ndimage as _ndi
                    _filled = _ndi.binary_fill_holes(m[0, 0].detach().cpu().numpy() > 0.5)
                    m = torch.maximum(
                        m,
                        torch.from_numpy(_filled.astype(np.float32)).to(device).view(1, 1, ch, cw),
                    )
                except Exception as _e:
                    print(f"[V71-Mask] 孔洞填充跳过: {type(_e).__name__}")

            # --- 主轴 / 眼睛角度：face landmarks > pose 眼点 > 遮罩孔洞 > PCA ---
            eye = None
            eye_src = "-"
            lm = None
            img_h = int(image.shape[1])
            img_w = int(image.shape[2])

            head_axis = None
            frame = _face_frame_robust(m, device)     # 行剖面法，抗不对称鼓包
            if self.eye_source in ("auto", "pose") and self.pose_data is not None:
                head_axis = _head_axis_from_pose(self.pose_data, img_w, img_h, seg.bbox)

            if self.eye_source in ("auto", "pose") and self.pose_data is not None:
                lm = _landmarks_from_pose(self.pose_data, img_w, img_h, seg.bbox)
                if lm is not None and self.pose_check != "关闭":
                    ok, why, detail = _lm_plausible(lm, frame, cx1, cy1)
                    if ok:
                        print(f"[V82] 姿势判定 OK | {detail}")
                    else:
                        tag = "弃用姿势" if self.pose_check == "强制" else "警告(仍使用)"
                        print(f"[V82] 姿势判定 REJECT({why}) [{tag}] | {detail}")
                        if self.pose_check == "强制":
                            lm = None
                if lm is not None:
                    e = lm["eyes"]
                    eye = (e[0] - cx1, e[1] - cy1, e[2] - cx1, e[3] - cy1)
                    lm["pts_crop"] = [(px - cx1, py - cy1) for (px, py) in lm["pts"]]
                    eye_src = "landmark" if lm["has_brow"] else "pose"
                else:
                    # 退回 body25/coco18 的裸眼点
                    ep = _eye_pair_from_pose(self.pose_data, img_w, img_h, seg.bbox)
                    if ep is not None:
                        e1t = (ep[0] - cx1, ep[1] - cy1)
                        e2t = (ep[2] - cx1, ep[3] - cy1)
                        ok2, why2, det2 = _lm_plausible(
                            {"eyes": (e1t[0], e1t[1], e2t[0], e2t[1])}, frame)
                        if ok2 or frame is None:
                            eye = (e1t[0], e1t[1], e2t[0], e2t[1])
                            eye_src = "pose-body"
                            print(f"[V82] body眼点 OK | {det2}")
                        elif self.pose_check == "仅警告":
                            eye = (e1t[0], e1t[1], e2t[0], e2t[1])
                            eye_src = "pose-body"
                            print(f"[V82] body眼点 REJECT({why2}) [警告(仍使用)] | {det2}")
                        else:
                            print(f"[V82] body眼点 REJECT({why2}) [跳过] | {det2}")

            # 【V8】色无关瞳色团块：在 SAM 遮罩内部找两只眼睛。
            # 先算出来，既可作为兜底，也用来交叉校验姿势眼点。
            blob_eye = None
            if self.eye_source in ("auto", "pose", "mask-holes", "mask-geo"):
                try:
                    _ic = image[0, int(cy1):int(cy2), int(cx1):int(cx2)]
                except Exception:
                    _ic = None
                if _ic is not None and _ic.shape[0] == ch and _ic.shape[1] == cw:
                    blob_eye = _eye_pair_from_blobs(_ic.detach().cpu().numpy(), m, device)

            # 交叉校验：姿势眼点与团块眼点差异过大 → 弃用姿势
            if blob_eye is not None and eye is not None and self.pose_check != "关闭":
                _pb = 0.5 * (blob_eye[0] + blob_eye[2])
                _py = 0.5 * (blob_eye[1] + blob_eye[3])
                _px = 0.5 * (eye[0] + eye[2])
                _ey = 0.5 * (eye[1] + eye[3])
                _fw = max(20.0, float(bw))
                # 小脸时 0.18*脸宽 会小于团块自身的质心偏差(~15px)，故加绝对下限
                _thr = max(0.18 * _fw, 14.0)
                _err_px = math.hypot(_px - _pb, _ey - _py)
                d_pose = math.hypot(eye[2] - eye[0], eye[3] - eye[1])
                d_blob = math.hypot(blob_eye[2] - blob_eye[0], blob_eye[3] - blob_eye[1])
                _dratio = abs(d_pose - d_blob) / d_blob if d_blob > 1e-3 else 0.0
                _bad = (_err_px > _thr) or (_dratio > 0.45)
                _msg = ("[V82] 姿势/团块比对: 中点偏差 %.1fpx(阈值 %.1f) 眼距 %.0f vs %.0f(差 %.0f%%)"
                        % (_err_px, _thr, d_pose, d_blob, _dratio * 100))
                if not _bad:
                    print(_msg + " -> 一致")
                else:
                    tag = "弃用姿势" if self.pose_check == "强制" else "警告(仍用姿势)"
                    print(_msg + " -> 不符 [" + tag + "]")
                    if self.pose_check == "强制":
                        eye = None
                        lm = None

            if eye is None and blob_eye is not None:
                eye = blob_eye
                eye_src = "eye-blob"
                _lm2 = _landmarks_from_eye_pair(blob_eye, cx1, cy1)
                if _lm2 is not None:
                    lm = _lm2
                    lm["pts_crop"] = [(px - cx1, py - cy1) for (px, py) in lm["pts"]]
                    eye = (blob_eye[0], blob_eye[1], blob_eye[2], blob_eye[3])

            if eye is None and self.eye_source in ("auto", "mask-holes"):
                eye = _eye_pair_from_holes(m_raw[0, 0], device)
                if eye is not None:
                    ok3, why3 = _lm_plausible(
                        {"eyes": (eye[0], eye[1], eye[2], eye[3])}, frame)
                    if not ok3 and frame is not None:
                        print(f"[V82] 眼孔结果不可信({why3})，跳过")
                        eye = None
                    else:
                        eye_src = "holes"

            # 【V7.1.2】姿势不可用时的可靠兜底：只用 SAM 遮罩几何推导五官。
            # 异色瞳 / 闭眼 / 表情差异不会影响 SAM，所以这条路最稳。
            if eye is None and self.eye_source in ("auto", "pose", "mask-holes", "mask-geo"):
                lm_g = _landmarks_from_mask_geometry(m, device, cx1, cy1)
                if lm_g is not None:
                    e = lm_g["eyes"]
                    eye = (e[0] - cx1, e[1] - cy1, e[2] - cx1, e[3] - cy1)
                    lm = lm_g
                    lm["pts_crop"] = [(px - cx1, py - cy1) for (px, py) in lm["pts"]]
                    eye_src = "mask-geo"
                    print("[V82] 使用 SAM 几何推导五官（不依赖姿势）")


            if eye is None and self.eye_source == "insightface":
                e2 = _eye_pair_from_insightface(image if image.ndim == 3 else image[0],
                                                seg.bbox, device)
                if e2 is not None:
                    eye = (e2[0] - cx1, e2[1] - cy1, e2[2] - cx1, e2[3] - cy1)
                    eye_src = "insight"

            cx_, cy_, theta = _principal_angle(m[0, 0], ch, cw, device)
            if cx_ is None or cy_ is None:
                # 遮罩为空时用 bbox 中心兜底，避免后续 None 参与运算
                cx_ = (lbx1 + lbx2) * 0.5
                cy_ = (lby1 + lby2) * 0.5
            tilt_deg = 0.0
            eye_mid_uv = None

            if eye is not None:
                ex1, ey1, ex2, ey2 = eye
                vx = ex2 - ex1
                vy = ey2 - ey1
                vn = math.hypot(vx, vy)
                if vn < 1e-3:
                    vx, vy, vn = 1.0, 0.0, 1.0
                vx /= vn
                vy /= vn

                # 默认：u 轴 = 眼线的法线，朝下
                ux, uy = -vy, vx
                if uy < 0:
                    ux, uy = -ux, -uy

                # 若有骨骼 鼻-颈 轴，则正交化后作为 u 轴（更能反映头部朝向）
                if head_axis is not None:
                    hx, hy = head_axis
                    d = hx * vx + hy * vy
                    hx2, hy2 = hx - d * vx, hy - d * vy
                    n2 = math.hypot(hx2, hy2)
                    if n2 > 0.15:
                        ux, uy = -hx2 / n2, -hy2 / n2        # 取反 → 指向下巴
                        if uy < 0:
                            ux, uy = -ux, -uy

                theta = math.atan2(uy, ux)
                tilt_deg = math.degrees(math.atan2(vy, vx))

                if not self.tilt_follow or abs(tilt_deg) > self.max_tilt_deg:
                    theta = math.pi / 2.0
                    tilt_deg = 0.0
                    eye_mid_uv = None
                else:
                    eye_mid_uv = ((ex1 + ex2) * 0.5, (ey1 + ey2) * 0.5)
            elif theta is None:
                theta = math.pi / 2.0
            else:
                # 主轴有 180° 二义性：把 tilt 归一化到 (-90°, 90°]
                tilt = theta - math.pi / 2.0
                tilt = (tilt + math.pi / 2.0) % math.pi - math.pi / 2.0
                tilt_deg = math.degrees(tilt)
                if self.tilt_follow and abs(tilt) <= self.max_tilt:
                    theta = math.pi / 2.0 + tilt
                else:
                    theta = math.pi / 2.0
                    tilt_deg = 0.0

            ys = torch.arange(ch, dtype=torch.float32, device=device).view(ch, 1)
            xs = torch.arange(cw, dtype=torch.float32, device=device).view(1, cw)
            dx = xs - cx_
            dy = ys - cy_
            u = dx * math.cos(theta) + dy * math.sin(theta)
            v = -dx * math.sin(theta) + dy * math.cos(theta)
            u = u.expand(ch, cw)
            v = v.expand(ch, cw)

            ext = _mask_extent(m[0, 0], theta, cx_, cy_, device, ch, cw)
            if ext is None:
                # 遮罩为空：用 bbox 四角在 (u,v) 上的投影兜底，继续走椭圆/眉眼带逻辑
                ct0 = math.cos(theta)
                st0 = math.sin(theta)
                corners = [(lbx1, lby1), (lbx2, lby1), (lbx1, lby2), (lbx2, lby2)]
                us_c = [(px - cx_) * ct0 + (py - cy_) * st0 for px, py in corners]
                vs_c = [-(px - cx_) * st0 + (py - cy_) * ct0 for px, py in corners]
                ext = (min(us_c), max(us_c), min(vs_c), max(vs_c))
                mode = "geo"
            u_min, u_max, v_min, v_max = ext
            L = max(1.0, u_max - u_min)
            W = max(1.0, v_max - v_min)

            # 无 SAM 时：用椭圆【替换】bbox 矩形（不是并集，否则矩形会保留下来）
            if mode == "geo":
                frx = max(1.0, 0.5 * W * self.face_width_ratio)
                taper = 1.0 - self.chin_taper * ((u - u_min) / L - 0.5).clamp(0.0, 1.0) * 2.0
                ny = (u - (u_min + u_max) * 0.5) / max(1.0, 0.5 * L)
                nx = v / (frx * taper).clamp(min=1.0)
                face = (nx ** 2 + ny ** 2 <= 1.0).float().view(1, 1, ch, cw)
                m = face

            e1c = (lm["eyes"][0] - cx1, lm["eyes"][1] - cy1) if lm is not None else None
            e2c = (lm["eyes"][2] - cx1, lm["eyes"][3] - cy1) if lm is not None else None

            # --- 眉眼带 ---
            # A) 有眉毛关键点 -> 平行四边形（上边=修正后眉线，下边=过鼻子的平行线，短边=鼻颈轴）
            # B) 只有眼点      -> 椭圆（眼宽定宽）
            # C) 都没有        -> bbox 比例椭圆
            bt = min(self.brow_top_ratio, self.brow_bottom_ratio - 0.05)
            bb = max(self.brow_bottom_ratio, bt + 0.05)
            band = None
            band_kind = "none"
            u0 = u1 = 0.0
            vh = 1.0
            nose_c = None
            vv_band = None
            uu_band = None
            u_mid = u_half = 0.0
            v_c = 0.0
            v_half = 1.0

            if lm is not None and lm.get("has_brow") and lm.get("r_brow") and lm.get("l_brow"):
                # 1) 眉毛修正：只沿"垂直于眼线"的方向拉长，避免同时把眉毛水平推开
                evx = e2c[0] - e1c[0]
                evy = e2c[1] - e1c[1]
                evn = math.hypot(evx, evy)
                if evn > 1e-3:
                    evx /= evn
                    evy /= evn
                else:
                    evx, evy = 1.0, 0.0

                # 眼线法线（朝额头方向，y 更小）
                up_x, up_y = evy, -evx
                if up_y > 0:
                    up_x, up_y = -up_x, -up_y

                # 【V7.1】眉高自动兜底：二次元眉毛常画在额头中上部，
                # 用"眼心到人脸框顶边"的距离做下限，摆脱画风常数依赖
                eye_mid_y = 0.5 * (e1c[1] + e2c[1])
                dist_to_top = max(4.0, eye_mid_y - lby1)
                lift_floor = max(0.0, self.brow_lift_floor) * dist_to_top

                def corrected(brow_pts, eye_c):
                    out = []
                    for (px, py) in brow_pts:
                        dx = (px - cx1) - eye_c[0]
                        dy = (py - cy1) - eye_c[1]
                        proj = dx * evx + dy * evy          # 平行眼线的分量（保留不动）
                        perp_x = dx - proj * evx            # 垂直眼线的分量（拉长）
                        perp_y = dy - proj * evy
                        mag = math.hypot(perp_x, perp_y)
                        lift = mag * self.brow_offset
                        if mag > 1e-3:
                            if lift < lift_floor:
                                lift = lift_floor
                            sgn = 1.0 if (perp_x * up_x + perp_y * up_y) >= 0 else -1.0
                            perp_x = up_x * lift * sgn
                            perp_y = up_y * lift * sgn
                        else:
                            perp_x = 0.0
                            perp_y = 0.0
                        out.append((eye_c[0] + proj * evx + perp_x,
                                    eye_c[1] + proj * evy + perp_y))
                    return out

                rb = corrected(lm["r_brow"], e1c)
                lb = corrected(lm["l_brow"], e2c)
                if rb and lb:
                    bc_r = self._median_pt(rb)
                    bc_l = self._median_pt(lb)
                    # 2) 眉线方向 = v 轴
                    vx = bc_l[0] - bc_r[0]
                    vy = bc_l[1] - bc_r[1]
                    vn = math.hypot(vx, vy)
                    if vn > 1e-3:
                        vx /= vn
                        vy /= vn

                        # 3) 短边方向：以「十字定位」的正交垂线为基准，
                        #    鼻颈轴只作为修正项，偏差越大权重越高。
                        #    理由：画师用十字定位五官，眉毛本来就不会斜太多；
                        #          只有当鼻颈轴与垂线差异明显时（真·侧脸/俯仰）才以它为准。
                        ux, uy = -vy, vx
                        if uy < 0:
                            ux, uy = -ux, -uy
                        perp_x, perp_y = ux, uy          # 正交十字的垂线（基准）
                        perp_ang = math.atan2(perp_y, perp_x)
                        nose_c = None
                        if lm.get("nose") is not None:
                            nose_c = (lm["nose"][0] - cx1, lm["nose"][1] - cy1)
                        if lm.get("neck") is not None and nose_c is not None:
                            hx = (lm["neck"][0] - cx1) - nose_c[0]     # 鼻子 -> 脖子（朝下巴）
                            hy = (lm["neck"][1] - cy1) - nose_c[1]
                            hn = math.hypot(hx, hy)
                            if hn > 1e-3:
                                cand_ang = math.atan2(hy, hx)
                                # 立到与基准同向（±180° 内选最近的等价角）
                                d0 = (cand_ang - perp_ang + math.pi) % (2 * math.pi) - math.pi
                                cand_ang = perp_ang + d0

                                # 偏差先夹到上限，再用连续曲线算鼻颈轴权重。
                                # 正交十字是"人脸 + 绘画"的共通基准，永远占多数。
                                #   a <= knee : w = slope × a            （线性段，过 (0,0)）
                                #   a >  knee : w = wmax - (wmax-w_knee)·exp(-(a-knee)/tau)
                                #   tau 由 C¹ 连续推出，不需要额外参数
                                lim = math.radians(self.max_axis_dev)
                                dev = max(-lim, min(lim, d0))
                                a = math.degrees(abs(dev))
                                slope = max(0.0, float(self.axis_slope))
                                knee = max(0.5, float(self.axis_knee))
                                wmax = min(1.0, max(0.0, float(self.axis_max_w)))
                                w_knee = min(wmax, slope * knee)
                                if a <= knee:
                                    w = slope * a
                                else:
                                    tau = max(1.0, (wmax - w_knee) / max(slope, 1e-6))
                                    w = wmax - (wmax - w_knee) * math.exp(-(a - knee) / tau)
                                w = min(w, wmax)

                                final_ang = perp_ang + w * dev
                                ux, uy = math.cos(final_ang), math.sin(final_ang)

                        # 4) 以眉线中点为原点，用斜基 [u, v] 的逆矩阵求坐标
                        #    点乘只在 u⊥v 时成立；斜基必须用对偶基（逆矩阵）
                        ox = (bc_r[0] + bc_l[0]) * 0.5
                        oy = (bc_r[1] + bc_l[1]) * 0.5
                        det = ux * vy - uy * vx
                        if abs(det) < 1e-3:
                            det = 1e-3 if det >= 0 else -1e-3
                        rx = xs - ox
                        ry = ys - oy
                        uu = (rx * vy - ry * vx) / det
                        vv = (ux * ry - uy * rx) / det

                        u_top = 0.0
                        if nose_c is not None:
                            u_bot = ((nose_c[0] - ox) * vy - (nose_c[1] - oy) * vx) / det
                        else:
                            u_bot = 0.45 * L
                        if u_bot <= u_top + 4:
                            u_bot = u_top + max(8.0, 0.35 * L)
                        span_u = u_bot - u_top
                        # 上边界：眉线以上 band_top_extend 倍
                        # 下边界：朝鼻子方向走 band_bottom_ratio 倍（1.0 = 正好到鼻子）
                        u0 = u_top - self.band_top_extend * span_u
                        u1 = u_top + self.band_bottom_ratio * span_u

                        # 【V7.1.1】稳健中心：以两只眼睛的中点为锚，
                        # 眉毛点只用于扩大宽度；离群点（被检测到头发里）被过滤掉，
                        # 不能再把整个带子拖偏。
                        def _u_of(pt):
                            return ((pt[0] - ox) * vy - (pt[1] - oy) * vx) / det

                        def _v_of(pt):
                            return (ux * (pt[1] - oy) - uy * (pt[0] - ox)) / det

                        v_e1, v_e2 = _v_of(e1c), _v_of(e2c)
                        v_eye_mid = 0.5 * (v_e1 + v_e2)
                        v_eye_half = 0.5 * abs(v_e2 - v_e1)

                        brow_vs = [_v_of(p) for p in (rb + lb)]
                        lim_v = max(4.0, 2.3 * v_eye_half)
                        kept = [x for x in brow_vs if abs(x - v_eye_mid) <= lim_v]
                        if kept:
                            v_lo, v_hi = min(kept), max(kept)
                            v_brow_half = max(abs(v_lo - v_eye_mid), abs(v_hi - v_eye_mid))
                        else:
                            v_brow_half = v_eye_half

                        v_c = v_eye_mid          # 中心强制锚定眼心，保证左右对称
                        v_half = max(2.0, max(v_brow_half, v_eye_half) * self.landmark_pad)
                        vh = v_half
                        vv_band = (vv - v_c).abs()
                        uu_band = uu

                        # 超椭圆（p 越大越接近直角矩形，p=4~6 圆角且能包住大圆眼）
                        u_mid = 0.5 * (u0 + u1)
                        u_half = max(2.0, 0.5 * (u1 - u0))
                        nu = (uu - u_mid) / u_half
                        nv = (vv - v_c) / v_half
                        p_pow = max(2.0, float(self.band_p))
                        band = ((nu.abs() ** p_pow + nv.abs() ** p_pow) <= 1.0).float().view(1, 1, ch, cw)
                        band_kind = "para"

            if band is None and lm is not None and lm.get("pts_crop"):
                pts = lm["pts_crop"]
                ct = math.cos(theta)
                st = math.sin(theta)
                us_ = [(px - cx_) * ct + (py - cy_) * st for px, py in pts]
                vs_ = [-(px - cx_) * st + (py - cy_) * ct for px, py in pts]
                uc = 0.5 * (max(us_) + min(us_))
                vc = 0.5 * (max(vs_) + min(vs_))
                uh_lm = max(1.0, 0.5 * (max(us_) - min(us_)) * self.landmark_pad)
                vh_lm = max(1.0, 0.5 * (max(vs_) - min(vs_)) * self.landmark_pad)
                uh = max(uh_lm, max(1.0, 0.5 * (bb - bt) * L))
                vh = max(vh_lm, max(1.0, 0.5 * self.brow_width_ratio * W))
                band = (((v - vc) / vh) ** 2 + ((u - uc) / uh) ** 2 <= 1.0).float().view(1, 1, ch, cw)

                band_kind = "ellipse-lm"
            if band is None and eye_mid_uv is not None:
                emx, emy = eye_mid_uv
                uc = (emx - cx_) * math.cos(theta) + (emy - cy_) * math.sin(theta)
                uh = max(1.0, 0.5 * (bb - bt) * L)
                eye_w = math.hypot(eye[2] - eye[0], eye[3] - eye[1])
                vh = max(1.0, 0.5 * eye_w * 1.35)
                band = (((v - 0.0) / vh) ** 2 + ((u - uc) / uh) ** 2 <= 1.0).float().view(1, 1, ch, cw)
                band_kind = "ellipse-eye"

            if band is None:
                uc = u_min + 0.5 * (bt + bb) * L
                uh = max(1.0, 0.5 * (bb - bt) * L)
                vh = max(1.0, 0.5 * self.brow_width_ratio * W)
                band = (((v - 0.0) / vh) ** 2 + ((u - uc) / uh) ** 2 <= 1.0).float().view(1, 1, ch, cw)

                band_kind = "ellipse-bbox"
            # 眼睛补圆（SAM 常把眼睛挖掉，这里按眼距定半径强制覆盖）
            eye_part = None
            if e1c is not None and e2c is not None and self.eye_radius_ratio > 0.001:
                eye_w = math.hypot(e2c[0] - e1c[0], e2c[1] - e1c[1])
                rr = max(2.0, eye_w * self.eye_radius_ratio)
                for (ecx_, ecy_) in (e1c, e2c):
                    circ = (((xs - ecx_) ** 2 + (ys - ecy_) ** 2) <= rr * rr).float().view(1, 1, ch, cw)
                    eye_part = circ if eye_part is None else torch.maximum(eye_part, circ)
            band2d = band[0, 0]

            # --- 边界裁剪（只裁 SAM 部分，几何带不裁，否则会切出直角边） ---
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

            # --- 向上膨胀（只对 SAM 部分，限制在眉眼带宽度内） ---
            uu = int(self.extend_up)
            lr = int(self.extend_lr)
            if uu > 0 or lr > 0:
                padded = F.pad(m, (lr, lr, 0, uu), mode="constant", value=0.0)
                grown = F.max_pool2d(padded, kernel_size=(uu + 1, lr * 2 + 1), stride=1)
                if self.band_limited_dilate:
                    # 【V7.1 修复】limit 必须用平行四边形自己的横向坐标，
                    # 之前用的是以 SAM 质心为原点的 v，与 vh 不同源 → 会切掉一侧
                    if vv_band is not None:
                        limit = (vv_band <= vh * 1.15).float().view(1, 1, ch, cw)
                    else:
                        limit = (v.abs() <= vh * 1.15).float().view(1, 1, ch, cw)
                    m = m + (grown - m).clamp(min=0.0) * limit
                else:
                    m = torch.maximum(m, grown)

            # --- 并集（几何带最后合并，不被上面任何裁剪影响） ---
            face_part = m.clone()
            m = torch.maximum(m, band)
            if eye_part is not None:
                m = torch.maximum(m, eye_part)

            # 【V7.1.2】截掉超椭圆上边界以上的所有识别。
            # 漂染/挑染的刘海会让 SAM 顺着发丝往上爬，形成"角"状泄漏。
            # 用超椭圆自身的上边界曲线当"天花板"，并把它向左右延伸：
            #   |nv| <= 1  ->  u_ceil = u_mid - u_half * (1 - |nv|^p)^(1/p)
            #   |nv| >  1  ->  天花板收到 u_mid（带子外侧不允许高于中线）
            if self.cut_above_band and uu_band is not None and band_kind == "para" \
                    and abs(self.band_p - 0.0) > 0.01:
                p_pow = max(2.0, float(self.band_p))
                nv_c = ((vv - v_c) / v_half).abs().clamp(max=1.0)
                val = (1.0 - nv_c ** p_pow).clamp(min=0.0) ** (1.0 / p_pow)
                u_ceil = u_mid - u_half * val
                keep = (uu_band >= u_ceil).float().view(1, 1, ch, cw)
                m = m * keep
                if parts is not None:
                    face_part = face_part * keep

            if parts is not None:
                parts["face"].append((seg.crop_region, face_part[0, 0].cpu().numpy().copy()))
                parts["brow"].append((seg.crop_region, band[0, 0].cpu().numpy().copy()))
                if eye_part is not None:
                    parts["eye"].append((seg.crop_region, eye_part[0, 0].cpu().numpy().copy()))

            m = _blur(m, self.feather)

            info = (mode, eye_src, round(tilt_deg, 1), head_axis is not None, band_kind)
            self.last_eye_src = eye_src
            if info != self.last_info:
                print(f"[V82-Mask] mode={mode} eye={eye_src} tilt={tilt_deg:.1f}deg "
                      f"head_axis={head_axis is not None} band={band_kind} "
                      f"L={L:.0f} W={W:.0f}")
                if band_kind == "para":
                    print(f"[V82-Mask]   para u=[{u0:.0f},{u1:.0f}] v±{vh:.0f} "
                          f"nose={'Y' if nose_c is not None else 'N'} "
                          f"offset={self.brow_offset:.2f} vpad={self.landmark_pad:.2f}")
                self.last_info = info

            items.append(_make_seg(seg, m.squeeze(0).squeeze(0).cpu().numpy().astype(np.float32)))

        out = (segs[0], items)
        if parts is not None:
            self.last_parts = parts
        if _key is not None:
            keep_parts = parts if parts is not None else (
                self._cache[3] if (self._cache and self._cache[0] == _key) else None)
            self._cache = (_key, out[0], list(out[1]), keep_parts)
        if detailer_hook is not None and hasattr(detailer_hook, "post_detection"):
            out = detailer_hook.post_detection(out)
        return out

    def setAux(self, x):
        if hasattr(self.bbox_detector, "setAux"):
            self.bbox_detector.setAux(x)

    def __getattr__(self, name):
        return getattr(self.bbox_detector, name)


def _render_parts(image_tensor, parts, mode, opacity, edge_width,
                  colors=None, base_hw=None):
    """把 parts（face/brow/eye）渲染成三色叠加图 + 三个去重叠 MASK。"""
    import numpy as np
    import torch

    colors = colors or {
        "face": (0.10, 0.55, 1.00),
        "brow": (1.00, 0.50, 0.00),
        "eye":  (0.10, 1.00, 0.30),
    }
    src = image_tensor
    if src.shape[0] > 1:
        src = src[:1]
    hh, ww = int(src.shape[1]), int(src.shape[2])

    def assemble(key):
        acc = np.zeros((hh, ww), dtype=np.float32)
        for (cx1, cy1, cx2, cy2), t in (parts.get(key) or []):
            th_, tw_ = t.shape
            ey, ex = min(cy2, hh), min(cx2, ww)
            dy, dx = ey - cy1, ex - cx1
            if dy > 0 and dx > 0:
                acc[cy1:ey, cx1:ex] = np.maximum(acc[cy1:ey, cx1:ex], t[:dy, :dx])
        return acc

    fm = assemble("face")
    bm = assemble("brow")
    em = assemble("eye")

    out = src[0].cpu().numpy().astype(np.float32).copy()
    try:
        import scipy.ndimage as ndi
    except Exception:
        ndi = None

    order = [("face", fm), ("brow", bm), ("eye", em)]
    if mode == "仅脸部(SAM)":
        order = [("face", fm)]
    elif mode == "仅眉眼带":
        order = [("brow", bm)]
    elif mode == "仅眼圆":
        order = [("eye", em)]

    single = mode in ("仅脸部(SAM)", "仅眉眼带", "仅眼圆")
    fill_a = 0.65 if single else float(opacity)

    if mode != "仅轮廓":
        for key, msk in order:
            if float(msk.max()) <= 0:
                continue
            col = np.array(colors[key], dtype=np.float32).reshape(1, 1, 3)
            al = (msk > 0.5).astype(np.float32)[:, :, None] * fill_a
            out = out * (1 - al) + col * al

    w = int(max(1, edge_width))
    for key, msk in order:
        binm = msk > 0.5
        if not binm.any():
            continue
        edge = (binm & (~ndi.binary_erosion(binm, iterations=w))) if ndi is not None else binm
        out[edge] = np.array(colors[key], dtype=np.float32)

    out = np.clip(out, 0, 1)

    face_only = np.clip(fm - np.maximum(bm, em), 0, 1)
    brow_only = np.clip(bm - fm, 0, 1)
    eye_only = np.clip(em - np.maximum(fm, bm), 0, 1)

    return (torch.from_numpy(out).unsqueeze(0),
            torch.from_numpy(face_only).unsqueeze(0),
            torch.from_numpy(brow_only).unsqueeze(0),
            torch.from_numpy(eye_only).unsqueeze(0))


class IllustriousPosePreviewV82:
    """把 POSE_KEYPOINT 画到图像上：骨骼线 + 脸部点 + 眼线 + 眉毛线。

    用来核对最终图上的 pose 检测质量（脸点是否落在正确位置）。
    """

    _BODY_LINKS = [(0, 1), (1, 2), (2, 3), (3, 4), (1, 5), (5, 6), (6, 7)]
    _R_BROW = list(range(17, 22))
    _L_BROW = list(range(22, 27))
    _R_EYE = list(range(36, 42))
    _L_EYE = list(range(42, 48))

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "keypoints": ("POSE_KEYPOINT",),
                "show_body": ("BOOLEAN", {"default": True}),
                "show_face": ("BOOLEAN", {"default": True}),
                "show_eye_line": ("BOOLEAN", {"default": True}),
                "point_size": ("INT", {"default": 3, "min": 1, "max": 12, "step": 1}),
                "opacity": ("FLOAT", {"default": 0.9, "min": 0.1, "max": 1.0, "step": 0.05}),
                "score_threshold": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.05}),
            },
            "optional": {
                "image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("preview",)
    FUNCTION = "draw"
    CATEGORY = "Illustrious/Utils"

    def draw(self, keypoints, show_body, show_face, show_eye_line,
             point_size, opacity, score_threshold, image=None):
        import numpy as np
        from PIL import Image, ImageDraw

        items = keypoints if isinstance(keypoints, (list, tuple)) else [keypoints]
        flat = []
        for it in items:
            if isinstance(it, dict):
                flat.append(it)
            elif isinstance(it, (list, tuple)):
                flat.extend([x for x in it if isinstance(x, dict)])
        if not flat:
            raise ValueError("[PosePreview] keypoints 为空")
        it = flat[0]

        cw = float(it.get("canvas_width") or 512)
        chh = float(it.get("canvas_height") or 512)

        if image is not None:
            base = image
            if base.ndim == 4:
                base = base[0]
            arr = (base.detach().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
            H, W = int(arr.shape[0]), int(arr.shape[1])
        else:
            W, H = int(cw), int(chh)
            arr = np.zeros((H, W, 3), dtype=np.uint8)

        pil = Image.fromarray(arr).convert("RGB")
        ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        dr = ImageDraw.Draw(ov)
        sx = W / max(1.0, cw)
        sy = H / max(1.0, chh)
        alpha = int(255 * float(opacity))
        r = int(point_size)
        th = max(1, int(round(point_size * 0.7)))

        people = []
        for item in flat:
            people.extend(item.get("people") or [])

        n_body_drawn = 0
        n_face_drawn = 0

        for person in people:
            body = person.get("pose_keypoints_2d") or []
            pts = {}
            if body:
                for i in range(len(body) // 3):
                    x, y, c = body[3 * i], body[3 * i + 1], body[3 * i + 2]
                    if c >= score_threshold and (x or y):
                        pts[i] = (x * sx, y * sy)

            if show_body:
                for a, b in self._BODY_LINKS:
                    if a in pts and b in pts:
                        dr.line([pts[a], pts[b]], fill=(0, 235, 235, alpha), width=th)
                for p in pts.values():
                    dr.ellipse([p[0] - r, p[1] - r, p[0] + r, p[1] + r],
                               fill=(0, 200, 255, alpha))
                n_body_drawn += len(pts)

            fk = person.get("face_keypoints_2d") or []
            fpts = {}
            if fk:
                for i in range(len(fk) // 3):
                    x, y, c = fk[3 * i], fk[3 * i + 1], fk[3 * i + 2]
                    if c >= score_threshold and (x or y):
                        fpts[i] = (x * sx, y * sy)

            if show_face and fpts:
                r2 = max(1, r - 1)
                for p in fpts.values():
                    dr.ellipse([p[0] - r2, p[1] - r2, p[0] + r2, p[1] + r2],
                               fill=(255, 255, 255, alpha))
                n_face_drawn += len(fpts)

                # 眉毛折线（洋红 / 青）
                for idxs, col in ((self._R_BROW, (255, 90, 200, alpha)),
                                  (self._L_BROW, (160, 120, 255, alpha))):
                    seq = [fpts[i] for i in idxs if i in fpts]
                    if len(seq) >= 2:
                        dr.line(seq, fill=col, width=max(1, th))

                # 眼线（黄）
                if show_eye_line:
                    def mean_of(idxs):
                        ps = [fpts[i] for i in idxs if i in fpts]
                        if not ps:
                            return None
                        return (sum(p[0] for p in ps) / len(ps),
                                sum(p[1] for p in ps) / len(ps))
                    e1 = mean_of(self._R_EYE)
                    e2 = mean_of(self._L_EYE)
                    if e1 and e2:
                        dr.line([e1, e2], fill=(255, 230, 40, alpha), width=max(2, th + 1))
                        for e in (e1, e2):
                            rr = r + 2
                            dr.ellipse([e[0] - rr, e[1] - rr, e[0] + rr, e[1] + rr],
                                       outline=(255, 230, 40, alpha), width=max(1, th))

        out = Image.alpha_composite(pil.convert("RGBA"), ov).convert("RGB")
        res = np.asarray(out).astype(np.float32) / 255.0
        print(f"[PosePreview] 画布 {W}x{H} 缩放({sx:.2f},{sy:.2f}) "
              f"body点={n_body_drawn} face点={n_face_drawn} people={len(people)}")

        import torch
        return (torch.from_numpy(res).unsqueeze(0),)


class IllustriousMaskDebuggerV82:
    """把 脸部(SAM) / 眉眼带 / 眼睛圆 三个部件用不同颜色叠加显示，便于定位是哪一层出问题。

    注意：本节点会自己再跑一次检测（含 SAM），仅用于调试，可随时静音。
    """

    COLORS = {
        "face": (0.10, 0.55, 1.00),   # 蓝 = SAM 脸部
        "brow": (1.00, 0.50, 0.00),   # 橙 = 眉眼带
        "eye":  (0.10, 1.00, 0.30),   # 绿 = 眼睛补圆
    }

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "bbox_detector": ("BBOX_DETECTOR",),
                "brow_offset": ("FLOAT", {"default": 1.70, "min": 0.8, "max": 3.5, "step": 0.05}),
                "band_top_extend": ("FLOAT", {"default": 0.45, "min": 0.0, "max": 1.5, "step": 0.05}),
                "band_bottom_ratio": ("FLOAT", {"default": 0.85, "min": 0.3, "max": 1.3, "step": 0.05}),
                "landmark_pad": ("FLOAT", {"default": 1.25, "min": 0.8, "max": 2.5, "step": 0.05}),
                "eye_radius_ratio": ("FLOAT", {"default": 0.50, "min": 0.0, "max": 1.2, "step": 0.05}),
                "trim_bottom_ratio": ("FLOAT", {"default": 0.02, "min": -0.10, "max": 0.25, "step": 0.01}),
                "brow_width_ratio": ("FLOAT", {"default": 0.72, "min": 0.40, "max": 0.98, "step": 0.02}),
                "brow_top_ratio": ("FLOAT", {"default": 0.12, "min": 0.0, "max": 0.40, "step": 0.02}),
                "brow_bottom_ratio": ("FLOAT", {"default": 0.56, "min": 0.30, "max": 0.85, "step": 0.02}),
                "face_width_ratio": ("FLOAT", {"default": 1.00, "min": 0.4, "max": 1.5, "step": 0.02}),
                "chin_taper": ("FLOAT", {"default": 0.30, "min": 0.0, "max": 0.8, "step": 0.02}),
                "max_tilt_deg": ("FLOAT", {"default": 45.0, "min": 0.0, "max": 90.0, "step": 1.0}),
                "eye_source": (["auto", "pose", "mask-holes", "pca", "insightface"],
                               {"default": "auto"}),
                "mode": (["分色填充+轮廓", "仅轮廓", "仅脸部(SAM)", "仅眉眼带", "仅眼圆"],
                         {"default": "分色填充+轮廓"}),
                "opacity": ("FLOAT", {"default": 0.35, "min": 0.1, "max": 1.0, "step": 0.05}),
                "edge_width": ("INT", {"default": 2, "min": 1, "max": 8, "step": 1}),
                "fill_holes": ("BOOLEAN", {"default": True}),
                "band_p": ("FLOAT", {"default": 4.0, "min": 2.0, "max": 12.0, "step": 0.5}),
                "brow_lift_floor": ("FLOAT", {"default": 0.38, "min": 0.0, "max": 1.0, "step": 0.02}),
                "cut_above_band": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "image": ("IMAGE",),
                "sam_model": ("SAM_MODEL",),
                "pose_keypoints": ("POSE_KEYPOINT",),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "MASK", "MASK")
    RETURN_NAMES = ("preview", "face_mask", "brow_mask", "eye_mask")
    FUNCTION = "run"
    CATEGORY = "Illustrious/Utils"

    def run(self, bbox_detector, brow_offset, band_top_extend, band_bottom_ratio,
            landmark_pad, eye_radius_ratio,
            trim_bottom_ratio, brow_width_ratio, brow_top_ratio, brow_bottom_ratio,
            face_width_ratio, chin_taper, max_tilt_deg, eye_source,
            mode, opacity, edge_width, fill_holes, band_p, brow_lift_floor,
            cut_above_band,
            image=None, sam_model=None, pose_keypoints=None):
        import numpy as np
        import torch

        det = _TiltAwareBrowDetectorV82(
            bbox_detector=bbox_detector, sam_model=sam_model,
            extend_up=0, extend_lr=0, brow_width_ratio=brow_width_ratio,
            brow_top_ratio=brow_top_ratio, brow_bottom_ratio=brow_bottom_ratio,
            trim_bottom_ratio=trim_bottom_ratio, feather=0,
            tilt_follow=True, max_tilt_deg=max_tilt_deg,
            band_limited_dilate=True,
            face_width_ratio=face_width_ratio, chin_taper=chin_taper,
            eye_source=eye_source, pose_data=pose_keypoints,
            landmark_pad=landmark_pad, brow_offset=brow_offset,
            band_top_extend=band_top_extend, band_bottom_ratio=band_bottom_ratio,
            eye_radius_ratio=eye_radius_ratio,
            fill_holes=fill_holes, band_p=band_p, brow_lift_floor=brow_lift_floor,
            cut_above_band=cut_above_band,
            axis_slope=axis_slope,
            axis_knee=axis_knee,
            axis_max_w=axis_max_w,
        )
        det.collect_parts = True

        src = image
        if src is None:
            raise ValueError("[MaskDebugger] 需要连接 image")
        if src.shape[0] > 1:
            src = src[:1]
        det.detect(src, 0.35, 0, 2.5, 10, None)

        hh = int(src.shape[1])
        ww = int(src.shape[2])
        parts = det.last_parts or {"face": [], "brow": [], "eye": []}

        def assemble(key):
            acc = np.zeros((hh, ww), dtype=np.float32)
            for (cx1, cy1, cx2, cy2), t in parts.get(key, []):
                sub = t[:, :]
                th_, tw_ = sub.shape
                ey = min(cy2, hh)
                ex = min(cx2, ww)
                dy, dx = ey - cy1, ex - cx1
                if dy > 0 and dx > 0:
                    acc[cy1:ey, cx1:ex] = np.maximum(
                        acc[cy1:ey, cx1:ex], sub[:dy, :dx])
            return acc

        fm = assemble("face")
        bm = assemble("brow")
        em = assemble("eye")

        base = src[0].cpu().numpy()
        out = base.astype(np.float32).copy()

        try:
            import scipy.ndimage as ndi
        except Exception:
            ndi = None

        order = [("face", fm), ("brow", bm), ("eye", em)]
        if mode == "仅脸部(SAM)":
            order = [("face", fm)]
        elif mode == "仅眉眼带":
            order = [("brow", bm)]
        elif mode == "仅眼圆":
            order = [("eye", em)]

        single = mode in ("仅脸部(SAM)", "仅眉眼带", "仅眼圆")
        fill_a = 0.65 if single else float(opacity)
        draw_edge = mode != "分色填充+轮廓" or True   # 所有模式都画轮廓

        # 1) 先铺半透明填充（单层模式用更高不透明度）
        if mode != "仅轮廓":
            for key, msk in order:
                if float(msk.max()) <= 0:
                    continue
                col = np.array(self.COLORS[key], dtype=np.float32).reshape(1, 1, 3)
                al = (msk > 0.5).astype(np.float32)[:, :, None] * fill_a
                out = out * (1 - al) + col * al

        # 2) 再画高对比轮廓线（满不透明度），保证重叠区边界依然可分辨
        if draw_edge:
            w = int(max(1, edge_width))
            for key, msk in order:
                binm = msk > 0.5
                if not binm.any():
                    continue
                if ndi is not None:
                    er = ndi.binary_erosion(binm, iterations=w)
                    edge = binm & (~er)
                else:
                    edge = binm
                col = np.array(self.COLORS[key], dtype=np.float32).reshape(1, 1, 3)
                out[edge] = col[0, 0]

        out = np.clip(out, 0, 1)

        # 各层单独输出：face 去掉眉眼带/眼圆，brow 去掉脸部，eye 去掉其余
        face_only = np.clip(fm - np.maximum(bm, em), 0, 1)
        brow_only = np.clip(bm - fm, 0, 1)
        eye_only = np.clip(em - np.maximum(fm, bm), 0, 1)

        print("[MaskDebugger] face=%d brow=%d eye=%d | 仅SAM=%d 仅眉眼带=%d 仅眼圆=%d"
              % (int((fm > 0.5).sum()), int((bm > 0.5).sum()), int((em > 0.5).sum()),
                 int((face_only > 0.5).sum()), int((brow_only > 0.5).sum()),
                 int((eye_only > 0.5).sum())))

        return (torch.from_numpy(out).unsqueeze(0),
                torch.from_numpy(face_only).unsqueeze(0),
                torch.from_numpy(brow_only).unsqueeze(0),
                torch.from_numpy(eye_only).unsqueeze(0))


class IllustriousSamBrowDetectorV82:
    """SAM 精确边缘 + 跟随头倾斜的眉眼带（主轴分析）。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "bbox_detector": ("BBOX_DETECTOR",),
                "extend_up": ("INT", {"default": 12, "min": 0, "max": 96, "step": 2,
                                      "tooltip": "向上膨胀像素（band_limited 开启时只在眉眼带宽度内）"}),
                "extend_lr": ("INT", {"default": 2, "min": 0, "max": 32, "step": 1,
                                      "tooltip": "左右微扩像素"}),
                "brow_width_ratio": ("FLOAT", {"default": 0.72, "min": 0.40, "max": 0.98, "step": 0.02,
                                               "tooltip": "眉眼带宽度占脸宽比例"}),
                "brow_top_ratio": ("FLOAT", {"default": 0.12, "min": 0.0, "max": 0.40, "step": 0.02,
                                             "tooltip": "眉眼带上沿（沿主轴，0=额头端）"}),
                "brow_bottom_ratio": ("FLOAT", {"default": 0.56, "min": 0.30, "max": 0.85, "step": 0.02,
                                                "tooltip": "眉眼带下沿（沿主轴）"}),
                "trim_bottom_ratio": ("FLOAT", {"default": 0.02, "min": -0.10, "max": 0.25, "step": 0.01,
                                                "tooltip": "下巴/脖子截断比例"}),
                "feather": ("INT", {"default": 6, "min": 0, "max": 24, "step": 1,
                                    "tooltip": "边缘羽化像素"}),
                "tilt_follow": ("BOOLEAN", {"default": True,
                                            "tooltip": "是否让眉眼带跟随头部倾斜（主轴分析）"}),
                "max_tilt_deg": ("FLOAT", {"default": 45.0, "min": 0.0, "max": 90.0, "step": 1.0,
                                           "tooltip": "超过该倾斜角就退回轴对齐（防止误判）"}),
                "band_limited_dilate": ("BOOLEAN", {"default": True,
                                                    "tooltip": "向上膨胀限制在眉眼带宽度内，避免顺刘海爬升"}),
                "face_width_ratio": ("FLOAT", {"default": 1.00, "min": 0.4, "max": 1.5, "step": 0.02,
                                               "tooltip": "无 SAM 时回退椭圆的宽度比例"}),
                "chin_taper": ("FLOAT", {"default": 0.30, "min": 0.0, "max": 0.8, "step": 0.02,
                                         "tooltip": "回退椭圆的下半脸收窄"}),
                "sam_threshold": ("FLOAT", {"default": 0.93, "min": 0.5, "max": 1.0, "step": 0.01,
                                            "tooltip": "SAM 阈值"}),
                "sam_bbox_expansion": ("INT", {"default": 0, "min": 0, "max": 200, "step": 2,
                                               "tooltip": "SAM 提示框外扩像素"}),
                "sam_dilation": ("INT", {"default": 0, "min": -64, "max": 64, "step": 1,
                                         "tooltip": "SAM mask 均匀膨胀像素"}),
                "eye_source": (["auto", "pose", "mask-holes", "pca", "insightface"],
                               {"default": "auto",
                                "tooltip": "倾角来源：auto=pose眉眼点→遮罩孔洞→PCA"}),
                "landmark_pad": ("FLOAT", {"default": 1.25, "min": 0.8, "max": 2.5, "step": 0.05,
                                           "tooltip": "眉眼带宽度外扩倍数"}),
                "brow_offset": ("FLOAT", {"default": 1.70, "min": 0.8, "max": 3.5, "step": 0.05,
                                          "tooltip": "眉毛位置修正：从眼心沿(眼->眉)方向拉长的倍数（openpose 眉毛偏低，1.7 左右）"}),
                "band_top_extend": ("FLOAT", {"default": 0.45, "min": 0.0, "max": 1.5, "step": 0.05,
                                              "tooltip": "眉眼带上边界超出眉线的距离（× 眉到鼻距离）"}),
                "band_bottom_ratio": ("FLOAT", {"default": 0.85, "min": 0.3, "max": 1.3, "step": 0.05,
                                                "tooltip": "眉眼带下边界走多少（1.0=到鼻子，0.85=提前收住）"}),
                "eye_radius_ratio": ("FLOAT", {"default": 0.50, "min": 0.0, "max": 1.2, "step": 0.05,
                                               "tooltip": "眼睛补圆半径 = 眼距 × 该值（0 关闭）"}),
                "fill_holes": ("BOOLEAN", {"default": True,
                                           "tooltip": "填平 SAM 皮肤遮罩内部的孔洞（眼睛/嘴巴）"}),
                "band_p": ("FLOAT", {"default": 4.0, "min": 2.0, "max": 12.0, "step": 0.5,
                                     "tooltip": "眉眼带超椭圆指数：2=椭圆，4~6=圆角矩形（包住大圆眼且不戳鬓角）"}),
                "brow_lift_floor": ("FLOAT", {"default": 0.38, "min": 0.0, "max": 1.0, "step": 0.02,
                                              "tooltip": "眉高自动下限 = 该值 × (眼心到人脸框顶边距离)，0=关闭"}),
                "preview_mode": (["分色填充+轮廓", "仅轮廓", "仅脸部(SAM)", "仅眉眼带", "仅眼圆"],
                                 {"default": "分色填充+轮廓",
                                  "tooltip": "本节点自带预览的模式（接上 image 才会出图）"}),
                "cut_above_band": ("BOOLEAN", {"default": True,
                                               "tooltip": "截掉眉眼带上边界以上的识别（修漂染刘海把 SAM 带成尖角）"}),
                "axis_slope": ("FLOAT", {"default": 0.010, "min": 0.0, "max": 0.05, "step": 0.001,
                                         "tooltip": "线性段斜率：每 1° 偏差增加多少鼻颈轴权重（0.01 → 10°时10%、30°时30%）"}),
                "axis_knee": ("FLOAT", {"default": 30.0, "min": 1.0, "max": 90.0, "step": 0.5,
                                        "tooltip": "拐点角度：超过它改为指数饱和（度）"}),
                "axis_max_w": ("FLOAT", {"default": 0.50, "min": 0.0, "max": 1.0, "step": 0.01,
                                         "tooltip": "鼻颈轴权重上限（0.5 = 正交永远至少占一半）"}),
                "pose_check": (["强制", "仅警告", "关闭"], {"default": "强制",
                                                            "tooltip": "强制=判不可信就弃用姿势；"
                                                                       "仅警告=只打印指标与判定，照常使用姿势点；"
                                                                       "关闭=完全不校验"}),
            },
            "optional": {
                "sam_model": ("SAM_MODEL",),
                "pose_keypoints": ("POSE_KEYPOINT",),
                "image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("BBOX_DETECTOR", "IMAGE", "MASK", "MASK", "MASK")
    RETURN_NAMES = ("bbox_detector", "preview", "face_mask", "brow_mask", "eye_mask")
    FUNCTION = "build"
    CATEGORY = "Illustrious/Image"

    def build(self, bbox_detector, extend_up, extend_lr, brow_width_ratio,
              brow_top_ratio, brow_bottom_ratio, trim_bottom_ratio, feather,
              tilt_follow, max_tilt_deg, band_limited_dilate,
              face_width_ratio, chin_taper, sam_threshold, sam_bbox_expansion,
              sam_dilation, eye_source="auto", landmark_pad=1.25,
              brow_offset=1.70, eye_radius_ratio=0.50,
              band_top_extend=0.45, band_bottom_ratio=0.85,
              fill_holes=True, band_p=4.0, brow_lift_floor=0.38,
              preview_mode="分色填充+轮廓", cut_above_band=True,
              axis_slope=0.01, axis_knee=30.0, axis_max_w=0.50, pose_check="强制",
              image=None,
              sam_model=None, pose_keypoints=None):
        det = _TiltAwareBrowDetectorV82(
            bbox_detector=bbox_detector, sam_model=sam_model,
            extend_up=extend_up, extend_lr=extend_lr,
            brow_width_ratio=brow_width_ratio,
            brow_top_ratio=brow_top_ratio, brow_bottom_ratio=brow_bottom_ratio,
            trim_bottom_ratio=trim_bottom_ratio, feather=feather,
            tilt_follow=tilt_follow, max_tilt_deg=max_tilt_deg,
            band_limited_dilate=band_limited_dilate,
            face_width_ratio=face_width_ratio, chin_taper=chin_taper,
            sam_threshold=sam_threshold,
            sam_bbox_expansion=sam_bbox_expansion, sam_dilation=sam_dilation,
            eye_source=eye_source,
            pose_data=pose_keypoints,
            landmark_pad=landmark_pad,
            brow_offset=brow_offset,
            eye_radius_ratio=eye_radius_ratio,
            band_top_extend=band_top_extend,
            band_bottom_ratio=band_bottom_ratio,
            fill_holes=fill_holes,
            band_p=band_p,
            brow_lift_floor=brow_lift_floor,
            cut_above_band=cut_above_band,
            axis_slope=axis_slope,
            axis_knee=axis_knee,
            axis_max_w=axis_max_w,
            pose_check=pose_check,
        )

        # 自带预览：用【本节点自己的参数】跑一遍，输出三色分色图 + 三个遮罩。
        # 这样预览永远和实际重绘范围一致，不会出现"改了参数预览不动"。
        if image is None:
            import torch as _t
            dummy_img = _t.zeros((1, 8, 8, 3), dtype=_t.float32)
            dummy_m = _t.zeros((1, 8, 8), dtype=_t.float32)
            return (det, dummy_img, dummy_m, dummy_m, dummy_m)

        det.collect_parts = True
        try:
            det.detect(image, 0.35, 0, 2.5, 10, None)
            parts = det.last_parts or {"face": [], "brow": [], "eye": []}
            prev, fm, bm, em = _render_parts(image, parts, preview_mode, 0.35, 2)
        finally:
            det.collect_parts = False

        print("[V82] 预览已生成 mode=%s" % preview_mode)
        return (det, prev, fm, bm, em)
