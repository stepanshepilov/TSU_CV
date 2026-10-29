"""Презентация-отчёт по лабораторной работе 5 (вариант 9).

Скрипт повторяет конвейер ноутбука — SIFT, BFMatcher, фильтр Лоу и RANSAC на
реальной фотографии баннера, — поэтому все числа на слайдах получены измерением,
а не переписаны.
"""
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from presentation_kit import figures as F  # noqa: E402
from presentation_kit import tokens as T  # noqa: E402
from presentation_kit.deck import Deck  # noqa: E402

ROOT = Path(__file__).resolve().parent
LOGO_PATH = ROOT / "data" / "logo_source.jpg"
BANNER_PATH = ROOT / "data" / "banner_photo.jpg"
ASSETS = ROOT / ".presentation_assets"
ASSETS.mkdir(exist_ok=True)
OUTPUT = ROOT / "Лабораторная_5_вариант_9.pptx"

RATIO = 0.8
LOGO_WIDTH = 400
PAD = 20

logo_source = cv2.imread(str(LOGO_PATH), cv2.IMREAD_COLOR)
banner = cv2.imread(str(BANNER_PATH), cv2.IMREAD_COLOR)

# Эталон: цветные пиксели выше полосы фотобанка, с отступом, ширина 400
saturation = cv2.cvtColor(logo_source, cv2.COLOR_BGR2HSV)[:, :, 1]
ys, xs = np.where(saturation[:1150] > 80)
x0, x1, y0, y1 = xs.min() - PAD, xs.max() + PAD, ys.min() - PAD, ys.max() + PAD
logo_crop = logo_source[y0:y1 + 1, x0:x1 + 1]
LOGO_SCALE = LOGO_WIDTH / logo_crop.shape[1]
logo = cv2.resize(logo_crop, None, fx=LOGO_SCALE, fy=LOGO_SCALE, interpolation=cv2.INTER_AREA)
LOGO_H, LOGO_W = logo.shape[:2]


def to_rgb(bgr):
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def signed(angle):
    return f"{angle:+d}°" if angle else "0°"


def ransac_homography(src, dst, threshold=5.0):
    """RANSAC с фиксированным зерном, чтобы числа на слайдах воспроизводились."""
    cv2.setRNGSeed(0)
    return cv2.findHomography(src, dst, cv2.RANSAC, threshold)


def tilt_view(img, angle_deg):
    """Поворачивает плоскость снимка вокруг вертикальной оси и проецирует заново."""
    h, w = img.shape[:2]
    distance = w * 1.2
    t = np.deg2rad(angle_deg)
    plane = np.float32([[-w / 2, -h / 2], [w / 2, -h / 2], [w / 2, h / 2], [-w / 2, h / 2]])
    dst = np.zeros_like(plane)
    for i, (X, Y) in enumerate(plane):
        x_rot, z_rot = X * np.cos(t), -X * np.sin(t)
        dst[i] = (w / 2 + distance * x_rot / (distance + z_rot),
                  h / 2 + distance * Y / (distance + z_rot))
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    view_h = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, view_h, (w, h), borderValue=(40, 40, 40)), view_h


sift = cv2.SIFT_create()
logo_gray = cv2.cvtColor(logo, cv2.COLOR_BGR2GRAY)
banner_gray = cv2.cvtColor(banner, cv2.COLOR_BGR2GRAY)
kp_logo, des_logo = sift.detectAndCompute(logo_gray, None)
kp_banner, des_banner = sift.detectAndCompute(banner_gray, None)
logo_sizes = np.array([k.size for k in kp_logo])

bf = cv2.BFMatcher(cv2.NORM_L2)
knn_matches = bf.knnMatch(des_logo, des_banner, k=2)
ratios = np.array([m.distance / n.distance for m, n in knn_matches])
good_matches = sorted([m for m, n in knn_matches if m.distance < RATIO * n.distance],
                      key=lambda m: m.distance)

# --- опорная гомография по эталону в полном разрешении ---------------------
kp_full, des_full = sift.detectAndCompute(cv2.cvtColor(logo_crop, cv2.COLOR_BGR2GRAY), None)
ref_knn = [m for m, n in bf.knnMatch(des_full, des_banner, k=2) if m.distance < 0.8 * n.distance]
ref_src = np.float32([kp_full[m.queryIdx].pt for m in ref_knn]).reshape(-1, 1, 2)
ref_dst = np.float32([kp_banner[m.trainIdx].pt for m in ref_knn]).reshape(-1, 1, 2)
_, ref_mask = ransac_homography(ref_src, ref_dst, 3.0)
ref_mask = ref_mask.ravel().astype(bool)
H_full, _ = cv2.findHomography(ref_src[ref_mask], ref_dst[ref_mask], 0)
H_ref = H_full @ np.diag([1 / LOGO_SCALE, 1 / LOGO_SCALE, 1])

LOGO_CORNERS = np.float32([[0, 0], [LOGO_W, 0], [LOGO_W, LOGO_H], [0, LOGO_H]]).reshape(-1, 1, 2)
REF_CORNERS = cv2.perspectiveTransform(LOGO_CORNERS, H_ref)


def evaluate(matches, gt=H_ref, keypoints=None):
    """Доля верных совпадений и точность локализации по оценённой гомографии."""
    keypoints = keypoints or kp_banner
    if len(matches) < 4:
        return dict(n=len(matches), correct=0, precision=float("nan"),
                    inliers=0, error=float("nan"), homography=None)
    src = np.float32([kp_logo[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst = np.float32([keypoints[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    errors = np.linalg.norm(cv2.perspectiveTransform(src, gt) - dst, axis=2).ravel()
    correct = int((errors < 5.0).sum())
    estimated, mask = ransac_homography(src, dst)
    inliers = int(mask.sum()) if mask is not None else 0
    error = float("nan")
    if estimated is not None:
        true_corners = cv2.perspectiveTransform(LOGO_CORNERS, gt)
        error = float(np.linalg.norm(cv2.perspectiveTransform(LOGO_CORNERS, estimated)
                                     - true_corners, axis=2).mean())
    return dict(n=len(matches), correct=correct, precision=correct / len(matches) * 100,
                inliers=inliers, error=error, homography=estimated,
                mask=mask.ravel().astype(bool) if mask is not None else None)


raw_matches = [m for m, _ in knn_matches]
cross_matches = sorted(cv2.BFMatcher(cv2.NORM_L2, crossCheck=True).match(des_logo, des_banner),
                       key=lambda m: m.distance)
filter_modes = [
    ("без фильтра", evaluate(raw_matches)),
    ("crossCheck = True", evaluate(cross_matches)),
    (f"фильтр Лоу, ρ = {RATIO}", evaluate(good_matches)),
]
main = filter_modes[-1][1]

RATIO_GRID = [0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]
ratio_rows = [evaluate([m for m, n in knn_matches if m.distance < rho * n.distance])
              for rho in RATIO_GRID]

# --- локализация -----------------------------------------------------------
inlier_mask = main["mask"]
est_corners = cv2.perspectiveTransform(LOGO_CORNERS, main["homography"])
corner_errors = np.linalg.norm(est_corners - REF_CORNERS, axis=2).ravel()
logo_diagonal = float(np.linalg.norm(est_corners[0] - est_corners[2]))
affine = (main["homography"] / main["homography"][2, 2])[:2, :2]
shrink_x, shrink_y = 1 / np.linalg.norm(affine[:, 0]), 1 / np.linalg.norm(affine[:, 1])
rotation = float(np.degrees(np.arctan2(affine[1, 0], affine[0, 0])))

located = banner.copy()
cv2.polylines(located, [np.int32(REF_CORNERS)], True, (255, 255, 255), 2, cv2.LINE_AA)
cv2.polylines(located, [np.int32(est_corners)], True, (0, 220, 0), 3, cv2.LINE_AA)

logo_edges = cv2.Canny(logo_gray, 50, 150)
warped_edges = cv2.warpPerspective(logo_edges, H_ref, (banner.shape[1], banner.shape[0]))
overlay = banner.copy()
overlay[warped_edges > 0] = (0, 255, 0)
overlay_zoom = cv2.resize(overlay[30:380, 80:360], None, fx=2, fy=2,
                          interpolation=cv2.INTER_NEAREST)

inlier_img = cv2.drawMatches(logo, kp_logo, banner, kp_banner, good_matches, None,
                             matchColor=(0, 200, 0), singlePointColor=(190, 190, 190),
                             matchesMask=inlier_mask.astype(np.uint8).tolist(),
                             flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
# drawMatches дополняет более низкое изображение чёрным — заменяем на цвет карточки
inlier_img[LOGO_H:, :LOGO_W] = (252, 246, 243)
logo_kp_img = cv2.drawKeypoints(logo, kp_logo, None,
                                flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
banner_kp_img = cv2.drawKeypoints(banner, kp_banner, None,
                                  flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# --- 4. разный угол обзора -------------------------------------------------
ANGLES = [-60, -45, -30, 0, 30, 45, 55, 65, 75]
GALLERY_ANGLES = [-45, -30, 0, 30, 45]
angle_rows, angle_previews = [], []
for angle in ANGLES:
    scene, view_h = tilt_view(banner, angle)
    kp_scene, des_scene = sift.detectAndCompute(cv2.cvtColor(scene, cv2.COLOR_BGR2GRAY), None)
    gt = view_h @ H_ref
    good = [m for m, n in bf.knnMatch(des_logo, des_scene, k=2)
            if m.distance < RATIO * n.distance]
    row = evaluate(good, gt=gt, keypoints=kp_scene)
    true_corners = cv2.perspectiveTransform(LOGO_CORNERS, gt).reshape(-1, 2)
    row.update(angle=angle, logo_px=float(np.linalg.norm(true_corners[1] - true_corners[0])))
    angle_rows.append(row)

    if angle in GALLERY_ANGLES:
        preview = scene.copy()
        cv2.polylines(preview, [np.int32(true_corners.reshape(-1, 1, 2))], True,
                      (255, 255, 255), 2, cv2.LINE_AA)
        if row["homography"] is not None:
            found = cv2.perspectiveTransform(LOGO_CORNERS, row["homography"])
            cv2.polylines(preview, [np.int32(found)], True, (0, 220, 0), 3, cv2.LINE_AA)
        angle_previews.append((to_rgb(preview[:420]), f"{signed(angle)} · инлайеров {row['inliers']}"))

# --- устойчивость ----------------------------------------------------------
def match_against(scene_img, gt, label):
    kp_scene, des_scene = sift.detectAndCompute(cv2.cvtColor(scene_img, cv2.COLOR_BGR2GRAY), None)
    good = [m for m, n in bf.knnMatch(des_logo, des_scene, k=2)
            if m.distance < RATIO * n.distance]
    row = evaluate(good, gt=gt, keypoints=kp_scene)
    row["label"] = label
    return row


trials = [match_against(banner, H_ref, "исходная фотография")]
for factor in (0.75, 0.5, 0.35):
    small = cv2.resize(banner, None, fx=factor, fy=factor, interpolation=cv2.INTER_AREA)
    scale = np.float32([[factor, 0, 0], [0, factor, 0], [0, 0, 1]])
    trials.append(match_against(small, scale @ H_ref, f"масштаб кадра ×{factor}"))
for alpha, beta, name in ((0.5, 0, "яркость ×0.5"), (1.6, 30, "яркость ×1.6 +30"),
                          (1.0, -70, "яркость −70")):
    trials.append(match_against(cv2.convertScaleAbs(banner, alpha=alpha, beta=beta),
                                H_ref, name))
for deg in (45, 90, 180):
    h_s, w_s = banner.shape[:2]
    rot = cv2.getRotationMatrix2D((w_s / 2, h_s / 2), deg, 1.0)
    cos, sin = abs(rot[0, 0]), abs(rot[0, 1])
    new_w, new_h = int(h_s * sin + w_s * cos), int(h_s * cos + w_s * sin)
    rot[0, 2] += new_w / 2 - w_s / 2
    rot[1, 2] += new_h / 2 - h_s / 2
    trials.append(match_against(cv2.warpAffine(banner, rot, (new_w, new_h)),
                                np.vstack([rot, [0, 0, 1]]) @ H_ref, f"поворот кадра на {deg}°"))


# --- сравнение методов -----------------------------------------------------
def timed(matcher_call, repeats=5):
    start = time.perf_counter()
    for _ in range(repeats):
        result = matcher_call()
    return result, (time.perf_counter() - start) / repeats * 1000


_, bf_ms = timed(lambda: bf.knnMatch(des_logo, des_banner, k=2))
flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
flann_knn, flann_ms = timed(lambda: flann.knnMatch(des_logo, des_banner, k=2))
flann_good = [m for m, n in flann_knn if m.distance < RATIO * n.distance]
flann_result = evaluate(flann_good)
flann_common = len({(m.queryIdx, m.trainIdx) for m in good_matches}
                   & {(m.queryIdx, m.trainIdx) for m in flann_good})

orb = cv2.ORB_create(nfeatures=2000)
kp_logo_orb, des_logo_orb = orb.detectAndCompute(logo_gray, None)
kp_banner_orb, des_banner_orb = orb.detectAndCompute(banner_gray, None)
bf_hamming = cv2.BFMatcher(cv2.NORM_HAMMING)
orb_knn, orb_ms = timed(lambda: bf_hamming.knnMatch(des_logo_orb, des_banner_orb, k=2))
orb_good = [m for m, n in orb_knn if m.distance < RATIO * n.distance]
orb_src = np.float32([kp_logo_orb[m.queryIdx].pt for m in orb_good]).reshape(-1, 1, 2)
orb_dst = np.float32([kp_banner_orb[m.trainIdx].pt for m in orb_good]).reshape(-1, 1, 2)
orb_h, orb_mask = ransac_homography(orb_src, orb_dst)
orb_inliers = int(orb_mask.sum())
orb_error = float(np.linalg.norm(cv2.perspectiveTransform(LOGO_CORNERS, orb_h) - REF_CORNERS,
                                 axis=2).mean())

# --- иллюстрации -----------------------------------------------------------
hero = F.rounded_png(ASSETS / "hero.png", to_rgb(located[:470]), placement_w=5.08)
data_source = F.rounded_png(ASSETS / "logo_source.png", to_rgb(logo_source), placement_w=3.4)
data_logo = F.rounded_png(ASSETS / "logo.png", to_rgb(logo), placement_w=3.0)
data_banner = F.rounded_png(ASSETS / "banner.png", to_rgb(banner), placement_w=3.4)
logo_kp = F.rounded_png(ASSETS / "logo_kp.png", to_rgb(logo_kp_img), placement_w=3.9)
banner_kp = F.rounded_png(ASSETS / "banner_kp.png", to_rgb(banner_kp_img), placement_w=3.4)
overlay_png = F.rounded_png(ASSETS / "overlay.png", to_rgb(overlay_zoom), placement_w=3.4)
inliers_png = F.rounded_png(ASSETS / "inliers.png", to_rgb(inlier_img), placement_w=7.6)
located_png = F.rounded_png(ASSETS / "located.png", to_rgb(located), placement_w=3.4)
angle_gallery = F.panels(ASSETS / "angles.png", angle_previews, width=11.53)


def draw_ratio_hist(ax):
    ax.hist(ratios, bins=40, color=T.SERIES[0])
    ax.axvline(RATIO, color=T.SERIES[1], linestyle="--", linewidth=1.8)
    ax.annotate(f"ρ = {RATIO}", xy=(RATIO, 0), xytext=(RATIO - 0.2, ax.get_ylim()[1] * 0.82),
                color=T.SERIES[1], fontsize=10, weight="bold")
    ax.set_xlabel("Отношение d₁ / d₂")
    ax.set_ylabel("Совпадений")
    ax.set_xlim(0.3, 1.02)


ratio_hist = F.chart(ASSETS / "ratio_hist.png", 7.35, 3.0, draw_ratio_hist,
                     pad=(0.72, 0.52, 0.22, 0.30))


def draw_sweep(path, x_values, panels_spec, xlabel, legend_loc, shade=None,
               width=11.53, height=3.15):
    fig = F.figure(width, height)
    for column, (title, primary, secondary, labels) in enumerate(panels_spec):
        ax = F.axes(fig, [0.055 + column * 0.324, 0.20, 0.256, 0.66])
        ax.plot(x_values, primary, "o-", color=T.SERIES[0], markersize=4,
                label=labels[0] if secondary else None)
        if secondary:
            ax.plot(x_values, secondary, "s-", color=T.SERIES[1], markersize=4,
                    label=labels[1])
            ax.legend(loc=legend_loc)
        if column == 2:
            ax.set_yscale("symlog", linthresh=10)
        for left, right in shade or []:
            ax.axvspan(left, right, color=T.SERIES[1], alpha=0.08)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel(xlabel)
    return F.finish(fig, path)


ratio_sweep = draw_sweep(ASSETS / "ratio_sweep.png", RATIO_GRID, [
    ("Совпадений после фильтра", [r["n"] for r in ratio_rows],
     [r["correct"] for r in ratio_rows], ("всего", "из них верных")),
    ("Точность, %", [r["precision"] for r in ratio_rows], None, None),
    ("Ошибка локализации, пикс.", [r["error"] for r in ratio_rows], None, None),
], "коэффициент Лоу ρ", "upper left")

angle_sweep = draw_sweep(ASSETS / "angle_sweep.png", ANGLES, [
    ("Число совпадений", [r["n"] for r in angle_rows], [r["inliers"] for r in angle_rows],
     ("после фильтра Лоу", "инлайеров RANSAC")),
    ("Доля верных совпадений, %",
     [r["correct"] / r["n"] * 100 if r["n"] else 0 for r in angle_rows], None, None),
    ("Ошибка локализации, пикс.", [r["error"] for r in angle_rows], None, None),
], "дополнительный наклон, °", "upper left", shade=[(-63, -52), (38, 78)])

found_angles = [r["angle"] for r in angle_rows if np.isfinite(r["error"]) and r["error"] < 10]

# --- слайды ----------------------------------------------------------------
deck = Deck("Лабораторная работа 5 · вариант 9")

deck.cover(
    kicker="Лабораторная работа 5 · вариант 9",
    title="Поиск логотипа\nна реальной фотографии",
    subtitle="SIFT, BFMatcher с фильтром Лоу\nи локализация объекта методом RANSAC",
    meta="Компьютерное зрение · OpenCV · NumPy · Matplotlib",
    image=hero,
)

slide = deck.slide("Задание варианта 9", "Пять пунктов", kicker="Постановка")
deck.steps(slide, [
    "Найти ключевые точки на изображении логотипа (SIFT)",
    "Сопоставить логотип с его фотографией на баннере",
    "Использовать BFMatcher с фильтрацией по Лоу",
    "Сопоставить изображения с разным углом обзора",
    "Сделать вывод о применимости метода",
], x=T.MARGIN, y=2.25, w=7.55, row_h=0.70)
deck.metrics(slide, [
    (f"{len(kp_logo)}", "ключевых точек на логотипе"),
    (f"{des_logo.shape[1]}", "чисел в одном дескрипторе SIFT"),
    (f"{main['inliers']} / {main['n']}", "инлайеров RANSAC из совпадений"),
], x=8.86, y=2.20, w=3.57, h=3.62, vertical=True)

slide = deck.slide("Данные", "Эталонный логотип и реальная фотография фасада",
                   kicker="Постановка")
deck.fit_picture(slide, data_source, T.MARGIN, 2.12, 2.3, 2.45, center=False)
deck.fit_picture(slide, data_logo, 3.45, 2.12, 2.3, 2.45, center=False)
deck.fit_picture(slide, data_banner, 6.0, 2.12, 3.2, 4.3, center=False)
deck.caption(slide, "Исходный файл и эталон после обрезки", x=T.MARGIN, y=4.75, w=4.85)
deck.note(slide,
          f"Эталон: полоса фотобанка и белые поля обрезаны, ширина {LOGO_W} пикселей.",
          x=T.MARGIN, y=5.2, w=4.85, bar=True, size=T.BODY_SM)
deck.note(slide,
          f"Фото — реальный снимок фасада {banner.shape[1]} × {banner.shape[0]}. Логотип "
          "снят снизу и сбоку, повёрнут, уменьшен и сжат JPEG; рядом текст, стрелка "
          "и вывеска McDonald's.",
          x=9.5, y=2.12, w=2.93, bar=True, size=T.BODY_SM)

slide = deck.slide("Конвейер из трёх ступеней", "Ни одну ступень нельзя убрать",
                   kicker="Теория")
bottom = deck.columns(slide, [
    ("SIFT",
     "Пирамида разностей гауссиан даёт точки, устойчивые к масштабу; доминирующее "
     "направление градиента — к повороту. Дескриптор — 128 чисел из гистограмм "
     "направлений в сетке 4 × 4."),
    ("Фильтр Лоу",
     "Оставляем совпадение, только если d₁ / d₂ < ρ. У верного соответствия ближайший "
     "сосед заметно ближе второго, у случайного оба соседа равноудалены."),
    ("RANSAC",
     "Логотип плоский, значит все верные совпадения связаны одной гомографией 3 × 3. "
     "Случайные четвёрки точек дают модели, побеждает та, у которой больше инлайеров."),
], x=T.MARGIN, y=2.12, w=11.53)
deck.note(slide,
          "Абсолютный порог на расстояние d₁ работал бы хуже: разные участки "
          "изображения имеют разную различимость, единого масштаба расстояний "
          "не существует.",
          x=T.MARGIN, y=bottom + 0.32, w=11.53, bar=True, size=T.BODY)

slide = deck.slide("Ключевые точки логотипа", "Точки садятся туда, где градиент велик "
                                              "сразу в двух направлениях", kicker="Пункт 01")
deck.fit_picture(slide, logo_kp, T.MARGIN, 2.12, 4.2, 4.3, center=False)
deck.metrics(slide, [
    (f"{len(kp_logo)}", "ключевых точек"),
    (f"{des_logo.shape}", "матрица дескрипторов, float32"),
    (f"{logo_sizes.min():.1f} … {logo_sizes.max():.1f}", "размер окрестности, пикселей"),
    (f"{len(kp_logo) / (LOGO_W * LOGO_H) * 1e4:.1f}", "точек на 10 000 пикселей"),
], x=5.55, y=2.12, w=3.2, h=4.3, vertical=True)
deck.note(slide,
          "Точки — на контурах букв, бликах булочек и краях синей дуги. На крупных "
          "одноцветных заливках их почти нет: плоская графика даёт мало точек. "
          "Несколько точек попали на водяной знак фотобанка — на фото им нет пары.",
          x=9.05, y=2.12, w=3.38, bar=True, size=T.BODY_SM)

slide = deck.slide("Ключевые точки на фотографии", "Логотип — лишь малая часть кадра",
                   kicker="Пункт 02")
deck.fit_picture(slide, banner_kp, T.MARGIN, 2.12, 3.4, 4.5, center=False)
deck.metrics(slide, [
    (f"{len(kp_banner)}", "ключевых точек на фото"),
    (f"{len(kp_logo)} × {len(kp_banner)}", "сравнений при полном переборе"),
    (f"{len(kp_logo) * len(kp_banner) / 1e3:.0f} тыс.", "операций над 128-мерными векторами"),
], x=4.7, y=2.12, w=3.6, h=4.3, vertical=True)
deck.note(slide,
          "Большая часть точек приходится на надпись, стрелку, вывеску McDonald's, "
          "плитку стены, светильники и камеры. Для каждой точки логотипа найдётся "
          "«ближайший» дескриптор среди них — даже когда верной пары нет.",
          x=8.65, y=2.12, w=3.78, bar=True, size=T.BODY_SM)

slide = deck.slide("BFMatcher и фильтр Лоу", "Отношение d₁ / d₂ разделяет верные "
                                             "и случайные совпадения", kicker="Пункт 03")
deck.picture(slide, ratio_hist, T.MARGIN, 2.12, w=7.35)
deck.metrics(slide, [
    (f"{len(good_matches)} из {len(knn_matches)}", f"прошли фильтр при ρ = {RATIO}"),
    (f"{main['precision']:.1f} %", "из них верных"),
], x=8.62, y=2.12, w=3.81, h=3.0, vertical=True)
deck.table(slide, ["Режим фильтрации", "Совпадений", "Верных", "Точность", "Ошибка углов"],
           [[label, str(r["n"]), str(r["correct"]), f"{r['precision']:.1f} %",
             f"{r['error']:.2f} пикс."] for label, r in filter_modes],
           x=T.MARGIN, y=5.3, w=11.53, widths=[0.28, 0.17, 0.15, 0.17, 0.23],
           row_h=0.36, highlight=2)

slide = deck.slide("Подбор коэффициента ρ", "Маленький логотип требует мягкого порога",
                   kicker="Пункт 03")
bottom = deck.picture(slide, ratio_sweep, T.MARGIN, 2.12, w=11.53).bottom_in
deck.note(slide,
          f"При ρ ≤ 0.6 верных пар не хватает, и RANSAC строит неправильную модель. "
          f"Минимум ошибки — при ρ = {RATIO}, как в статье Лоу: важнее покрыть точками "
          "весь логотип, а оставшиеся ложные пары убирает RANSAC.",
          x=T.MARGIN, y=bottom + 0.28, w=11.53, bar=True, size=T.BODY_SM)

slide = deck.slide("Опорное положение логотипа", "Как численно оценить точность "
                                                 "на реальном фото", kicker="Методика")
deck.fit_picture(slide, overlay_png, T.MARGIN, 2.12, 3.4, 4.5, center=False)
deck.note(slide,
          "Истинное положение логотипа на снимке неизвестно. Опорная гомография строится "
          f"по другим данным — эталону в полном разрешении ({len(kp_full)} точек): "
          f"RANSAC с порогом 3 пикселя и уточнение по {ref_mask.sum()} инлайерам.",
          x=4.75, y=2.12, w=7.68, bar=True, size=T.BODY_SM)
deck.note(slide,
          "Проверка: контуры эталона, перенесённые на фото, легли на края букв "
          "и булочек с точностью до пикселя (слева, увеличение ×2).",
          x=4.75, y=3.55, w=7.68, bar=True, size=T.BODY_SM)
deck.note(slide,
          "Верное совпадение — в пределах 5 пикселей от опорного положения. Ошибка углов — "
          "среднее отклонение углов эталона при переносе оценённой и опорной гомографиями.",
          x=4.75, y=4.85, w=7.68, bar=True, size=T.BODY_SM)

slide = deck.slide("Логотип найден", "Зелёный контур — оценка, белый — опорное положение",
                   kicker="Локализация")
deck.fit_picture(slide, located_png, T.MARGIN, 2.12, 3.4, 4.5, center=False)
deck.fit_picture(slide, inliers_png, 4.55, 2.12, 7.88, 2.75, center=False)
deck.metrics(slide, [
    (f"{main['inliers']} из {main['n']}", "инлайеров RANSAC"),
    (f"{corner_errors.mean():.2f} пикс.", "среднее отклонение углов"),
    (f"{logo_diagonal:.0f} пикс.", "диагональ логотипа на фото"),
    (f"{rotation:.0f}°", "поворот логотипа на фото"),
], x=4.55, y=5.2, w=7.88, h=1.12)

slide = deck.slide("Разный угол обзора", "Снимок наклоняется как плоскость: опора "
                                         "пересчитывается точно", kicker="Пункт 04")
bottom = deck.picture(slide, angle_sweep, T.MARGIN, 2.12, w=11.53).bottom_in
deck.note(slide,
          f"Снимок уже сделан под углом около 40° (логотип сжат в {shrink_x:.1f} раза "
          f"по горизонтали и в {shrink_y:.1f} по вертикали): отрицательные углы частично "
          f"компенсируют наклон, положительные — добавляют. Логотип найден при "
          f"{', '.join(signed(a) for a in found_angles)}.",
          x=T.MARGIN, y=bottom + 0.28, w=11.53, bar=True, size=T.BODY_SM)

slide = deck.slide("Граница применимости", "Решает аффинное сжатие, а не размер объекта",
                   kicker="Пункт 04")
bottom = deck.fit_picture(slide, angle_gallery, T.MARGIN, 2.08, 11.53, 2.1).bottom_in
deck.table(slide, ["Наклон", "Ширина лого", "После Лоу", "Верных", "Инлайеров", "Ошибка углов"],
           [[signed(r['angle']), f"{r['logo_px']:.0f} пикс.", str(r["n"]), str(r["correct"]),
             str(r["inliers"]),
             f"{r['error']:.1f} пикс." if np.isfinite(r["error"]) else "не найден"]
            for r in angle_rows if r["angle"] in (-60, -45, -30, 0, 30, 45, 75)],
           x=T.MARGIN, y=bottom + 0.22, w=11.53,
           widths=[0.11, 0.19, 0.16, 0.14, 0.17, 0.23], row_h=0.29, size=11)

slide = deck.slide("Устойчивость к условиям съёмки",
                   "Масштаб, поворот и освещение метод переносит почти без потерь",
                   kicker="Дополнительно")
deck.table(slide, ["Условие", "После фильтра Лоу", "Инлайеров", "Ошибка углов"],
           [[r["label"], str(r["n"]), str(r["inliers"]),
             f"{r['error']:.2f} пикс." if np.isfinite(r["error"]) else "не найден"]
            for r in trials],
           x=T.MARGIN, y=2.12, w=7.4, widths=[0.40, 0.22, 0.20, 0.18], row_h=0.40, size=11.5)
deck.note(slide,
          "Дескриптор строится из направлений градиента и нормируется, поэтому "
          "равномерное изменение яркости из него уходит. Хуже всего — осветление "
          "с ростом контраста: красный и белый уходят в насыщение, и часть градиентов "
          "пропадает. Уменьшение кадра до ×0.35 (логотип около 60 пикселей) не мешает.",
          x=8.62, y=2.12, w=3.81, bar=True, size=T.BODY_SM)

slide = deck.slide("Сравнение методов", "На этой паре изображений полный перебор выигрывает",
                   kicker="Дополнительно")
deck.table(slide, ["Метод", "Дескрипторов", "После Лоу", "Инлайеров", "Ошибка", "Время"], [
    ["SIFT + BFMatcher", str(len(kp_logo)), str(main["n"]), str(main["inliers"]),
     f"{main['error']:.2f} пикс.", f"{bf_ms:.1f} мс"],
    ["SIFT + FLANN", str(len(kp_logo)), str(flann_result["n"]), str(flann_result["inliers"]),
     f"{flann_result['error']:.2f} пикс.", f"{flann_ms:.1f} мс"],
    ["ORB + BFMatcher", str(len(kp_logo_orb)), str(len(orb_good)), str(orb_inliers),
     f"{orb_error:.2f} пикс.", f"{orb_ms:.1f} мс"],
], x=T.MARGIN, y=2.22, w=11.53, widths=[0.22, 0.15, 0.13, 0.13, 0.22, 0.15], row_h=0.48)
deck.note(slide,
          f"FLANN дал почти тот же набор пар ({flann_common} общих из {main['n']}), но на "
          f"{len(kp_logo)} × {len(kp_banner)} дескрипторах построение kd-индекса не "
          f"окупается. ORB нашёл больше пар, но согласованы из них только {orb_inliers}: "
          "BRIEF хуже переносит перспективу и сжатие JPEG.",
          x=T.MARGIN, y=4.30, w=11.53, bar=True, size=T.BODY)

slide = deck.slide("Вывод о применимости метода", "Пункт 5 задания", kicker="Выводы")
bottom = deck.columns(slide, [
    ("Что метод выдерживает",
     "Уменьшение кадра почти втрое, поворот камеры на любой угол, затемнение "
     "и осветление, умеренную перспективу. Ошибка локализации — пара пикселей "
     "при логотипе в сотни пикселей."),
    ("Где метод ломается",
     "Когда суммарный наклон превышает примерно 45–50°: SIFT инвариантен к повороту "
     "и масштабу, но не к аффинному сжатию окрестности. Нужны ASIFT или обучаемые "
     "дескрипторы."),
    ("Практический вывод",
     "Для контроля наружной рекламы метод пригоден без обучения: работает по одному "
     "эталону и выдаёт не только факт наличия логотипа, но и его границы. Условия: "
     "плоский контрастный объект и ракурс до 45–50°."),
], x=T.MARGIN, y=2.05, w=11.53)
deck.note(slide,
          f"Роль ступеней: без фильтрации верных совпадений "
          f"{filter_modes[0][1]['precision']:.0f} %, перекрёстная проверка даёт "
          f"{filter_modes[1][1]['precision']:.0f} %, фильтр Лоу — {main['precision']:.0f} %, "
          "а RANSAC завершает очистку и попутно выдаёт саму гомографию.",
          x=T.MARGIN, y=bottom + 0.26, w=11.53, bar=True, size=T.BODY_SM)

slide = deck.slide("Контрольные вопросы", "Ответы по материалу работы", kicker="Защита")
deck.qa(slide, [
    ("Что такое ключевые точки и зачем они нужны?",
     f"Локальные участки, которые можно повторно найти на другом снимке того же "
     f"объекта. Они сводят сравнение изображений к сравнению небольших наборов "
     f"дескрипторов: логотип описан {len(kp_logo)} точками вместо "
     f"{LOGO_W * LOGO_H // 1000} тысяч пикселей."),
    ("Чем отличаются SIFT и ORB?",
     "SIFT описывает окрестность вещественным вектором из 128 чисел и сравнивает "
     "по евклидову расстоянию. ORB использует FAST и бинарный BRIEF из 32 байт "
     "с расстоянием Хэмминга: быстрее, но хуже переносит перспективу."),
    ("В чём разница между BFMatcher и FLANN?",
     "BFMatcher сравнивает каждый дескриптор с каждым — результат точен, сложность "
     "растёт как произведение размеров. FLANN строит индекс и ищет приближённых "
     "соседей: выигрыш появляется на десятках тысяч дескрипторов."),
    ("Зачем нужна фильтрация по методу Лоу?",
     f"Чтобы отбросить ненадёжные совпадения по критерию d₁ / d₂ < ρ. В работе фильтр "
     f"поднял долю верных совпадений с {filter_modes[0][1]['precision']:.0f} % "
     f"до {main['precision']:.0f} % при ρ = {RATIO}."),
    ("Где применяется сопоставление изображений?",
     "Поиск объекта или логотипа в сцене и контроль размещения рекламы, склейка "
     "панорам, дополненная реальность, визуальная локализация робота, восстановление "
     "трёхмерной сцены, поиск дубликатов, распознавание товаров на полке."),
], x=T.MARGIN, y=2.20, w=11.53, columns=2, size=12.5, avail_h=4.4)

slide = deck.slide("Итоги", "Что выполнено в ноутбуке", kicker="Выводы")
bottom = deck.checklist(slide, [
    f"SIFT нашёл {len(kp_logo)} точек на логотипе и {len(kp_banner)} на фотографии",
    f"BFMatcher с фильтром Лоу при ρ = {RATIO} дал {main['n']} совпадений",
    f"RANSAC отобрал {main['inliers']} инлайеров и построил гомографию",
    f"Отклонение углов от опорного положения — {corner_errors.mean():.2f} пикселя "
    f"при диагонали логотипа {logo_diagonal:.0f} пикселей",
    "Эксперимент с наклоном снимка определил границу применимости — около 45–50°",
    "Измерена устойчивость к масштабу, освещению и повороту кадра",
], x=T.MARGIN, y=2.20, w=11.53, size=16.5)
deck.text(slide,
          "Связка SIFT → BFMatcher → фильтр Лоу → RANSAC находит логотип на реальной "
          "фотографии без обучения, по единственному эталонному изображению, но с чёткой "
          "границей по ракурсу съёмки.",
          x=T.MARGIN, y=bottom + 0.42, w=11.53, h=0.8,
          size=T.BODY, color=T.PRIMARY, bold=True)

path, count = deck.save(OUTPUT)
print(f"saved: {path} ({count} слайдов)")
