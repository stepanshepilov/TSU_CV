"""Презентация-отчёт по лабораторной работе 5 (вариант 9).

Скрипт повторяет конвейер ноутбука — синтез сцены, SIFT, BFMatcher, фильтр Лоу
и RANSAC, — поэтому все числа на слайдах получены измерением, а не переписаны.
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
POSTER_PATH = ROOT.parent / "lab2" / "image.png"
ASSETS = ROOT / ".presentation_assets"
ASSETS.mkdir(exist_ok=True)
OUTPUT = ROOT / "Лабораторная_5_вариант_9.pptx"

RATIO = 0.75
LOGO_X, LOGO_Y, LOGO_W, LOGO_H = 95, 70, 260, 450

poster = cv2.imread(str(POSTER_PATH), cv2.IMREAD_COLOR)
logo = poster[LOGO_Y:LOGO_Y + LOGO_H, LOGO_X:LOGO_X + LOGO_W].copy()


def to_rgb(bgr):
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def make_banner_scene(poster, angle_deg=30.0, out_size=(1280, 860), seed=0):
    """Синтезирует фотографию баннера и точную гомографию постер → сцена."""
    rng = np.random.default_rng(seed)
    W, H = out_size

    wall = cv2.resize(poster, (W, H), interpolation=cv2.INTER_AREA)
    wall = cv2.GaussianBlur(wall, (0, 0), 25)
    wall = cv2.cvtColor(cv2.cvtColor(wall, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    wall = (wall * 0.45 + 30).astype(np.uint8)
    yy, xx = np.mgrid[0:H, 0:W]
    light = (0.65 + 0.5 * (1 - xx / W) * (1 - 0.4 * yy / H))[:, :, None]
    wall = np.clip(wall * light, 0, 255).astype(np.uint8)

    ph, pw = poster.shape[:2]
    bw = W * 0.62
    bh = bw * ph / pw
    cx, cy = W * 0.5, H * 0.5
    D = bw * 1.15

    t = np.deg2rad(angle_deg)
    corners_plane = np.float32([[-bw / 2, -bh / 2], [bw / 2, -bh / 2],
                                [bw / 2, bh / 2], [-bw / 2, bh / 2]])
    dst = np.zeros_like(corners_plane)
    for i, (X, Y) in enumerate(corners_plane):
        x_rot, z_rot = X * np.cos(t), -X * np.sin(t)
        dst[i, 0] = cx + D * x_rot / (D + z_rot)
        dst[i, 1] = cy + D * Y / (D + z_rot)

    src_pts = np.float32([[0, 0], [pw, 0], [pw, ph], [0, ph]])
    homography = cv2.getPerspectiveTransform(src_pts, dst)

    warped = cv2.warpPerspective(poster, homography, (W, H))
    mask = cv2.warpPerspective(np.full((ph, pw), 255, np.uint8), homography, (W, H))
    frame = cv2.dilate(mask, np.ones((17, 17), np.uint8))

    scene = wall.copy()
    scene[frame > 0] = (35, 35, 40)
    scene[mask > 0] = warped[mask > 0]
    scene = np.clip(scene.astype(np.float32) * 0.9 + 12, 0, 255).astype(np.uint8)
    scene = cv2.GaussianBlur(scene, (3, 3), 0.8)
    scene = np.clip(scene.astype(np.float32) + rng.normal(0, 4, scene.shape),
                    0, 255).astype(np.uint8)
    return scene, homography


banner, poster_to_scene = make_banner_scene(poster, angle_deg=30.0)
logo_to_poster = np.float32([[1, 0, LOGO_X], [0, 1, LOGO_Y], [0, 0, 1]])
H_true = poster_to_scene @ logo_to_poster

LOGO_CORNERS = np.float32([[0, 0], [LOGO_W, 0], [LOGO_W, LOGO_H], [0, LOGO_H]]).reshape(-1, 1, 2)
TRUE_CORNERS = cv2.perspectiveTransform(LOGO_CORNERS, H_true)

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


def evaluate(matches, gt=H_true, keypoints=None):
    """Доля верных совпадений и точность локализации по оценённой гомографии."""
    keypoints = keypoints or kp_banner
    if len(matches) < 4:
        return dict(n=len(matches), correct=0, precision=float("nan"),
                    inliers=0, error=float("nan"), homography=None)
    src = np.float32([kp_logo[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst = np.float32([keypoints[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    errors = np.linalg.norm(cv2.perspectiveTransform(src, gt) - dst, axis=2).ravel()
    correct = int((errors < 5.0).sum())
    estimated, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
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
corner_errors = np.linalg.norm(est_corners - TRUE_CORNERS, axis=2).ravel()
logo_diagonal = float(np.linalg.norm(est_corners[0] - est_corners[2]))

located = banner.copy()
cv2.polylines(located, [np.int32(est_corners)], True, (0, 255, 0), 4, cv2.LINE_AA)
cv2.polylines(located, [np.int32(TRUE_CORNERS)], True, (255, 255, 255), 2, cv2.LINE_AA)

matches_img = cv2.drawMatches(logo, kp_logo, banner, kp_banner, good_matches[:40], None,
                              matchColor=(0, 200, 0), singlePointColor=(190, 190, 190),
                              flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
# drawMatches дополняет более низкое изображение чёрным — заменяем на цвет карточки
matches_img[LOGO_H:, :LOGO_W] = (252, 246, 243)
logo_kp_img = cv2.drawKeypoints(logo, kp_logo, None,
                                flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
banner_kp_img = cv2.drawKeypoints(banner, kp_banner, None,
                                  flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# --- 4. разный угол обзора -------------------------------------------------
ANGLES = [0, 15, 30, 45, 55, 65, 75]
angle_rows, angle_previews = [], []
for angle in ANGLES:
    scene, scene_homography = make_banner_scene(poster, angle_deg=angle, seed=angle)
    kp_scene, des_scene = sift.detectAndCompute(cv2.cvtColor(scene, cv2.COLOR_BGR2GRAY), None)
    gt = scene_homography @ logo_to_poster
    good = [m for m, n in bf.knnMatch(des_logo, des_scene, k=2)
            if m.distance < RATIO * n.distance]
    row = evaluate(good, gt=gt, keypoints=kp_scene)
    true_corners = cv2.perspectiveTransform(LOGO_CORNERS, gt).reshape(-1, 2)
    row.update(angle=angle, logo_px=float(np.linalg.norm(true_corners[1] - true_corners[0])))
    angle_rows.append(row)

    preview = scene.copy()
    if row["homography"] is not None:
        cv2.polylines(preview, [np.int32(cv2.perspectiveTransform(LOGO_CORNERS,
                                                                  row["homography"]))],
                      True, (0, 220, 0), 5, cv2.LINE_AA)
    cv2.polylines(preview, [np.int32(true_corners.reshape(-1, 1, 2))], True,
                  (255, 255, 255), 2, cv2.LINE_AA)
    angle_previews.append((to_rgb(preview), f"{angle}° · инлайеров {row['inliers']}"))

# --- устойчивость ----------------------------------------------------------
base_scene, base_homography = make_banner_scene(poster, angle_deg=20.0, seed=7)
base_gt = base_homography @ logo_to_poster


def match_against(scene_img, gt, label):
    kp_scene, des_scene = sift.detectAndCompute(cv2.cvtColor(scene_img, cv2.COLOR_BGR2GRAY), None)
    good = [m for m, n in bf.knnMatch(des_logo, des_scene, k=2)
            if m.distance < RATIO * n.distance]
    row = evaluate(good, gt=gt, keypoints=kp_scene)
    row["label"] = label
    return row


trials = [match_against(base_scene, base_gt, "исходная сцена, наклон 20°")]
for factor in (0.6, 0.4, 0.25):
    small = cv2.resize(base_scene, None, fx=factor, fy=factor, interpolation=cv2.INTER_AREA)
    scale = np.float32([[factor, 0, 0], [0, factor, 0], [0, 0, 1]])
    trials.append(match_against(small, scale @ base_gt, f"масштаб сцены ×{factor}"))
for alpha, beta, name in ((0.5, 0, "яркость ×0.5"), (1.6, 30, "яркость ×1.6 +30"),
                          (1.0, -70, "яркость −70")):
    trials.append(match_against(cv2.convertScaleAbs(base_scene, alpha=alpha, beta=beta),
                                base_gt, name))

# --- сравнение методов -----------------------------------------------------
def timed(matcher_call, repeats=3):
    start = time.perf_counter()
    for _ in range(repeats):
        result = matcher_call()
    return result, (time.perf_counter() - start) / repeats * 1000


_, bf_ms = timed(lambda: bf.knnMatch(des_logo, des_banner, k=2))
flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
flann_knn, flann_ms = timed(lambda: flann.knnMatch(des_logo, des_banner, k=2))
flann_good = [m for m, n in flann_knn if m.distance < RATIO * n.distance]
flann_result = evaluate(flann_good)

orb = cv2.ORB_create(nfeatures=2000)
kp_logo_orb, des_logo_orb = orb.detectAndCompute(logo_gray, None)
kp_banner_orb, des_banner_orb = orb.detectAndCompute(banner_gray, None)
bf_hamming = cv2.BFMatcher(cv2.NORM_HAMMING)
orb_knn, orb_ms = timed(lambda: bf_hamming.knnMatch(des_logo_orb, des_banner_orb, k=2))
orb_good = [m for m, n in orb_knn if m.distance < RATIO * n.distance]

# --- иллюстрации -----------------------------------------------------------
hero = F.rounded_png(ASSETS / "hero.png", to_rgb(located), placement_w=5.08)
data_logo = F.rounded_png(ASSETS / "logo.png", to_rgb(logo), placement_w=2.1)
data_banner = F.rounded_png(ASSETS / "banner.png", to_rgb(banner), placement_w=8.6)
logo_kp = F.rounded_png(ASSETS / "logo_kp.png", to_rgb(logo_kp_img), placement_w=2.4)
banner_kp = F.rounded_png(ASSETS / "banner_kp.png", to_rgb(banner_kp_img), placement_w=8.05)
matches_png = F.rounded_png(ASSETS / "matches.png", to_rgb(matches_img), placement_w=11.53)
located_png = F.rounded_png(ASSETS / "located.png", to_rgb(located), placement_w=8.05)
angle_gallery = F.panels(ASSETS / "angles.png", angle_previews[:4], width=11.53)


def draw_ratio_hist(ax):
    ax.hist(ratios, bins=50, color=T.SERIES[0])
    ax.axvline(RATIO, color=T.SERIES[1], linestyle="--", linewidth=1.8)
    ax.annotate(f"ρ = {RATIO}", xy=(RATIO, 0), xytext=(RATIO - 0.235, ax.get_ylim()[1] * 0.82),
                color=T.SERIES[1], fontsize=10, weight="bold")
    ax.set_xlabel("Отношение d₁ / d₂")
    ax.set_ylabel("Совпадений")
    ax.set_xlim(0, 1.02)


ratio_hist = F.chart(ASSETS / "ratio_hist.png", 7.35, 3.55, draw_ratio_hist,
                     pad=(0.72, 0.52, 0.22, 0.30))


def draw_ratio_sweep(path, width=11.53, height=3.15):
    fig = F.figure(width, height)
    panels_spec = [
        ("Совпадений после фильтра", [r["n"] for r in ratio_rows],
         [r["correct"] for r in ratio_rows]),
        ("Точность, %", [r["precision"] for r in ratio_rows], None),
        ("Ошибка локализации, пикс.", [r["error"] for r in ratio_rows], None),
    ]
    for column, (title, primary, secondary) in enumerate(panels_spec):
        ax = F.axes(fig, [0.055 + column * 0.324, 0.20, 0.256, 0.66])
        ax.plot(RATIO_GRID, primary, "o-", color=T.SERIES[0], markersize=4,
                label="всего" if secondary else None)
        if secondary:
            ax.plot(RATIO_GRID, secondary, "s-", color=T.SERIES[1], markersize=4,
                    label="из них верных")
            ax.legend(loc="upper left")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("коэффициент Лоу ρ")
    return F.finish(fig, path)


ratio_sweep = draw_ratio_sweep(ASSETS / "ratio_sweep.png")


def draw_angle_sweep(path, width=11.53, height=3.15):
    fig = F.figure(width, height)
    angles = [r["angle"] for r in angle_rows]
    panels_spec = [
        ("Число совпадений", [r["n"] for r in angle_rows], [r["inliers"] for r in angle_rows]),
        ("Доля верных совпадений, %",
         [r["correct"] / r["n"] * 100 if r["n"] else 0 for r in angle_rows], None),
        ("Ошибка локализации, пикс.", [r["error"] for r in angle_rows], None),
    ]
    for column, (title, primary, secondary) in enumerate(panels_spec):
        ax = F.axes(fig, [0.055 + column * 0.324, 0.20, 0.256, 0.66])
        ax.plot(angles, primary, "o-", color=T.SERIES[0], markersize=4,
                label="после фильтра Лоу" if secondary else None)
        if secondary:
            ax.plot(angles, secondary, "s-", color=T.SERIES[1], markersize=4,
                    label="инлайеров RANSAC")
            ax.legend(loc="upper right")
        if column == 2:
            ax.set_yscale("symlog", linthresh=10)
        ax.axvspan(50, 78, color=T.SERIES[1], alpha=0.08)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("угол наклона баннера, °")
    return F.finish(fig, path)


angle_sweep = draw_angle_sweep(ASSETS / "angle_sweep.png")

# --- слайды ----------------------------------------------------------------
deck = Deck("Лабораторная работа 5 · вариант 9")

deck.cover(
    kicker="Лабораторная работа 5 · вариант 9",
    title="Поиск логотипа\nв реальной сцене",
    subtitle="SIFT, дескрипторы, BFMatcher с фильтрацией по Лоу\nи локализация объекта методом RANSAC",
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

slide = deck.slide("Данные", "Эталон и сцена, в которой его нужно найти", kicker="Постановка")
deck.fit_picture(slide, data_logo, T.MARGIN, 2.15, 1.7, 2.90, center=False)
bottom = deck.fit_picture(slide, data_banner, 3.00, 2.15, 9.43, 2.90).bottom_in
deck.note(slide,
          f"Логотип {LOGO_W} × {LOGO_H} вырезан из кадра с позиции ({LOGO_X}, {LOGO_Y}). "
          f"Сцена {banner.shape[1]} × {banner.shape[0]} синтезирована: наклон 30°, "
          "проекция камерой-обскурой, изменение яркости, расфокусировка и гауссов шум. "
          "Синтез вместо съёмки выбран ради точной опорной гомографии — качество "
          "сопоставления оценивается численно, а не «на глаз».",
          x=T.MARGIN, y=bottom + 0.32, w=11.53, bar=True, size=T.BODY_SM)

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
deck.picture(slide, logo_kp, T.MARGIN, 2.15, w=2.4)
deck.metrics(slide, [
    (f"{len(kp_logo)}", "ключевых точек"),
    (f"{des_logo.shape}", "матрица дескрипторов, float32"),
    (f"{logo_sizes.min():.1f} … {logo_sizes.max():.1f}", "размер окрестности, пикселей"),
    (f"{len(kp_logo) / (LOGO_W * LOGO_H) * 1e4:.1f}", "точек на 10 000 пикселей"),
], x=4.05, y=2.15, w=8.38, h=3.9, vertical=True)
deck.caption(slide, "На однородном жёлтом фоне точек практически нет: такую область "
                    "невозможно отличить от соседних.",
             x=T.MARGIN, y=6.34, w=11.53)

slide = deck.slide("Ключевые точки сцены", "Кроме логотипа в кадре есть текст, рамка "
                                           "и границы стены", kicker="Пункт 02")
deck.fit_picture(slide, banner_kp, T.MARGIN, 2.15, 8.05, 3.9)
deck.metrics(slide, [
    (f"{len(kp_banner)}", "ключевых точек в сцене"),
    (f"{len(kp_logo)} × {len(kp_banner)}", "сравнений при полном переборе"),
    (f"{len(kp_logo) * len(kp_banner) / 1e6:.2f} млн", "операций над 128-мерными векторами"),
], x=9.15, y=2.15, w=3.28, h=3.55, vertical=True)

slide = deck.slide("BFMatcher и фильтр Лоу", f"ρ = {RATIO}: 40 лучших совпадений из "
                                             f"{len(good_matches)}", kicker="Пункт 03")
bottom = deck.fit_picture(slide, matches_png, T.MARGIN, 2.12, 11.53, 3.30).bottom_in
deck.metrics(slide, [
    (f"{len(good_matches)} из {len(knn_matches)}", "прошли фильтр Лоу"),
    (f"{len(good_matches) / len(knn_matches) * 100:.1f} %", "доля прошедших"),
    (f"{main['precision']:.1f} %", "из них верных"),
], x=T.MARGIN, y=bottom + 0.30, w=11.53, h=1.02)

slide = deck.slide("Почему фильтр работает", "Отношение d₁ / d₂ разделяет верные "
                                             "и случайные совпадения", kicker="Пункт 03")
deck.picture(slide, ratio_hist, T.MARGIN, 2.15, w=7.35)
deck.note(slide,
          "Высокий правый горб — совпадения разных объектов: ближайший и второй сосед "
          "почти равноудалены. Длинный левый хвост — верные совпадения, у которых "
          "ближайший сосед заметно ближе. Порог рассекает распределение между "
          "этими группами.",
          x=8.62, y=2.15, w=3.81, bar=True, size=T.BODY_SM)
deck.table(slide, ["Режим фильтрации", "Совпадений", "Верных", "Точность", "Ошибка углов"],
           [[label, str(r["n"]), str(r["correct"]), f"{r['precision']:.1f} %",
             f"{r['error']:.2f} пикс."] for label, r in filter_modes],
           x=T.MARGIN, y=5.28, w=11.53, widths=[0.28, 0.17, 0.15, 0.17, 0.23],
           row_h=0.38, highlight=2)

slide = deck.slide("Подбор коэффициента ρ", "Компромисс между количеством и чистотой",
                   kicker="Пункт 03")
bottom = deck.picture(slide, ratio_sweep, T.MARGIN, 2.12, w=11.53).bottom_in
deck.note(slide,
          f"При ρ = 0.5 остаются только безошибочные пары ({ratio_rows[0]['precision']:.0f} % "
          f"точности), но их всего {ratio_rows[0]['n']} — они хуже покрывают площадь "
          "объекта, поэтому ошибка локализации ведёт себя немонотонно и при ρ = 0.7 "
          "оказывается меньше, чем при ρ = 0.5.",
          x=T.MARGIN, y=bottom + 0.28, w=11.53, bar=True, size=T.BODY_SM)

slide = deck.slide("Логотип найден", "Зелёный контур — оценка, белый — истина",
                   kicker="Локализация")
deck.fit_picture(slide, located_png, T.MARGIN, 2.15, 8.05, 2.75)
deck.metrics(slide, [
    (f"{main['inliers']} из {main['n']}", f"инлайеров RANSAC ({main['inliers'] / main['n'] * 100:.1f} %)"),
    (f"{corner_errors.mean():.2f} пикс.", "средняя ошибка положения углов"),
    (f"{logo_diagonal:.0f} пикс.", "диагональ логотипа в сцене"),
], x=9.15, y=2.15, w=3.28, h=3.55, vertical=True)
deck.caption(slide, "Ошибки четырёх углов: "
                    + ", ".join(f"{value:.2f}" for value in corner_errors)
                    + " пикселя — оценённая гомография практически совпала с эталонной.",
             x=T.MARGIN, y=5.06, w=11.53)

slide = deck.slide("Разный угол обзора", "Наклон баннера от 0° до 75°", kicker="Пункт 04")
bottom = deck.picture(slide, angle_sweep, T.MARGIN, 2.12, w=11.53).bottom_in
deck.note(slide,
          "До 45° логотип находится надёжно. Начиная с 55° верных совпадений почти "
          "не остаётся, RANSAC собирает случайную модель, и ошибка положения углов "
          "подскакивает на два порядка — эта область закрашена на графиках.",
          x=T.MARGIN, y=bottom + 0.28, w=11.53, bar=True, size=T.BODY_SM)

slide = deck.slide("Граница применимости", "Контур вырождается там, где кончаются "
                                           "верные совпадения", kicker="Пункт 04")
bottom = deck.fit_picture(slide, angle_gallery, T.MARGIN, 2.08, 11.53, 1.80).bottom_in
deck.table(slide, ["Угол", "Ширина лого", "После Лоу", "Верных", "Инлайеров", "Ошибка углов"],
           [[f"{r['angle']}°", f"{r['logo_px']:.0f} пикс.", str(r["n"]), str(r["correct"]),
             str(r["inliers"]),
             f"{r['error']:.2f} пикс." if np.isfinite(r["error"]) else "не найден"]
            for r in angle_rows],
           x=T.MARGIN, y=bottom + 0.34, w=11.53,
           widths=[0.11, 0.19, 0.16, 0.14, 0.17, 0.23], row_h=0.32, size=11.5)

slide = deck.slide("Устойчивость к съёмочным условиям",
                   "Масштаб и освещение метод переносит почти без потерь",
                   kicker="Дополнительно")
deck.table(slide, ["Условие", "После фильтра Лоу", "Инлайеров", "Ошибка углов"],
           [[r["label"], str(r["n"]), str(r["inliers"]),
             f"{r['error']:.2f} пикс." if np.isfinite(r["error"]) else "не найден"]
            for r in trials],
           x=T.MARGIN, y=2.25, w=7.4, widths=[0.40, 0.22, 0.20, 0.18], row_h=0.46)
deck.note(slide,
          "Дескриптор строится из направлений градиента и нормируется, поэтому "
          "равномерное изменение яркости из него уходит. Уменьшение сцены в четыре раза "
          "сокращает число инлайеров, но ошибка локализации остаётся в пределах "
          "пары пикселей.",
          x=8.62, y=2.20, w=3.81, bar=True, size=T.BODY_SM)

slide = deck.slide("Сравнение методов", "На этой паре изображений полный перебор выигрывает",
                   kicker="Дополнительно")
deck.table(slide, ["Метод", "Дескрипторов", "После Лоу", "Инлайеров", "Ошибка", "Время"], [
    ["SIFT + BFMatcher", str(len(kp_logo)), str(main["n"]), str(main["inliers"]),
     f"{main['error']:.2f} пикс.", f"{bf_ms:.1f} мс"],
    ["SIFT + FLANN", str(len(kp_logo)), str(flann_result["n"]), str(flann_result["inliers"]),
     f"{flann_result['error']:.2f} пикс.", f"{flann_ms:.1f} мс"],
    ["ORB + BFMatcher", str(len(kp_logo_orb)), str(len(orb_good)), "—",
     "гомография не построена" if len(orb_good) < 4 else "—", f"{orb_ms:.1f} мс"],
], x=T.MARGIN, y=2.22, w=11.53, widths=[0.22, 0.15, 0.13, 0.13, 0.22, 0.15], row_h=0.48)
deck.note(slide,
          f"FLANN дал тот же набор пар, но на {len(kp_logo)} × {len(kp_banner)} "
          "дескрипторах построение kd-индекса не окупается. ORB непригоден: "
          "BRIEF слишком чувствителен к перспективе и размытию.",
          x=T.MARGIN, y=4.30, w=11.53, bar=True, size=T.BODY)

slide = deck.slide("Вывод о применимости метода", "Пункт 5 задания", kicker="Выводы")
bottom = deck.columns(slide, [
    ("Что метод выдерживает",
     "Перспективу до 45°, уменьшение сцены вчетверо, затемнение и осветление, поворот "
     "кадра на любой угол. Ошибка локализации при этом остаётся в пределах пары "
     "пикселей при размере объекта в сотни пикселей."),
    ("Где метод ломается",
     "С наклона 55°: SIFT инвариантен к повороту и масштабу, но не к аффинному скосу — "
     "окрестность точки сжимается вдоль одного направления, и дескриптор перестаёт "
     "совпадать. Нужны ASIFT или обучаемые дескрипторы."),
    ("Практический вывод",
     "Для контроля размещения рекламы метод пригоден без обучения: он работает по "
     "одному эталону и выдаёт не только факт наличия объекта, но и его границы. "
     "Условия: плоский текстурный логотип и ракурс до 45°."),
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
    f"SIFT нашёл {len(kp_logo)} точек на логотипе и {len(kp_banner)} в сцене",
    f"BFMatcher с фильтром Лоу при ρ = {RATIO} дал {main['n']} совпадений",
    f"RANSAC отобрал {main['inliers']} инлайеров и построил гомографию",
    f"Ошибка локализации углов — {corner_errors.mean():.2f} пикселя при диагонали "
    f"логотипа {logo_diagonal:.0f} пикселей",
    "Эксперимент с наклоном от 0° до 75° определил границу применимости — около 45°",
    "Измерена устойчивость к масштабу, освещению и повороту кадра",
], x=T.MARGIN, y=2.20, w=11.53, size=16.5)
deck.text(slide,
          "Связка SIFT → BFMatcher → фильтр Лоу → RANSAC решает задачу поиска логотипа "
          "без обучения, по единственному эталонному изображению, но с чёткой границей "
          "по ракурсу съёмки.",
          x=T.MARGIN, y=bottom + 0.42, w=11.53, h=0.8,
          size=T.BODY, color=T.PRIMARY, bold=True)

path, count = deck.save(OUTPUT)
print(f"saved: {path} ({count} слайдов)")
