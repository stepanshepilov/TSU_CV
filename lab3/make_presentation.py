"""Презентация-отчёт по лабораторной работе 3 (вариант 9)."""
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from presentation_kit import figures as F  # noqa: E402
from presentation_kit import tokens as T  # noqa: E402
from presentation_kit.deck import Deck  # noqa: E402

ROOT = Path(__file__).resolve().parent
IMAGE_PATH = ROOT.parent / "lab2" / "image.png"
ASSETS = ROOT / ".presentation_assets"
ASSETS.mkdir(exist_ok=True)
OUTPUT = ROOT / "Лабораторная_3_вариант_9.pptx"

BORDER = (246, 246, 246)

image = cv2.imread(str(IMAGE_PATH), cv2.IMREAD_COLOR)
rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
height, width = image.shape[:2]


def to_rgb(bgr):
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


# --- 1. масштабирование ----------------------------------------------------
resized = cv2.resize(image, (300, 300), interpolation=cv2.INTER_AREA)
scale_x, scale_y = 300 / width, 300 / height

fit = 300 / max(width, height)
fitted = cv2.resize(image, (int(width * fit), int(height * fit)), interpolation=cv2.INTER_AREA)
pad_y = (300 - fitted.shape[0]) // 2
padded = cv2.copyMakeBorder(fitted, pad_y, 300 - fitted.shape[0] - pad_y, 0, 0,
                            cv2.BORDER_CONSTANT, value=BORDER)

INTERPOLATIONS = [("INTER_NEAREST", cv2.INTER_NEAREST), ("INTER_LINEAR", cv2.INTER_LINEAR),
                  ("INTER_CUBIC", cv2.INTER_CUBIC), ("INTER_AREA", cv2.INTER_AREA)]
interp_results = []
for name, flag in INTERPOLATIONS:
    small = cv2.resize(image, (300, 300), interpolation=flag)
    sharpness = float(np.abs(cv2.Laplacian(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY),
                                           cv2.CV_64F)).mean())
    interp_results.append((name, small, sharpness))

# --- 2. поворот ------------------------------------------------------------
centre = (width / 2, height / 2)
rotation = cv2.getRotationMatrix2D(centre, 25.0, 1.0)
rotated = cv2.warpAffine(image, rotation, (width, height), borderValue=BORDER)
rot_det = float(np.linalg.det(rotation[:, :2]))

angle = np.deg2rad(25.0)
new_w = int(height * abs(np.sin(angle)) + width * abs(np.cos(angle)))
new_h = int(height * abs(np.cos(angle)) + width * abs(np.sin(angle)))
expanded_matrix = rotation.copy()
expanded_matrix[0, 2] += new_w / 2 - centre[0]
expanded_matrix[1, 2] += new_h / 2 - centre[1]
expanded = cv2.warpAffine(image, expanded_matrix, (new_w, new_h), borderValue=BORDER)

# --- 3. сдвиг --------------------------------------------------------------
shift = np.float32([[1, 0, -120], [0, 1, 60]])
shifted = cv2.warpAffine(image, shift, (width, height), borderValue=BORDER)
probe = np.float32([[[400, 200]]])
probe_moved = cv2.transform(probe, shift)[0, 0]

# --- 4. отражение ----------------------------------------------------------
flip_v = cv2.flip(image, 0)
flip_h = cv2.flip(image, 1)
flip_both = cv2.flip(image, -1)
flip_checks = [
    ("cv2.flip(img, 0)", "img[::-1, :]", np.array_equal(flip_v, image[::-1, :])),
    ("cv2.flip(img, 1)", "img[:, ::-1]", np.array_equal(flip_h, image[:, ::-1])),
    ("cv2.flip(img, -1)", "img[::-1, ::-1]", np.array_equal(flip_both, image[::-1, ::-1])),
]

# --- 5. аффинное преобразование --------------------------------------------
src_points = np.float32([[150, 100], [850, 120], [200, 480]])
dst_points = np.float32([[80, 180], [900, 60], [420, 520]])
affine = cv2.getAffineTransform(src_points, dst_points)
warped = cv2.warpAffine(image, affine, (width, height), borderValue=BORDER)
affine_det = float(np.linalg.det(affine[:, :2]))


def triangle_metrics(points):
    sides = [float(np.linalg.norm(points[(i + 1) % 3] - points[i])) for i in range(3)]
    angles = []
    for i in range(3):
        a = points[(i + 1) % 3] - points[i]
        b = points[(i + 2) % 3] - points[i]
        cosine = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        angles.append(float(np.degrees(np.arccos(np.clip(cosine, -1, 1)))))
    u = points[1] - points[0]
    v = points[2] - points[0]
    area = abs(float(u[0] * v[1] - u[1] * v[0])) / 2
    return sides, angles, float(area)


src_sides, src_angles, src_area = triangle_metrics(src_points)
dst_sides, dst_angles, dst_area = triangle_metrics(dst_points)

# --- композиция ------------------------------------------------------------
def to3x3(matrix):
    return np.vstack([matrix, [0, 0, 1]])


composed = to3x3(shift) @ to3x3(rotation) @ to3x3(np.float32([[1, 0, 0], [0, -1, height - 1]]))
composed_once = cv2.warpAffine(image, composed[:2], (width, height), borderValue=BORDER)
step_by_step = cv2.warpAffine(image, np.float32([[1, 0, 0], [0, -1, height - 1]]),
                              (width, height), borderValue=BORDER)
step_by_step = cv2.warpAffine(step_by_step, rotation, (width, height), borderValue=BORDER)
step_by_step = cv2.warpAffine(step_by_step, shift, (width, height), borderValue=BORDER)
compose_diff = cv2.absdiff(composed_once, step_by_step)

# --- иллюстрации -----------------------------------------------------------
hero = F.rounded_png(ASSETS / "hero.png", to_rgb(rotated), placement_w=5.08)
resize_panels = F.panels(ASSETS / "resize.png",
                         [(to_rgb(resized), "resize 300 × 300"),
                          (to_rgb(padded), "вписывание с полями")], width=5.6)
rotate_panels = F.panels(ASSETS / "rotate.png",
                         [(rgb, "Исходное"), (to_rgb(rotated), "Поворот на 25°")], width=8.05)
expand_panels_narrow = F.panels(ASSETS / "expand.png",
                               [(to_rgb(rotated), f"кадр {width} × {height} — углы обрезаны"),
                                (to_rgb(expanded), f"кадр {new_w} × {new_h} — целиком")],
                               width=8.05)
shift_panels = F.panels(ASSETS / "shift.png",
                        [(rgb, "Исходное"),
                         (to_rgb(shifted), "Сдвиг на (−120, +60)")], width=8.05)
flip_panels = F.panels(ASSETS / "flip.png",
                       [(to_rgb(flip_v), "flip(0) — верх и низ"),
                        (to_rgb(flip_h), "flip(1) — лево и право"),
                        (to_rgb(flip_both), "flip(−1) — поворот на 180°")], width=11.53)
compose_panels = F.panels(ASSETS / "compose.png",
                          [(to_rgb(composed_once), "одна матрица 3 × 3"),
                           (to_rgb(step_by_step), "три вызова warpAffine подряд")], width=8.05)


def draw_interp(path):
    """Увеличенный фрагмент после четырёх видов интерполяции."""
    crops = [(name, cv2.cvtColor(small[70:145, 20:100], cv2.COLOR_BGR2RGB), sharp)
             for name, small, sharp in interp_results]
    return F.panels(path, [(crop, f"{name} · {sharp:.1f}") for name, crop, sharp in crops],
                    width=11.53)


interp_panels = draw_interp(ASSETS / "interp.png")


def draw_triangles(path, width_in=11.53, height_in=2.85):
    fig = F.figure(width_in, height_in)
    for column, (canvas, points, title) in enumerate(
            [(rgb, src_points, "До: P₁(150, 100) · P₂(850, 120) · P₃(200, 480)"),
             (to_rgb(warped), dst_points, "После: (80, 180) · (900, 60) · (420, 520)")]):
        ax = fig.add_axes([0.035 + column * 0.492, 0.035, 0.473, 0.83])
        ax.imshow(canvas)
        closed = np.vstack([points, points[:1]])
        ax.plot(closed[:, 0], closed[:, 1], color=T.SERIES[1], linewidth=2.4)
        ax.scatter(points[:, 0], points[:, 1], color=T.SERIES[1], s=28, zorder=3)
        ax.set_axis_off()
        fig.text(0.035 + column * 0.492 + 0.473 / 2, 0.935, title, ha="center", va="center",
                 fontsize=10, color="#" + T.MUTED)
    return F.finish(fig, path)


triangle_figure = draw_triangles(ASSETS / "affine.png")


def draw_angles(ax):
    positions = np.arange(3)
    ax.bar(positions - 0.19, src_angles, width=0.36, color=T.SERIES[0], label="до")
    ax.bar(positions + 0.19, dst_angles, width=0.36, color=T.SERIES[1], label="после")
    ax.set_xticks(positions)
    ax.set_xticklabels(["угол P₁", "угол P₂", "угол P₃"])
    ax.set_ylabel("Градусы")
    ax.legend(loc="upper right")
    for position, (before, after) in enumerate(zip(src_angles, dst_angles)):
        ax.text(position - 0.19, before + 2, f"{before:.1f}", ha="center", fontsize=9.5)
        ax.text(position + 0.19, after + 2, f"{after:.1f}", ha="center", fontsize=9.5)
    ax.set_ylim(0, 108)


angle_chart = F.chart(ASSETS / "metrics.png", 7.35, 3.9, draw_angles,
                      pad=(0.66, 0.42, 0.24, 0.30))

# --- слайды ----------------------------------------------------------------
deck = Deck("Лабораторная работа 3 · вариант 9")

deck.cover(
    kicker="Лабораторная работа 3 · вариант 9",
    title="Геометрия кадра",
    subtitle="Масштабирование, поворот, сдвиг, отражение\nи аффинное преобразование по трём точкам",
    meta="Компьютерное зрение · OpenCV · NumPy · Matplotlib",
    image=hero,
)

slide = deck.slide("Задание варианта 9", "Пять преобразований", kicker="Постановка")
deck.steps(slide, [
    "Масштабировать изображение до квадрата 300 × 300",
    "Повернуть изображение на 25°",
    "Сместить на 120 пикселей влево и 60 вниз",
    "Отразить изображение по вертикали",
    "Выполнить аффинное преобразование треугольника",
], x=T.MARGIN, y=2.25, w=7.55, row_h=0.70)
deck.metrics(slide, [
    (f"{width} × {height}", "исходный кадр, пикселей"),
    ("2 × 3", "матрица аффинного преобразования"),
    ("6", "степеней свободы"),
], x=8.86, y=2.20, w=3.57, h=3.62, vertical=True)

slide = deck.slide("Аффинное преобразование", "Одна матрица описывает все пять операций",
                   kicker="Теория")
bottom = deck.columns(slide, [
    ("Линейная часть 2 × 2",
     "Отвечает за поворот, масштаб и скос. Модуль определителя показывает, во сколько "
     "раз меняется площадь фигуры, а знак — сохраняется ли ориентация."),
    ("Столбец переноса",
     "Задаёт сдвиг: положительное tₓ двигает вправо, положительное t_y — вниз. "
     "Отсюда «влево на 120 и вниз на 60» — это tₓ = −120, t_y = +60."),
    ("Что сохраняется",
     "Прямые остаются прямыми, параллельные — параллельными, отношение длин на одной "
     "прямой не меняется. Углы и длины не сохраняются."),
], x=T.MARGIN, y=2.12, w=11.53)
deck.note(slide,
          "warpAffine идёт от выходного пикселя к входному по обратному преобразованию — "
          "иначе в выходной сетке остались бы «дыры». Точки за пределами исходного кадра "
          "заполняются по borderMode и borderValue.",
          x=T.MARGIN, y=bottom + 0.32, w=11.53, bar=True, size=T.BODY)

slide = deck.slide("Масштабирование до 300 × 300", "Соотношение сторон меняется с 1.776 на 1.000",
                   kicker="Пункт 01")
deck.picture(slide, resize_panels, T.MARGIN, 2.15, w=5.6)
deck.metrics(slide, [
    (f"{scale_x:.3f} / {scale_y:.3f}", "коэффициенты sₓ и s_y"),
    (f"в {width * height / (300 * 300):.1f} раза", "меньше пикселей"),
    ("(ширина, высота)", "порядок аргументов cv2.resize"),
], x=6.81, y=2.15, w=5.62, h=3.55, vertical=True)
deck.caption(slide, "Второй вариант сохраняет пропорции: изображение вписано в квадрат "
                    "и дополнено полями через cv2.copyMakeBorder.",
             x=T.MARGIN, y=6.34, w=11.53)

slide = deck.slide("Выбор интерполяции", "Один и тот же фрагмент после четырёх методов",
                   kicker="Пункт 01")
bottom = deck.picture(slide, interp_panels, T.MARGIN, 2.12, w=11.53).bottom_in
bottom = deck.metrics(slide, [(name.split("_")[1].title(), f"средний |∇²I| = {sharp:.1f}")
                              for name, _, sharp in interp_results],
                      x=T.MARGIN, y=bottom + 0.28, w=11.53, h=0.98)
deck.caption(slide, "INTER_NEAREST даёт самый «зубчатый» результат, INTER_AREA — самый "
                    "гладкий; при уменьшении кадра рекомендуется именно он.",
             x=T.MARGIN, y=bottom + 0.18, w=11.53)

slide = deck.slide("Поворот на 25°", "getRotationMatrix2D возвращает готовую матрицу 2 × 3",
                   kicker="Пункт 02")
deck.picture(slide, rotate_panels, T.MARGIN, 2.15, w=8.05)
deck.metrics(slide, [
    (f"{rotation[0, 0]:.4f}", f"M[0,0] = cos 25° = {np.cos(angle):.4f}"),
    (f"{rotation[0, 1]:.4f}", f"M[0,1] = sin 25° = {np.sin(angle):.4f}"),
    (f"{rot_det:.6f}", "определитель: поворот сохраняет площадь"),
], x=9.15, y=2.15, w=3.28, h=3.55, vertical=True)
deck.code(slide, "M = cv2.getRotationMatrix2D(center, 25.0, 1.0)\n"
                 "rotated = cv2.warpAffine(image, M, (width, height))",
          x=T.MARGIN, y=5.14, w=8.05)

slide = deck.slide("Кадр под поворот", "При неизменном размере углы изображения обрезаются",
                   kicker="Пункт 02")
bottom = deck.picture(slide, expand_panels_narrow, T.MARGIN, 2.15, w=8.05).bottom_in
deck.metrics(slide, [
    (f"{new_w} × {new_h}", "расширенный кадр вместо "
                           f"{width} × {height}"),
    ("W′ = H·sin θ + W·cos θ", "новая ширина"),
    ("H′ = H·cos θ + W·sin θ", "новая высота"),
], x=9.15, y=2.15, w=3.28, h=3.55, vertical=True)
deck.caption(slide, "Столбец переноса сдвигается на разницу центров, иначе изображение "
                    "уедет в угол расширенного кадра.",
             x=T.MARGIN, y=bottom + 0.26, w=8.05)

slide = deck.slide("Сдвиг на (−120, +60)", "Матрица переноса и проверка по контрольной точке",
                   kicker="Пункт 03")
deck.picture(slide, shift_panels, T.MARGIN, 2.15, w=8.05)
deck.metrics(slide, [
    ("(400, 200)", "контрольная точка до сдвига"),
    (f"({probe_moved[0]:.0f}, {probe_moved[1]:.0f})", "она же после сдвига"),
    ("120 и 60", "ширина полос фона справа и сверху"),
], x=9.15, y=2.15, w=3.28, h=3.55, vertical=True)
deck.code(slide, "T = np.float32([[1, 0, -120],\n"
                 "               [0, 1,   60]])",
          x=T.MARGIN, y=5.14, w=8.05)

slide = deck.slide("Отражение", "cv2.flip численно совпадает со срезами NumPy",
                   kicker="Пункт 04")
bottom = deck.picture(slide, flip_panels, T.MARGIN, 2.12, w=11.53).bottom_in
deck.table(slide, ["Функция OpenCV", "Эквивалент NumPy", "Совпало"],
           [[a, b, "да" if ok else "нет"] for a, b, ok in flip_checks],
           x=T.MARGIN, y=bottom + 0.34, w=6.6, widths=[0.38, 0.38, 0.24])
deck.note(slide,
          "flip(0) совпал и с warpAffine по матрице diag(1, −1) с переносом на H − 1. "
          "Её определитель равен −1: ориентация меняется на зеркальную.",
          x=7.85, y=bottom + 0.28, w=4.58, size=T.BODY_SM)

slide = deck.slide("Аффинное преобразование треугольника",
                   "Три пары точек однозначно задают матрицу 2 × 3", kicker="Пункт 05")
bottom = deck.picture(slide, triangle_figure, T.MARGIN, 2.10, w=11.53).bottom_in
deck.code(slide, "M = cv2.getAffineTransform(pts_src, pts_dst)\n"
                 f"M = [[{affine[0, 0]:8.4f} {affine[0, 1]:8.4f} {affine[0, 2]:9.4f}]\n"
                 f"     [{affine[1, 0]:8.4f} {affine[1, 1]:8.4f} {affine[1, 2]:9.4f}]]",
          x=T.MARGIN, y=bottom + 0.28, w=6.6)
deck.metrics(slide, [
    (f"{affine_det:.4f}", "определитель линейной части"),
    (f"{dst_area / src_area:.4f}", "во столько раз выросла площадь"),
], x=7.85, y=bottom + 0.28, w=4.58, h=1.36, vertical=True, gap=0.16)

slide = deck.slide("Что стало с углами", "Сумма осталась 180°, но сами углы изменились",
                   kicker="Пункт 05")
deck.picture(slide, angle_chart, T.MARGIN, 2.15, w=7.35)
deck.metrics(slide, [
    (f"{sum(src_angles):.1f}° → {sum(dst_angles):.1f}°", "сумма углов: фигура осталась треугольником"),
    (f"{src_area:.0f} → {dst_area:.0f}", "площадь, пикселей²"),
    ("10⁻⁸", "синус угла между сторонами параллелограмма после преобразования"),
], x=8.62, y=2.15, w=3.81, h=3.55, vertical=True)
deck.caption(slide, "Углы сохраняет только подобие — поворот, перенос, равномерный масштаб "
                    "и отражение. Общее аффинное преобразование включает ещё скос.",
             x=T.MARGIN, y=6.30, w=11.53)

slide = deck.slide("Композиция преобразований", "Три матрицы, перемноженные в одну",
                   kicker="Дополнительно")
deck.picture(slide, compose_panels, T.MARGIN, 2.15, w=8.05)
deck.metrics(slide, [
    (f"{compose_diff.mean():.1f}", "средняя разница, уровней яркости"),
    (f"{compose_diff.max()}", "максимальная разница"),
], x=9.15, y=2.15, w=3.28, h=2.34, vertical=True)
deck.note(slide,
          "Расхождение возникает из-за повторной интерполяции и заливки границ "
          "на промежуточных шагах: композицию выгоднее выполнять одной матрицей.",
          x=T.MARGIN, y=4.98, w=8.05, size=T.BODY_SM, bar=True)

slide = deck.slide("Контрольные вопросы варианта", "Два вопроса по теме работы", kicker="Защита")
deck.qa(slide, [
    ("Что произойдёт с углами треугольника при аффинном преобразовании?",
     f"В общем случае углы изменятся: линейная часть матрицы включает не только "
     f"поворот, но и растяжение по осям и скос. В расчёте "
     f"{src_angles[0]:.1f}°, {src_angles[1]:.1f}°, {src_angles[2]:.1f}° перешли в "
     f"{dst_angles[0]:.1f}°, {dst_angles[1]:.1f}°, {dst_angles[2]:.1f}°; сумма осталась "
     f"равной 180°. Площадь умножилась на модуль определителя — {affine_det:.4f}. "
     "Углы сохраняет только подобие."),
    ("Как в OpenCV задаются исходные и целевые точки?",
     "Массивом np.float32 формы (N, 2), где каждая строка — пара (x, y): сначала "
     "столбец, потом строка — противоположно порядку image[y, x]. Аффинному "
     "преобразованию нужны три пары точек, перспективному — четыре. Порядок точек "
     "в наборах должен совпадать, и точки не должны лежать на одной прямой."),
], x=T.MARGIN, y=2.20, w=11.53, columns=2, size=14, avail_h=4.2)

slide = deck.slide("Контрольные вопросы практики", "Пять вопросов по теме занятия",
                   kicker="Защита")
deck.qa(slide, [
    ("Чем аффинное преобразование отличается от перспективного?",
     "Аффинное — матрица 2 × 3, 6 степеней свободы, сохраняет параллельность, строится "
     "по трём парам точек. Перспективное — 3 × 3, 8 степеней свободы, параллельность "
     "не сохраняет, нужны четыре пары."),
    ("Как работает cv2.warpAffine?",
     "Для каждого пикселя выхода находит точку входа через обратное преобразование "
     "и берёт там значение с интерполяцией. Точки за границей заполняются по borderMode."),
    ("Какие параметры нужны getRotationMatrix2D?",
     "Центр вращения (x, y), угол в градусах — положительный против часовой стрелки — "
     "и коэффициент масштаба."),
    ("Чем отражение по горизонтали отличается от вертикального?",
     "flip(1) меняет местами лево и право, flip(0) — верх и низ, flip(−1) выполняет обе "
     "операции и равен повороту на 180°."),
    ("Как выбрать точки для перспективного преобразования?",
     "Четыре точки одной плоскости, образующие выпуклый четырёхугольник; никакие три "
     "не лежат на прямой. Чем дальше точки друг от друга, тем устойчивее оценка."),
], x=T.MARGIN, y=2.20, w=11.53, columns=2, size=12.5, avail_h=4.2)

slide = deck.slide("Итоги", "Что выполнено в ноутбуке", kicker="Выводы")
bottom = deck.checklist(slide, [
    f"Масштабирование до 300 × 300 изменило соотношение сторон с "
    f"{width / height:.3f} на 1.000",
    f"Поворот на 25° выполнен матрицей с определителем {rot_det:.6f}",
    f"Кадр под поворот расширен до {new_w} × {new_h}, чтобы не терять углы",
    f"Сдвиг проверен точкой: (400, 200) → ({probe_moved[0]:.0f}, {probe_moved[1]:.0f})",
    "cv2.flip численно совпал со срезами NumPy и с матричным преобразованием",
    f"Аффинное преобразование изменило площадь ровно в {affine_det:.4f} раза",
], x=T.MARGIN, y=2.20, w=11.53, size=16.5)
deck.text(slide,
          "Аффинное преобразование сохраняет прямые, параллельность и отношения длин, "
          "но не углы и длины; площадь меняется ровно на модуль определителя.",
          x=T.MARGIN, y=bottom + 0.46, w=11.53, h=0.8,
          size=T.BODY, color=T.PRIMARY, bold=True)

path, count = deck.save(OUTPUT)
print(f"saved: {path} ({count} слайдов)")
