"""Презентация-отчёт по лабораторной работе 4 (вариант 9)."""
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
OUTPUT = ROOT / "Лабораторная_4_вариант_9.pptx"

SIGMA = 45

image = cv2.imread(str(IMAGE_PATH), cv2.IMREAD_GRAYSCALE)
height, width = image.shape


def mad(a, b):
    """Среднее абсолютное отклонение двух изображений."""
    return float(np.abs(a.astype(np.float64) - b.astype(np.float64)).mean())


def norm(values):
    return cv2.normalize(values, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


# --- 1. усреднение ---------------------------------------------------------
blur_5 = cv2.blur(image, (5, 5))
blur_11 = cv2.blur(image, (11, 11))
mad_5, mad_11 = mad(blur_5, image), mad(blur_11, image)

# --- 2. импульсный шум и медиана -------------------------------------------
rng = np.random.default_rng(42)
noise_mask = rng.random(image.shape)
noisy = image.copy()
noisy[noise_mask < 0.025] = 0
noisy[noise_mask > 0.975] = 255
median_5 = cv2.medianBlur(noisy, 5)
mad_noisy, mad_median = mad(noisy, image), mad(median_5, image)

# --- 3. Собель -------------------------------------------------------------
sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
gradient = np.sqrt(sobel_x ** 2 + sobel_y ** 2)

# --- 4. повышение резкости -------------------------------------------------
sharpen_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
sharpened = cv2.filter2D(image, -1, sharpen_kernel)
mad_sharpen = mad(sharpened, image)

# --- 5. гауссов ВЧ-фильтр --------------------------------------------------
spectrum = np.fft.fftshift(np.fft.fft2(image))
yy, xx = np.mgrid[0:height, 0:width]
distance_sq = (yy - height / 2) ** 2 + (xx - width / 2) ** 2
gaussian_lpf = np.exp(-distance_sq / (2 * SIGMA ** 2))
gaussian_hpf = 1.0 - gaussian_lpf
highpass = np.abs(np.fft.ifft2(np.fft.ifftshift(spectrum * gaussian_hpf)))
lowpass = np.abs(np.fft.ifft2(np.fft.ifftshift(spectrum * gaussian_lpf)))
log_spectrum = np.log1p(np.abs(spectrum))

# --- иллюстрации -----------------------------------------------------------
hero = F.rounded_png(ASSETS / "hero.png",
                     cv2.cvtColor(norm(gradient), cv2.COLOR_GRAY2RGB), placement_w=5.08)
blur_panels = F.panels(ASSETS / "blur.png",
                       [(image, "Исходное"), (blur_5, "Усреднение 5 × 5"),
                        (blur_11, "Усреднение 11 × 11")], width=11.53)
median_panels = F.panels(ASSETS / "median.png",
                         [(noisy, "Шум «соль и перец», 5 %"),
                          (median_5, "Медианный фильтр 5 × 5")], width=8.05)
sobel_panels = F.panels(ASSETS / "sobel.png",
                        [(norm(np.abs(sobel_x)), "Собель X — вертикальные границы"),
                         (norm(np.abs(sobel_y)), "Собель Y — горизонтальные границы"),
                         (norm(gradient), "Модуль градиента")], width=11.53)
sharpen_panels = F.panels(ASSETS / "sharpen.png",
                          [(image[120:400, 60:560], "Исходное"),
                           (sharpened[120:400, 60:560], "После ядра sharpen")], width=8.05)
spectrum_panels = F.panels(ASSETS / "spectrum.png",
                           [(norm(log_spectrum), "log(1 + |F|) — спектр"),
                            (norm(gaussian_hpf), f"Маска ВЧ, σ = {SIGMA}"),
                            (norm(highpass), "Результат: контуры")], width=11.53)
compare_panels = F.panels(ASSETS / "compare.png",
                          [(norm(lowpass), "НЧ: плавный фон"),
                           (norm(highpass), "ВЧ: детали и границы")], width=5.6)


def draw_mask(ax):
    radius = np.arange(0, 260)
    ax.plot(radius, np.exp(-radius ** 2 / (2 * SIGMA ** 2)), color=T.SERIES[0],
            label="H_НЧ(D)")
    ax.plot(radius, 1 - np.exp(-radius ** 2 / (2 * SIGMA ** 2)), color=T.SERIES[1],
            label="H_ВЧ(D) = 1 − H_НЧ(D)")
    ax.axvline(SIGMA, color=T.SERIES[2], linestyle="--", linewidth=1.4)
    ax.annotate(f"σ = {SIGMA}", xy=(SIGMA, 0.5), xytext=(SIGMA + 10, 0.53),
                color=T.SERIES[2], fontsize=10, weight="bold")
    ax.set_xlabel("Расстояние D до центра спектра")
    ax.set_ylabel("Коэффициент передачи")
    ax.set_xlim(0, 259)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="center right")


mask_chart = F.chart(ASSETS / "mask.png", 5.62, 3.05, draw_mask,
                     pad=(0.70, 0.52, 0.20, 0.26))


def draw_effect(ax):
    labels = ["Усреднение\n5 × 5", "Усреднение\n11 × 11", "Медиана 5 × 5\nна шуме", "Sharpen"]
    values = [mad_5, mad_11, mad_median, mad_sharpen]
    colors = [T.SERIES[0], T.SERIES[0], T.SERIES[1], T.SERIES[1]]
    bars = ax.bar(labels, values, color=colors, width=0.58)
    F.bar_labels(ax, bars, values, fmt="{:.2f}", dy=0.03)
    ax.set_ylabel("Среднее абсолютное отклонение")
    ax.set_ylim(0, max(values) * 1.2)


effect_chart = F.chart(ASSETS / "effect.png", 7.35, 3.55, draw_effect,
                       pad=(0.80, 0.66, 0.22, 0.30))

# --- слайды ----------------------------------------------------------------
deck = Deck("Лабораторная работа 4 · вариант 9")

deck.cover(
    kicker="Лабораторная работа 4 · вариант 9",
    title="Фильтрация\nизображений",
    subtitle="Пространственные фильтры, оператор Собеля\nи частотная фильтрация через БПФ",
    meta="Компьютерное зрение · OpenCV · NumPy · Matplotlib",
    image=hero,
)

slide = deck.slide("Задание варианта 9", "Пять фильтров", kicker="Постановка")
deck.steps(slide, [
    "Применить усредняющий фильтр 5 × 5 и 11 × 11",
    "Удалить шум «соль и перец» медианным фильтром",
    "Выделить границы оператором Собеля",
    "Реализовать фильтр повышения резкости sharpen",
    "Реализовать гауссов высокочастотный фильтр",
], x=T.MARGIN, y=2.25, w=7.55, row_h=0.70)
deck.metrics(slide, [
    (f"{width} × {height}", "размер кадра, пикселей"),
    ("0 … 255", "диапазон значений яркости"),
    (f"σ = {SIGMA}", "параметр гауссова ВЧ-фильтра"),
], x=8.86, y=2.20, w=3.57, h=3.62, vertical=True)

slide = deck.slide("Два взгляда на фильтрацию", "Одно и то же действие в разных пространствах",
                   kicker="Теория")
bottom = deck.columns(slide, [
    ("Пространственная область",
     "g(x, y) = ΣΣ h(i, j)·f(x − i, y − j). Ядро идёт по изображению, а его "
     "коэффициенты задают вклад соседей. Удобно для локальных операций "
     "и небольших ядер."),
    ("Нелинейные фильтры",
     "Медианный фильтр сортирует значения в окне и берёт центральное. Он не "
     "линейная комбинация, поэтому лучше сохраняет границы при импульсном шуме."),
    ("Частотная область",
     "Низкие частоты — плавный фон и крупные формы, высокие — резкие переходы, "
     "мелкие детали и шум. Свёртка превращается в поэлементное умножение спектров."),
], x=T.MARGIN, y=2.12, w=11.53)
deck.note(slide,
          "Большое окно сильнее подавляет шум, но сильнее размывает границы и мелкие "
          "детали — этот компромисс виден во всех результатах ниже.",
          x=T.MARGIN, y=bottom + 0.32, w=11.53, bar=True, size=T.BODY)

slide = deck.slide("Усредняющий фильтр", "Размер окна решает, что останется от деталей",
                   kicker="Пункт 01")
bottom = deck.picture(slide, blur_panels, T.MARGIN, 2.12, w=11.53).bottom_in
bottom = deck.metrics(slide, [
    ("25 → 121", "пикселей в окне 5 × 5 и 11 × 11"),
    (f"{mad_5:.2f}", "среднее изменение после 5 × 5"),
    (f"{mad_11:.2f}", "среднее изменение после 11 × 11"),
], x=T.MARGIN, y=bottom + 0.28, w=11.53, h=0.98)
deck.caption(slide, "Каждое значение в окне 11 × 11 зависит от 121 пикселя, поэтому "
                    "локальная структура меняется заметно сильнее.",
             x=T.MARGIN, y=bottom + 0.18, w=11.53)

slide = deck.slide("Медианный фильтр", "Импульсный шум удаляется, граница остаётся",
                   kicker="Пункт 02")
deck.picture(slide, median_panels, T.MARGIN, 2.15, w=8.05)
deck.metrics(slide, [
    (f"{mad_noisy:.2f}", "ошибка зашумлённого кадра"),
    (f"{mad_median:.2f}", "ошибка после медианы 5 × 5"),
    ("42", "фиксированное зерно генератора"),
], x=9.15, y=2.15, w=3.28, h=3.55, vertical=True)
deck.code(slide, "noisy[noise_mask < 0.025] = 0\n"
                 "noisy[noise_mask > 0.975] = 255\n"
                 "median_5 = cv2.medianBlur(noisy, 5)",
          x=T.MARGIN, y=5.14, w=8.05)

slide = deck.slide("Оператор Собеля", "Первая производная яркости по двум направлениям",
                   kicker="Пункт 03")
bottom = deck.picture(slide, sobel_panels, T.MARGIN, 2.12, w=11.53).bottom_in
bottom = deck.metrics(slide, [
    (f"{np.abs(sobel_x).max():.0f}", "максимум |Gₓ|"),
    (f"{np.abs(sobel_y).max():.0f}", "максимум |G_y|"),
    ("G = √(Gₓ² + G_y²)", "модуль градиента объединяет направления"),
], x=T.MARGIN, y=bottom + 0.28, w=11.53, h=0.98)
deck.caption(slide, "Большое значение G означает резкий переход яркости; для показа "
                    "модуль нормируется в диапазон 0…255.",
             x=T.MARGIN, y=bottom + 0.18, w=11.53)

slide = deck.slide("Повышение резкости", "Центр ядра больше единицы, соседи отрицательны",
                   kicker="Пункт 04")
bottom = deck.picture(slide, sharpen_panels, T.MARGIN, 2.15, w=8.05).bottom_in
deck.metrics(slide, [
    (f"{mad_sharpen:.2f}", "среднее изменение относительно исходного"),
    ("I + α(I − G*I)", "эквивалент через unsharp masking"),
], x=9.15, y=2.15, w=3.28, h=2.34, vertical=True)
bottom = deck.code(slide, "kernel = np.array([[ 0, -1,  0],\n"
                          "                   [-1,  5, -1],\n"
                          "                   [ 0, -1,  0]], dtype=np.float32)",
                   x=T.MARGIN, y=bottom + 0.26, w=8.05)
deck.caption(slide, "Ядро усиливает разницу между центром и соседями: границы становятся "
                    "контрастнее, но вместе с ними усиливаются шум и ореолы.",
             x=T.MARGIN, y=bottom + 0.24, w=11.53)

slide = deck.slide("Гауссов высокочастотный фильтр",
                   "H_ВЧ(D) = 1 − exp(−D² / 2σ²): плавный переход без ringing",
                   kicker="Пункт 05")
chart_bottom = deck.picture(slide, mask_chart, T.MARGIN, 2.15, w=5.62).bottom_in
panels_bottom = deck.picture(slide, compare_panels, 6.81, 2.15, w=5.62).bottom_in
deck.note(slide,
          "В центре спектра маска ВЧ близка к нулю, поэтому плавный фон подавляется; "
          "дальше от центра она стремится к единице, и границы проходят сильнее.",
          x=6.81, y=panels_bottom + 0.26, w=5.62, size=T.BODY_SM)
deck.metrics(slide, [
    (f"σ = {SIGMA}", "ширина гауссовой маски в спектре"),
    ("fftshift", "нулевая частота переносится в центр спектра"),
], x=T.MARGIN, y=chart_bottom + 0.28, w=11.53, h=0.98)

slide = deck.slide("Путь через спектр", "БПФ, маска, обратное БПФ", kicker="Пункт 05")
bottom = deck.picture(slide, spectrum_panels, T.MARGIN, 2.12, w=11.53).bottom_in
deck.code(slide, "spectrum = np.fft.fftshift(np.fft.fft2(image))\n"
                 "result = np.fft.ifft2(np.fft.ifftshift(spectrum * gaussian_hpf))",
          x=T.MARGIN, y=bottom + 0.28, w=11.53)

slide = deck.slide("Насколько сильно меняется кадр",
                   "Среднее абсолютное отклонение от исходного изображения",
                   kicker="Сравнение")
deck.picture(slide, effect_chart, T.MARGIN, 2.15, w=7.35)
deck.note(slide,
          "Усреднение 11 × 11 меняет кадр почти вдвое сильнее, чем 5 × 5. Медианный "
          "фильтр, наоборот, приближает зашумлённый кадр к исходному: ошибка падает "
          f"с {mad_noisy:.2f} до {mad_median:.2f}. Sharpen меняет немного, но "
          "прицельно — только на границах.",
          x=8.62, y=2.15, w=3.81, bar=True, size=T.BODY_SM)

slide = deck.slide("Сравнение методов", "Сильные стороны и цена каждого фильтра",
                   kicker="Сравнение")
deck.table(slide, ["Метод", "Что делает", "Сильная сторона", "Недостаток"], [
    ["Усреднение 5 × 5", "среднее по окну", "умеренное подавление шума", "размывает границы"],
    ["Усреднение 11 × 11", "среднее по большому окну", "сильное сглаживание", "сильное размытие"],
    ["Медианный 5 × 5", "медиана окна", "эффективен против соли и перца", "убирает мелкие детали"],
    ["Собель", "первая производная", "выделяет направленные границы", "чувствителен к шуму"],
    ["Sharpen", "усиливает локальный контраст", "повышает резкость", "усиливает шум и ореолы"],
    ["Гауссов ВЧ", "подавляет низкие частоты", "выделяет детали и контуры", "нужна настройка σ"],
], x=T.MARGIN, y=2.25, w=11.53, widths=[0.19, 0.25, 0.31, 0.25], row_h=0.58, size=12.5)

QUESTIONS = [
    ("Что такое фильтрация и зачем она нужна?",
     "Преобразование изображения по значениям соседних пикселей или по спектру — для "
     "подавления шума, сглаживания, повышения резкости и выделения границ."),
    ("Чем линейная фильтрация отличается от нелинейной?",
     "Линейная считает взвешенную сумму значений окна, нелинейная применяет другое "
     "правило — например, выбирает медиану или минимум."),
    ("Какие бывают шумы?",
     "Гауссов даёт небольшие случайные отклонения, импульсный «соль и перец» — чёрные "
     "и белые выбросы; бывают равномерный, периодический, мультипликативный."),
    ("Как размер окна влияет на результат?",
     "Большое окно сильнее сглаживает шум, но размывает границы, уничтожает мелкие "
     "детали и стоит дороже по вычислениям."),
    ("Когда пространственная, а когда частотная фильтрация?",
     "Пространственная — для локальных операций и небольших ядер, частотная — для "
     "анализа спектра, удаления выбранных диапазонов и больших ядер."),

    ("Как работает усредняющий фильтр?",
     "Каждый пиксель заменяется средним значением пикселей внутри заданного окна."),
    ("Чем гауссов фильтр отличается от усреднения?",
     "Он придаёт больший вес центру окна и меньший дальним пикселям, поэтому "
     "сглаживание мягче и структура сохраняется лучше."),
    ("Для какого шума эффективен медианный фильтр?",
     "Прежде всего для импульсного шума «соль и перец»."),
    ("Что делают Собель и Превитт?",
     "Это градиентные операторы первой производной: они оценивают изменения яркости "
     "по горизонтали и вертикали и выделяют границы."),
    ("Чем Лаплас отличается от Собеля?",
     "Лаплас основан на второй производной и не выделяет направление; Собель даёт "
     "направленные компоненты X и Y."),

    ("Как реализуется sharpen или unsharp masking?",
     "Ядром с положительным центром и отрицательными соседями либо добавлением "
     "к кадру усиленной разницы с его размытой копией."),
    ("Как интерпретировать низкие и высокие частоты?",
     "Низкие описывают плавный фон и крупные формы, высокие — резкие переходы, "
     "границы, мелкие детали и часть шума."),
    ("Как выполняется переход в частотную область?",
     "Дискретным преобразованием Фурье, на практике — его быстрой реализацией "
     "np.fft.fft2."),
    ("Какие шаги выполняются при частотной фильтрации?",
     "БПФ, перенос нулевой частоты в центр, построение маски, умножение спектра "
     "на маску, обратный перенос и обратное БПФ."),
    ("Чем идеальный НЧ-фильтр отличается от гауссова?",
     "У идеального резкая граница: внутри радиуса частоты проходят полностью, "
     "снаружи отбрасываются. Гауссов уменьшает пропускание плавно."),

    ("Какие артефакты даёт идеальный НЧ-фильтр?",
     "Ringing — концентрические колебательные ореолы около резких границ."),
    ("Как работает ВЧ-фильтр?",
     "Он подавляет область низких частот и пропускает высокие, поэтому выделяет "
     "контуры, текстуры и резкие детали."),
    ("Что такое полосовой фильтр?",
     "Фильтр, пропускающий только диапазон частот между нижней и верхней границей: "
     "им выделяют текстуры и структуры заданного масштаба."),
    ("Какие преимущества даёт частотная фильтрация?",
     "Она позволяет явно управлять диапазонами частот и эффективно работать "
     "с большими ядрами через БПФ."),
    ("Почему БПФ эффективно при больших ядрах?",
     "Прямая свёртка с большим ядром дорога, а в спектре она превращается "
     "в поэлементное умножение."),
]

for part in range(4):
    slide = deck.slide("Контрольные вопросы", f"Часть {part + 1} из 4", kicker="Защита")
    deck.qa(slide, QUESTIONS[part * 5:(part + 1) * 5],
            x=T.MARGIN, y=2.20, w=11.53, columns=2, size=12.5, avail_h=4.4)

slide = deck.slide("Итоги", "Что выполнено в ноутбуке", kicker="Выводы")
bottom = deck.checklist(slide, [
    f"Усреднение 5 × 5 и 11 × 11 изменило кадр на {mad_5:.2f} и {mad_11:.2f} уровня",
    f"Медианный фильтр снизил ошибку зашумлённого кадра с {mad_noisy:.2f} до {mad_median:.2f}",
    f"Собель выделил границы: максимальный отклик {np.abs(sobel_x).max():.0f} по обоим осям",
    f"Ядро sharpen повысило локальный контраст, изменив кадр на {mad_sharpen:.2f}",
    f"Гауссов ВЧ-фильтр с σ = {SIGMA} подавил плавный фон и оставил контуры",
    "Показан весь путь через спектр: БПФ, маска, обратное БПФ",
], x=T.MARGIN, y=2.20, w=11.53, size=16.5)
deck.text(slide,
          "Выбор фильтра — это всегда компромисс: чем сильнее подавляется шум, "
          "тем больше теряется мелких деталей и резкости границ.",
          x=T.MARGIN, y=bottom + 0.46, w=11.53, h=0.8,
          size=T.BODY, color=T.PRIMARY, bold=True)

path, count = deck.save(OUTPUT)
print(f"saved: {path} ({count} слайдов)")
