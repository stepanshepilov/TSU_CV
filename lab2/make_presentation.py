"""Презентация-отчёт по лабораторной работе 2 (вариант 9)."""
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from presentation_kit import figures as F  # noqa: E402
from presentation_kit import tokens as T  # noqa: E402
from presentation_kit.deck import Deck  # noqa: E402

ROOT = Path(__file__).resolve().parent
IMAGE_PATH = ROOT / "image.png"
ASSETS = ROOT / ".presentation_assets"
ASSETS.mkdir(exist_ok=True)
OUTPUT = ROOT / "Лабораторная_2_вариант_9.pptx"

# --- расчёты ---------------------------------------------------------------
image = cv2.imread(str(IMAGE_PATH), cv2.IMREAD_GRAYSCALE)
height, width = image.shape

histogram = cv2.calcHist([image], [0], None, [256], [0, 256]).ravel()
peak_intensity = int(np.argmax(histogram))
peak_count = int(histogram[peak_intensity])

laplacian = np.abs(cv2.Laplacian(image, cv2.CV_64F))
lap_h_profile = laplacian.mean(axis=1)
lap_v_profile = laplacian.mean(axis=0)
lap_row = int(np.argmax(lap_h_profile))
lap_col = int(np.argmax(lap_v_profile))

h_projection = image.sum(axis=1)
v_projection = image.sum(axis=0)
proj_row = int(np.argmax(h_projection))
proj_col = int(np.argmax(v_projection))

otsu_threshold, binary = cv2.threshold(image, 0, 255,
                                       cv2.THRESH_BINARY + cv2.THRESH_OTSU)
white_share = float((binary == 255).mean()) * 100
bin_h_projection = binary.sum(axis=1)
bin_v_projection = binary.sum(axis=0)
bin_row = int(np.argmax(bin_h_projection))
bin_col = int(np.argmax(bin_v_projection))
full_rows = int((bin_h_projection == bin_h_projection.max()).sum())
full_cols = int((bin_v_projection == bin_v_projection.max()).sum())

# --- иллюстрации -----------------------------------------------------------
hero = F.rounded_png(ASSETS / "hero.png",
                     cv2.cvtColor(image, cv2.COLOR_GRAY2RGB), placement_w=5.08)
lap_panels = F.panels(ASSETS / "laplacian.png",
                      [(image, "Исходное, оттенки серого"),
                       (np.clip(laplacian, 0, 255).astype(np.uint8), "|∇²I| — отклик Лапласа")],
                      width=8.05)
binary_panels = F.panels(ASSETS / "binary.png",
                         [(image, "Оттенки серого"),
                          (binary, f"Бинаризация Отсу, порог {otsu_threshold:.0f}")],
                         width=8.05)


def draw_histogram(ax):
    ax.bar(np.arange(256), histogram, color=T.SERIES[0], width=1.0)
    ax.axvline(peak_intensity, color=T.SERIES[1], linestyle="--", linewidth=1.6)
    ax.annotate(f"пик: {peak_intensity}", xy=(peak_intensity, peak_count),
                xytext=(peak_intensity - 92, peak_count * 0.92),
                color=T.SERIES[1], fontsize=10, weight="bold")
    ax.set_xlabel("Уровень яркости")
    ax.set_ylabel("Количество пикселей")
    ax.set_xlim(0, 255)


hist_chart = F.chart(ASSETS / "hist.png", 7.35, 4.15, draw_histogram,
                     pad=(0.86, 0.52, 0.24, 0.30))


def profile_pair(path, horizontal, vertical, h_index, v_index, h_label, v_label,
                 width=11.53, height=3.05):
    """Два профиля рядом: по строкам и по столбцам."""
    fig = F.figure(width, height)
    for column, (values, index, label) in enumerate(
            [(horizontal, h_index, h_label), (vertical, v_index, v_label)]):
        left = 0.075 + column * 0.49
        ax = F.axes(fig, [left, 0.185, 0.415, 0.72])
        ax.plot(np.arange(len(values)), values, color=T.SERIES[0], linewidth=1.4)
        ax.axvline(index, color=T.SERIES[1], linestyle="--", linewidth=1.4)
        ax.set_xlim(0, len(values) - 1)
        ax.set_xlabel(label)
        ax.annotate(f"{index}", xy=(index, values[index]),
                    xytext=(index + len(values) * 0.02, values.max() * 0.9),
                    color=T.SERIES[1], fontsize=10, weight="bold")
    return F.finish(fig, path)


lap_profiles = profile_pair(ASSETS / "lap_profiles.png", lap_h_profile, lap_v_profile,
                            lap_row, lap_col,
                            "Строка y · средний отклик", "Столбец x · средний отклик")
projections = profile_pair(ASSETS / "projections.png", h_projection, v_projection,
                           proj_row, proj_col,
                           "Строка y · сумма яркостей", "Столбец x · сумма яркостей")
bin_projections = profile_pair(ASSETS / "binary_projections.png",
                               bin_h_projection, bin_v_projection, bin_row, bin_col,
                               "Строка y · сумма по бинарному",
                               "Столбец x · сумма по бинарному")

# --- слайды ----------------------------------------------------------------
deck = Deck("Лабораторная работа 2 · вариант 9")

deck.cover(
    kicker="Лабораторная работа 2 · вариант 9",
    title="Одномерный портрет\nизображения",
    subtitle="Гистограмма, профили, проекции, фильтр Лапласа\nи бинаризация методом Отсу",
    meta="Компьютерное зрение · OpenCV · NumPy · Matplotlib",
    image=hero,
)

slide = deck.slide("Задание варианта 9", "Пять пунктов", kicker="Постановка")
deck.steps(slide, [
    "Построить гистограмму и найти пик интенсивности",
    "Построить профили после применения фильтра Лапласа",
    "Построить горизонтальную проекцию и найти строку максимума",
    "Построить вертикальную проекцию и найти столбец максимума",
    "Повторить профили и проекции после бинаризации Отсу",
], x=T.MARGIN, y=2.25, w=7.55, row_h=0.70)
deck.metrics(slide, [
    (f"{width} × {height}", "размер кадра, пикселей"),
    ("0 … 255", "диапазон значений яркости"),
    ("1 канал", "чтение с IMREAD_GRAYSCALE"),
], x=8.86, y=2.20, w=3.57, h=3.62, vertical=True)

slide = deck.slide("Три способа посмотреть на кадр",
                   "Каждая характеристика отвечает на свой вопрос", kicker="Теория")
bottom = deck.columns(slide, [
    ("Гистограмма",
     "H(k) — сколько пикселей имеют яркость k. Отвечает на вопрос «какая яркость "
     "встречается чаще всего» и показывает контраст и динамический диапазон."),
    ("Профиль",
     "Pₕ(y) = среднее по строке, Pᵥ(x) = среднее по столбцу. Нормирован на длину, "
     "поэтому профили разных изображений одного размера сравнимы напрямую."),
    ("Проекция",
     "HP(y) = сумма по строке, VP(x) = сумма по столбцу. Зависит от размера кадра "
     "и хранит суммарную энергию строки или столбца."),
], x=T.MARGIN, y=2.12, w=11.53)
deck.note(slide,
          "Фильтр Лапласа ∇²I — вторая производная яркости. Он усиливает резкие "
          "перепады, поэтому профиль по его модулю показывает не «где ярче», "
          "а «где сильнее меняется яркость».",
          x=T.MARGIN, y=bottom + 0.32, w=11.53, bar=True, size=T.BODY)

slide = deck.slide("Гистограмма исходного кадра", "Пик показывает доминирующий тон",
                   kicker="Пункт 01")
deck.picture(slide, hist_chart, T.MARGIN, 2.15, w=7.35)
deck.metrics(slide, [
    (f"{peak_intensity}", "уровень яркости пика"),
    (f"{peak_count}", "пикселей на этом уровне"),
    (f"{image.mean():.2f}", "средняя яркость кадра"),
], x=8.62, y=2.15, w=3.81, h=3.55, vertical=True)
deck.caption(slide, "Пик не обязан совпадать с яркостью интересующего объекта: "
                    "большой фон легко перевешивает небольшой объект.",
             x=T.MARGIN, y=6.42, w=11.53)

slide = deck.slide("Фильтр Лапласа", "Отклик берётся по модулю: важна сила, а не знак",
                   kicker="Пункт 02")
deck.picture(slide, lap_panels, T.MARGIN, 2.15, w=8.05)
deck.metrics(slide, [
    (f"{laplacian.max():.0f}", "максимум |∇²I|"),
    (f"{laplacian.mean():.2f}", "средний отклик по кадру"),
], x=9.15, y=2.15, w=3.28, h=2.34, vertical=True)
deck.code(slide, "laplacian = np.abs(cv2.Laplacian(image, cv2.CV_64F))",
          x=T.MARGIN, y=5.14, w=8.05)

slide = deck.slide("Профили после Лапласа", "Где в среднем сосредоточены перепады яркости",
                   kicker="Пункт 02")
bottom = deck.picture(slide, lap_profiles, T.MARGIN, 2.15, w=11.53).bottom_in
deck.metrics(slide, [
    (f"строка {lap_row}", f"максимум профиля: {lap_h_profile[lap_row]:.2f}"),
    (f"столбец {lap_col}", f"максимум профиля: {lap_v_profile[lap_col]:.2f}"),
], x=T.MARGIN, y=bottom + 0.30, w=11.53, h=1.06)

slide = deck.slide("Проекции исходного кадра", "Сумма яркостей вдоль строки и вдоль столбца",
                   kicker="Пункты 03–04")
bottom = deck.picture(slide, projections, T.MARGIN, 2.05, w=11.53).bottom_in
bottom = deck.metrics(slide, [
    (f"строка {proj_row}", f"HP = {h_projection[proj_row]}"),
    (f"столбец {proj_col}", f"VP = {v_projection[proj_col]}"),
], x=T.MARGIN, y=bottom + 0.28, w=11.53, h=0.98)
deck.caption(slide, "Проекция отвечает на вопрос, где накоплено больше общей яркости.",
             x=T.MARGIN, y=bottom + 0.18, w=11.53)

slide = deck.slide("Бинаризация методом Отсу", "Порог подбирается по межклассовой дисперсии",
                   kicker="Пункт 05")
deck.picture(slide, binary_panels, T.MARGIN, 2.15, w=8.05)
deck.metrics(slide, [
    (f"{otsu_threshold:.0f}", "порог Отсу"),
    (f"{white_share:.2f} %", "белых пикселей"),
    ("0 и 255", "все значения после порога"),
], x=9.15, y=2.15, w=3.28, h=3.55, vertical=True)
deck.code(slide, "otsu, binary = cv2.threshold(\n"
                 "    image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)",
          x=T.MARGIN, y=5.14, w=8.05)

slide = deck.slide("Проекции бинарного кадра", "Теперь проекция пропорциональна числу белых пикселей",
                   kicker="Пункт 05")
bottom = deck.picture(slide, bin_projections, T.MARGIN, 2.05, w=11.53).bottom_in
bottom = deck.metrics(slide, [
    (f"строка {bin_row}", f"первая из {full_rows} полностью белых строк"),
    (f"столбец {bin_col}", f"первый из {full_cols} полностью белых столбцов"),
], x=T.MARGIN, y=bottom + 0.28, w=11.53, h=0.98)
deck.caption(slide, "По пикам и провалам таких проекций находят строки текста и границы объектов.",
             x=T.MARGIN, y=bottom + 0.18, w=11.53)

slide = deck.slide("Контрольные вопросы", "Часть первая", kicker="Защита")
deck.qa(slide, [
    ("Что такое гистограмма и зачем она нужна?",
     "Распределение количества пикселей по уровням яркости. Нужна для анализа тонов, "
     "контраста и выбора порога сегментации."),
    ("Чем гистограмма цветного кадра отличается от серого?",
     "У серого один канал и одна гистограмма 0…255. У цветного строят три — по R, G и B; "
     "одинаковая яркость получается разными сочетаниями цветов."),
    ("Что такое профиль изображения?",
     "Одномерный массив средних интенсивностей по строкам или по столбцам."),
    ("Как вычисляются горизонтальный и вертикальный профили?",
     "Горизонтальный — усреднением по оси столбцов (axis=1), вертикальный — по оси "
     "строк (axis=0)."),
    ("Что такое проекция изображения?",
     "Сумма интенсивностей вдоль одной координаты: она сворачивает двумерный кадр "
     "в одномерный сигнал."),
], x=T.MARGIN, y=2.20, w=11.53, columns=2, size=13, avail_h=4.3)

slide = deck.slide("Контрольные вопросы", "Часть вторая", kicker="Защита")
deck.qa(slide, [
    ("В чём разница между профилем и проекцией?",
     "Профиль — среднее, нормированное на длину строки или столбца. Проекция — сумма, "
     "она зависит от размера изображения."),
    ("Как гистограмма помогает улучшить контраст?",
     "По ней видно сжатие диапазона яркостей; отсюда решение о растяжении контраста "
     "или эквализации гистограммы."),
    ("Зачем нужны проекции бинарного изображения?",
     "По пикам и провалам находят строки текста, границы объектов, пустые интервалы, "
     "положение и размеры объектов."),
    ("Как изменится гистограмма после сглаживания?",
     "Высокочастотный шум уменьшается, экстремальные значения встречаются реже, "
     "распределение становится компактнее и менее «зубчатым»."),
    ("Что показывает пик на гистограмме?",
     "Самый частый уровень яркости и количество пикселей на нём. Он не обязан "
     "совпадать с яркостью интересующего объекта."),
], x=T.MARGIN, y=2.20, w=11.53, columns=2, size=13, avail_h=4.3)

slide = deck.slide("Итоги", "Что выполнено в ноутбуке", kicker="Выводы")
bottom = deck.checklist(slide, [
    f"Гистограмма построена: доминирующая интенсивность {peak_intensity}, "
    f"{peak_count} пикселей",
    f"Профили после Лапласа дали строку {lap_row} и столбец {lap_col}",
    f"Горизонтальная проекция: максимум в строке {proj_row}",
    f"Вертикальная проекция: максимум в столбце {proj_col}",
    f"Отсу разделил кадр порогом {otsu_threshold:.0f}; белых пикселей {white_share:.2f} %",
    "Бинарные профили и проекции интерпретируются как распределение белой области",
], x=T.MARGIN, y=2.20, w=11.53, size=16.5)
deck.text(slide,
          "Гистограмма отвечает «какая яркость встречается чаще», Лаплас — «где сильнее "
          "меняется яркость», проекция — «где накоплено больше общей яркости».",
          x=T.MARGIN, y=bottom + 0.46, w=11.53, h=0.8,
          size=T.BODY, color=T.PRIMARY, bold=True)

path, count = deck.save(OUTPUT)
print(f"saved: {path} ({count} слайдов)")
