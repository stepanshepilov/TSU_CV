"""Презентация-отчёт по лабораторной работе 1 (вариант 9).

Все числа считаются заново из исходного изображения, поэтому слайды не могут
разойтись с ноутбуком.
"""
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
OUTPUT = ROOT / "Лабораторная_1_вариант_9.pptx"

# --- расчёты ---------------------------------------------------------------
image = cv2.imread(str(IMAGE_PATH), cv2.IMREAD_COLOR)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
height, width, channels = image.shape

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
otsu_threshold, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
white_ratio = float((binary == 255).mean()) * 100

hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
hue, sat, val = cv2.split(hsv)
blue, green, red = cv2.split(image)

edited = image.copy()
edited[:, :20] = (255, 0, 0)

levels = np.arange(256)
hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
p = hist / hist.sum()
w0 = np.cumsum(p)
w1 = 1.0 - w0
cum_mean = np.cumsum(levels * p)
mu_total = float(cum_mean[-1])
mu0 = np.divide(cum_mean, w0, out=np.zeros(256), where=w0 > 0)
mu1 = np.divide(mu_total - cum_mean, w1, out=np.zeros(256), where=w1 > 0)
sigma_b = w0 * w1 * (mu0 - mu1) ** 2
manual_threshold = int(np.argmax(sigma_b))

hue_hist = cv2.calcHist([hsv], [0], None, [180], [0, 180]).ravel()
dominant_hue = int(np.argmax(hue_hist))
# Нижняя граница 26, а не 20: у оранжевого сектора шлема H ≈ 19…25, и весь
# жёлтый диапазон срезал бы половину головы вместе с фоном.
YELLOW_LOW = np.array([26, 80, 80])
YELLOW_HIGH = np.array([35, 255, 255])
yellow_mask = cv2.inRange(hsv, YELLOW_LOW, YELLOW_HIGH)
yellow_share = float((yellow_mask > 0).mean()) * 100

open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
alpha = cv2.bitwise_not(yellow_mask)
alpha = cv2.morphologyEx(alpha, cv2.MORPH_OPEN, open_kernel)
alpha = cv2.morphologyEx(alpha, cv2.MORPH_CLOSE, close_kernel)
opaque_share = float((alpha > 0).mean()) * 100

ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
diff = cv2.absdiff(ycrcb[:, :, 0], gray)

checker = np.indices((height, width)).sum(axis=0) // 24 % 2
checker = np.where(checker == 0, 236, 214).astype(np.uint8)
alpha_f = (alpha / 255.0)[:, :, None]
composited = (image_rgb.astype(np.float32) * alpha_f
              + cv2.cvtColor(checker, cv2.COLOR_GRAY2RGB).astype(np.float32) * (1 - alpha_f))
composited = composited.astype(np.uint8)

objects_only = cv2.cvtColor(
    cv2.bitwise_and(image, image, mask=cv2.bitwise_not(yellow_mask)), cv2.COLOR_BGR2RGB)

# --- иллюстрации -----------------------------------------------------------
hero = F.rounded_png(ASSETS / "hero.png", image_rgb, placement_w=5.08)
original = F.rounded_png(ASSETS / "original.png", image_rgb, placement_w=7.45)

otsu_panels = F.panels(ASSETS / "otsu.png",
                       [(gray, "Градации серого"),
                        (binary, f"Бинаризация Оцу, порог {otsu_threshold:.0f}")], width=8.05)
hsv_panels = F.panels(ASSETS / "hsv.png",
                      [(hue, "H — тон"), (sat, "S — насыщенность"), (val, "V — яркость")],
                      width=11.53)
edge_panels = F.panels(ASSETS / "edge.png",
                       [(image_rgb, "Исходное"),
                        (cv2.cvtColor(edited, cv2.COLOR_BGR2RGB), "Левый край 20 пикс. — синий")],
                       width=8.05)
channel_panels = F.panels(ASSETS / "channels.png",
                          [(blue, f"B · среднее {blue.mean():.1f}"),
                           (green, f"G · среднее {green.mean():.1f}"),
                           (red, f"R · среднее {red.mean():.1f}")], width=11.53)
mask_panels = F.panels(ASSETS / "mask.png",
                       [(yellow_mask, "Маска фона"), (objects_only, "Только объекты")],
                       width=5.62)
model_panels = F.panels(ASSETS / "models.png",
                        [(cv2.cvtColor(image, cv2.COLOR_BGR2HLS)[:, :, 1], "HLS · L"),
                         (ycrcb[:, :, 0], "YCrCb · Y"),
                         (cv2.cvtColor(image, cv2.COLOR_BGR2LAB)[:, :, 2], "LAB · b")],
                        width=11.53)
alpha_panels = F.panels(ASSETS / "alpha.png",
                        [(alpha, "Альфа-канал"), (composited, "RGBA поверх подложки")],
                        width=8.05)


def draw_sigma(ax):
    ax.plot(levels, sigma_b, color=T.SERIES[0])
    ax.axvline(manual_threshold, color=T.SERIES[1], linestyle="--", linewidth=1.6)
    ax.annotate(f"t* = {manual_threshold}", xy=(manual_threshold, sigma_b.max()),
                xytext=(manual_threshold - 96, sigma_b.max() * 0.94),
                color=T.SERIES[1], fontsize=10, weight="bold")
    ax.set_xlabel("Порог t")
    ax.set_ylabel("Межклассовая дисперсия σ²b(t)")
    ax.set_xlim(0, 255)
    ax.set_ylim(0, sigma_b.max() * 1.12)


sigma_chart = F.chart(ASSETS / "sigma.png", 7.35, 4.15, draw_sigma,
                      pad=(0.92, 0.52, 0.26, 0.30))


def draw_hue(ax):
    ax.bar(np.arange(180), hue_hist, color=T.SERIES[0], width=1.0)
    ax.axvline(dominant_hue, color=T.SERIES[1], linestyle="--", linewidth=1.6)
    ax.annotate(f"H = {dominant_hue}", xy=(dominant_hue, hue_hist.max()),
                xytext=(dominant_hue + 12, hue_hist.max() * 0.88),
                color=T.SERIES[1], fontsize=10, weight="bold")
    ax.set_xlabel("Тон H, 0…179")
    ax.set_ylabel("Пикселей")
    ax.set_xlim(0, 179)


hue_chart = F.chart(ASSETS / "hue_hist.png", 5.62, 2.85, draw_hue,
                    pad=(0.86, 0.50, 0.20, 0.26))

# --- слайды ----------------------------------------------------------------
deck = Deck("Лабораторная работа 1 · вариант 9")

deck.cover(
    kicker="Лабораторная работа 1 · вариант 9",
    title="Изображение\nкак массив чисел",
    subtitle="Цветовые модели, бинаризация методом Оцу,\nпрямое редактирование пикселей и альфа-канал",
    meta="Компьютерное зрение · OpenCV · NumPy · Matplotlib",
    image=hero,
)

# Задание
slide = deck.slide("Задание варианта 9", "Пять обязательных пунктов", kicker="Постановка")
deck.steps(slide, [
    "Загрузить изображение, вывести размеры и число каналов",
    "Перевести в градации серого и бинаризовать методом Оцу",
    "Преобразовать в HSV и вывести каналы H, S, V",
    "Закрасить синим левый край шириной 20 пикселей",
    "Сохранить изображение с изменениями",
], x=T.MARGIN, y=2.25, w=7.55, row_h=0.70)
deck.metrics(slide, [
    (f"{width} × {height}", "размер кадра, пикселей"),
    (f"{channels} · uint8", "каналы BGR и тип данных"),
    (f"{image.nbytes / 1024 / 1024:.2f} МБ", "массив в оперативной памяти"),
], x=8.86, y=2.20, w=3.57, h=3.62, vertical=True)

slide = deck.slide("Собственные задачи 6–10", "Пять задач, сформулированных самостоятельно",
                   kicker="Постановка")
deck.steps(slide, [
    "Сравнить порог Оцу с фиксированным и проверить его ручным расчётом",
    "Разделить кадр на каналы B, G, R и собрать их статистику",
    "Построить гистограмму тона H и выделить фон бинарной маской",
    "Сравнить модели HSV, HLS, YCrCb и LAB между собой",
    "Убрать фон по маске и сохранить «стикер» в формате RGBA",
], x=T.MARGIN, y=2.35, w=11.53, row_h=0.80, start=6, size=T.BODY_LG)

# Теория
slide = deck.slide("Три опоры работы", "Что нужно знать, чтобы прочитать результаты",
                   kicker="Теория")
bottom = deck.columns(slide, [
    ("Массив пикселей",
     "Цветное изображение — матрица (H, W, 3) типа uint8. OpenCV хранит каналы "
     "в порядке BGR, а обращение image[y, x] идёт сначала по строке, потом по столбцу."),
    ("Порог Оцу",
     "σ²b(t) = w₀·w₁·(μ₀ − μ₁)²\n"
     "Перебираются все пороги и выбирается тот, что разводит фон и объект "
     "максимально далеко друг от друга."),
    ("Модель HSV",
     "H — тон (0…179), S — насыщенность, V — яркость. Весь цвет собран в одном "
     "канале, поэтому маска по цвету — это один диапазон вместо трёх."),
], x=T.MARGIN, y=2.10, w=11.53)
deck.note(slide,
          "Полутоновое изображение считается как Y = 0.299·R + 0.587·G + 0.114·B — "
          "коэффициенты отражают разную чувствительность глаза к каналам. "
          "Та же формула стоит за каналом Y модели YCrCb, что проверяется в задаче 9.",
          x=T.MARGIN, y=bottom + 0.34, w=11.53, bar=True, size=T.BODY)

# Результаты
slide = deck.slide("Размеры и число каналов", "image.shape возвращает (высота, ширина, каналы)",
                   kicker="Пункт 01")
deck.picture(slide, original, T.MARGIN, 2.15, w=7.45)
deck.metrics(slide, [
    (f"{width} × {height}", "ширина × высота"),
    (f"{height * width // 1000} тыс.", "пикселей в кадре"),
    (f"{image[0, 0].tolist()}", "пиксель [0, 0] в BGR"),
], x=8.72, y=2.15, w=3.71, h=3.55, vertical=True)

slide = deck.slide("Градации серого и метод Оцу", "Порог подбирается автоматически по гистограмме",
                   kicker="Пункт 02")
deck.picture(slide, otsu_panels, T.MARGIN, 2.15, w=8.05)
deck.metrics(slide, [
    (f"{otsu_threshold:.0f}", "порог Оцу"),
    (f"{white_ratio:.2f} %", "белых пикселей"),
    (f"{gray.mean():.2f}", "средняя яркость кадра"),
], x=9.35, y=2.15, w=3.08, h=3.55, vertical=True)
deck.code(slide, "otsu, binary = cv2.threshold(\n"
                 "    gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)",
          x=T.MARGIN, y=5.14, w=8.05)

slide = deck.slide("Проверка метода Оцу вручную", "Максимум межклассовой дисперсии совпал с OpenCV",
                   kicker="Задача 06")
deck.picture(slide, sigma_chart, T.MARGIN, 2.15, w=7.35)
deck.metrics(slide, [
    (f"{otsu_threshold:.0f} = {manual_threshold}", "OpenCV и ручной расчёт"),
    (f"{sigma_b.max():.0f}", "максимум σ²b"),
], x=8.62, y=2.15, w=3.81, h=2.30, vertical=True)
fixed_100 = float((gray > 100).mean()) * 100
fixed_200 = float((gray > 200).mean()) * 100
deck.table(slide, ["Порог", "Доля белого"], [
    ["100", f"{fixed_100:.2f} %"],
    [f"Оцу · {otsu_threshold:.0f}", f"{white_ratio:.2f} %"],
    ["200", f"{fixed_200:.2f} %"],
], x=8.76, y=4.86, w=3.53, widths=[0.52, 0.48], highlight=1)
deck.caption(slide, "Фиксированный порог пришлось бы подбирать заново для каждого кадра.",
             x=T.MARGIN, y=6.36, w=7.35)

slide = deck.slide("Каналы HSV", "Цвет и яркость разнесены по разным каналам", kicker="Пункт 03")
bottom = deck.picture(slide, hsv_panels, T.MARGIN, 2.12, w=11.53).bottom_in
bottom = deck.metrics(slide, [
    (f"{hue.mean():.1f}", "среднее H — тон"),
    (f"{sat.mean():.1f}", "среднее S — насыщенность"),
    (f"{val.mean():.1f}", "среднее V — яркость"),
], x=T.MARGIN, y=bottom + 0.30, w=11.53, h=1.06)
deck.caption(slide, "Высокая средняя насыщенность означает, что кадр состоит из чистых, "
                    "а не приглушённых цветов.", x=T.MARGIN, y=bottom + 0.28, w=11.53)

slide = deck.slide("Правка пикселей и сохранение", "Срез NumPy и запись файла в порядке BGR",
                   kicker="Пункты 04–05")
deck.picture(slide, edge_panels, T.MARGIN, 2.15, w=8.05)
deck.metrics(slide, [
    (f"{height * 20}", "изменённых пикселей"),
    ("побитово", "файл прочитан обратно и совпал"),
], x=9.15, y=2.15, w=3.28, h=2.34, vertical=True)
deck.code(slide, "edited = image.copy()\n"
                 "edited[:, :20] = (255, 0, 0)      # BGR: синий\n"
                 "cv2.imwrite('output/edited_blue_edge.png', edited)",
          x=T.MARGIN, y=5.20, w=8.05)

slide = deck.slide("Каналы B, G, R и их статистика", "Числовой портрет жёлтого фона",
                   kicker="Задача 07")
bottom = deck.picture(slide, channel_panels, T.MARGIN, 2.12, w=11.53).bottom_in
deck.table(slide, ["Канал", "Среднее", "Медиана", "Ст. отклонение"], [
    ["B", f"{blue.mean():.2f}", f"{np.median(blue):.1f}", f"{blue.std():.2f}"],
    ["G", f"{green.mean():.2f}", f"{np.median(green):.1f}", f"{green.std():.2f}"],
    ["R", f"{red.mean():.2f}", f"{np.median(red):.1f}", f"{red.std():.2f}"],
], x=T.MARGIN, y=bottom + 0.34, w=5.4, widths=[0.16, 0.28, 0.28, 0.28])
deck.note(slide,
          "Низкое среднее синего при высоких средних зелёного и красного — это и есть "
          "жёлтый: в RGB он складывается из красного и зелёного при почти нулевом синем.",
          x=6.85, y=bottom + 0.28, w=5.58, size=T.BODY_SM)

slide = deck.slide("Выделение фона по цвету", "Один диапазон тона вместо трёх диапазонов BGR",
                   kicker="Задача 08")
chart_bottom = deck.picture(slide, hue_chart, T.MARGIN, 2.15, w=5.62).bottom_in
deck.picture(slide, mask_panels, 6.81, 2.15, w=5.62)
deck.code(slide, "yellow_mask = cv2.inRange(hsv,\n"
                 "    np.array([26, 80, 80]), np.array([35, 255, 255]))",
          x=6.81, y=4.02, w=5.62)
deck.metrics(slide, [
    (f"H = {dominant_hue}", f"доминирующий тон, ≈ {dominant_hue * 2}° на цветовом круге"),
    (f"{yellow_share:.2f} %", "площади кадра занимает фон"),
    ("26 … 35", "нижняя граница поднята: у шлема H ≈ 19…25"),
], x=T.MARGIN, y=chart_bottom + 0.30, w=11.53, h=1.06)

slide = deck.slide("Сравнение цветовых моделей", "Канал яркости у всех моделей один и тот же",
                   kicker="Задача 09")
bottom = deck.picture(slide, model_panels, T.MARGIN, 2.12, w=11.53).bottom_in
bottom = deck.metrics(slide, [
    (f"{diff.max()}", "максимум |Y − gray|"),
    (f"{diff.mean():.4f}", "средняя разница"),
    (f"{(diff == 0).mean() * 100:.2f} %", "пикселей совпали точно"),
], x=T.MARGIN, y=bottom + 0.30, w=11.53, h=1.06)
deck.caption(slide, "Канал Y модели YCrCb и полутоновое изображение считаются по одной формуле — "
                    "расхождение возникает только из-за целочисленного округления.",
             x=T.MARGIN, y=bottom + 0.28, w=11.53)

slide = deck.slide("Стикер без фона", "Маска фона переиспользуется как прозрачность",
                   kicker="Задача 10")
deck.picture(slide, alpha_panels, T.MARGIN, 2.15, w=8.05)
deck.metrics(slide, [
    (f"{opaque_share:.2f} %", "непрозрачных пикселей"),
    ("(563, 1000, 4)", "чтение с IMREAD_UNCHANGED"),
    ("(563, 1000, 3)", "без флага — альфа теряется"),
], x=9.15, y=2.15, w=3.28, h=3.55, vertical=True)
deck.code(slide, "alpha = cv2.bitwise_not(yellow_mask)\n"
                 "rgba = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)\n"
                 "rgba[:, :, 3] = alpha",
          x=T.MARGIN, y=5.24, w=8.05)

# Вопросы и итоги
slide = deck.slide("Контрольные вопросы", "Ответы по материалу работы", kicker="Защита")
deck.qa(slide, [
    ("Что такое пиксель?",
     "Наименьший элемент растра. В цветном изображении — вектор из трёх uint8, "
     "в OpenCV в порядке B, G, R."),
    ("Чем RGB отличается от HSV?",
     "RGB смешивает три излучения, и при смене освещения меняются все три канала. "
     "HSV отделяет тон от яркости, поэтому цвет задаётся одним каналом."),
    ("Как получить отдельный канал?",
     "cv2.split(image) либо срез image[:, :, 0] — срез быстрее, так как не копирует "
     "данные. Обратно — cv2.merge."),
    ("Как перевести в градации серого?",
     "cv2.cvtColor(image, cv2.COLOR_BGR2GRAY); внутри считается "
     "Y = 0.299R + 0.587G + 0.114B."),
    ("Что такое альфа-канал?",
     "Четвёртый канал непрозрачности 0…255. Поддерживают PNG и TIFF, но не JPEG; "
     "читается только с флагом IMREAD_UNCHANGED."),
], x=T.MARGIN, y=2.20, w=11.53, columns=2, size=13.5, avail_h=4.4)

slide = deck.slide("Итоги", "Что выполнено в ноутбуке", kicker="Выводы")
bottom = deck.checklist(slide, [
    f"Изображение прочитано как массив ({height}, {width}, {channels}) типа uint8",
    f"Метод Оцу выбрал порог {otsu_threshold:.0f}; ручной расчёт дисперсии дал то же значение",
    "Выведены каналы H, S, V и подтверждена высокая насыщенность кадра",
    f"Полоса шириной 20 пикселей закрашена синим — {height * 20} пикселей — и сохранена",
    f"Фон выделен одним диапазоном тона: {yellow_share:.2f} % площади",
    f"Канал Y и полутоновое изображение совпали на {(diff == 0).mean() * 100:.2f} % пикселей",
    f"Получен PNG с альфа-каналом: непрозрачны {opaque_share:.2f} % пикселей",
], x=T.MARGIN, y=2.15, w=11.53, size=16.5)
deck.text(slide,
          "Изображение — обычный числовой массив, а цветовые модели — разные системы "
          "координат для одного и того же цвета: выбор модели определяет, насколько "
          "простой окажется задача.",
          x=T.MARGIN, y=bottom + 0.42, w=11.53, h=0.8,
          size=T.BODY, color=T.PRIMARY, bold=True)

path, count = deck.save(OUTPUT)
print(f"saved: {path} ({count} слайдов)")
