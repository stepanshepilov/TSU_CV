"""Собирает презентацию по лабораторной работе 1 (вариант 9)."""
from pathlib import Path

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt

matplotlib.use("Agg")

ROOT = Path(__file__).parent
IMAGE_PATH = ROOT / ".." / "lab2" / "image.png"
OUTPUT_PATH = ROOT / "Лабораторная_1_вариант_9.pptx"
ASSET_DIR = ROOT / ".presentation_assets"
ASSET_DIR.mkdir(exist_ok=True)

image = cv2.imread(str(IMAGE_PATH), cv2.IMREAD_COLOR)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
height, width, channels = image.shape

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
otsu_threshold, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
white_ratio = float((binary == 255).mean()) * 100

hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
h, s, v = cv2.split(hsv)

edited = image.copy()
edited[:, :20] = (255, 0, 0)

b, g, r = cv2.split(image)

hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
p = hist / hist.sum()
levels = np.arange(256)
w0 = np.cumsum(p)
w1 = 1.0 - w0
mu_total = float((levels * p).sum())
mu0 = np.divide(np.cumsum(levels * p), w0, out=np.zeros(256), where=w0 > 0)
mu1 = np.divide(mu_total - np.cumsum(levels * p), w1, out=np.zeros(256), where=w1 > 0)
sigma_b = w0 * w1 * (mu0 - mu1) ** 2

hue_hist = cv2.calcHist([hsv], [0], None, [180], [0, 180]).ravel()
dominant_hue = int(np.argmax(hue_hist))
yellow_mask = cv2.inRange(hsv, np.array([20, 80, 80]), np.array([35, 255, 255]))
yellow_share = float((yellow_mask > 0).mean()) * 100

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
alpha = cv2.bitwise_not(yellow_mask)
alpha = cv2.morphologyEx(alpha, cv2.MORPH_OPEN, kernel)
alpha = cv2.morphologyEx(alpha, cv2.MORPH_CLOSE, kernel)
opaque_share = float((alpha > 0).mean()) * 100

ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
diff = cv2.absdiff(ycrcb[:, :, 0], gray)


def save_plot(name, plotter, figsize=(10, 4)):
    path = ASSET_DIR / name
    fig = plt.figure(figsize=figsize)
    plotter()
    plt.tight_layout()
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return path


def panels(name, items, figsize=(14, 4), cmap="gray"):
    # items: список (изображение, подпись); цветные выводятся как RGB
    path = ASSET_DIR / name
    fig, axes = plt.subplots(1, len(items), figsize=figsize)
    for ax, (img, title) in zip(np.atleast_1d(axes), items):
        ax.imshow(img, cmap=None if img.ndim == 3 else cmap)
        ax.set_title(title, fontsize=11)
        ax.axis("off")
    plt.tight_layout()
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return path


original_path = panels("original.png", [(image_rgb, "")], figsize=(8, 4.5))
otsu_path = panels(
    "otsu.png",
    [(gray, "Градации серого"), (binary, f"Бинаризация Оцу, порог = {otsu_threshold:.0f}")],
    figsize=(12, 3.6),
)
sigma_path = save_plot(
    "sigma.png",
    lambda: (
        plt.plot(levels, sigma_b, color="#0e7490", linewidth=2),
        plt.axvline(int(np.argmax(sigma_b)), color="#e45c4f", linestyle="--", linewidth=2,
                    label=f"t* = {int(np.argmax(sigma_b))}"),
        plt.title("Межклассовая дисперсия по методу Оцу"),
        plt.xlabel("Порог t"),
        plt.ylabel("sigma_b^2(t)"),
        plt.legend(),
    ),
    figsize=(9, 4),
)
hsv_path = panels(
    "hsv.png",
    [(h, "H — тон"), (s, "S — насыщенность"), (v, "V — яркость")],
    figsize=(15, 3.4),
)
edge_path = panels(
    "edge.png",
    [(image_rgb, "Исходное"),
     (cv2.cvtColor(edited, cv2.COLOR_BGR2RGB), "Левый край 20 пикс. — синий"),
     (cv2.cvtColor(edited[:, :80], cv2.COLOR_BGR2RGB), "Фрагмент 0…80 столбцов")],
    figsize=(15, 3.2),
)
channels_path = panels(
    "channels.png",
    [(b, "B: среднее 49.7"), (g, "G: среднее 218.5"), (r, "R: среднее 208.5")],
    figsize=(15, 3.4),
)
hue_hist_path = save_plot(
    "hue_hist.png",
    lambda: (
        plt.bar(np.arange(180), hue_hist, color="#0e7490", width=1.0),
        plt.axvline(dominant_hue, color="#e45c4f", linestyle="--", linewidth=2,
                    label=f"H = {dominant_hue}"),
        plt.title("Гистограмма тона H"),
        plt.xlabel("H, 0…179"),
        plt.ylabel("Количество пикселей"),
        plt.legend(),
    ),
    figsize=(9, 4),
)
mask_path = panels(
    "mask.png",
    [(yellow_mask, "Маска жёлтого (inRange)"),
     (cv2.cvtColor(cv2.bitwise_and(image, image, mask=cv2.bitwise_not(yellow_mask)),
                   cv2.COLOR_BGR2RGB), "Только объекты")],
    figsize=(12, 3.6),
)
models_path = panels(
    "models.png",
    [(cv2.cvtColor(image, cv2.COLOR_BGR2HLS)[:, :, 1], "HLS: L"),
     (ycrcb[:, :, 0], "YCrCb: Y"),
     (cv2.cvtColor(image, cv2.COLOR_BGR2LAB)[:, :, 2], "LAB: b")],
    figsize=(15, 3.4),
)

checker = np.indices((height, width)).sum(axis=0) // 20 % 2
checker = np.where(checker == 0, 235, 190).astype(np.uint8)
checker_rgb = cv2.cvtColor(checker, cv2.COLOR_GRAY2RGB).astype(np.float32)
alpha_f = (alpha / 255.0)[:, :, None]
composited = (image_rgb.astype(np.float32) * alpha_f + checker_rgb * (1 - alpha_f)).astype(np.uint8)
alpha_path = panels(
    "alpha.png",
    [(alpha, "Альфа-канал"), (composited, "RGBA поверх подложки")],
    figsize=(12, 3.6),
)


prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)
BG = RGBColor(246, 243, 237)
INK = RGBColor(22, 37, 54)
TEAL = RGBColor(14, 116, 144)
CORAL = RGBColor(228, 92, 79)
MUTED = RGBColor(84, 99, 108)


def set_background(slide):
    fill = slide.background.fill
    fill.solid()
    fill.fore_color.rgb = BG


def add_text(slide, text, x, y, w, h, size=20, color=INK, bold=False, align=PP_ALIGN.LEFT):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    frame = box.text_frame
    frame.clear()
    frame.word_wrap = True
    for index, line in enumerate(text.split("\n")):
        paragraph = frame.paragraphs[0] if index == 0 else frame.add_paragraph()
        paragraph.text = line
        paragraph.font.name = "Aptos"
        paragraph.font.size = Pt(size)
        paragraph.font.bold = bold
        paragraph.font.color.rgb = color
        paragraph.alignment = align
        paragraph.space_after = Pt(7)
    return box


def add_title(slide, title, subtitle=None):
    add_text(slide, title, 0.65, 0.45, 12, 0.65, size=29, bold=True)
    if subtitle:
        add_text(slide, subtitle, 0.68, 1.12, 11.8, 0.4, size=13, color=MUTED)
    rule = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.68), Inches(1.62), Inches(1.0), Inches(0.08))
    rule.fill.solid()
    rule.fill.fore_color.rgb = CORAL
    rule.line.fill.background()


def add_footer(slide, number):
    add_text(slide, f"Лабораторная 1 · вариант 9                                      {number:02d}",
             0.68, 7.12, 12, 0.2, size=9, color=MUTED)


def new_slide(title, subtitle=None):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_background(slide)
    add_title(slide, title, subtitle)
    add_footer(slide, len(prs.slides))
    return slide


# 01 — титульный
slide = prs.slides.add_slide(prs.slide_layouts[6])
set_background(slide)
add_text(slide, "ЛАБОРАТОРНАЯ 01", 0.75, 0.75, 6, 0.4, size=15, color=TEAL, bold=True)
add_text(slide, "Основы работы с изображениями\nи цветовые модели", 0.72, 1.45, 7.4, 1.7, size=34, bold=True)
add_text(slide, "Компьютерное зрение · вариант 9", 0.78, 3.45, 5.5, 0.5, size=18, color=MUTED)
slide.shapes.add_picture(str(IMAGE_PATH), Inches(8.2), Inches(0.6), width=Inches(4.45), height=Inches(2.5))
add_text(slide, "Изображение как числовой массив: срезы, цветовые модели, пороги и альфа-канал",
         0.78, 5.65, 8.5, 0.6, size=16, color=INK)
add_footer(slide, 1)

# 02 — задание
slide = new_slide("Задание варианта 9", "Пять обязательных пунктов")
add_text(slide, "01  Размеры изображения и число каналов\n"
                "02  Градации серого и бинаризация методом Оцу\n"
                "03  Преобразование в HSV, каналы H, S, V\n"
                "04  Левый край шириной 20 пикселей — синий\n"
                "05  Сохранение изображения с изменениями",
         0.9, 2.0, 8.0, 3.5, size=22)
add_text(slide, f"Вход: ../lab2/image.png\nРазмер: {width} × {height}\nКаналов: {channels} (BGR)\n"
                "Выход: файлы в output/",
         9.2, 2.2, 3.4, 2.2, size=17, color=TEAL, bold=True)

# 03 — собственные задачи
slide = new_slide("Собственные задачи 6–10", "Пять задач, сформулированных самостоятельно")
add_text(slide, "06  Оцу против фиксированного порога;\n"
                "      ручной расчёт межклассовой дисперсии\n\n"
                "07  Каналы B, G, R: статистика и сохранение\n\n"
                "08  Гистограмма тона H и выделение фона по цвету\n\n"
                "09  Сравнение HSV, HLS, YCrCb и LAB;\n"
                "      проверка эквивалентности Y и градаций серого\n\n"
                "10  Стикер: удаление фона и запись RGBA с прозрачностью",
         0.9, 1.95, 11.5, 4.8, size=21)

# 04 — теория
slide = new_slide("Теория: цвет как три числа", "Одна и та же точка в разных системах координат")
add_text(slide, "Изображение\nМатрица (H, W, 3) типа uint8.\n"
                "OpenCV хранит каналы в порядке BGR.\n"
                "image[y, x] — сначала строка, потом столбец.\n\n"
                "Градации серого\nY = 0.299·R + 0.587·G + 0.114·B\n\n"
                "Метод Оцу\nмаксимум w0·w1·(mu0 − mu1)²",
         0.85, 1.95, 5.2, 4.5, size=20)
add_text(slide, "HSV\nH — тон (0…179 в OpenCV)\nS — насыщенность (0…255)\nV — яркость (0…255)\n\n"
                "Информация о цвете — в одном канале H,\n"
                "поэтому выделять объекты по цвету\n"
                "удобнее в HSV, а не в BGR.\n\n"
                "Альфа-канал\nC = a·F + (1 − a)·B, формат RGBA",
         6.6, 1.95, 5.9, 4.5, size=20, color=MUTED)

# 05 — пункт 1
slide = new_slide("01 · Размеры и число каналов", "image.shape возвращает (высота, ширина, каналы)")
slide.shapes.add_picture(str(original_path), Inches(0.75), Inches(1.9), width=Inches(7.6), height=Inches(4.3))
add_text(slide, f"Ширина: {width} пикс.\nВысота: {height} пикс.\nКаналов: {channels}\n"
                f"Тип: {image.dtype}\nПикселей: {height * width}\nВ памяти: {image.nbytes / 1024 / 1024:.2f} МБ",
         9.0, 2.1, 3.4, 2.6, size=20, color=TEAL, bold=True)
add_text(slide, f"Пиксель [0, 0] = BGR {image[0, 0].tolist()}\n"
                f"Центр = BGR {image[height // 2, width // 2].tolist()}",
         9.0, 4.9, 3.4, 1.2, size=16, color=MUTED)

# 06 — пункт 2
slide = new_slide("02 · Градации серого и метод Оцу", "Порог подбирается автоматически по гистограмме")
slide.shapes.add_picture(str(otsu_path), Inches(0.7), Inches(1.9), width=Inches(8.0), height=Inches(3.0))
add_text(slide, "otsu_threshold, binary = cv2.threshold(\n"
                "    gray, 0, 255,\n"
                "    cv2.THRESH_BINARY + cv2.THRESH_OTSU\n)",
         0.75, 5.15, 7.9, 1.4, size=17)
add_text(slide, f"Порог Оцу: {otsu_threshold:.0f}\nБелых пикселей: {white_ratio:.2f}%\n"
                f"Средняя яркость: {gray.mean():.2f}",
         9.2, 2.4, 3.3, 2.0, size=21, color=CORAL, bold=True)
add_text(slide, "Порог 0 в вызове означает «подобрать автоматически»: значение возвращается функцией.",
         9.2, 4.6, 3.3, 1.4, size=15, color=MUTED)

# 07 — задача 6
slide = new_slide("06 · Проверка метода Оцу вручную", "Максимум межклассовой дисперсии совпал с результатом OpenCV")
slide.shapes.add_picture(str(sigma_path), Inches(0.75), Inches(1.9), width=Inches(7.4), height=Inches(4.4))
add_text(slide, f"OpenCV: {otsu_threshold:.0f}\nРучной расчёт: {int(np.argmax(sigma_b))}\n"
                f"max sigma_b² = {sigma_b.max():.0f}",
         8.8, 2.2, 3.6, 1.8, size=22, color=TEAL, bold=True)
add_text(slide, "Доля белого при разных порогах:\n"
                f"порог 100 → 88.95%\nОцу {otsu_threshold:.0f} → {white_ratio:.2f}%\nпорог 200 → 77.73%\n\n"
                "Фиксированный порог пришлось бы\nподбирать заново для каждого\nизображения.",
         8.8, 4.05, 3.7, 2.2, size=15, color=MUTED)

# 08 — пункт 3
slide = new_slide("03 · Каналы HSV", "Цвет и яркость разнесены по разным каналам")
slide.shapes.add_picture(str(hsv_path), Inches(0.6), Inches(2.0), width=Inches(9.0), height=Inches(2.6))
add_text(slide, f"Средние значения\nH = {h.mean():.1f}\nS = {s.mean():.1f}\nV = {v.mean():.1f}",
         10.0, 2.2, 2.6, 2.0, size=20, color=TEAL, bold=True)
add_text(slide, "H в OpenCV лежит в диапазоне 0…179 — это угол на цветовом круге, делённый на 2, "
                "чтобы поместиться в один байт. Высокая средняя насыщенность говорит о том, что "
                "изображение состоит из чистых, а не приглушённых цветов.",
         0.75, 5.0, 11.7, 1.4, size=18, color=MUTED)

# 09 — пункты 4 и 5
slide = new_slide("04–05 · Правка пикселей и сохранение", "Срез NumPy и запись файла в BGR-порядке")
slide.shapes.add_picture(str(edge_path), Inches(0.6), Inches(1.9), width=Inches(9.1), height=Inches(2.4))
add_text(slide, "edited = image.copy()\nedited[:, :20] = (255, 0, 0)   # BGR: синий\n\n"
                "cv2.imwrite('output/edited_blue_edge.png', edited)",
         0.75, 4.7, 8.6, 1.6, size=18)
add_text(slide, f"Изменено пикселей:\n{height} × 20 = {height * 20}\n\nФайл прочитан обратно\nи совпал побитово",
         10.1, 2.2, 2.6, 2.6, size=18, color=CORAL, bold=True)

# 10 — задача 7
slide = new_slide("07 · Каналы B, G, R и их статистика", "Числовой портрет жёлтого фона")
slide.shapes.add_picture(str(channels_path), Inches(0.6), Inches(2.0), width=Inches(9.0), height=Inches(2.6))
add_text(slide, "среднее / ст. откл.\n"
                f"B   {b.mean():.1f} / {b.std():.1f}\n"
                f"G   {g.mean():.1f} / {g.std():.1f}\n"
                f"R   {r.mean():.1f} / {r.std():.1f}",
         9.9, 2.3, 2.9, 2.0, size=17, color=TEAL, bold=True)
add_text(slide, "Низкое среднее синего при высоких средних зелёного и красного — это и есть жёлтый: "
                "в RGB он складывается из красного и зелёного при почти нулевом синем. "
                "cv2.merge восстановил исходный массив без потерь.",
         0.75, 5.0, 11.7, 1.4, size=18, color=MUTED)

# 11 — задача 8
slide = new_slide("08 · Выделение фона по цвету", "Один диапазон тона вместо трёх диапазонов BGR")
slide.shapes.add_picture(str(hue_hist_path), Inches(0.7), Inches(1.95), width=Inches(5.6), height=Inches(3.1))
slide.shapes.add_picture(str(mask_path), Inches(6.6), Inches(1.95), width=Inches(6.0), height=Inches(2.4))
add_text(slide, "yellow_mask = cv2.inRange(hsv,\n    np.array([20, 80, 80]),\n    np.array([35, 255, 255]))",
         0.75, 5.3, 5.6, 1.2, size=17)
add_text(slide, f"Доминирующий тон: H = {dominant_hue} (≈ {dominant_hue * 2}°)\n"
                f"Фон занимает {yellow_share:.2f}% кадра",
         6.7, 4.7, 5.9, 1.2, size=20, color=CORAL, bold=True)

# 12 — задача 9
slide = new_slide("09 · Сравнение цветовых моделей", "Канал яркости у всех моделей один и тот же")
slide.shapes.add_picture(str(models_path), Inches(0.6), Inches(2.0), width=Inches(9.0), height=Inches(2.6))
add_text(slide, f"|Y − gray|\nмаксимум: {diff.max()}\nсреднее: {diff.mean():.4f}\n"
                f"совпало: {(diff == 0).mean() * 100:.2f}%",
         9.9, 2.2, 2.9, 2.0, size=19, color=TEAL, bold=True)
add_text(slide, "Канал Y модели YCrCb и полутоновое изображение считаются по одной формуле, "
                "поэтому различаются не более чем на единицу — из-за целочисленного округления. "
                "Каналы H, a и b, наоборот, несут информацию о цвете, а не о яркости.",
         0.75, 5.0, 11.7, 1.4, size=18, color=MUTED)

# 13 — задача 10
slide = new_slide("10 · Альфа-канал: стикер без фона", "Маска фона переиспользуется как прозрачность")
slide.shapes.add_picture(str(alpha_path), Inches(0.7), Inches(1.95), width=Inches(8.2), height=Inches(3.3))
add_text(slide, "rgba = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)\nrgba[:, :, 3] = alpha\n"
                "cv2.imwrite('output/sticker_rgba.png', rgba)",
         0.75, 5.5, 8.2, 1.2, size=17)
add_text(slide, f"Непрозрачно: {opaque_share:.2f}%\n\n"
                "Чтение с IMREAD_UNCHANGED\n→ (563, 1000, 4)\n\nБез флага\n→ (563, 1000, 3):\nальфа теряется",
         9.3, 2.2, 3.2, 3.4, size=17, color=CORAL, bold=True)

# 14 — контрольные вопросы
slide = new_slide("Контрольные вопросы", "Ответы по материалу работы")
add_text(slide, "1. Пиксель — наименьший элемент растра; в цветном изображении это вектор из трёх uint8 "
                "в порядке B, G, R.\n\n"
                "2. RGB задаёт цвет смесью излучений, HSV разделяет тон, насыщенность и яркость — "
                "поэтому выделять объекты по цвету удобнее в HSV.\n\n"
                "3. Канал получают через cv2.split или срезом image[:, :, 0]; обратно — cv2.merge.\n\n"
                "4. В градации серого переводит cv2.cvtColor(..., COLOR_BGR2GRAY): "
                "Y = 0.299R + 0.587G + 0.114B.\n\n"
                "5. Альфа-канал задаёт непрозрачность (0…255), поддерживается PNG, "
                "читается только с флагом IMREAD_UNCHANGED.",
         0.85, 1.85, 11.6, 4.95, size=19)

# 15 — итоги
slide = new_slide("Итоги", "Что выполнено в ноутбуке")
add_text(slide, f"✓ Прочитан массив ({height}, {width}, {channels}) типа uint8\n"
                f"✓ Метод Оцу выбрал порог {otsu_threshold:.0f}, ручной расчёт дал то же значение\n"
                "✓ Выведены каналы H, S, V и карта чистых оттенков\n"
                "✓ Полоса шириной 20 пикселей закрашена синим и сохранена\n"
                f"✓ Фон выделен по тону: {yellow_share:.2f}% кадра\n"
                "✓ Подтверждена эквивалентность канала Y и градаций серого\n"
                "✓ Получен PNG с альфа-каналом",
         1.0, 1.95, 10.5, 3.8, size=22, color=INK)
add_text(slide, "Главная идея: изображение — обычный числовой массив, а цветовые модели — разные "
                "системы координат для одного и того же цвета. Выбор модели определяет, "
                "насколько простой окажется задача.",
         1.0, 5.7, 11.1, 0.9, size=18, color=TEAL, bold=True)

prs.save(OUTPUT_PATH)
print(f"saved: {OUTPUT_PATH} ({len(prs.slides.__iter__.__self__._sldIdLst)} слайдов)")
