"""Презентация-отчёт по лабораторной работе 6 (вариант 9).

Обучение здесь не повторяется: числа и матрицы ошибок взяты из фактического
запуска ноутбука, а иллюстрации перерисованы в оформлении презентации.
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
ASSETS = ROOT / ".presentation_assets"
ASSETS.mkdir(exist_ok=True)
OUTPUT = ROOT / "Лабораторная_6_вариант_9.pptx"
SAMPLES_PNG = ASSETS / "nb_samples.png"

CLASSES = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
           "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

TRAIN_SIZE, TEST_SIZE, EPOCHS, BATCH = 10000, 2000, 2, 128

# --- фактические результаты запуска ноутбука -------------------------------
MODELS = [
    ("Linear SVM", 0.8190, 5.50),
    ("Базовая CNN", 0.8255, 4.36),
    ("CNN + L2", 0.7840, 4.41),
    ("CNN + Dropout", 0.8080, 4.35),
]

CURVES = {                      # test accuracy по эпохам
    "Базовая CNN": [0.7140, 0.8255],
    "CNN + L2": [0.7705, 0.7840],
    "CNN + Dropout": [0.7735, 0.8080],
}

# precision, recall, f1, support по классам для Linear SVM
SVM_REPORT = [
    (0.80, 0.75, 0.77, 200), (0.94, 0.95, 0.94, 203), (0.72, 0.74, 0.73, 214),
    (0.79, 0.77, 0.78, 190), (0.73, 0.73, 0.73, 219), (0.93, 0.91, 0.92, 195),
    (0.58, 0.59, 0.58, 197), (0.88, 0.91, 0.89, 200), (0.90, 0.91, 0.91, 194),
    (0.95, 0.95, 0.95, 188),
]

CONFUSION_L2 = np.array([
    [141, 1, 1, 10, 1, 0, 46, 0, 0, 0],
    [0, 194, 0, 6, 0, 0, 3, 0, 0, 0],
    [1, 0, 137, 0, 8, 0, 68, 0, 0, 0],
    [3, 3, 0, 158, 0, 0, 26, 0, 0, 0],
    [0, 0, 26, 8, 64, 0, 121, 0, 0, 0],
    [0, 0, 0, 0, 0, 177, 1, 16, 0, 1],
    [12, 0, 8, 4, 1, 1, 171, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 190, 1, 8],
    [0, 0, 1, 2, 1, 2, 27, 0, 161, 0],
    [0, 0, 0, 0, 0, 2, 0, 11, 0, 175],
])
CONFUSION_DROPOUT = np.array([
    [163, 0, 0, 23, 2, 1, 9, 0, 2, 0],
    [0, 190, 0, 13, 0, 0, 0, 0, 0, 0],
    [2, 0, 151, 5, 35, 0, 20, 0, 1, 0],
    [4, 2, 0, 177, 1, 0, 5, 0, 1, 0],
    [0, 1, 28, 18, 158, 0, 13, 0, 1, 0],
    [0, 0, 0, 1, 0, 179, 0, 13, 0, 2],
    [55, 0, 35, 8, 39, 1, 57, 0, 2, 0],
    [0, 0, 0, 0, 0, 5, 0, 173, 1, 21],
    [1, 1, 0, 1, 0, 1, 1, 0, 189, 0],
    [0, 0, 0, 0, 0, 3, 0, 6, 0, 179],
])

assert CONFUSION_L2.sum() == TEST_SIZE and CONFUSION_DROPOUT.sum() == TEST_SIZE
assert abs(np.trace(CONFUSION_L2) / TEST_SIZE - 0.7840) < 1e-9
assert abs(np.trace(CONFUSION_DROPOUT) / TEST_SIZE - 0.8080) < 1e-9


def top_confusions(matrix, count=4):
    """Самые частые пары «истинный класс → предсказанный»."""
    pairs = [(matrix[i, j], i, j) for i in range(10) for j in range(10) if i != j]
    return sorted(pairs, reverse=True)[:count]


# --- примеры изображений из фигуры ноутбука --------------------------------
def sample_tiles():
    """Вырезает десять образцов из иллюстрации, сохранённой в ноутбуке."""
    figure = cv2.imread(str(SAMPLES_PNG), cv2.IMREAD_GRAYSCALE)
    labels = [["Ankle boot", "T-shirt/top", "T-shirt/top", "Dress", "T-shirt/top"],
              ["Pullover", "Sneaker", "Pullover", "Sandal", "Sandal"]]
    tiles = []
    for row, top in enumerate((31, 270)):
        for column, left in enumerate((10, 247, 484, 721, 958)):
            tile = figure[top:top + 217, left:left + 217]
            tiles.append((tile, labels[row][column]))
    return tiles


tiles = sample_tiles()
grid = np.vstack([np.hstack([tile for tile, _ in tiles[:5]]),
                  np.hstack([tile for tile, _ in tiles[5:]])])

# --- иллюстрации -----------------------------------------------------------
hero = F.rounded_png(ASSETS / "hero.png",
                     cv2.cvtColor(grid, cv2.COLOR_GRAY2RGB), placement_w=5.08)
sample_strip = F.panels(ASSETS / "samples.png", tiles, width=11.53, wspace=0.07)


def draw_results(path, width=11.53, height=3.05):
    fig = F.figure(width, height)
    names = [name for name, _, _ in MODELS]
    accuracy = [value for _, value, _ in MODELS]
    seconds = [value for _, _, value in MODELS]
    short = [name.replace("Базовая ", "").replace(" + ", "\n+ ").replace("Linear ", "")
             for name in names]

    ax = F.axes(fig, [0.055, 0.20, 0.40, 0.68])
    bars = ax.bar(short, accuracy, color=[T.SERIES[0], T.SERIES[0], T.SERIES[1], T.SERIES[1]],
                  width=0.58)
    for bar, value in zip(bars, accuracy):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.004, f"{value:.4f}",
                ha="center", fontsize=9.5, weight="bold", color="#" + T.INK)
    ax.set_ylim(0.74, 0.845)
    ax.set_ylabel("Accuracy на тесте")
    ax.set_title("Точность", fontsize=10)

    ax = F.axes(fig, [0.555, 0.20, 0.40, 0.68])
    bars = ax.bar(short, seconds, color=T.SERIES[2], width=0.58)
    for bar, value in zip(bars, seconds):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.08, f"{value:.2f} с",
                ha="center", fontsize=9.5, weight="bold", color="#" + T.INK)
    ax.set_ylim(0, 6.4)
    ax.set_ylabel("Время обучения, с")
    ax.set_title("Стоимость обучения", fontsize=10)
    return F.finish(fig, path)


results_chart = draw_results(ASSETS / "results.png")


def draw_curves(ax):
    epochs = [1, 2]
    for index, (name, values) in enumerate(CURVES.items()):
        ax.plot(epochs, values, "o-", color=T.SERIES[index], markersize=5, label=name)
    ax.axhline(MODELS[0][1], color=T.SERIES[3], linestyle="--", linewidth=1.4)
    ax.annotate("Linear SVM", xy=(1.02, MODELS[0][1]), xytext=(1.02, MODELS[0][1] + 0.006),
                color=T.SERIES[3], fontsize=9.5, weight="bold")
    ax.set_xticks(epochs)
    ax.set_xlabel("Эпоха")
    ax.set_ylabel("Accuracy на тесте")
    ax.set_xlim(0.9, 2.1)
    ax.legend(loc="lower right")


curves_chart = F.chart(ASSETS / "curves.png", 5.62, 3.15, draw_curves,
                       pad=(0.78, 0.52, 0.22, 0.26))


def draw_f1(ax):
    f1 = [row[2] for row in SVM_REPORT]
    order = np.argsort(f1)
    colors = [T.SERIES[1] if f1[i] < 0.75 else T.SERIES[0] for i in order]
    bars = ax.barh([CLASSES[i] for i in order], [f1[i] for i in order], color=colors,
                   height=0.62)
    for bar, index in zip(bars, order):
        ax.text(f1[index] + 0.008, bar.get_y() + bar.get_height() / 2, f"{f1[index]:.2f}",
                va="center", fontsize=9.5, color="#" + T.INK, weight="bold")
    ax.set_xlim(0, 1.08)
    ax.set_xlabel("F1-score, Linear SVM")


f1_chart = F.chart(ASSETS / "f1.png", 7.35, 4.05, draw_f1, pad=(1.15, 0.48, 0.16, 0.26))


def draw_confusions(path, width=11.53, height=4.20):
    fig = F.figure(width, height)
    ticks = np.arange(10)
    for column, (matrix, title) in enumerate([(CONFUSION_L2, "CNN + L2"),
                                              (CONFUSION_DROPOUT, "CNN + Dropout")]):
        ax = fig.add_axes([0.105 + column * 0.455, 0.12, 0.34, 0.78])
        ax.imshow(matrix, cmap="Blues", vmin=0, vmax=matrix.max())
        for i in range(10):
            for j in range(10):
                value = matrix[i, j]
                if value == 0:
                    continue
                ax.text(j, i, str(value), ha="center", va="center", fontsize=7.4,
                        color="white" if value > matrix.max() * 0.55 else "#" + T.MUTED)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xticklabels(ticks, fontsize=8)
        ax.set_yticklabels(ticks, fontsize=8)
        ax.set_xlabel("Предсказанный класс", fontsize=9)
        if column == 0:
            ax.set_ylabel("Истинный класс", fontsize=9)
        ax.set_title(title, fontsize=10.5)
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_visible(False)
    return F.finish(fig, path)


confusion_figure = draw_confusions(ASSETS / "confusion.png")

# --- слайды ----------------------------------------------------------------
deck = Deck("Лабораторная работа 6 · вариант 9")

deck.cover(
    kicker="Лабораторная работа 6 · вариант 9",
    title="Классификация\nизображений одежды",
    subtitle="Fashion-MNIST на PyTorch: SVM против CNN,\nL2-регуляризация, Dropout и матрицы ошибок",
    meta="Компьютерное зрение · PyTorch · scikit-learn · seaborn",
    image=hero,
)

slide = deck.slide("Задание варианта 9", "Пять пунктов", kicker="Постановка")
deck.steps(slide, [
    "Построить классификатор SVM для Fashion-MNIST",
    "Реализовать CNN и обучить её на тех же данных",
    "Сравнить время обучения и точность SVM и CNN",
    "Сравнить CNN с L2-регуляризацией и с Dropout",
    "Построить confusion matrix для двух CNN",
], x=T.MARGIN, y=2.25, w=7.55, row_h=0.70)
deck.metrics(slide, [
    (f"{TRAIN_SIZE} / {TEST_SIZE}", "изображений на обучение и тест"),
    (f"{EPOCHS} эпохи", f"на каждую CNN, батч {BATCH}"),
    ("CPU", "воспроизводимый учебный запуск"),
], x=8.86, y=2.20, w=3.57, h=3.62, vertical=True)

slide = deck.slide("Fashion-MNIST", "28 × 28 пикселей, оттенки серого, десять классов",
                   kicker="Данные")
bottom = deck.picture(slide, sample_strip, T.MARGIN, 2.12, w=11.53).bottom_in
bottom = deck.metrics(slide, [
    ("10 классов", "T-shirt/top · Trouser · Pullover · Dress · Coat"),
    ("28 × 28", "Sandal · Shirt · Sneaker · Bag · Ankle boot"),
], x=T.MARGIN, y=bottom + 0.30, w=11.53, h=1.02)
deck.note(slide,
          "Ноутбук скачивает IDX-архивы с зеркала Fashion-MNIST и распаковывает их "
          "через gzip, без torchvision: загрузка не зависит от системных расширений. "
          "Пиксели делятся на 255, метки — целые числа от 0 до 9.",
          x=T.MARGIN, y=bottom + 0.28, w=11.53, bar=True, size=T.BODY_SM)

slide = deck.slide("Две модели и две регуляризации", "Что именно сравнивается",
                   kicker="Теория")
bottom = deck.columns(slide, [
    ("Linear SVM",
     "Изображение разворачивается в вектор из 784 признаков, и модель ищет "
     "разделяющую гиперплоскость с максимальным зазором. Быстрый и понятный "
     "базовый результат."),
    ("CNN",
     "Свёртки сохраняют двумерную структуру и учатся находить края, текстуры "
     "и части объектов; MaxPool делает признаки устойчивее к небольшим сдвигам."),
    ("L2 против Dropout",
     "L2 добавляет к потерям штраф λ·Σw² и плавно ограничивает большие веса. "
     "Dropout при обучении случайно зануляет часть активаций, создавая ансамбль "
     "случайных подмоделей."),
], x=T.MARGIN, y=2.12, w=11.53)
deck.note(slide,
          "Обе регуляризованные модели имеют одинаковую архитектуру и одинаковый режим "
          "обучения, поэтому сравнивается именно способ регуляризации, а не сеть.",
          x=T.MARGIN, y=bottom + 0.32, w=11.53, bar=True, size=T.BODY)

slide = deck.slide("Архитектура сети", "Две свёртки, два пулинга, два полносвязных слоя",
                   kicker="Пункт 02")
deck.code(slide, "Conv2d(1 → 32, 3×3)  →  ReLU  →  MaxPool(2)\n"
                 "Conv2d(32 → 64, 3×3) →  ReLU  →  MaxPool(2)\n"
                 "Flatten → Linear(64·7·7 → 128) → ReLU → Linear(128 → 10)",
          x=T.MARGIN, y=2.20, w=7.55)
deck.metrics(slide, [
    ("CrossEntropyLoss", "принимает сырые logits — softmax не нужен"),
    ("Adam, lr = 1e-3", "оптимизатор и шаг обучения"),
    ("model.eval()", "перед подсчётом accuracy Dropout выключается"),
], x=8.86, y=2.20, w=3.57, h=3.62, vertical=True)
deck.code(slide, "for values, targets in train_loader:\n"
                 "    optimizer.zero_grad()\n"
                 "    loss = criterion(model(values), targets)\n"
                 "    loss.backward()\n"
                 "    optimizer.step()",
          x=T.MARGIN, y=3.80, w=7.55)

slide = deck.slide("Результаты эксперимента", "CPU, две эпохи, одинаковые подвыборки",
                   kicker="Пункт 03")
bottom = deck.picture(slide, results_chart, T.MARGIN, 2.12, w=11.53).bottom_in
deck.note(slide,
          f"Базовая CNN обошла SVM и по точности ({MODELS[1][1]:.4f} против "
          f"{MODELS[0][1]:.4f}), и по времени обучения ({MODELS[1][2]:.2f} с против "
          f"{MODELS[0][2]:.2f} с).",
          x=T.MARGIN, y=bottom + 0.28, w=11.53, bar=True, size=T.BODY_SM)

slide = deck.slide("Обучение по эпохам", "За две эпохи ни одна модель не вышла на плато",
                   kicker="Пункт 04")
bottom = deck.picture(slide, curves_chart, T.MARGIN, 2.15, w=5.62).bottom_in
deck.metrics(slide, [
    (f"{MODELS[2][1]:.4f}", "CNN + L2: weight_decay = 1e-4"),
    (f"{MODELS[3][1]:.4f}", "CNN + Dropout: p = 0.5"),
    (f"{MODELS[1][1]:.4f}", "базовая CNN без регуляризации"),
], x=6.81, y=2.15, w=5.62, h=3.15, vertical=True)
deck.note(slide,
          "Меньшая accuracy не означает, что регуляризация бесполезна: за две эпохи "
          "переобучение ещё не наступило, а штраф за большие веса и зануление "
          "активаций только замедляют подгонку.",
          x=T.MARGIN, y=bottom + 0.28, w=11.53, bar=True, size=T.BODY_SM)

slide = deck.slide("Разбор по классам", "Linear SVM: обувь и сумки даются легко, "
                                        "верхняя одежда — нет", kicker="Метрики")
deck.picture(slide, f1_chart, T.MARGIN, 2.15, w=7.35)
deck.metrics(slide, [
    (f"{SVM_REPORT[9][2]:.2f}", "лучший класс: Ankle boot"),
    (f"{SVM_REPORT[6][2]:.2f}", "худший класс: Shirt"),
    (f"{np.mean([r[2] for r in SVM_REPORT]):.2f}", "macro avg F1 по десяти классам"),
], x=8.62, y=2.15, w=3.81, h=3.55, vertical=True)
deck.caption(slide, "Shirt, T-shirt/top, Pullover и Coat имеют похожие силуэты, "
                    "и линейная граница по сырым пикселям их не разделяет.",
             x=T.MARGIN, y=6.34, w=11.53)

slide = deck.slide("Матрицы ошибок", "Строки — истинные классы, столбцы — предсказанные",
                   kicker="Пункт 05")
bottom = deck.picture(slide, confusion_figure, T.MARGIN, 2.06, w=11.53).bottom_in
deck.caption(slide, "0 T-shirt/top · 1 Trouser · 2 Pullover · 3 Dress · 4 Coat · "
                    "5 Sandal · 6 Shirt · 7 Sneaker · 8 Bag · 9 Ankle boot",
             x=T.MARGIN, y=bottom + 0.22, w=11.53)

slide = deck.slide("Что именно путается", "Обе сети ошибаются на одной и той же группе классов",
                   kicker="Пункт 05")
deck.table(slide, ["CNN + L2: истинный → предсказанный", "Ошибок"],
           [[f"{CLASSES[i]} → {CLASSES[j]}", str(int(value))]
            for value, i, j in top_confusions(CONFUSION_L2)],
           x=T.MARGIN, y=2.25, w=5.4, widths=[0.74, 0.26], row_h=0.46)
deck.table(slide, ["CNN + Dropout: истинный → предсказанный", "Ошибок"],
           [[f"{CLASSES[i]} → {CLASSES[j]}", str(int(value))]
            for value, i, j in top_confusions(CONFUSION_DROPOUT)],
           x=7.03, y=2.25, w=5.4, widths=[0.74, 0.26], row_h=0.46)
deck.note(slide,
          "Диагональ у обеих моделей плотная, а вне диагонали видна одна и та же "
          "группа: Shirt, T-shirt/top, Pullover и Coat. L2-модель сваливает "
          "спорные случаи в класс Shirt, Dropout-модель — наоборот, теряет сам Shirt. "
          "Обувь, сумки и брюки обе сети различают почти безошибочно.",
          x=T.MARGIN, y=4.62, w=11.53, bar=True, size=T.BODY)

slide = deck.slide("Контрольные вопросы", "Ответы по материалу работы", kicker="Защита")
deck.qa(slide, [
    ("Что такое задача классификации изображений?",
     "Определение класса, к которому относится входное изображение, — здесь вида "
     "одежды из десяти возможных."),
    ("Чем отличаются kNN, SVM и Random Forest?",
     "kNN хранит обучающие объекты и голосует по ближайшим; SVM строит границу "
     "с максимальным зазором; Random Forest усредняет предсказания множества "
     "деревьев решений."),
    ("В чём преимущества CNN?",
     "Она сама извлекает признаки, учитывает локальность и пространственную структуру "
     "и переиспользует веса свёрточных ядер — для изображений это работает лучше, "
     "чем плоский вектор пикселей."),
    ("Что такое confusion matrix?",
     "Таблица соответствий истинных и предсказанных классов: диагональ показывает "
     "правильные ответы, остальные клетки — типы ошибок."),
    ("Какие метрики используются кроме Accuracy?",
     "Precision, Recall, F1-score, ROC-AUC, а для многоклассовых задач ещё "
     "macro-, micro- и weighted-усреднения."),
], x=T.MARGIN, y=2.20, w=11.53, columns=2, size=13, avail_h=4.4)

slide = deck.slide("Итоги", "Что выполнено в ноутбуке", kicker="Выводы")
bottom = deck.checklist(slide, [
    f"Linear SVM на 784 признаках дал accuracy {MODELS[0][1]:.4f} за {MODELS[0][2]:.2f} с",
    f"CNN из двух свёрточных блоков — {MODELS[1][1]:.4f} за {MODELS[1][2]:.2f} с",
    f"CNN + L2 (weight_decay = 1e-4) — {MODELS[2][1]:.4f}",
    f"CNN + Dropout (p = 0.5) — {MODELS[3][1]:.4f}",
    "Построены и разобраны матрицы ошибок обеих регуляризованных сетей",
    "Ограничения запуска зафиксированы явно: CPU, 10 000 / 2 000, две эпохи",
], x=T.MARGIN, y=2.20, w=11.53, size=16.5)
deck.text(slide,
          "Accuracy показывает, сколько ошибок сделала модель, а confusion matrix — "
          "какие именно: обе сети путают одежду с похожим силуэтом и почти "
          "не ошибаются на обуви и сумках.",
          x=T.MARGIN, y=bottom + 0.42, w=11.53, h=0.8,
          size=T.BODY, color=T.PRIMARY, bold=True)

path, count = deck.save(OUTPUT)
print(f"saved: {path} ({count} слайдов)")
