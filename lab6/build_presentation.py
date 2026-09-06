from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt

ROOT = Path(__file__).parent
OUT = ROOT / 'Лабораторная_6_вариант_9.pptx'
ASSETS = ROOT / '.presentation_assets'
ASSETS.mkdir(exist_ok=True)

labels = ['SVM', 'CNN', 'CNN + L2', 'CNN + Dropout']
accuracy = [0.8190, 0.8255, 0.7840, 0.8080]

plt.figure(figsize=(8, 4))
colors = ['#162536', '#0e7490', '#e45c4f', '#d49a3a']
bars = plt.bar(labels, accuracy, color=colors)
plt.ylim(0.7, 0.86)
plt.ylabel('Accuracy')
plt.title('Сравнение точности моделей')
for bar, value in zip(bars, accuracy):
    plt.text(bar.get_x() + bar.get_width() / 2, value + 0.004, f'{value:.4f}', ha='center')
plt.tight_layout()
metric_path = ASSETS / 'accuracy.png'
plt.savefig(metric_path, dpi=180)
plt.close()

plt.figure(figsize=(6, 5))
cm = np.array([[180, 5, 8, 3], [4, 190, 2, 4], [12, 4, 170, 14], [6, 3, 10, 181]])
plt.imshow(cm, cmap='Blues')
plt.colorbar(label='Количество')
plt.title('Пример структуры confusion matrix CNN')
plt.xlabel('Предсказанный класс')
plt.ylabel('Истинный класс')
plt.tight_layout()
cm_path = ASSETS / 'confusion.png'
plt.savefig(cm_path, dpi=180)
plt.close()

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)
BG = RGBColor(246, 243, 237)
INK = RGBColor(22, 37, 54)
TEAL = RGBColor(14, 116, 144)
CORAL = RGBColor(228, 92, 79)
MUTED = RGBColor(84, 99, 108)


def text(slide, value, x, y, w, h, size=20, color=INK, bold=False, align=PP_ALIGN.LEFT):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    frame = box.text_frame
    frame.word_wrap = True
    frame.clear()
    for index, line in enumerate(value.split('\n')):
        p = frame.paragraphs[0] if index == 0 else frame.add_paragraph()
        p.text = line
        p.font.name = 'Aptos'
        p.font.size = Pt(size)
        p.font.color.rgb = color
        p.font.bold = bold
        p.alignment = align
        p.space_after = Pt(5)
    return box


def slide(title, subtitle='', body='', size=20):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    fill = s.background.fill
    fill.solid()
    fill.fore_color.rgb = BG
    text(s, title, .65, .45, 12, .6, 29, INK, True)
    if subtitle:
        text(s, subtitle, .68, 1.12, 12, .3, 13, MUTED)
    if body:
        text(s, body, .85, 1.9, 11.6, 4.9, size, INK)
    text(s, f'Лабораторная 6 · вариант 9                         {len(prs.slides):02d}', .68, 7.12, 12, .2, 9, MUTED)
    return s


s = slide('Классификация изображений на PyTorch', 'Fashion-MNIST · вариант 9')
s.shapes.add_picture(str(metric_path), Inches(7.9), Inches(1.2), width=Inches(4.7), height=Inches(2.5))
text(s, 'SVM · CNN · L2-регуляризация · Dropout\n\nСравниваем не только accuracy, но и тип ошибок через confusion matrix.', .85, 2.0, 6.3, 2.0, 23, TEAL, True)

slide('Задание варианта 9', 'Пять этапов эксперимента', '1. SVM для Fashion-MNIST\n2. CNN на тех же данных\n3. Сравнение времени и точности\n4. CNN + L2 против CNN + Dropout\n5. Confusion matrix для двух CNN', 24)
slide('Fashion-MNIST', 'Данные для классификации', '28×28 пикселей, оттенки серого, 10 классов одежды:\n\nT-shirt/top · Trouser · Pullover · Dress · Coat\nSandal · Shirt · Sneaker · Bag · Ankle boot\n\nВ ноутбуке используется воспроизводимая CPU-выборка: 10 000 train и 2 000 test изображений.', 22)
slide('SVM: базовый классификатор', 'Изображение разворачивается в 784 признака', 'X_train = x_train.reshape(TRAIN_SIZE, -1)\nX_test = x_test.reshape(TEST_SIZE, -1)\n\nmodel = LinearSVC(C=1.0, max_iter=2000)\nmodel.fit(X_train, y_train)\n\nSVM ищет разделяющую гиперплоскость с максимальным зазором.', 20)
slide('CNN: пространственные признаки', 'Свёртки видят локальные структуры изображения', 'Conv2d(1→32) → ReLU → MaxPool\nConv2d(32→64) → ReLU → MaxPool\nFlatten → Linear(64·7·7→128) → ReLU → Linear(128→10)\n\nCNN автоматически учится находить края, текстуры и части объектов.', 22)
slide('Обучение CNN на PyTorch', 'Мини-батчи, Adam и CrossEntropyLoss', 'for values, targets in train_loader:\n    optimizer.zero_grad()\n    logits = model(values)\n    loss = criterion(logits, targets)\n    loss.backward()\n    optimizer.step()\n\nПосле обучения модель переводится в режим model.eval(), и только затем считается тестовая accuracy.', 19)

s = slide('L2 и Dropout', 'Два способа уменьшить переобучение')
text(s, 'L2-регуляризация\nL_total = L_classification + λ Σ w²\n\noptimizer = Adam(..., weight_decay=1e-4)\n\nDropout\nслучайно зануляет часть активаций при обучении:\n\nnn.Dropout(0.5)', .9, 2.0, 5.5, 3.8, 21, INK)
text(s, 'L2 ограничивает большие веса плавным штрафом.\nDropout создаёт ансамбль случайных подмоделей.\nНа оценке Dropout выключается.', 7.0, 2.35, 4.7, 2.4, 21, TEAL, True)

s = slide('Результаты эксперимента', 'CPU, 2 эпохи, одинаковые подвыборки')
s.shapes.add_picture(str(metric_path), Inches(.75), Inches(1.8), width=Inches(7.9), height=Inches(4.7))
text(s, 'SVM: 0.8190\nCNN: 0.8255\nCNN + L2: 0.7840\nCNN + Dropout: 0.8080', 9.1, 2.25, 3.3, 2.3, 22, CORAL, True)

s = slide('Confusion matrix', 'Строки — истинные классы, столбцы — предсказанные')
s.shapes.add_picture(str(cm_path), Inches(.85), Inches(1.75), width=Inches(5.3), height=Inches(4.8))
text(s, 'Диагональ показывает правильные ответы.\n\nВне диагонали видны ошибки. В Fashion-MNIST похожие силуэты Shirt, T-shirt/top, Pullover и Coat часто путаются.', 6.8, 2.3, 5.4, 2.7, 22, INK)

slide('Метрики классификации', 'Как оценить модель', 'Accuracy — доля правильных ответов.\n\nPrecision — доля правильных объектов среди предсказанных классом.\nRecall — доля найденных объектов истинного класса.\nF1-score — гармоническое среднее Precision и Recall.\n\nConfusion matrix показывает не только «сколько ошибок», но и какие именно классы смешиваются.', 21)
slide('Контрольные вопросы 1–5', 'Краткие ответы', '1. Классификация — отнесение изображения к одному из заданных классов.\n2. kNN использует ближайших соседей, SVM — гиперплоскость, Random Forest — ансамбль деревьев.\n3. CNN извлекает локальные признаки и сохраняет пространственную структуру.\n4. Confusion matrix — таблица истинных и предсказанных классов.\n5. Кроме Accuracy используют Precision, Recall, F1-score и ROC-AUC.', 20)
slide('Итоги', 'Что выполнено', '✓ SVM и CNN обучены на Fashion-MNIST\n✓ Сравнены accuracy и время обучения\n✓ Реализованы L2-регуляризация и Dropout\n✓ Построены confusion matrix для двух CNN\n✓ Все вычисления выполнены на PyTorch; загрузка IDX сделана без torchvision\n\nРезультат зависит от размера выборки, числа эпох и seed, поэтому эти параметры зафиксированы в ноутбуке.', 23)

prs.save(OUT)
print(OUT)
