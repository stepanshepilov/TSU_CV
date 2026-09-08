"""Отрисовка иллюстраций в том же стиле, что и слайды.

Главное соглашение: размер фигуры в дюймах равен размеру её места на слайде,
dpi фиксировано. Тогда кегль подписей внутри графика совпадает с кеглем текста
на слайде, а скругление углов получается одинаковым у всех картинок.
"""
import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from . import tokens as T  # noqa: E402

DPI = 200


def use_style():
    plt.rcParams.update({
        "font.family": T.FONT,
        "font.size": 10,
        "text.color": "#" + T.INK,
        "axes.facecolor": "#" + T.CARD,
        "figure.facecolor": "#" + T.CARD,
        "axes.edgecolor": "#" + T.OUTLINE,
        "axes.labelcolor": "#" + T.MUTED,
        "axes.labelsize": 9.5,
        "axes.titlesize": 11,
        "axes.titlecolor": "#" + T.INK,
        "axes.titleweight": "bold",
        "axes.titlepad": 8,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": "#" + T.OUTLINE,
        "grid.linewidth": 0.7,
        "grid.alpha": 0.9,
        "xtick.color": "#" + T.MUTED,
        "ytick.color": "#" + T.MUTED,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "xtick.major.size": 0,
        "ytick.major.size": 0,
        "legend.frameon": False,
        "legend.fontsize": 9.5,
        "lines.linewidth": 2.0,
        "savefig.facecolor": "#" + T.CARD,
    })


use_style()


def despine(ax, keep=("bottom", "left")):
    for side, spine in ax.spines.items():
        spine.set_visible(side in keep)
        spine.set_linewidth(0.8)


def _rounded_alpha(height, width, radius, ss=4):
    mask = np.zeros((height * ss, width * ss), np.uint8)
    r = max(int(radius * ss), 1)
    h, w = mask.shape
    cv2.rectangle(mask, (r, 0), (w - r, h), 255, -1)
    cv2.rectangle(mask, (0, r), (w, h - r), 255, -1)
    for centre in ((r, r), (w - r, r), (r, h - r), (w - r, h - r)):
        cv2.circle(mask, centre, r, 255, -1)
    return cv2.resize(mask, (width, height), interpolation=cv2.INTER_AREA)


def finish(fig, path, radius_in=T.RADIUS):
    """Сохраняет фигуру в PNG со скруглёнными прозрачными углами."""
    fig.canvas.draw()
    buffer = np.asarray(fig.canvas.buffer_rgba()).copy()
    height, width = buffer.shape[:2]
    alpha = _rounded_alpha(height, width, radius_in * DPI)
    buffer[:, :, 3] = (buffer[:, :, 3].astype(np.float32) * alpha / 255.0).astype(np.uint8)
    cv2.imwrite(str(path), cv2.cvtColor(buffer, cv2.COLOR_RGBA2BGRA))
    plt.close(fig)
    return path


def figure(width, height):
    """Фигура ровно того размера, которое она займёт на слайде."""
    return plt.figure(figsize=(width, height), dpi=DPI)


def axes(fig, rect, keep=("bottom", "left")):
    """Ось внутри фигуры: rect задаётся в долях фигуры."""
    ax = fig.add_axes(rect)
    despine(ax, keep)
    return ax


def chart(path, width, height, draw, pad=(0.62, 0.42, 0.30, 0.34)):
    """Одна ось с сеткой. pad — отступы слева/снизу/сверху/справа в дюймах."""
    fig = figure(width, height)
    left, bottom, top, right = pad
    ax = fig.add_axes([left / width, bottom / height,
                       1 - (left + right) / width, 1 - (bottom + top) / height])
    despine(ax)
    draw(ax)
    return finish(fig, path)


def panels_height(width, count, aspect, pad_top=0.30, wspace=0.05):
    """Высота ряда из count картинок, при которой они займут его целиком."""
    cell_in = (width * 0.94 - wspace * (count - 1)) / count
    return (pad_top + cell_in * aspect) / 0.94


def panels(path, items, width, height=None, cmap="gray", pad_top=0.30, wspace=0.05):
    """Ряд изображений с короткими подписями сверху."""
    count = len(items)
    if height is None:
        aspect = items[0][0].shape[0] / items[0][0].shape[1]
        height = panels_height(width, count, aspect, pad_top, wspace)
    fig = figure(width, height)
    gap = wspace / width
    cell = (1 - 0.03 * 2 - gap * (count - 1)) / count
    label_h = pad_top / height
    for index, (image, title) in enumerate(items):
        left = 0.03 + index * (cell + gap)
        ax = fig.add_axes([left, 0.035, cell, 1 - label_h - 0.06])
        ax.imshow(image, cmap=None if image.ndim == 3 else cmap,
                  vmin=None if image.ndim == 3 else 0,
                  vmax=None if image.ndim == 3 else 255)
        ax.set_axis_off()
        if title:
            fig.text(left + cell / 2, 1 - label_h * 0.72, title, ha="center", va="center",
                     fontsize=10, color="#" + T.MUTED)
    return finish(fig, path)


def image_card(path, image, width, height=None, cmap=None):
    """Одно изображение без подписей, вписанное в карточку."""
    aspect = image.shape[0] / image.shape[1]
    height = height or width * aspect
    fig = figure(width, height)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(image, cmap=cmap if image.ndim == 2 else None)
    ax.set_axis_off()
    return finish(fig, path)


def rounded_png(path, image_rgb, placement_w=None, radius_px=None):
    """Скругляет обычную картинку; radius подбирается под ширину места на слайде."""
    height, width = image_rgb.shape[:2]
    if radius_px is None:
        radius_px = int(T.RADIUS * width / placement_w) if placement_w else int(width * 0.02)
    alpha = _rounded_alpha(height, width, radius_px)
    rgba = np.dstack([cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR), alpha])
    cv2.imwrite(str(path), rgba)
    return path


def bar_labels(ax, bars, values, fmt="{:.2f}", dy=0.01, size=10):
    span = max(values) - min(min(values), 0)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + span * dy, fmt.format(value),
                ha="center", va="bottom", fontsize=size, color="#" + T.INK, weight="bold")
