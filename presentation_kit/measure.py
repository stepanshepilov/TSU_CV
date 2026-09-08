"""Оценка размеров текста, чтобы блоки на слайде не переполнялись.

python-pptx не умеет переносить строки заранее, поэтому ширина строк меряется
тем же шрифтом, каким текст будет отрисован, и по ней считается высота блока.
"""
from functools import lru_cache

from PIL import ImageFont

from . import tokens as T

_FACES = {
    (T.FONT, False): ("/System/Library/Fonts/HelveticaNeue.ttc", 0),
    (T.FONT, True): ("/System/Library/Fonts/HelveticaNeue.ttc", 1),
    (T.FONT_MONO, False): ("/System/Library/Fonts/Menlo.ttc", 0),
    (T.FONT_MONO, True): ("/System/Library/Fonts/Menlo.ttc", 1),
}
_SCALE = 8  # меряем в 8× кегле, чтобы округление шрифтового движка не мешало


@lru_cache(maxsize=None)
def _font(name, bold, size_pt):
    path, index = _FACES.get((name, bold), _FACES[(T.FONT, bold)])
    return ImageFont.truetype(path, int(size_pt * _SCALE), index=index)


def width_pt(text, size, font=T.FONT, bold=False):
    """Ширина строки в пунктах."""
    return _font(font, bold, size).getlength(text) / _SCALE


def wrap(text, width_in, size, font=T.FONT, bold=False):
    """Разбивает абзац на строки так же, как это сделает PowerPoint."""
    limit = width_in * 72
    lines = []
    for paragraph in str(text).split("\n"):
        words = paragraph.split(" ")
        if not words:
            lines.append("")
            continue
        current = words[0]
        for word in words[1:]:
            candidate = f"{current} {word}"
            if width_pt(candidate, size, font, bold) <= limit:
                current = candidate
            else:
                lines.append(current)
                current = word
        lines.append(current)
    return lines


def line_count(text, width_in, size, font=T.FONT, bold=False):
    return len(wrap(text, width_in, size, font, bold))


def text_height(text, width_in, size, line_spacing=1.28, space_after=6,
                font=T.FONT, bold=False):
    """Высота текстового блока в дюймах."""
    paragraphs = str(text).split("\n")
    lines = sum(line_count(p, width_in, size, font, bold) for p in paragraphs)
    body = lines * size * 1.2 * line_spacing
    gaps = space_after * max(len(paragraphs) - 1, 0)
    return (body + gaps) / 72


def fits(text, width_in, height_in, size, line_spacing=1.28, space_after=6,
         font=T.FONT, bold=False):
    return text_height(text, width_in, size, line_spacing, space_after, font, bold) <= height_in
