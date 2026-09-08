"""Дизайн-токены: единая палитра, типографика и сетка для всех презентаций.

Стиль — минималистичный Material Design: белая поверхность, один основной цвет,
светлые карточки-контейнеры со скруглением 12 pt и мягкой тенью, много воздуха.
"""

# --- Цвет -------------------------------------------------------------------
# Роли названы как в Material 3: surface / on-surface / primary / outline.
SURFACE = "FFFFFF"          # фон слайда
CARD = "F3F6FC"             # surface-container: карточки и подложки графиков
CARD_STRONG = "E7EEFA"      # surface-container-high: акцентные карточки
PRIMARY = "0B57D0"          # основной цвет
PRIMARY_CONTAINER = "D7E3FC"
ON_PRIMARY_CONTAINER = "0A2E6B"
INK = "15171C"              # on-surface: основной текст
MUTED = "5B6070"            # on-surface-variant: пояснения и подписи
OUTLINE = "C9D0DE"          # тонкие разделители
ACCENT = "B3541E"           # tertiary: единственный тёплый акцент
POSITIVE = "146B3A"

# Палитра для графиков: основной, тёплый акцент, приглушённый серый.
SERIES = ["#0B57D0", "#B3541E", "#8A90A0", "#146B3A", "#6B4FA0"]

# --- Шрифты -----------------------------------------------------------------
FONT = "Helvetica Neue"
FONT_MONO = "Menlo"

# --- Типографика (pt) -------------------------------------------------------
DISPLAY = 42
TITLE = 27
SUBTITLE = 13.5
KICKER = 10
BODY_LG = 18
BODY = 16
BODY_SM = 13.5
CAPTION = 10.5
METRIC = 25
METRIC_LABEL = 10
CODE = 13

# --- Сетка (дюймы) ----------------------------------------------------------
SLIDE_W = 13.333
SLIDE_H = 7.5
MARGIN = 0.9
CONTENT_W = SLIDE_W - 2 * MARGIN

KICKER_Y = 0.56
TITLE_Y = 0.80
SUBTITLE_Y = 1.42
BODY_TOP = 2.02          # верх контента, когда есть подзаголовок
BODY_TOP_BARE = 1.72     # верх контента без подзаголовка
BODY_BOTTOM = 6.72
FOOTER_Y = 6.94

RADIUS = 0.14            # скругление карточек, дюймы
GAP = 0.26               # стандартный зазор между элементами
