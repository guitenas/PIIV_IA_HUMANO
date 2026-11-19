"""Gera icone.ico (balão azul/vermelho com '?') para o executável e o instalador. Uso: python gerar_icone.py"""
import os

from PIL import Image, ImageDraw, ImageFont

AQUI = os.path.dirname(os.path.abspath(__file__))
S = 256
img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
d = ImageDraw.Draw(img)
d.rounded_rectangle([8, 8, S - 8, S - 8], radius=56, fill="#2a78d6")
d.polygon([(S // 2, 8), (S - 8, 8), (S - 8, S - 8), (S // 2, S - 8)], fill="#e34948")
d.rounded_rectangle([8, 8, S - 8, S - 8], radius=56, outline="#ffffff", width=6)
try:
    fonte = ImageFont.truetype("segoeuib.ttf", 170)
except OSError:
    fonte = ImageFont.load_default()
d.text((S // 2, S // 2 + 6), "?", font=fonte, fill="#ffffff", anchor="mm")
img.save(os.path.join(AQUI, "icone.ico"), sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
print("icone.ico gerado")
