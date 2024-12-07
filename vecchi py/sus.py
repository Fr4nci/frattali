import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

# Parametri
lettera = 'α'  # Lettera greca da disegnare
frase = "Questa è una frase che forma la lettera greca "
font_path = "arial.ttf"  # Percorso al file del font

# Creare immagine di base
img_size = (800, 800)
image = Image.new("RGB", img_size, "white")
draw = ImageDraw.Draw(image)

# Caricare il font
font_size = 50
font = ImageFont.truetype(font_path, font_size)

# Scrivere la frase ripetuta
text = frase * 20
draw.multiline_text((50, 50), text, fill="black", font=font, spacing=10)

# Disegnare sopra la lettera greca
plt.figure(figsize=(8, 8))
plt.imshow(image)
plt.text(0.5, 0.5, lettera, fontsize=300, color="black", ha="center", va="center", alpha=0.8)
plt.axis("off")
plt.show()