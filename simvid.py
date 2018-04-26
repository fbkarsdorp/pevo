import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import sys
import numpy as np
from models import NeutralModel, ConformistModel, AntiConformistModel, ExemplarModel

from moviepy.video.io.bindings import mplfig_to_npimage
import moviepy.editor as mpy
import icons

icons = list(icons.icons.values())
FONTAWESOME_FILE = 'font/FontAwesome.otf'

if sys.argv[1] == "neutral":
    model = NeutralModel(N=64, mu=0.01, yield_population=True)
elif sys.argv[1] == "conformist":
    model = ConformistModel(N=100, mu=0.01, yield_population=True)
elif sys.argv[1] == "anti-conformist":
    model = AntiConformistModel(N=100, mu=0.01, yield_population=True)
elif sys.argv[1] == "exemplar":
    model = ExemplarModel(N=100, mu=0.01, yield_population=True)
else:
    raise ValueError("model unknown...")


model_run = model.run()
print(model.max_traits)
f = plt.figure(figsize=(8, 8))
ax = plt.subplot(aspect='equal')
traits = next(model_run).reshape(8, 8)
im = ax.imshow(np.ones((8, 8)), cmap="Greys")
for (i, j), z in np.ndenumerate(traits):
    ax.text(j, i, icons[z], ha='center', va='center', fontproperties=fm.FontProperties(fname=FONTAWESOME_FILE, size=20), color="#3498db")
plt.axis('tight')
plt.axis('off')

def make_frame_mpl(t):
    traits = next(model_run)
    im.set_data(np.ones((8, 8)))
    for i, trait in enumerate(traits):
        ax.texts[i].set_text(icons[trait])
    return mplfig_to_npimage(f)


animation = mpy.VideoClip(make_frame_mpl,
                          duration=model.T - 50)
animation.write_videofile("images/%s_matrix.mp4" % sys.argv[1], fps=10)



