import matplotlib
matplotlib.use('TkAgg')
import seaborn as sns
import sys
import numpy as np
from models import NeutralModel, ConformistModel, AntiConformistModel, ExemplarModel
from colormaps import viridis

from moviepy.video.io.bindings import mplfig_to_npimage
import moviepy.editor as mpy

if sys.argv[1] == "neutral":
    model = NeutralModel(N=100, mu=0.01, yield_population=True)
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
f = sns.plt.figure(figsize=(10, 10))
ax = sns.plt.subplot(aspect='equal')
traits = model_run.next().reshape(10, 10)
im = ax.imshow(traits, cmap=viridis)
for (i, j), z in np.ndenumerate(traits):
    ax.text(j, i, '%s' % z, ha='center', va='center')
sns.plt.axis('tight')
sns.plt.axis('off')

def make_frame_mpl(t):
    traits = model_run.next()
    im.set_data(traits.reshape(10, 10))
    for i, trait in enumerate(traits):
        ax.texts[i].set_text("%s" % trait)
    return mplfig_to_npimage(f)


animation = mpy.VideoClip(make_frame_mpl,
                          duration=model.T - 50)
animation.write_gif("images/%s_matrix.gif" % sys.argv[1], fps=10)


