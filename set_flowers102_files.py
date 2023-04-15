import shutil, os
import numpy as np

data_path = 'data/flowers102'

try:
    shutil.rmtree(data_path)
    print("directory was removed successfully")
except OSError as x:
    print("An error occured: %s : %s" % (data_path, x.strerror))

os.mkdir(data_path)
print("Directory '% s' created" % data_path)

ORIGINAL_CLASS_NAMES = ["pink primrose", "hard-leaved pocket orchid", "canterbury bells",
    "sweet pea", "english marigold", "tiger lily", "moon orchid",
    "bird of paradise", "monkshood", "globe thistle", "snapdragon",
    "colt's foot", "king protea", "spear thistle", "yellow iris",
    "globe-flower", "purple coneflower", "peruvian lily", "balloon flower",
    "giant white arum lily", "fire lily", "pincushion flower", "fritillary",
    "red ginger", "grape hyacinth", "corn poppy", "prince of wales feathers",
    "stemless gentian", "artichoke", "sweet william", "carnation",
    "garden phlox", "love in the mist", "mexican aster", "alpine sea holly",
    "ruby-lipped cattleya", "cape flower", "great masterwort", "siam tulip",
    "lenten rose", "barbeton daisy", "daffodil", "sword lily", "poinsettia",
    "bolero deep blue", "wallflower", "marigold", "buttercup", "oxeye daisy",
    "common dandelion", "petunia", "wild pansy", "primula", "sunflower",
    "pelargonium", "bishop of llandaff", "gaura", "geranium", "orange dahlia",
    "pink-yellow dahlia?", "cautleya spicata", "japanese anemone",
    "black-eyed susan", "silverbush", "californian poppy", "osteospermum",
    "spring crocus", "bearded iris", "windflower", "tree poppy", "gazania",
    "azalea", "water lily", "rose", "thorn apple", "morning glory",
    "passion flower", "lotus", "toad lily", "anthurium", "frangipani",
    "clematis", "hibiscus", "columbine", "desert-rose", "tree mallow",
    "magnolia", "cyclamen", "watercress", "canna lily", "hippeastrum",
    "bee balm", "ball moss", "foxglove", "bougainvillea", "camellia", "mallow",
    "mexican petunia", "bromelia", "blanket flower", "trumpet creeper",
    "blackberry lily"]

CLASS_NAMES = []

for clazz in ORIGINAL_CLASS_NAMES:
    clazz = clazz.replace(" ", "_")
    CLASS_NAMES.append(clazz)
    path = os.path.join(data_path, clazz)
    os.mkdir(path)

import scipy.io
mat = scipy.io.loadmat('data/imagelabels.mat')

labels = mat["labels"][0] - 1

print(labels[0:4])
print(len(labels))
print(np.unique(labels))

files = [os.path.join(r,file) for r,d,f in os.walk("data/jpg") for file in f]

for f in files:
    file_index = int(f[f.rindex("_") + 1:-4]) - 1
    label_index = labels[file_index]
    label = CLASS_NAMES[label_index]
    file = f[f.rindex("_") + 1:]
    dest = os.path.join(data_path, label, file)
    # print(dest)
    shutil.copyfile(f, dest)
