
from min_dalle import MinDalle
import torch
from IPython.display import display



model = MinDalle(
    models_root='./pretrained',
    dtype=torch.float32,
    device='cpu',
    is_mega=True, 
    is_reusable=True
)

image = model.generate_image(
    text='Nuclear explosion',
    seed=-1,
    grid_size=4,
    is_seamless=False,
    temperature=1,
    top_k=256,
    supercondition_factor=32,
    is_verbose=False
)

display(image)
image.save('mi_imagen_guardada.jpg')