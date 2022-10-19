import numpy as np
import torch
from torch import nn

from dataset.dataloader import CustomImageDataset
from deepsvg.svglib.svg import SVG
from model import SimpleDenoiser
import pydiffvg

dataset = CustomImageDataset('../data/tensors')

def make_image(inp):
    # return el
    canvas_width, canvas_height = 50, 50
    el = (inp + 0.5)
    el *= int(0.8 * canvas_width)
    el += int(0.1 * canvas_width)
    pathes = []
    groups = []
    for j, row in enumerate(el):
        num_control_points = []
        points = []
        points.append(0)
        points.append(1)
        for i in range(2, row.shape[0], 6):
            num_control_points.append(2)
            points.append(i + 2)
            points.append(i + 3)
            points.append(i)
            points.append(i + 1)
            points.append(i + 4)
            points.append(i + 5)
        points = row[points].reshape(-1, 2)
        pathes.append(pydiffvg.Path(torch.Tensor(num_control_points), points, False))
    groups.append(pydiffvg.ShapeGroup(shape_ids=torch.tensor([0, 1, 2, 3, 4]), fill_color=None,
                                      stroke_color=torch.tensor([0, 0, 0, 1])))
    scene_args = pydiffvg.RenderFunction.serialize_scene(canvas_width, canvas_height, pathes, groups)
    render = pydiffvg.RenderFunction.apply
    img = render(canvas_width,  # width
                 canvas_height,  # height
                 2,  # num_samples_x
                 2,  # num_samples_y
                 1,  # seed
                 None,
                 *scene_args)
    return img

def run():
    def add_noise(tensor, mult):
        return torch.normal(0, 1, size=tensor.shape) * mult

    def make_svg(tensor):
        tensor -= torch.min(tensor) - 0.5
        tensor *= 10
        data = []
        for row in tensor:
            command = [-1] * 14
            command[0] = 0
            command[-2] = row[0]
            command[-1] = row[1]
            lastX, lastY = row[0], row[1]
            data.append(command)
            for i in range(2, row.shape[0], 6):
                command = [-1] * 14
                command[0] = 2
                command[-8] = lastX
                command[-7] = lastY
                command[-6] = row[i]
                command[-5] = row[i + 1]
                command[-4] = row[i + 2]
                command[-3] = row[i + 3]
                command[-2] = row[i + 4]
                command[-1] = row[i + 5]
                if (lastX - command[-2]) ** 2 + (lastY - command[-1]) ** 2 < 0.1:
                    continue
                lastX, lastY = command[-2], command[-1]
                data.append(command)
        return SVG.from_tensor(torch.Tensor(data))


    model = SimpleDenoiser()
    model.load_state_dict(torch.load('../model_weights'))
    model.eval()
    for i, img in enumerate(dataset):
        if i == 3:
            break
        noise = add_noise(img, 0.01)
        new_img = img + noise
        pred_noise = model(make_image(new_img), torch.Tensor(1))
        make_svg(new_img - pred_noise[0]).save_svg('../trash/' + 'test' + str(i) + '.svg')
        make_svg(img).save_svg('../trash/' + 'real' + str(i) + '.svg')
        make_svg(new_img).save_svg('../trash/' + 'inp' + str(i) + '.svg')
run()
