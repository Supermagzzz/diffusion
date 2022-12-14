from deepsvg.svglib.geom import Bbox
from deepsvg.svglib.svg import SVG
import torch
import pickle

MAX_CNT_LINES = 10
MAX_CNT_MOVES = 5


def check_image(svg):
    svg_tensor = svg.to_tensor()
    cnt_move_to = 0
    max_cnt_lines = 0
    cur_cnt_lines = 0
    for row in svg_tensor.data:
        if row[0] == 0:
            cnt_move_to += 1
            cur_cnt_lines = 0
        else:
            cur_cnt_lines += 1
        max_cnt_lines = max(max_cnt_lines, cur_cnt_lines)

    return max_cnt_lines <= MAX_CNT_LINES and cnt_move_to == MAX_CNT_MOVES


def prepare_svg(svg):
    global MAX_CNT_LINES
    answer = []
    svg_tensor = svg.to_tensor()

    def append_all(elems):
        for var in elems:
            answer[-1].append(var)

    index = 0
    cnt = MAX_CNT_LINES

    def expand(cnt):
        while cnt < MAX_CNT_LINES:
            real_answer = []
            real_answer.append(answer[-1][0])
            real_answer.append(answer[-1][1])
            for x in range(2, len(answer[-1]), 6):
                if cnt == MAX_CNT_LINES:
                    real_answer.append(answer[-1][x])
                    real_answer.append(answer[-1][x + 1])
                    real_answer.append(answer[-1][x + 2])
                    real_answer.append(answer[-1][x + 3])
                    real_answer.append(answer[-1][x + 4])
                    real_answer.append(answer[-1][x + 5])
                else:
                    cnt += 1
                    Ax, Ay = answer[-1][x - 2], answer[-1][x - 1]
                    Bx, By = answer[-1][x], answer[-1][x + 1]
                    Cx, Cy = answer[-1][x + 2], answer[-1][x + 3]
                    Dx, Dy = answer[-1][x + 4], answer[-1][x + 5]
                    Ex, Ey = (Ax + Bx) / 2, (Ay + By) / 2
                    Fx, Fy = (Bx + Cx) / 2, (By + Cy) / 2
                    Gx, Gy = (Cx + Dx) / 2, (Cy + Dy) / 2
                    Hx, Hy = (Ex + Fx) / 2, (Ey + Fy) / 2
                    Jx, Jy = (Fx + Gx) / 2, (Fy + Gy) / 2
                    Kx, Ky = (Hx + Jx) / 2, (Hy + Jy) / 2

                    real_answer.append(Ex); real_answer.append(Ey)
                    real_answer.append(Hx); real_answer.append(Hy)
                    real_answer.append(Kx); real_answer.append(Ky)
                    real_answer.append(Jx); real_answer.append(Jy)
                    real_answer.append(Gx); real_answer.append(Gy)
                    real_answer.append(Dx); real_answer.append(Dy)
            answer[-1] = real_answer

    for row in svg_tensor.data:
        if row[0] == 0:
            expand(cnt)
            answer.append([])
            append_all([row[-2], row[-1]])
            cnt = 0
        elif row[0] == 1:
            lastX, lastY = answer[-1][-2], answer[-1][-1]
            append_all([
                lastX + (row[-2] - lastX) / 3,
                lastY + (row[-1] - lastY) / 3,
                lastX + 2 * (row[-2] - lastX) / 3,
                lastY + 2 * (row[-1] - lastY) / 3,
                row[-2],
                row[-1]
            ])
            cnt += 1
        elif row[0] == 2:
            append_all([row[-6], row[-5], row[-4], row[-3], row[-2], row[-1]])
            cnt += 1
        index += 1
    expand(cnt)
    while len(answer) < MAX_CNT_MOVES:
        answer.append([0] * (MAX_CNT_LINES * 6 + 2))
    answer = torch.Tensor(answer)
    return answer

for i in range(0, 99508):
    print(i)
    with open('data/icons_tensor/' + str(i) + '.pkl', 'rb') as f:
        data = pickle.load(f)

    svg = SVG.from_tensors(data["tensors"][0], Bbox(300)).normalize().zoom(0.9).canonicalize().simplify_heuristic()
    if check_image(svg):
        tensor = prepare_svg(svg)
        tensor -= torch.sum(tensor) / tensor.shape[0] / tensor.shape[1]
        tensor /= torch.max(tensor) - torch.min(tensor)
        with open('data/tensors/' + str(i) + '.pkl', 'wb') as fout:
            pickle.dump(tensor, fout)
