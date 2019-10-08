from PIL import Image
import numpy as np
a = Image.open('/Users/qiuhaonan/Downloads/gtFine_trainvaltest/new_candidates/p9/p94/lindau_000011_000019_gtFine_instanceIds.png')
b = np.array(a, dtype=np.int32)
c = Image.open('/Users/qiuhaonan/Downloads/gtFine_trainvaltest/new_candidates/p9/p94/lindau_000011_000019_gtFine_labelIds.png')
d = np.array(c, dtype=np.int32)
e = b[519,1117]
print(e)
f = d[374,1189]
# f = b[539,747]
print(f)

id = e
# id_target = f
#
ys,xs = np.where(b==id)
ymin, ymax, xmin, xmax = \
                    ys.min(), ys.max(), xs.min(), xs.max()

print(xmin.item(), ymin.item(), xmax.item(), ymax.item())

ys,xs = np.where(b==id)
for i in range(len(ys)):
    if b[ys[i],xs[i]] != id:
        print(ys[i],xs[i])
    # b[ys[i], xs[i]] = b[ys[i], xs[i]] + 50
    # if d[ys[i],xs[i]] != id and d[ys[i],xs[i]] != id + 1 and d[ys[i],xs[i]] != id - 1:
    b[ys[i], xs[i]] = 21
    d[ys[i], xs[i]] = 21
    # if xs[i] > 990:
    #     print(ys[i],xs[i])

ys,xs = np.where(d==21)
ymin, ymax, xmin, xmax = \
                    ys.min(), ys.max(), xs.min(), xs.max()

print(xmin.item(), ',', ymin.item(), ',', xmax.item(), ',', ymax.item())
#
#
#
#
# ys,xs = np.where(b==f)
# ymin, ymax, xmin, xmax = \
#                     ys.min(), ys.max(), xs.min(), xs.max()
#
# print(xmin.item(), ymin.item(), xmax.item(), ymax.item())
#
# ys,xs = np.where(b==f)
# for i in range(len(ys)):
#     if b[ys[i],xs[i]] != f:
#         print(ys[i],xs[i])
#     # b[ys[i], xs[i]] = b[ys[i], xs[i]] + 50
#     if d[ys[i],xs[i]] == f:
#         b[ys[i], xs[i]] = b[ys[i], xs[i]] + 50
#     # if xs[i] > 990:
#     #     print(ys[i],xs[i])
#
# ys,xs = np.where(b==f+50)
# ymin, ymax, xmin, xmax = \
#                     ys.min(), ys.max(), xs.min(), xs.max()
#
# print(xmin.item(), ',', ymin.item(), ',', xmax.item(), ',', ymax.item())

x = Image.fromarray(b)
x.save('/Users/qiuhaonan/Downloads/gtFine_trainvaltest/new_candidates/p9/p94/lindau_000011_000019_gtFine_instanceIds1.png')

print(np.min(d))
y = Image.fromarray(d.astype(np.uint8))
y.save('/Users/qiuhaonan/Downloads/gtFine_trainvaltest/new_candidates/p9/p94/lindau_000011_000019_gtFine_labelIds1.png')