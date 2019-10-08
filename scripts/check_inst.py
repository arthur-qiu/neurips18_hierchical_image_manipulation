from PIL import Image
import numpy as np
a = Image.open('/Users/qiuhaonan/Downloads/gtFine_trainvaltest/new_candidates/p93/frankfurt_000000_019607_gtFine_instanceIds3.png')
b = np.array(a, dtype=np.int32)
c = Image.open('/Users/qiuhaonan/Downloads/gtFine_trainvaltest/new_candidates/p93/frankfurt_000000_019607_gtFine_instanceIds.png')
d = np.array(c, dtype=np.int32)
e = b[422,330]
print(e)
f = b[414,153]
# f = b[539,747]
print(f)

id = e
id_target = f

ys,xs = np.where(b==id)
ymin, ymax, xmin, xmax = \
                    ys.min(), ys.max(), xs.min(), xs.max()

print(xmin.item(), ymin.item(), xmax.item(), ymax.item())

ys,xs = np.where(b==id)
for i in range(len(ys)):
    if b[ys[i],xs[i]] != id:
        print(ys[i],xs[i])
    # b[ys[i], xs[i]] = b[ys[i], xs[i]] + 50
    if d[ys[i],xs[i]] != id and d[ys[i],xs[i]] != id + 1 and d[ys[i],xs[i]] != id - 1:
        b[ys[i], xs[i]] = b[ys[i], xs[i]] + 50
    # if xs[i] > 990:
    #     print(ys[i],xs[i])

ys,xs = np.where(b==id+50)
ymin, ymax, xmin, xmax = \
                    ys.min(), ys.max(), xs.min(), xs.max()

print(xmin.item(), ',', ymin.item(), ',', xmax.item(), ',', ymax.item())




ys,xs = np.where(b==f)
ymin, ymax, xmin, xmax = \
                    ys.min(), ys.max(), xs.min(), xs.max()

print(xmin.item(), ymin.item(), xmax.item(), ymax.item())

ys,xs = np.where(b==f)
for i in range(len(ys)):
    if b[ys[i],xs[i]] != f:
        print(ys[i],xs[i])
    # b[ys[i], xs[i]] = b[ys[i], xs[i]] + 50
    if d[ys[i],xs[i]] == f:
        b[ys[i], xs[i]] = b[ys[i], xs[i]] + 50
    # if xs[i] > 990:
    #     print(ys[i],xs[i])

ys,xs = np.where(b==f+50)
ymin, ymax, xmin, xmax = \
                    ys.min(), ys.max(), xs.min(), xs.max()

print(xmin.item(), ',', ymin.item(), ',', xmax.item(), ',', ymax.item())



d = Image.fromarray(b)
d.save('/Users/qiuhaonan/Downloads/gtFine_trainvaltest/new_candidates/p93/frankfurt_000000_019607_gtFine_instanceIds1.png')