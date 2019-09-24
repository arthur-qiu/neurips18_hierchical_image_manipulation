from PIL import Image
import numpy as np
a = Image.open('/Users/qiuhaonan/Downloads/gtFine_trainvaltest/p3/p31/munster_000123_000019_gtFine_instanceIds4.png')
b = np.array(a, dtype=np.int32)
c = Image.open('/Users/qiuhaonan/Downloads/gtFine_trainvaltest/p3/p31/munster_000123_000019_gtFine_instanceIds.png')
d = np.array(c, dtype=np.int32)
print(b[540,644])
print(b[531,720])

id = 33001

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
#
# d = Image.fromarray(b)
# d.save('/Users/qiuhaonan/Downloads/gtFine_trainvaltest/p3/p31/munster_000123_000019_gtFine_instanceIds1.png')
#
ys,xs = np.where(b==id+50)
ymin, ymax, xmin, xmax = \
                    ys.min(), ys.max(), xs.min(), xs.max()

print(xmin.item(), ymin.item(), xmax.item(), ymax.item())