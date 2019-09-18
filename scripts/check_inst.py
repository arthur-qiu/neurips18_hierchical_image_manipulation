from PIL import Image
import numpy as np
a = Image.open('/Users/qiuhaonan/Downloads/gtFine_trainvaltest/p1/p11/munster_000014_000019_gtFine_instanceIds3.png')
b = np.array(a, dtype=np.int32)
print(b[480,980])
print(b[480,1180])

c = b[:,:1195]
ys,xs = np.where(b==26004)
ymin, ymax, xmin, xmax = \
                    ys.min(), ys.max(), xs.min(), xs.max()

print(xmin.item(), ymin.item(), xmax.item(), ymax.item())

ys,xs = np.where(c==26004)
for i in range(len(ys)):
    if b[ys[i],xs[i]] != 26004:
        print(ys[i],xs[i])
    b[ys[i], xs[i]] = b[ys[i], xs[i]] + 50

d = Image.fromarray(b)
d.save('/Users/qiuhaonan/Downloads/gtFine_trainvaltest/p1/p11/munster_000014_000019_gtFine_instanceIds1.png')

ys,xs = np.where(b==26054)
ymin, ymax, xmin, xmax = \
                    ys.min(), ys.max(), xs.min(), xs.max()

print(xmin.item(), ymin.item(), xmax.item(), ymax.item())