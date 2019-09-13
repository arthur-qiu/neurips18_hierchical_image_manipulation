from PIL import Image
import numpy as np
a = Image.open('/Users/qiuhaonan/Downloads/gtFine_trainvaltest/semantic/munster_000014_000019_gtFine_instanceIds2.png')
b = np.array(a, dtype=np.int32)
print(b[480,950])
print(b[464,1185])

c = b[:,:1200]
ys,xs = np.where(c==26006)
ymin, ymax, xmin, xmax = \
                    ys.min(), ys.max(), xs.min(), xs.max()

for i in range(len(ys)):
    if b[ys[i],xs[i]] != 26006:
        print(ys[i],xs[i])
    b[ys[i], xs[i]] = b[ys[i], xs[i]] + 50
#
# d = Image.fromarray(b)
# d.save('/Users/qiuhaonan/Downloads/gtFine_trainvaltest/semantic/munster_000014_000019_gtFine_instanceIds1.png')

print(xmin.item(), ymin.item(), xmax.item(), ymax.item())