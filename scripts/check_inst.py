from PIL import Image
import numpy as np
a = Image.open('/Users/qiuhaonan/Downloads/gtFine_trainvaltest/p2/munster_000123_000019_gtFine_instanceIds3.png')
b = np.array(a, dtype=np.int32)
# c = Image.open('/Users/qiuhaonan/Downloads/gtFine_trainvaltest/p2/munster_000123_000019_gtFine_instanceIds.png')
# d = np.array(c, dtype=np.int32)
print(b[465,880])
print(b[473,727])

id = 26000

c = b[:,:995]
ys,xs = np.where(b==id)
ymin, ymax, xmin, xmax = \
                    ys.min(), ys.max(), xs.min(), xs.max()

print(xmin.item(), ymin.item(), xmax.item(), ymax.item())

ys,xs = np.where(c==id)
for i in range(len(ys)):
    if b[ys[i],xs[i]] != id:
        print(ys[i],xs[i])
    b[ys[i], xs[i]] = b[ys[i], xs[i]] + 50
    # if d[ys[i],xs[i]] != 26001:
    #     b[ys[i], xs[i]] = b[ys[i], xs[i]] + 50
    # if xs[i] > 990:
    #     print(ys[i],xs[i])

d = Image.fromarray(b)
d.save('/Users/qiuhaonan/Downloads/gtFine_trainvaltest/p2/munster_000123_000019_gtFine_instanceIds1.png')

ys,xs = np.where(b==id+50)
ymin, ymax, xmin, xmax = \
                    ys.min(), ys.max(), xs.min(), xs.max()

print(xmin.item(), ymin.item(), xmax.item(), ymax.item())