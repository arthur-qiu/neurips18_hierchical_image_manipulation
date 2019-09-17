from PIL import Image
import numpy as np
a = Image.open('/Users/qiuhaonan/Downloads/gtFine_trainvaltest/pick1/pair11/munster_000090_000019_gtFine_instanceIds3.png')
b = np.array(a, dtype=np.int32)
print(b[481,886])
print(b[465,742])

c = b[:,928:]
ys,xs = np.where(b==26000)
ymin, ymax, xmin, xmax = \
                    ys.min(), ys.max(), xs.min(), xs.max()

print(xmin.item(), ymin.item(), xmax.item(), ymax.item())

# ys,xs = np.where(c==26004)
# for i in range(len(ys)):
#     if b[ys[i],xs[i]+928] != 26004:
#         print(ys[i],xs[i])
#     b[ys[i], xs[i]+928] = b[ys[i], xs[i]+928] + 50
#
# d = Image.fromarray(b)
# d.save('/Users/qiuhaonan/Downloads/gtFine_trainvaltest/pick2/pair21/frankfurt_000001_023369_gtFine_instanceIds1.png')
#
# ys,xs = np.where(b==26054)
# ymin, ymax, xmin, xmax = \
#                     ys.min(), ys.max(), xs.min(), xs.max()
#
# print(xmin.item(), ymin.item(), xmax.item(), ymax.item())