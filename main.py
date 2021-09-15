import cv2
import numpy as np

img = cv2.imread('melons.tif', 0)
len_img = img.shape[0]
len_each_channel = np.int(np.ceil(len_img / 3))
img_2 = np.zeros([len_each_channel, img.shape[1], 3])

if len_img % 3 == 0:
    img_2[:, :, 0] = img[0:len_each_channel, :]
    img_2[:, :, 1] = img[len_each_channel:2 * len_each_channel, :]
    img_2[:, :, 2] = img[2 * len_each_channel:, :]
if len_img % 3 == 1:
    img_2[1:, :, 0] = img[0:len_each_channel-1, :]
    img_2[1:, :, 1] = img[len_each_channel-1:2 * len_each_channel-2, :]
    img_2[:, :, 2] = img[2 * len_each_channel-2:, :]
if len_img % 3 == 2:
    img_2[1:, :, 0] = img[0:len_each_channel - 1, :]
    img_2[:, :, 1] = img[len_each_channel - 1:2 * len_each_channel - 1, :]
    img_2[:, :, 2] = img[2 * len_each_channel - 1:, :]

cv2.imwrite('main.jpg', img_2)
width = 450
height = 400
dim = (width, height)
resized = np.zeros([501, 551, 3])
resized[50:450, 50:500, :] = cv2.resize(img_2, dim, interpolation=cv2.INTER_AREA)

mse_bg = 1000000
ind_i_bg = 0
ind_j_bg = 0
mse_br = 1000000
ind_i_br = 0
ind_j_br = 0
rows, cols = resized.shape[:2]
for i in range(-50, 51):
    for j in range(-50, 51):
        M = np.float32([[1, 0, i], [0, 1, j]])
        dst = np.zeros([501, 551, 3])
        dst[:, :, 0] = resized[:, :, 0]
        dst[:, :, 1] = cv2.warpAffine(resized[:, :, 1], M, (cols, rows))
        err = dst[:, :, 1] - dst[:, :, 0]
        err = np.power(err, 2)
        m_err = np.sum(err[max(50, 50 + i):min(450, 450 + i)+1,
                       max(50, 50 + i):min(500, 500 + i)+1])
        length = (min(450, 450 + i) - max(50, 50 + i)+1) * \
                 (min(500, 500 + i) - max(50, 50 + i)+1)
        m_err = m_err / length
        if m_err < mse_bg:
            mse_bg = m_err
            ind_i_bg = i
            ind_j_bg = j

        dst[:, :, 2] = cv2.warpAffine(resized[:, :, 2], M, (cols, rows))
        err = dst[:, :, 2] - dst[:, :, 0]
        err = np.power(err, 2)
        m_err = np.sum(err[max(50, 50 + i):min(450, 450 + i),
                       max(50, 50 + i):min(500, 500 + i)])
        length = (min(450, 450 + i) - max(50, 50 + i)) * \
                 (min(500, 500 + i) - max(50, 50 + i))
        m_err = m_err / length
        if m_err < mse_br:
            mse_br = m_err
            ind_i_br = i
            ind_j_br = j

ind_i_bg = np.round(ind_i_bg * img_2.shape[0] / 400)
ind_i_br = np.round(ind_i_br * img_2.shape[0] / 400)
ind_j_bg = np.round(ind_j_bg * img_2.shape[1] / 450)
ind_j_br = np.round(ind_j_br * img_2.shape[1] / 450)

img_out = img_2
img_out[:, :, 0] = img_2[:, :, 0]
rows, cols = img_2.shape[:2]
M = np.float32([[1, 0, ind_i_bg], [0, 1, ind_j_bg]])
img_out[:, :, 1] = cv2.warpAffine(img_2[:, :, 1], M, (cols, rows))
M = np.float32([[1, 0, ind_i_br], [0, 1, ind_j_br]])
img_out[:, :, 2] = cv2.warpAffine(img_2[:, :, 2], M, (cols, rows))
cv2.imwrite('res04.jpg', img_out)

print("green_blue: ", ind_i_bg, ind_j_bg)
print("red_blue: ", ind_i_br, ind_j_br)
