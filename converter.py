import imageio
import matplotlib.pyplot as plt

filename = './Gym-brake-delay-MBRL-DATS/out/brake_MBRL-2022-04-22_16.09.44.mp4'
vid = imageio.get_reader(filename,  'ffmpeg')
frame = []
for num, image in enumerate(vid.iter_data()):
    frame.append(image[410:635,555:870])

# plt.imshow(frame[60])
# plt.show()
# plt.imshow(frame[920])
# plt.show()
# print(num)

imageio.mimsave('./Gym-brake-delay-MBRL-DATS/brake-delay-MBRL-DATS.gif',frame[60:915],'GIF', duration=0.02)