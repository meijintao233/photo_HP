import numpy as np
from PIL import Image

L = np.asarray(Image.open('lam.jpg').convert('L')).astype('float')

depth = 10
grad = np.gradient(L)
grad_x,gray_y = grad
grad_x = grad_x*depth/100
grad_y = grad_x*depth/100

A = np.sqrt(grad_x**2+grad_y**2+1.)
uni_x = grad_x/A
uni_y = grad_y/A
uni_z = 1./A

vec_el = np.pi/2.2
vec_az = np.pi/4
dx = np.cos(vec_el)*np.cos(vec_az)
dy = np.cos(vec_el)*np.sin(vec_az)
dz = np.sin(vec_el)

g = 255*(dx*uni_x+dy*uni_y+dz*uni_z)
g = g.clip(0,255)

im = Image.fromarray(g.astype('uint8'))
im.save('lam_HP.jpg')