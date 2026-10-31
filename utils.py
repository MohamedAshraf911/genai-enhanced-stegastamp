import cv2
import itertools
import numpy as np
import random
from tf_compat import tf


def projective_transform(images, transforms, interpolation='BILINEAR'):
    transformed = tf.raw_ops.ImageProjectiveTransformV3(
        images=images,
        transforms=transforms,
        output_shape=tf.shape(images)[1:3],
        fill_value=tf.constant(0.0, dtype=tf.float32),
        interpolation=interpolation,
        fill_mode='CONSTANT')
    transformed.set_shape(images.shape)
    return transformed


def spatial_transformer_network(input_fmap, theta, out_dims=None):
    batch_size = tf.shape(input_fmap)[0]
    height = tf.shape(input_fmap)[1]
    width = tf.shape(input_fmap)[2]
    if out_dims:
        output_height, output_width = out_dims[:2]
    else:
        output_height, output_width = height, width

    theta = tf.reshape(theta, [batch_size, 2, 3])
    x = tf.linspace(-1.0, 1.0, output_width)
    y = tf.linspace(-1.0, 1.0, output_height)
    x_grid, y_grid = tf.meshgrid(x, y)
    x_grid = tf.reshape(x_grid, [-1])
    y_grid = tf.reshape(y_grid, [-1])
    sampling_grid = tf.stack([x_grid, y_grid, tf.ones_like(x_grid)])
    sampling_grid = tf.tile(
        tf.expand_dims(sampling_grid, 0),
        tf.stack([batch_size, 1, 1]))
    sampling_grid = tf.cast(sampling_grid, tf.float32)
    batch_grids = tf.matmul(tf.cast(theta, tf.float32), sampling_grid)
    batch_grids = tf.reshape(
        batch_grids, [batch_size, 2, output_height, output_width])

    x = batch_grids[:, 0]
    y = batch_grids[:, 1]
    max_y = tf.cast(height - 1, tf.int32)
    max_x = tf.cast(width - 1, tf.int32)
    x = 0.5 * ((x + 1.0) * tf.cast(max_x - 1, tf.float32))
    y = 0.5 * ((y + 1.0) * tf.cast(max_y - 1, tf.float32))

    x0 = tf.cast(tf.floor(x), tf.int32)
    x1 = x0 + 1
    y0 = tf.cast(tf.floor(y), tf.int32)
    y1 = y0 + 1
    x0 = tf.clip_by_value(x0, 0, max_x)
    x1 = tf.clip_by_value(x1, 0, max_x)
    y0 = tf.clip_by_value(y0, 0, max_y)
    y1 = tf.clip_by_value(y1, 0, max_y)

    def get_pixel_value(x_coords, y_coords):
        indices_shape = tf.shape(x_coords)
        batch_indices = tf.reshape(
            tf.range(indices_shape[0]), [indices_shape[0], 1, 1])
        batch_indices = tf.tile(
            batch_indices, [1, indices_shape[1], indices_shape[2]])
        return tf.gather_nd(
            input_fmap,
            tf.stack([batch_indices, y_coords, x_coords], axis=3))

    top_left = get_pixel_value(x0, y0)
    bottom_left = get_pixel_value(x0, y1)
    top_right = get_pixel_value(x1, y0)
    bottom_right = get_pixel_value(x1, y1)
    x0f, x1f = tf.cast(x0, tf.float32), tf.cast(x1, tf.float32)
    y0f, y1f = tf.cast(y0, tf.float32), tf.cast(y1, tf.float32)
    weights = [
        tf.expand_dims((x1f - x) * (y1f - y), 3),
        tf.expand_dims((x1f - x) * (y - y0f), 3),
        tf.expand_dims((x - x0f) * (y1f - y), 3),
        tf.expand_dims((x - x0f) * (y - y0f), 3),
    ]
    return tf.add_n([
        weights[0] * top_left,
        weights[1] * bottom_left,
        weights[2] * top_right,
        weights[3] * bottom_right,
    ])


def random_blur_kernel(probs, N_blur, sigrange_gauss, sigrange_line, wmin_line):
    N = N_blur
    coords = tf.to_float(tf.stack(tf.meshgrid(tf.range(N_blur), tf.range(N_blur), indexing='ij'), -1)) - (.5 * (N-1))
    # coords = tf.to_float(coords)
    manhat = tf.reduce_sum(tf.abs(coords), -1)

    # nothing, default

    vals_nothing = tf.to_float(manhat < .5)

    # gauss

    sig_gauss = tf.random.uniform([], sigrange_gauss[0], sigrange_gauss[1])
    vals_gauss = tf.exp(-tf.reduce_sum(coords**2, -1)/2./sig_gauss**2)

    # line

    theta = tf.random_uniform([], 0, 2.*np.pi)
    v = tf.convert_to_tensor([tf.cos(theta), tf.sin(theta)])
    dists = tf.reduce_sum(coords * v, -1)

    sig_line = tf.random.uniform([], sigrange_line[0], sigrange_line[1])
    w_line = tf.random.uniform([], wmin_line, .5 * (N-1) + .1)

    vals_line = tf.exp(-dists**2/2./sig_line**2) * tf.to_float(manhat < w_line)

    t = tf.random_uniform([])
    vals = vals_nothing
    vals = tf.cond(t < probs[0]+probs[1], lambda : vals_line, lambda : vals)
    vals = tf.cond(t < probs[0], lambda : vals_gauss, lambda : vals)

    v = vals / tf.reduce_sum(vals)
    z = tf.zeros_like(v)
    f = tf.reshape(tf.stack([v,z,z, z,v,z, z,z,v],-1), [N,N,3,3])

    return f


def get_rand_transform_matrix(image_size, d, batch_size):
    Ms = np.zeros((batch_size, 2, 8))

    for i in range(batch_size):
        tl_x = random.uniform(-d, d)     # Top left corner, top
        tl_y = random.uniform(-d, d)    # Top left corner, left
        bl_x = random.uniform(-d, d)  # Bot left corner, bot
        bl_y = random.uniform(-d, d)    # Bot left corner, left
        tr_x = random.uniform(-d, d)     # Top right corner, top
        tr_y = random.uniform(-d, d)   # Top right corner, right
        br_x = random.uniform(-d, d)  # Bot right corner, bot
        br_y = random.uniform(-d, d)   # Bot right corner, right

        rect = np.array([
            [tl_x, tl_y],
            [tr_x + image_size, tr_y],
            [br_x + image_size, br_y + image_size],
            [bl_x, bl_y +  image_size]], dtype = "float32")

        dst = np.array([
            [0, 0],
            [image_size, 0],
            [image_size, image_size],
            [0, image_size]], dtype = "float32")

        M = cv2.getPerspectiveTransform(rect, dst)
        M_inv = np.linalg.inv(M)
        Ms[i,0,:] = M_inv.flatten()[:8]
        Ms[i,1,:] = M.flatten()[:8]
    return Ms

def get_rnd_brightness_tf(rnd_bri, rnd_hue, batch_size):
    rnd_hue = tf.random.uniform((batch_size,1,1,3), -rnd_hue, rnd_hue)
    rnd_brightness = tf.random.uniform((batch_size,1,1,1), -rnd_bri, rnd_bri)
    return rnd_hue + rnd_brightness


## Differentiable JPEG, Source - https://github.com/rshin/differentiable-jpeg/blob/master/jpeg-tensorflow.ipynb

# 1. RGB -> YCbCr
# https://en.wikipedia.org/wiki/YCbCr
def rgb_to_ycbcr(image):
  matrix = np.array(
      [[65.481, 128.553, 24.966], [-37.797, -74.203, 112.],
       [112., -93.786, -18.214]],
      dtype=np.float32).T / 255
  shift = [16., 128., 128.]

  result = tf.tensordot(image, matrix, axes=1) + shift
  result.set_shape(image.shape.as_list())
  return result


def rgb_to_ycbcr_jpeg(image):
  matrix = np.array(
      [[0.299, 0.587, 0.114], [-0.168736, -0.331264, 0.5],
       [0.5, -0.418688, -0.081312]],
      dtype=np.float32).T
  shift = [0., 128., 128.]

  result = tf.tensordot(image, matrix, axes=1) + shift
  result.set_shape(image.shape.as_list())
  return result


# 2. Chroma subsampling
def downsampling_420(image):
  # input: batch x height x width x 3
  # output: tuple of length 3
  #   y:  batch x height x width
  #   cb: batch x height/2 x width/2
  #   cr: batch x height/2 x width/2
  y, cb, cr = tf.split(image, 3, axis=3)
  cb = tf.nn.avg_pool(
      cb, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
  cr = tf.nn.avg_pool(
      cr, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
  return (tf.squeeze(
      y, axis=-1), tf.squeeze(
          cb, axis=-1), tf.squeeze(
              cr, axis=-1))


# 3. Block splitting
# From https://stackoverflow.com/questions/41564321/split-image-tensor-into-small-patches
def image_to_patches(image):
  # input: batch x h x w
  # output: batch x h*w/64 x h x w
  k = 8
  height, width = image.shape.as_list()[1:3]
  batch_size = tf.shape(image)[0]
  image_reshaped = tf.reshape(image, [batch_size, height // k, k, -1, k])
  image_transposed = tf.transpose(image_reshaped, [0, 1, 3, 2, 4])
  return tf.reshape(image_transposed, [batch_size, -1, k, k])


# 4. DCT
def dct_8x8_ref(image):
  image = image - 128
  result = np.zeros((8, 8), dtype=np.float32)
  for u, v in itertools.product(range(8), range(8)):
    value = 0
    for x, y in itertools.product(range(8), range(8)):
      value += image[x, y] * np.cos((2 * x + 1) * u * np.pi / 16) * np.cos(
          (2 * y + 1) * v * np.pi / 16)
    result[u, v] = value
  alpha = np.array([1. / np.sqrt(2)] + [1] * 7)
  scale = np.outer(alpha, alpha) * 0.25
  return result * scale


def dct_8x8(image):
  image = image - 128
  tensor = np.zeros((8, 8, 8, 8), dtype=np.float32)
  for x, y, u, v in itertools.product(range(8), repeat=4):
    tensor[x, y, u, v] = np.cos((2 * x + 1) * u * np.pi / 16) * np.cos(
        (2 * y + 1) * v * np.pi / 16)
  alpha = np.array([1. / np.sqrt(2)] + [1] * 7)
  scale = np.outer(alpha, alpha) * 0.25
  result = scale * tf.tensordot(image, tensor, axes=2)
  result.set_shape(image.shape.as_list())
  return result


# 5. Quantizaztion
y_table = np.array(
    [[16, 11, 10, 16, 24, 40, 51, 61], [12, 12, 14, 19, 26, 58, 60,
                                        55], [14, 13, 16, 24, 40, 57, 69, 56],
     [14, 17, 22, 29, 51, 87, 80, 62], [18, 22, 37, 56, 68, 109, 103,
                                        77], [24, 35, 55, 64, 81, 104, 113, 92],
     [49, 64, 78, 87, 103, 121, 120, 101], [72, 92, 95, 98, 112, 100, 103, 99]],
    dtype=np.float32).T
c_table = np.empty((8, 8), dtype=np.float32)
c_table.fill(99)
c_table[:4, :4] = np.array([[17, 18, 24, 47], [18, 21, 26, 66],
                            [24, 26, 56, 99], [47, 66, 99, 99]]).T


def y_quantize(image, rounding, factor=1):
  image = image / (y_table * factor)
  image = rounding(image)
  return image


def c_quantize(image, rounding, factor=1):
  image = image / (c_table * factor)
  image = rounding(image)
  return image


# -5. Dequantization
def y_dequantize(image, factor=1):
  return image * (y_table * factor)


def c_dequantize(image, factor=1):
  return image * (c_table * factor)


# -4. Inverse DCT
def idct_8x8_ref(image):
  alpha = np.array([1. / np.sqrt(2)] + [1] * 7)
  alpha = np.outer(alpha, alpha)
  image = image * alpha

  result = np.zeros((8, 8), dtype=np.float32)
  for u, v in itertools.product(range(8), range(8)):
    value = 0
    for x, y in itertools.product(range(8), range(8)):
      value += image[x, y] * np.cos((2 * u + 1) * x * np.pi / 16) * np.cos(
          (2 * v + 1) * y * np.pi / 16)
    result[u, v] = value
  return result * 0.25 + 128


def idct_8x8(image):
  alpha = np.array([1. / np.sqrt(2)] + [1] * 7)
  alpha = np.outer(alpha, alpha)
  image = image * alpha

  tensor = np.zeros((8, 8, 8, 8), dtype=np.float32)
  for x, y, u, v in itertools.product(range(8), repeat=4):
    tensor[x, y, u, v] = np.cos((2 * u + 1) * x * np.pi / 16) * np.cos(
        (2 * v + 1) * y * np.pi / 16)
  result = 0.25 * tf.tensordot(image, tensor, axes=2) + 128
  result.set_shape(image.shape.as_list())
  return result


# -3. Block joining
def patches_to_image(patches, height, width):
  # input: batch x h*w/64 x h x w
  # output: batch x h x w
  k = 8
  batch_size = tf.shape(patches)[0]
  image_reshaped = tf.reshape(patches,
                              [batch_size, height // k, width // k, k, k])
  image_transposed = tf.transpose(image_reshaped, [0, 1, 3, 2, 4])
  return tf.reshape(image_transposed, [batch_size, height, width])


# -2. Chroma upsampling
def upsampling_420(y, cb, cr):
  # input:
  #   y:  batch x height x width
  #   cb: batch x height/2 x width/2
  #   cr: batch x height/2 x width/2
  # output:
  #   image: batch x height x width x 3
  def repeat(x, k=2):
    height, width = x.shape.as_list()[1:3]
    x = tf.expand_dims(x, -1)
    x = tf.tile(x, [1, 1, k, k])
    x = tf.reshape(x, [-1, height * k, width * k])
    return x

  cb = repeat(cb)
  cr = repeat(cr)
  return tf.stack((y, cb, cr), axis=-1)


# -1. YCbCr -> RGB
def ycbcr_to_rgb(image):
  matrix = np.array(
      [[298.082, 0, 408.583], [298.082, -100.291, -208.120],
       [298.082, 516.412, 0]],
      dtype=np.float32).T / 256
  shift = [-222.921, 135.576, -276.836]

  result = tf.tensordot(image, matrix, axes=1) + shift
  result.set_shape(image.shape.as_list())
  return result


def ycbcr_to_rgb_jpeg(image):
  matrix = np.array(
      [[1., 0., 1.402], [1, -0.344136, -0.714136], [1, 1.772, 0]],
      dtype=np.float32).T
  shift = [0, -128, -128]

  result = tf.tensordot(image + shift, matrix, axes=1)
  result.set_shape(image.shape.as_list())
  return result

def diff_round(x):
  return tf.round(x) + (x - tf.round(x))**3

def round_only_at_0(x):
  cond = tf.cast(tf.abs(x) < 0.5, tf.float32)
  return cond * (x ** 3) + (1 - cond) * x

def jpeg_compress_decompress(image,
                             downsample_c=True,
                             rounding=diff_round,
                             factor=1):
  image *= 255
  height, width = image.shape.as_list()[1:3]
  orig_height, orig_width = height, width
  if height % 16 != 0 or width % 16 != 0:
    # Round up to next multiple of 16
    height = ((height - 1) // 16 + 1) * 16
    width = ((width - 1) // 16 + 1) * 16

    vpad = height - orig_height
    wpad = width - orig_width
    top = vpad // 2
    bottom = vpad - top
    left = wpad // 2
    right = wpad - left

    #image = tf.pad(image, [[0, 0], [top, bottom], [left, right], [0, 0]], 'SYMMETRIC')
    image = tf.pad(image, [[0, 0], [0, vpad], [0, wpad], [0, 0]], 'SYMMETRIC')

  # "Compression"
  image = rgb_to_ycbcr_jpeg(image)
  if downsample_c:
    y, cb, cr = downsampling_420(image)
  else:
    y, cb, cr = tf.split(image, 3, axis=3)
  components = {'y': y, 'cb': cb, 'cr': cr}
  for k in components.keys():
    comp = components[k]
    comp = image_to_patches(comp)
    comp = dct_8x8(comp)
    comp = c_quantize(comp, rounding,
                      factor) if k in ('cb', 'cr') else y_quantize(
                          comp, rounding, factor)
    components[k] = comp

  # "Decompression"
  for k in components.keys():
    comp = components[k]
    comp = c_dequantize(comp, factor) if k in ('cb', 'cr') else y_dequantize(
        comp, factor)
    comp = idct_8x8(comp)
    if k in ('cb', 'cr'):
      if downsample_c:
        comp = patches_to_image(comp, int(height/2), int(width/2))
      else:
        comp = patches_to_image(comp, height, width)
    else:
      comp = patches_to_image(comp, height, width)
    components[k] = comp

  y, cb, cr = components['y'], components['cb'], components['cr']
  if downsample_c:
    image = upsampling_420(y, cb, cr)
  else:
    image = tf.stack((y, cb, cr), axis=-1)
  image = ycbcr_to_rgb_jpeg(image)

  # Crop to original size
  if orig_height != height or orig_width != width:
    #image = image[:, top:-bottom, left:-right]
    image = image[:, :-vpad, :-wpad]

  # Hack: RGB -> YUV -> RGB sometimes results in incorrect values
  #    min_value = tf.minimum(tf.reduce_min(image), 0.)
  #    max_value = tf.maximum(tf.reduce_max(image), 255.)
  #    value_range = max_value - min_value
  #    image = 255 * (image - min_value) / value_range
  image = tf.minimum(255., tf.maximum(0., image))
  image /= 255

  return image

def quality_to_factor(quality):
    if quality < 50:
        quality = 5000. / quality
    else:
        quality = 200. - quality*2
    return quality / 100.
