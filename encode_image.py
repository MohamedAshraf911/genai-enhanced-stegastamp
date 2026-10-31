import glob
import os
from PIL import Image,ImageOps
import numpy as np
from tf_compat import tf
from ecc import create_bch

SERVING_TAG = 'serve'
DEFAULT_SERVING_SIGNATURE = 'serving_default'

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str)
    parser.add_argument('--image', type=str, default=None)
    parser.add_argument('--images_dir', type=str, default=None)
    parser.add_argument('--save_dir', type=str, default=None)
    parser.add_argument('--secret', type=str, default='Stega!!')
    args = parser.parse_args()

    if args.image is not None:
        files_list = [args.image]
    elif args.images_dir is not None:
        files_list = glob.glob(os.path.join(args.images_dir, '*'))
    else:
        print('Missing input image')
        return

    sess = tf.InteractiveSession(graph=tf.Graph())

    model = tf.saved_model.loader.load(sess, [SERVING_TAG], args.model)

    input_secret_name = model.signature_def[DEFAULT_SERVING_SIGNATURE].inputs['secret'].name
    input_image_name = model.signature_def[DEFAULT_SERVING_SIGNATURE].inputs['image'].name
    input_secret = tf.get_default_graph().get_tensor_by_name(input_secret_name)
    input_image = tf.get_default_graph().get_tensor_by_name(input_image_name)

    output_stegastamp_name = model.signature_def[DEFAULT_SERVING_SIGNATURE].outputs['stegastamp'].name
    output_residual_name = model.signature_def[DEFAULT_SERVING_SIGNATURE].outputs['residual'].name
    output_stegastamp = tf.get_default_graph().get_tensor_by_name(output_stegastamp_name)
    output_residual = tf.get_default_graph().get_tensor_by_name(output_residual_name)

    width = 400
    height = 400

    bch = create_bch()

    if len(args.secret) > 7:
        print('Error: Can only encode 56bits (7 characters) with ECC')
        return

    data = bytearray(args.secret + ' '*(7-len(args.secret)), 'utf-8')
    ecc = bch.encode(data)
    packet = data + ecc

    packet_binary = ''.join(format(x, '08b') for x in packet)
    secret = [int(x) for x in packet_binary]
    secret.extend([0,0,0,0])

    if args.save_dir is not None:
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)
        size = (width, height)
        for filename in files_list:
            image = Image.open(filename).convert("RGB")
            image = np.array(ImageOps.fit(image,size),dtype=np.float32)
            image /= 255.

            feed_dict = {input_secret:[secret],
                         input_image:[image]}

            hidden_img, residual = sess.run([output_stegastamp, output_residual],feed_dict=feed_dict)

            rescaled = (hidden_img[0] * 255).astype(np.uint8)
            raw_img = (image * 255).astype(np.uint8)
            residual = residual[0]+.5

            residual = (residual * 255).astype(np.uint8)

            save_name = os.path.splitext(os.path.basename(filename))[0]

            im = Image.fromarray(np.array(rescaled))
            im.save(os.path.join(args.save_dir, save_name+'_hidden.png'))

            im = Image.fromarray(np.squeeze(np.array(residual)))
            im.save(os.path.join(args.save_dir, save_name+'_residual.png'))

if __name__ == "__main__":
    main()
