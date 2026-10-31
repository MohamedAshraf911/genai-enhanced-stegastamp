import glob
import os
from PIL import Image, ImageOps
import numpy as np
from tf_compat import tf
from ecc import create_bch, decode_inplace

SERVING_TAG = 'serve'
DEFAULT_SERVING_SIGNATURE = 'serving_default'

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str)
    parser.add_argument('--image', type=str, default=None)
    parser.add_argument('--images_dir', type=str, default=None)
    parser.add_argument('--secret_size', type=int, default=100)
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

    input_image_name = model.signature_def[DEFAULT_SERVING_SIGNATURE].inputs['image'].name
    input_image = tf.get_default_graph().get_tensor_by_name(input_image_name)

    output_secret_name = model.signature_def[DEFAULT_SERVING_SIGNATURE].outputs['decoded'].name
    output_secret = tf.get_default_graph().get_tensor_by_name(output_secret_name)

    bch = create_bch()

    for filename in files_list:
        image = Image.open(filename).convert("RGB")
        image = np.array(ImageOps.fit(image,(400, 400)),dtype=np.float32)
        image /= 255.

        feed_dict = {input_image:[image]}

        secret = sess.run([output_secret],feed_dict=feed_dict)[0][0]

        packet_binary = "".join([str(int(bit)) for bit in secret[:96]])
        packet = bytes(int(packet_binary[i : i + 8], 2) for i in range(0, len(packet_binary), 8))
        packet = bytearray(packet)

        data, ecc = packet[:-bch.ecc_bytes], packet[-bch.ecc_bytes:]

        bitflips = decode_inplace(bch, data, ecc)

        if bitflips != -1:
            try:
                code = data.decode("utf-8")
            except UnicodeDecodeError:
                print(filename, 'Failed to decode (invalid UTF-8 after BCH correction)')
                continue
            print(filename, code)
            continue
        print(filename, 'Failed to decode')


if __name__ == "__main__":
    main()
