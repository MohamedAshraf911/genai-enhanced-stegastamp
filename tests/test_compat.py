import unittest

import numpy as np

from ecc import create_bch, decode_inplace
from tf_compat import tf
import utils


class CompatibilityTests(unittest.TestCase):
    def test_bch_corrects_corrupted_message(self):
        bch = create_bch()
        original = bytearray(b"Hello  ")
        ecc = bytearray(bch.encode(original))
        damaged = bytearray(original)
        damaged[0] ^= 1

        self.assertEqual(decode_inplace(bch, damaged, ecc), 1)
        self.assertEqual(damaged, original)

    def test_projective_identity_and_spatial_transformer_gradient(self):
        with tf.Graph().as_default():
            image = tf.placeholder(tf.float32, [1, 8, 8, 3])
            theta = tf.placeholder(tf.float32, [1, 6])
            identity_transform = tf.constant(
                [[1., 0., 0., 0., 1., 0., 0., 0.]])
            projective = utils.projective_transform(image, identity_transform)
            spatial = utils.spatial_transformer_network(
                image, theta, [8, 8, 3])
            gradient = tf.gradients(tf.reduce_sum(spatial), theta)[0]
            sample = np.random.RandomState(0).rand(
                1, 8, 8, 3).astype(np.float32)
            identity_theta = np.array(
                [[1., 0., 0., 0., 1., 0.]], dtype=np.float32)

            with tf.Session() as session:
                projective_result, spatial_result, gradient_result = (
                    session.run(
                        [projective, spatial, gradient],
                        {image: sample, theta: identity_theta}))

        np.testing.assert_allclose(projective_result, sample, atol=1e-6)
        self.assertEqual(spatial_result.shape, sample.shape)
        self.assertTrue(np.isfinite(gradient_result).all())


if __name__ == "__main__":
    unittest.main()
