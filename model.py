import tensorflow as tf
import tensorflow_probability as tfp
import einops
from layers import *


class TRNet(tf.keras.models.Model):
    def __init__(self,
                 n_filters: int,
                 n_enc_layers: int,
                 n_transform_layers: int,
                 n_heads: int,
                 n_mlp_filters: int,
                 dropout_rate: float,
                 upscale_rate: int
                 ):
        super(TRNet, self).__init__()
        self.n_filters = n_filters
        self.n_enc_layers = n_enc_layers
        self.n_transformer_layers = n_transform_layers
        self.n_heads = n_heads
        self.n_mlp_filters = n_mlp_filters
        self.dropout_rate = dropout_rate
        self.upscale_rate = upscale_rate

        self.encoder = Encoder(
            self.n_filters, self.n_enc_layers
        )
        self.transformer = Transformer(
            self.n_filters, self.n_transformer_layers, self.n_heads, self.n_mlp_filters, self.dropout_rate
        )
        self.decoder = Decoder(
            self.upscale_rate
        )

    def forward(self, x, mask=None, training=False):
        b, h, w, t = tf.shape(x)
        if tf.is_tensor(mask):
            mask = tf.pad(mask, ((0, 0), (1, 0)), constant_values=1.)[:, tf.newaxis, tf.newaxis, :]
            mask = einops.rearrange(tf.broadcast_to(mask, (b, h, w, t+1)), 'b h w t_pad -> (b h w) t_pad')[:, tf.newaxis, tf.newaxis, :]

        # Encoder
        x = tf.expand_dims(x, axis=-1)
        x_ref = tf.broadcast_to(
            tfp.stats.percentile(x, 50., axis=-2, keepdims=True),
            (b, h, w, t, 1)
        )
        x = tf.concat([x, x_ref], axis=-1)
        x = einops.rearrange(x, 'b h w t c -> (b t) h w c')
        features = self.encoder(x, training=training)

        # Transformer
        features = tf.reshape(
            features, (b, t, h, w, self.n_filters)
        )
        features = tf.transpose(features, perm=[0, 2, 3, 1, 4])
        features = tf.reshape(features, (b*h*w, t, self.n_filters))
        features = self.transformer(features, mask=mask, training=training)

        # Decoder(PixelShuffle)
        features = tf.reshape(
            features, (b, h, w, 1, self.n_filters)
        )[:, :, :, 0, :]
        print(features.shape)
        recon = self.decoder(features, training=training)
        return recon

    def call(self, inputs, training=None, mask=None):
        lrs, mask = inputs
        if training is None:
            training = False
        return self.forward(lrs, mask=mask, training=training)