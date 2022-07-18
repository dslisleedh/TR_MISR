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

    def compile(self, ed_optimizer, fu_optimizer, **kwargs):
        super(TRNet, self).compile(**kwargs)
        self.ed_optimizer = ed_optimizer
        self.fu_optimizer = fu_optimizer

    def train_step(self, data):
        inputs, hr = data
        if tf.is_tensor(inputs):
            lr, attention_mask = inputs, None
        else:
            lr, attention_mask = inputs

        with tf.GradientTape(persistent=True) as tape:
            recon = self.forward(lr, attention_mask=attention_mask, training=True)
            loss = self.compiled_loss(hr, recon, regularization_losses=self.losses)
        grads_ed = tape.gradient(loss, self.encoder.trainable_variables + self.decoder.trainable_variables)
        grads_fu = tape.gradient(loss, self.transformer.trainable_variables)
        self.ed_optimizer.apply_gradients(
            zip(grads_ed, self.encoder.trainable_variables + self.decoder.trainable_variables)
        )
        self.fu_optimizer.apply_gradients(
            zip(grads_fu, self.transformer.trainable_variables)
        )

        self.compiled_metrics.update_state(hr, recon)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        inputs, hr = data
        if tf.is_tensor(inputs):
            lr, attention_mask = inputs, None
        else:
            lr, attention_mask = inputs

        recon = self.forward(lr, attention_mask=attention_mask, training=False)
        self.compiled_loss(hr, recon, regularization_losses=self.losses)
        self.compiled_metrics.update_state(hr, recon)

        val_psnr = [m.result() for m in self.metrics if m.name == 'psnr'][0]

        def good_cond():
            self.decay_patience = 0
            self.best_validation_psnr = val_psnr

        def bad_cond():
            self.decay_patience += 1
            if self.decay_patience > 3:
                self.decay_patience = 0
                self.fu_optimizer.learning_rate = .95 * self.fu_optimizer.learning_rate
                self.ed_optimizer.learning_rate = .95 * self.ed_optimizer.learning_rate

        tf.cond(
            tf.math.less(self.best_validation_psnr, val_psnr),
            good_cond,
            bad_cond
        )

        return {m.name: m.result() for m in self.metrics}

    def forward(self, x, attention_mask=None, training=False):
        b, h, w, t = x.get_shape().as_list()

        if tf.is_tensor(attention_mask):
            attention_mask = tf.pad(attention_mask, ((0, 0), (1, 0)), constant_values=1.)[:, tf.newaxis, tf.newaxis, :]
            attention_mask = einops.rearrange(
                tf.broadcast_to(attention_mask, (b, h, w, t+1)), 'b h w t_pad -> (b h w) t_pad'
            )[:, tf.newaxis, tf.newaxis, :]

        # Encoder
        x = tf.expand_dims(x, axis=-1)
        x_ref = einops.repeat(
            tfp.stats.percentile(x, 50., axis=-2, keepdims=True),
            'b h w () c -> b h w t c', t=t
        )
        x = tf.concat([x, x_ref], axis=-1)
        x = einops.rearrange(x, 'b h w t c -> (b t) h w c')
        features = self.encoder(x, training=training)

        # Transformer
        features = einops.rearrange(
            features, '(b t) h w c -> (b h w) t c', t=t
        )
        features = self.transformer(features, attention_mask=attention_mask, training=training)
        # Decoder(PixelShuffle)
        features = einops.rearrange(
            features, '(b h w) t c -> b h w t c', h=h, w=w
        )[:, :, :, 0, :]
        recon = self.decoder(features, training=training)
        return recon

    def call(self, inputs, training=None, mask=None):
        if tf.is_tensor(inputs):
            lrs = inputs
            attention_mask = None
        else:
            lrs, attention_mask = inputs
        if training is None:
            training = False
        return self.forward(lrs, attention_mask=attention_mask, training=training)