import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_probability as tfp
import einops


class PixelShuffle(tf.keras.layers.Layer):
    def __init__(self,
                 scale: int
                 ):
        super(PixelShuffle, self).__init__()
        self.scale = scale

    def call(self, inputs, *args, **kwargs):
        return tf.nn.depth_to_space(
            inputs, self.scale
        )


class ResidualBlock(tf.keras.layers.Layer):
    def __init__(self,
                 n_filters: int,
                 kernel_size: int
                 ):
        super(ResidualBlock, self).__init__()
        self.n_filters = n_filters
        self.kernel_size = kernel_size

        self.forward = tf.keras.Sequential([
            tf.keras.layers.Conv2D(
                self.n_filters,
                kernel_size=(self.kernel_size, self.kernel_size),
                strides=(1, 1),
                padding='SAME',
                kernel_initializer=tf.keras.initializers.VarianceScaling(.02)
            ),
            tf.keras.layers.PReLU(shared_axes=[1, 2]),
            tf.keras.layers.Conv2D(
                self.n_filters,
                kernel_size=(self.kernel_size, self.kernel_size),
                strides=(1, 1),
                padding='SAME',
                kernel_initializer=tf.keras.initializers.VarianceScaling(.02)
            ),
            tf.keras.layers.PReLU(shared_axes=[1, 2])
        ])

    def call(self, inputs, *args, **kwargs):
        return self.forward(inputs) + inputs


class Encoder(tf.keras.layers.Layer):
    def __init__(self,
                 n_filters,
                 n_layers,
                 kernel_size
                 ):
        super(Encoder, self).__init__()
        self.n_filters = n_filters
        self.n_layers = n_layers
        self.kernel_size = kernel_size

        self.intro = tf.keras.Sequential([
            tf.keras.layers.Conv2D(
                self.n_filters,
                kernel_size=(self.kernel_size, self.kernel_size),
                strides=(1, 1),
                padding='SAME',
                kernel_initializer=tf.keras.initializers.VarianceScaling(.02)
            ),
            tf.keras.layers.PReLU(shared_axes=[1, 2])
        ])
        self.blocks = tf.keras.Sequential([
            ResidualBlock(
                self.n_filters,
                self.kernel_size
            ) for _ in range(self.n_layers)
        ])
        self.out = tf.keras.layers.Conv2D(
            self.n_filters,
            kernel_size=(self.kernel_size, self.kernel_size),
            strides=(1, 1),
            padding='SAME',
            kernel_initializer=tf.keras.initializers.VarianceScaling(.02)
        )

    def call(self, inputs, *args, **kwargs):
        features = self.intro(inputs)
        features = self.blocks(features)
        return self.out(features)


class Decoder(tf.keras.layers.Layer):
    def __init__(self,
                 n_filters: int,
                 kernel_size: int,
                 scale: int
                 ):
        super(Decoder, self).__init__()
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.scale = scale

        self.forward = tf.keras.Sequential([
            tf.keras.layers.Conv2D(
                self.n_filters,
                kernel_size=(self.kernel_size, self.kernel_size),
                strides=(1, 1),
                padding='SAME',
                kernel_init=tf.keras.initializers.VarianceScaling(.02)
            ),
            tf.keras.layers.PReLU(shared_axes=[1, 2]),
            PixelShuffle(self.scale)
        ])

    def call(self, inputs, *args, **kwargs):
        return self.forward(inputs)


class FFN(tf.keras.layers.Layer):
    def __init__(self,
                 n_filters: int,
                 expansion_rate: int,
                 dropout_rate: float = 0.
                 ):
        super(FFN, self).__init__()
        self.n_filters = n_filters
        self.n_mlp_dims = n_filters * expansion_rate
        self.dropout_rate = dropout_rate

        self.forward = tf.keras.Sequential([
            tf.keras.layers.Dense(self.n_mlp_dims,
                                  kernel_initializer=tf.keras.initializers.VarianceScaling(.02),
                                  activation='gelu'
                                  ),
            tf.keras.layers.Dropout(self.dropout_rate),
            tf.keras.layers.Dense(self.n_filters),
            tf.keras.layers.Dropout(self.dropout_rate)
        ])

    def call(self, inputs, *args, **kwargs):
        return self.forward(inputs) + inputs


class Attention(tf.keras.layers.Layer):
    def __init__(self,
                 n_filters: int,
                 n_heads: int,
                 dropout_rate: float = 0.
                 ):
        super(Attention, self).__init__()
        self.n_filters = n_filters
        self.n_heads = n_heads
        self.scale = tf.Variable(
            float(self.n_filters) ** -0.5,
            trainable=False
        )
        self.dropout_rate = dropout_rate

        self.to_qkv = tf.keras.layers.Dense(
            self.n_filters * 3,
            kernel_initializer=tf.keras.initializers.VarianceScaling(.02)
        )
        self.to_out = tf.keras.Sequential([
            tf.keras.layers.Dense(
                self.n_filters,
                kernel_initializer=tf.keras.initializers.VarianceScaling(.02)
            ),
            tf.keras.layers.Dropout(self.dropout_rate)
        ])

    def call(self, inputs, *args, **kwargs):
        qkv = self.to_qkv(inputs)
        q, k, v = tf.unstack(
            einops.rearrange(
                qkv, 'b n (qkv_expansion n_heads c) -> b qkv_expansion n_heads n c',
                qkv_expansion=3, n_heads=8
            ),
            num=3,
            axis=1
        )
        attention_map = tf.matmul(q, k, transpose_b=True) * self.scale


#############################################################


class TRNet(tf.keras.models.Model):
    def __init__(self,
                 n_filters: int,
                 n_enc_layers: int,
                 n_transform_layers: int,
                 kernel_size: int = 3
                 ):
        super(TRNet, self).__init__()

    def forward(self, x, training=False):
        b, h, w, t = tf.shape(x)

        x = tf.expand_dims(x, axis=-1)
        x_ref = einops.repeat(
            tfp.stats.percentile(x, 50., axis=-2, keepdims=True),
            'b h w t c-> b h w (repeat t) c', repeat=t
        )
        x_ = tf.concat([x, x_ref], axis=-1)
        x_ = einops.rearrange(x_, 'b h w t c -> (b t) h w c')
        features = self.encoder(x_)




