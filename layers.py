import tensorflow as tf
import tensorflow_probability as tfp
import einops


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
                 kernel_size=3
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


class FFN(tf.keras.layers.Layer):
    def __init__(self,
                 n_filters: int,
                 n_mlp_filters: int,
                 dropout_rate: float = 0.
                 ):
        super(FFN, self).__init__()
        self.n_filters = n_filters
        self.n_mlp_filters = n_mlp_filters
        self.dropout_rate = dropout_rate

        self.forward = tf.keras.Sequential([
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.Dense(self.n_mlp_filters,
                                  kernel_initializer=tf.keras.initializers.VarianceScaling(.02),
                                  activation='gelu'
                                  ),
            tf.keras.layers.Dropout(self.dropout_rate),
            tf.keras.layers.Dense(self.n_filters,
                                  kernel_initializer=tf.keras.initializers.VarianceScaling(.02)
                                  ),
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

        self.to_qkv = tf.keras.Sequential([
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.Dense(
                self.n_filters * 3,
                use_bias=False,
                kernel_initializer=tf.keras.initializers.VarianceScaling(.02)
                )
        ])
        self.to_out = tf.keras.Sequential([
            tf.keras.layers.Dense(
                self.n_filters,
                kernel_initializer=tf.keras.initializers.VarianceScaling(.02)
            ),
            tf.keras.layers.Dropout(self.dropout_rate)
        ])

    def call(self, inputs, mask=None, *args, **kwargs):
        qkv = self.to_qkv(inputs)
        q, k, v = tf.unstack(
            einops.rearrange(
                qkv, 'b n (qkv_expansion h c) -> b qkv_expansion h n c',
                qkv_expansion=3, h=8
            ),
            num=3,
            axis=1
        )
        attention_map = tf.matmul(q, k, transpose_b=True) * self.scale
        if tf.is_tensor(mask):
            attention_map += mask * tf.DType(1.).min
        attention = tf.nn.softmax(attention_map, axis=-1)
        out = einops.rearrange(
            tf.matmul(attention, v), 'b h n c -> b n (h c)'
        )
        out = self.to_out(out)
        return inputs + out


class Transformer(tf.keras.layers.Layer):
    def __init__(self,
                 n_filters: int,
                 n_blocks: int,
                 n_heads: int,
                 n_mlp_filters: int,
                 dropout_rate: float,
                 ):
        super(Transformer, self).__init__()
        self.n_filters = n_filters
        self.n_blocks = n_blocks
        self.n_heads = n_heads
        self.n_mlp_filters = n_mlp_filters
        self.dropout_rate = dropout_rate

        self.cls_token = tf.Variable(
            tf.random.truncated_normal(shape=(1, 1, self.n_filters), stddev=0.02)
        )
        self.attns = [
            Attention(n_filters, n_heads, dropout_rate) for _ in range(n_blocks)
        ]
        self.ffns = [
            FFN(n_filters, n_mlp_filters, dropout_rate) for _ in range(n_blocks)
        ]

    def call(self, features, mask=None, **kwargs):
        b, _, _ = tf.shape(features)
        cls_token = tf.broadcast_to(
            self.cls_token, (b, 1, self.n_filters)
        )
        features = tf.concat([cls_token, features], axis=1)
        for a, f in zip(self.attns, self.ffns):
            features = a(features, mask)
            features = f(features)
        cls_token = tf.gather(features, [0], axis=1)
        return cls_token


class Decoder(tf.keras.layers.Layer):
    def __init__(self,
                 scale: int,
                 kernel_size: int = 3
                 ):
        super(Decoder, self).__init__()
        self.kernel_size = kernel_size
        self.scale = scale

        self.forward = tf.keras.Sequential([
            tf.keras.layers.Conv2D(
                self.scale**2,
                kernel_size=(self.kernel_size, self.kernel_size),
                strides=(1, 1),
                padding='SAME',
                kernel_initializer=tf.keras.initializers.VarianceScaling(.02)
            ),
            tf.keras.layers.PReLU(shared_axes=[1, 2])
        ])

    def call(self, inputs, *args, **kwargs):
        return tf.nn.depth_to_space(
            self.forward(inputs), self.scale
        )



