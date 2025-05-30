import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def get_positional_encoding(max_len, embed_dim):
    position = np.arange(max_len)[:, np.newaxis]
    div_term = np.exp(np.arange(0, embed_dim, 2) * -(np.log(10000.0) / embed_dim))
    pe = np.zeros((max_len, embed_dim))
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)
    return tf.constant(pe, dtype=tf.float32)
    
class MultiHeadSelfAttention(layers.Layer):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embedding dimension = {embed_dim} should be divisible by number of heads = {num_heads}"
            )
        self.projection_dim = embed_dim // num_heads
        self.query_dense = layers.Dense(embed_dim)
        self.key_dense = layers.Dense(embed_dim)
        self.value_dense = layers.Dense(embed_dim)
        self.combine_heads = layers.Dense(embed_dim)

    def attention(self, query, key, value):
        score = tf.matmul(query, key, transpose_b=True)
        dim_key = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_score = score / tf.math.sqrt(dim_key)
        weights = tf.nn.softmax(scaled_score, axis=-1)
        output = tf.matmul(weights, value)
        return output, weights

    def separate_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.projection_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        query = self.query_dense(inputs)  
        key = self.key_dense(inputs)  
        value = self.value_dense(inputs) 
        query = self.separate_heads(
            query, batch_size
        )  
        key = self.separate_heads(
            key, batch_size
        )  
        value = self.separate_heads(
            value, batch_size
        ) 
        attention, weights = self.attention(query, key, value)
        attention = tf.transpose(attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(attention, (batch_size, -1, self.embed_dim))
        output = self.combine_heads(concat_attention)
        return output
        
class TransformerEncoderLayer(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super().__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([
            layers.Dense(ff_dim, activation='relu'),
            layers.Dense(embed_dim),
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, x, training):
        attn_output = self.att(x, x)
        out1 = self.layernorm1(x + self.dropout1(attn_output, training=training))
        ffn_output = self.ffn(out1)
        return self.layernorm2(out1 + self.dropout2(ffn_output, training=training))
        
class TransformerFlipPredictor(tf.keras.Model):
    def __init__(self, num_layers, embed_dim, num_heads, ff_dim, max_len, rate=0.1):
        super().__init__()
        self.embed = layers.Dense(embed_dim)
        self.pos_encoding = get_positional_encoding(max_len, embed_dim)
        self.enc_layers = [TransformerEncoderLayer(embed_dim, num_heads, ff_dim, rate)
                           for _ in range(num_layers)]
        ##self.dropout = layers.Dropout(rate)
        self.final_layer = layers.Dense(max_len, activation='softmax')  # flip prediction

    def call(self, inputs, training=False):
        # inputs: (batch, seq_len)
        x = self.embed(inputs)  # (batch, seq_len, embed_dim)
        x += self.pos_encoding[:tf.shape(x)[1]]
        for enc_layer in self.enc_layers:
            x = enc_layer(x, training)
        x = tf.reduce_mean(x, axis=1)  # (batch, embed_dim)
        return self.final_layer(x)  # (batch, N)        


model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss=tf.keras.losses.CategoricalCrossentropy(),
    metrics=['accuracy']
)

model.fit(
    x=[received_llrs, decoded_result, transmitted_codeword],
    y=flip_position_onehot,
    batch_size=128,
    epochs=20,
    validation_split=0.1,
    callbacks=[...]
)