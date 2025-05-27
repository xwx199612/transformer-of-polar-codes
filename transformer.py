import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

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

class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadSelfAttention(embed_dim, num_heads)
        self.ffn = keras.Sequential([
            layers.Dense(ff_dim, activation="relu"), 
            layers.Dense(embed_dim)
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs):
        attn_output = self.att(inputs)
        attn_output = self.dropout1(attn_output)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)
        return self.layernorm2(out1 + ffn_output)

class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions

class TransformerEncoder(layers.Layer):
    def __init__(self, num_layers, embed_dim, num_heads, ff_dim, rate=0.1):
    ##def __init__(self, num_layers, embed_dim, num_heads, ff_dim, input_vocab_size, maximum_position_encoding, rate=0.1):
        super(TransformerEncoder, self).__init__()
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        ##self.embedding = TokenAndPositionEmbedding(maximum_position_encoding, input_vocab_size, self.embed_dim)
        self.enc_layers = [TransformerBlock(embed_dim, num_heads, ff_dim, rate) for _ in range(num_layers)]
        self.dropout = layers.Dropout(rate)

    def call(self, inputs, training):
        ##x = self.embedding(inputs)
        x = inputs* tf.math.sqrt(tf.cast(self.embed_dim, tf.float32))
        x = self.dropout(x, training=training)
        for i in range(self.num_layers):
            x = self.enc_layers[i](x)
        return x

class TransformerDecoder(layers.Layer):
    def __init__(self, num_layers, embed_dim, num_heads, ff_dim, rate=0.1):
    ##def __init__(self, num_layers, embed_dim, num_heads, ff_dim, target_vocab_size, maximum_position_encoding, rate=0.1):
        super(TransformerDecoder, self).__init__()
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        ##self.embedding = TokenAndPositionEmbedding(maximum_position_encoding, target_vocab_size, self.embed_dim)
        self.dec_layers = [TransformerBlock(embed_dim, num_heads, ff_dim, rate) for _ in range(num_layers)]
        self.dropout = layers.Dropout(rate)

    def call(self, inputs, encoder_outputs, training):
        ##x = self.embedding(inputs)
        x = inputs* tf.math.sqrt(tf.cast(self.embed_dim, tf.float32))
        x = self.dropout(x, training=training)
        for i in range(self.num_layers):
            x = self.dec_layers[i](x)
            attention_output = MultiHeadSelfAttention(self.embed_dim, num_heads)([x, encoder_outputs, encoder_outputs])
            x = layers.Concatenate()([x, attention_output])
        return x

class Transformer(keras.Model):
    def __init__(
        self,
        num_layers,
        embed_dim,
        num_heads,
        ff_dim,
        rate=0.1,
    ):
        super(Transformer, self).__init__()
        self.encoder = TransformerEncoder(num_layers, embed_dim, num_heads, ff_dim, rate)
        self.decoder = TransformerDecoder(num_layers, embed_dim, num_heads, ff_dim, rate)
        ##self.encoder = TransformerEncoder(num_layers, embed_dim, num_heads, ff_dim, input_vocab_size, pe_input, rate)
        ##self.decoder = TransformerDecoder(num_layers, embed_dim, num_heads, ff_dim, target_vocab_size, pe_target, rate)
        self.final_layer = layers.Dense(target_vocab_size, activation="softmax")
    
    @tf.function
    def call(self, inputs):
        input_seq, target_seq = inputs #LLR, codeword
        encoder_outputs = self.encoder(input_seq, training)
        decoder_outputs = self.decoder(target_seq, encoder_outputs, training)
        final_outputs = self.final_layer(decoder_outputs) #which bit to flip
        return final_outputs

class Transformer(keras.Model):
    def __init__(
        self,
        num_layers,
        embed_dim,
        num_heads,
        ff_dim,
        target_vocab_size,
        rate=0.1,
    ):
        super(Transformer, self).__init__()
        self.encoder = TransformerEncoder(num_layers, embed_dim, num_heads, ff_dim, rate)
        self.decoder = TransformerDecoder(num_layers, embed_dim, num_heads, ff_dim, rate)
        self.final_layer = layers.Dense(target_vocab_size, activation="softmax")

    def call(self, inputs, training=False):
        input_seq, target_seq = inputs  # input: (LLR, codeword)
        encoder_outputs = self.encoder(input_seq, training=training)
        decoder_outputs = self.decoder(target_seq, encoder_outputs, training=training)
        final_outputs = self.final_layer(decoder_outputs)  # output: which bit to flip
        return final_outputs
'''    
    @tf.function
    def train_step(self, data):
        cce = tf.keras.losses.CategoricalCrossentropy()
        
        with tf.GradientTape() as tape:
            y_pred = self(inputs, training=True) #forward
            # Compute loss
            loss = cce(y, y_pred)
            
        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        
         # Compute our own metrics
        loss_tracker.update_state(loss)
        mae_metric.update_state(y, y_pred)
        return {"loss": loss_tracker.result(), "mae": mae_metric.result()}

        train_loss(loss)
        train_accuracy(targets, predictions)'''