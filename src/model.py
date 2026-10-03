from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Embedding, Bidirectional, LSTM, Conv1D,
    GlobalMaxPooling1D, Dense, Dropout, Concatenate
)
from tensorflow.keras.optimizers import Adam

from config import MAX_WORDS, MAX_LEN, EMBEDDING_DIM

def build_hybrid_model():
    input_layer = Input(shape=(MAX_LEN,), name="input_text")

    # Embedding layer
    x = Embedding(
        input_dim=MAX_WORDS,
        output_dim=EMBEDDING_DIM,
        input_length=MAX_LEN,
        name="embedding"
    )(input_layer)

    # Branch 1: BiLSTM
    lstm_branch = Bidirectional(
        LSTM(64, return_sequences=True), name="bilstm"
    )(x)
    lstm_pool = GlobalMaxPooling1D(name="lstm_pool")(lstm_branch)

    # Branch 2: CNN
    cnn = Conv1D(filters=128, kernel_size=3, activation="relu", name="cnn")(x)
    cnn_pool = GlobalMaxPooling1D(name="cnn_pool")(cnn)

    # Concatenate branches
    merged = Concatenate(name="concat")([lstm_pool, cnn_pool])

    dense = Dense(128, activation="relu", name="dense1")(merged)
    drop = Dropout(0.5, name="dropout")(dense)
    output = Dense(1, activation="sigmoid", name="output")(drop)

    model = Model(inputs=input_layer, outputs=output, name="Hybrid_BiLSTM_CNN")

    model.compile(
        loss="binary_crossentropy",
        optimizer=Adam(learning_rate=1e-3),
        metrics=["accuracy"]
    )

    model.summary()
    return model
