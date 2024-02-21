import numpy as np
import tqdm
import json
from concurrent.futures import ThreadPoolExecutor
import mediapipe as mp


def rotate(data, rotation_matrix):
    frames, landmarks, _ = data.shape
    center = np.array([0.5, 0.5, 0])
    non_zero = np.argwhere(np.any(data[:, :, :2] != 0, axis=2))
    data = data.reshape(-1, 3)
    data[non_zero] -= center
    data[non_zero] = np.dot(data[non_zero], rotation_matrix.T)
    data[non_zero] += center
    data = data.reshape(frames, landmarks, 3)
    out_of_range = np.any((data[:, :, :2] < 0) | (data[:, :, :2] > 1), axis=2)
    data[out_of_range] = 0
    return data


def rotate_z(data):
    angle = np.random.choice([np.random.uniform(-30, -10), np.random.uniform(10, 30)])
    theta = np.radians(angle)
    rotation_matrix = np.array(
        [
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1],
        ]
    )
    return rotate(data, rotation_matrix)


def rotate_y(data):
    angle = np.random.choice([np.random.uniform(-30, -10), np.random.uniform(10, 30)])
    theta = np.radians(angle)
    rotation_matrix = np.array(
        [
            [np.cos(theta), 0, np.sin(theta)],
            [0, 1, 0],
            [-np.sin(theta), 0, np.cos(theta)],
        ]
    )
    return rotate(data, rotation_matrix)


def rotate_x(data):
    angle = np.random.choice([np.random.uniform(-30, -10), np.random.uniform(10, 30)])
    theta = np.radians(angle)
    rotation_matrix = np.array(
        [
            [1, 0, 0],
            [0, np.cos(theta), -np.sin(theta)],
            [0, np.sin(theta), np.cos(theta)],
        ]
    )
    return rotate(data, rotation_matrix)


def zoom(data):
    factor = np.random.uniform(0.8, 1.2)
    center = np.array([0.5, 0.5])
    non_zero = np.argwhere(np.any(data[:, :, :2] != 0, axis=2))
    data[non_zero[:, 0], non_zero[:, 1], :2] = (
        data[non_zero[:, 0], non_zero[:, 1], :2] - center
    ) * factor + center
    out_of_range = np.any((data[:, :, :2] < 0) | (data[:, :, :2] > 1), axis=2)
    data[out_of_range] = 0
    return data


def shift(data):
    x_shift = np.random.uniform(-0.2, 0.2)
    y_shift = np.random.uniform(-0.2, 0.2)
    non_zero = np.argwhere(np.any(data[:, :, :2] != 0, axis=2))
    data[non_zero[:, 0], non_zero[:, 1], 0] += x_shift
    data[non_zero[:, 0], non_zero[:, 1], 1] += y_shift
    out_of_range = np.any((data[:, :, :2] < 0) | (data[:, :, :2] > 1), axis=2)
    data[out_of_range] = 0
    return data


def mask(data):
    frames, landmarks, _ = data.shape
    num_hands = int(0.3 * 42)
    num_rest = int(0.6 * (landmarks - 42))

    mask = np.zeros(landmarks, dtype=bool)
    indices = np.concatenate(
        [
            np.random.choice(42, num_hands, replace=False),
            np.random.choice(landmarks - 42, num_rest, replace=False) + 42,
        ]
    )
    mask[indices] = True
    data[:, mask] = 0
    return data


def speedup(data):
    return data[::2]


def apply_augmentations(data):
    aug_functions = [rotate_x, rotate_y, rotate_z, zoom, shift, mask, speedup]
    np.random.shuffle(aug_functions)
    counter = 0
    for fun in aug_functions:
        if np.random.rand() < 0.5:
            data = fun(data)
            counter += 1

    if counter == 0:
        data = apply_augmentations(data)

    return data


def augment(X, Y, num=None):
    X_aug = X.copy()
    Y_aug = Y.copy()

    if num == None:
        for i in tqdm(range(len(Y)), ncols=100):
            num_aug = np.random.choice([1, 2, 3])
            for n in range(num_aug):
                X_aug.append(apply_augmentations(X[i].copy()))
                Y_aug.append(Y[i])
    elif num > 0:
        for i in tqdm(range(len(Y)), ncols=100):
            for n in range(num):
                X_aug.append(apply_augmentations(X[i].copy()))
                Y_aug.append(Y[i])

    return X_aug, Y_aug


def padding(X, length=None, pad=0):
    if length is None:
        length = max(len(x) for x in X)

    X_padded = []
    for x in X:
        if len(x) > length:
            X_padded.append(x[:length])  # truncate
        else:
            pad_length = length - len(x)
            X_padded.append(
                np.pad(
                    x,
                    ((0, pad_length), (0, 0), (0, 0)),
                    mode="constant",
                    constant_values=pad,
                )
            )

    X_padded = np.array(X_padded)
    # Y = np.array(Y)
    return X_padded


def padding11(X, length=None, pad=0):
    if length is None:
        length = max(len(x) for x in X)

    X_padded = []
    for x in X:
        if len(x) > length:
            start = (len(x) - length) // 2
            end = start + length
            X_padded.append(x[start:end])
        else:
            pad_before = (length - len(x)) // 2
            pad_after = length - len(x) - pad_before
            X_padded.append(
                np.pad(
                    x,
                    ((pad_before, pad_after), (0, 0), (0, 0)),
                    mode="constant",
                    constant_values=pad,
                )
            )

    X_padded = np.array(X_padded)
    return X_padded


def sequences(X, length=30, step=1, pad=0):
    X_sequences = []

    for inputs in X:
        num = inputs.shape[0]

        if num < length:
            padding = length - num
            inputs = np.pad(
                inputs,
                ((0, padding), (0, 0), (0, 0)),
                mode="constant",
                constant_values=pad,
            )
            num = length

        for start in range(0, num - length + 1, step):
            end = start + length
            sequence = inputs[start:end]
            X_sequences.append(sequence)

    X_sequences = np.array(X_sequences)
    return X_sequences


def interpolate(X, length=100):
    X_interpolated = [
        np.apply_along_axis(
            lambda arr: np.interp(
                np.linspace(0, 1, length), np.linspace(0, 1, arr.shape[0]), arr
            ),
            axis=0,
            arr=x,
        )
        for x in X
    ]

    X = np.array(X_interpolated)
    return X


def padding(X, length=None, pad=0):
    if length is None:
        length = max(len(x) for x in X)

    X_padded = []
    for x in X:
        if len(x) > length:
            start = (len(x) - length) // 2
            end = start + length
            X_padded.append(x[start:end])
        else:
            pad_before = (length - len(x)) // 2
            pad_after = length - len(x) - pad_before
            X_padded.append(
                np.pad(
                    x,
                    ((pad_before, pad_after), (0, 0), (0, 0)),
                    mode="constant",
                    constant_values=pad,
                )
            )

    X_padded = np.array(X_padded)
    return X_padded


# def padding_with_none(X, Y, length=None):
#     if length is None:
#         length = max(len(x) for x in X)

#     X_padded = []
#     for x in X:
#         if len(x) > length:
#             start = (len(x) - length) // 2
#             end = start + length
#             X_padded.append(x[start:end])
#         else:
#             pad_before = (length - len(x)) // 2
#             pad_after = length - len(x) - pad_before
#             idx = np.random.choice(len(None_list))
#             none_sample = None_list[idx]
#             selected_frames = np.random.choice(len(none_sample), size=(pad_before + pad_after,), replace=True)
#             padding_samples = none_sample[selected_frames]
#             padded_x = np.concatenate([padding_samples[:pad_before], x, padding_samples[pad_before:]], axis=0)
#             X_padded.append(padded_x)

#     X_padded = np.array(X_padded)
#     Y = np.array(Y)
#     return X_padded, Y
import tensorflow as tf
import tensorflow_addons as tfa
import matplotlib.pyplot as plt


class ECA(tf.keras.layers.Layer):
    def __init__(self, kernel_size=5, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True
        self.kernel_size = kernel_size
        self.conv = tf.keras.layers.Conv1D(
            1, kernel_size=kernel_size, strides=1, padding="same", use_bias=False
        )

    def call(self, inputs, mask=None):
        nn = tf.keras.layers.GlobalAveragePooling1D()(inputs, mask=mask)
        nn = tf.expand_dims(nn, -1)
        nn = self.conv(nn)
        nn = tf.squeeze(nn, -1)
        nn = tf.nn.sigmoid(nn)
        nn = nn[:, None, :]
        return inputs * nn


class LateDropout(tf.keras.layers.Layer):
    def __init__(self, rate, noise_shape=None, start_step=0, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True
        self.rate = rate
        self.start_step = start_step
        self.dropout = tf.keras.layers.Dropout(rate, noise_shape=noise_shape)

    def build(self, input_shape):
        super().build(input_shape)
        agg = tf.VariableAggregation.ONLY_FIRST_REPLICA
        self._train_counter = tf.Variable(
            0, dtype="int64", aggregation=agg, trainable=False
        )

    def call(self, inputs, training=False):
        x = tf.cond(
            self._train_counter < self.start_step,
            lambda: inputs,
            lambda: self.dropout(inputs, training=training),
        )
        if training:
            self._train_counter.assign_add(1)
        return x


class CausalDWConv1D(tf.keras.layers.Layer):
    def __init__(
        self,
        kernel_size=17,
        dilation_rate=1,
        use_bias=False,
        depthwise_initializer="glorot_uniform",
        name="",
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.causal_pad = tf.keras.layers.ZeroPadding1D(
            (dilation_rate * (kernel_size - 1), 0), name=name + "_pad"
        )
        self.dw_conv = tf.keras.layers.DepthwiseConv1D(
            kernel_size,
            strides=1,
            dilation_rate=dilation_rate,
            padding="valid",
            use_bias=use_bias,
            depthwise_initializer=depthwise_initializer,
            name=name + "_dwconv",
        )
        self.supports_masking = True

    def call(self, inputs):
        x = self.causal_pad(inputs)
        x = self.dw_conv(x)
        return x


def Conv1DBlock(
    channel_size,
    kernel_size,
    dilation_rate=1,
    drop_rate=0.0,
    expand_ratio=2,
    se_ratio=0.25,
    activation="swish",
    name=None,
):

    if name is None:
        name = str(tf.keras.backend.get_uid("mbblock"))

    # Expansion phase
    def apply(inputs):
        channels_in = tf.keras.backend.int_shape(inputs)[-1]
        channels_expand = channels_in * expand_ratio

        skip = inputs

        x = tf.keras.layers.Dense(
            channels_expand,
            use_bias=True,
            activation=activation,
            name=name + "_expand_conv",
        )(inputs)

        # Depthwise Convolution
        x = CausalDWConv1D(
            kernel_size,
            dilation_rate=dilation_rate,
            use_bias=False,
            name=name + "_dwconv",
        )(x)

        x = tf.keras.layers.BatchNormalization(momentum=0.95, name=name + "_bn")(x)

        x = ECA()(x)

        x = tf.keras.layers.Dense(
            channel_size, use_bias=True, name=name + "_project_conv"
        )(x)

        if drop_rate > 0:
            x = tf.keras.layers.Dropout(
                drop_rate, noise_shape=(None, 1, 1), name=name + "_drop"
            )(x)

        if channels_in == channel_size:
            x = tf.keras.layers.add([x, skip], name=name + "_add")
        return x

    return apply


class MultiHeadSelfAttention(tf.keras.layers.Layer):
    def __init__(self, dim=256, num_heads=4, dropout=0, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.scale = self.dim**-0.5
        self.num_heads = num_heads
        self.qkv = tf.keras.layers.Dense(3 * dim, use_bias=False)
        self.drop1 = tf.keras.layers.Dropout(dropout)
        self.proj = tf.keras.layers.Dense(dim, use_bias=False)
        self.supports_masking = True

    def call(self, inputs, mask=None):
        qkv = self.qkv(inputs)
        qkv = tf.keras.layers.Permute((2, 1, 3))(
            tf.keras.layers.Reshape(
                (-1, self.num_heads, self.dim * 3 // self.num_heads)
            )(qkv)
        )
        q, k, v = tf.split(qkv, [self.dim // self.num_heads] * 3, axis=-1)

        attn = tf.matmul(q, k, transpose_b=True) * self.scale

        if mask is not None:
            mask = mask[:, None, None, :]

        attn = tf.keras.layers.Softmax(axis=-1)(attn, mask=mask)
        attn = self.drop1(attn)

        x = attn @ v
        x = tf.keras.layers.Reshape((-1, self.dim))(
            tf.keras.layers.Permute((2, 1, 3))(x)
        )
        x = self.proj(x)
        return x


def TransformerBlock(
    dim=256, num_heads=4, expand=4, attn_dropout=0.2, drop_rate=0.2, activation="swish"
):
    def apply(inputs):
        x = inputs
        x = tf.keras.layers.BatchNormalization(momentum=0.95)(x)
        x = MultiHeadSelfAttention(dim=dim, num_heads=num_heads, dropout=attn_dropout)(
            x
        )
        x = tf.keras.layers.Dropout(drop_rate, noise_shape=(None, 1, 1))(x)
        x = tf.keras.layers.Add()([inputs, x])
        attn_out = x

        x = tf.keras.layers.BatchNormalization(momentum=0.95)(x)
        x = tf.keras.layers.Dense(dim * expand, use_bias=False, activation=activation)(
            x
        )
        x = tf.keras.layers.Dense(dim, use_bias=False)(x)
        x = tf.keras.layers.Dropout(drop_rate, noise_shape=(None, 1, 1))(x)
        x = tf.keras.layers.Add()([attn_out, x])
        return x

    return apply


class OneCycleLR(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(
        self,
        lr=1e-4,
        epochs=10,
        steps_per_epoch=100,
        steps_per_update=1,
        resume_epoch=0,
        decay_epochs=10,
        sustain_epochs=0,
        warmup_epochs=0,
        lr_start=0,
        lr_min=0,
        warmup_type="linear",
        decay_type="cosine",
        **kwargs
    ):

        super().__init__(**kwargs)
        self.lr = float(lr)
        self.epochs = float(epochs)
        self.steps_per_update = float(steps_per_update)
        self.resume_epoch = float(resume_epoch)
        self.steps_per_epoch = float(steps_per_epoch)
        self.decay_epochs = float(decay_epochs)
        self.sustain_epochs = float(sustain_epochs)
        self.warmup_epochs = float(warmup_epochs)
        self.lr_start = float(lr_start)
        self.lr_min = float(lr_min)
        self.decay_type = decay_type
        self.warmup_type = warmup_type

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        total_steps = self.epochs * self.steps_per_epoch
        warmup_steps = self.warmup_epochs * self.steps_per_epoch
        sustain_steps = self.sustain_epochs * self.steps_per_epoch
        decay_steps = self.decay_epochs * self.steps_per_epoch

        if self.resume_epoch > 0:
            step = step + self.resume_epoch * self.steps_per_epoch

        step = tf.cond(step > decay_steps, lambda: decay_steps, lambda: step)
        step = tf.math.truediv(step, self.steps_per_update) * self.steps_per_update

        warmup_cond = step < warmup_steps
        decay_cond = step >= (warmup_steps + sustain_steps)

        if self.warmup_type == "linear":
            lr = tf.cond(
                warmup_cond,
                lambda: tf.math.divide_no_nan(self.lr - self.lr_start, warmup_steps)
                * step
                + self.lr_start,
                lambda: self.lr,
            )
        elif self.warmup_type == "exponential":
            factor = tf.pow(self.lr_start, 1 / warmup_steps)
            lr = tf.cond(
                warmup_cond,
                lambda: (self.lr - self.lr_start) * factor ** (warmup_steps - step)
                + self.lr_start,
                lambda: self.lr,
            )
        elif self.warmup_type == "cosine":
            lr = tf.cond(
                warmup_cond,
                lambda: 0.5
                * (self.lr - self.lr_start)
                * (1 + tf.cos(3.14159265359 * (warmup_steps - step) / warmup_steps))
                + self.lr_start,
                lambda: self.lr,
            )
        else:
            raise NotImplementedError

        if self.decay_type == "linear":
            lr = tf.cond(
                decay_cond,
                lambda: self.lr
                + (self.lr_min - self.lr)
                / (decay_steps - warmup_steps - sustain_steps)
                * (step - warmup_steps - sustain_steps),
                lambda: lr,
            )
        elif self.decay_type == "exponential":
            factor = tf.pow(
                self.lr_min, 1 / (decay_steps - warmup_steps - sustain_steps)
            )
            lr = tf.cond(
                decay_cond,
                lambda: (self.lr - self.lr_min)
                * factor ** (step - warmup_steps - sustain_steps)
                + self.lr_min,
                lambda: lr,
            )
        elif self.decay_type == "cosine":
            lr = tf.cond(
                decay_cond,
                lambda: 0.5
                * (self.lr - self.lr_min)
                * (
                    1
                    + tf.cos(
                        3.14159265359
                        * (step - warmup_steps - sustain_steps)
                        / (decay_steps - warmup_steps - sustain_steps)
                    )
                )
                + self.lr_min,
                lambda: lr,
            )
        else:
            raise NotImplementedError

        return lr

    def plot(self):
        step = max(
            1, int(self.epochs * self.steps_per_epoch) // 1000
        )  # 1 for total_steps < 1000, total_steps//1000 else
        eps = list(range(0, int(self.epochs * self.steps_per_epoch), step))
        learning_rates = [self(x) for x in eps]
        plt.scatter(eps, learning_rates, 2)
        plt.show()

    def get_config(self):
        config = {
            "lr": self.lr,
            "epochs": self.epochs,
            "steps_per_update": self.steps_per_update,
            "resume_epoch": self.resume_epoch,
            "steps_per_epoch": self.steps_per_epoch,
            "decay_epochs": self.decay_epochs,
            "sustain_epochs": self.sustain_epochs,
            "warmup_epochs": self.warmup_epochs,
            "lr_start": self.lr_start,
            "lr_min": self.lr_min,
            "decay_type": self.decay_type,
            "warmup_type": self.warmup_type,
        }
        return config


# schedule = OneCycleLR(
#     4e-3,
#     200,
#     steps_per_epoch=192,
#     decay_epochs=200,
#     lr_min=1e-6,
#     decay_type="cosine",
#     warmup_type="linear",
# )
# decay_schedule = OneCycleLR(
#     4e-4,
#     200,
#     steps_per_epoch=192,
#     decay_epochs=200,
#     lr_min=1e-7,
#     decay_type="cosine",
#     warmup_type="linear",
# )
# optimizer = tfa.optimizers.RectifiedAdam(
#     learning_rate=schedule, weight_decay=decay_schedule, sma_threshold=4
# )
# optimizer = tfa.optimizers.Lookahead(optimizer, sync_period=5)
from keras_radam import RAdam

# import keras.optimizers as optimizers

# optimizer = RAdam()

# # import tensorflow as tf
# from keras.optimizers import legacy


# optimizer = legacy.RAdam()
def get_model():
    optimizer = tfa.optimizers.RectifiedAdam
    model = tf.keras.models.load_model(
        r"D:\College\Project\SignSpeak Converter\unoptimized model\model.h5",
        custom_objects={
            "ECA": ECA,
            "LateDropout": LateDropout,
            "CausalDWConv1D": CausalDWConv1D,
            "Conv1DBlock": Conv1DBlock,
            "MultiHeadSelfAttention": MultiHeadSelfAttention,
            "TransformerBlock": TransformerBlock,
            "OneCycleLR": OneCycleLR,
            "RectifiedAdam": optimizer,
        },
    )
    return model
