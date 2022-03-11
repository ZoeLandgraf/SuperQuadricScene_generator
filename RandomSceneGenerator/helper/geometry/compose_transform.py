import chainer
from chainer.backends import cuda


class ComposeTransform(chainer.Function):
    def check_type_forward(self, in_types):
        chainer.utils.type_check.expect(in_types.size() == 2)

        R_type, t_type = in_types
        chainer.utils.type_check.expect(
            R_type.ndim == 3,  # N, 3, 3
            R_type.shape[0] == t_type.shape[0],
            R_type.shape[1:3] == (3, 3),
            t_type.ndim == 2,  # N, 3
            t_type.shape[1] == 3,
        )

    def forward(self, inputs):
        xp = cuda.get_array_module(*inputs)
        R, t = inputs

        batch_size = R.shape[0]
        T = xp.eye(4, dtype=R.dtype)
        T = T[None].repeat(batch_size, axis=0)

        T[:, :3, :3] = R
        T[:, :3, 3] = t
        return (T,)

    def backward(self, inputs, gy):
        (gT,) = gy
        gR = gT[:, :3, :3]
        gt = gT[:, :3, 3]
        return gR, gt


def compose_transform_function(R, t):
    squeeze_axis0 = False
    if R.ndim == 2 and t.ndim == 1:
        R = R[None]  # 3x3 -> 1x3x3
        t = t[None]  # 4 -> 1x4
        squeeze_axis0 = True

    matrix = ComposeTransform()(R, t)

    if squeeze_axis0:
        matrix = matrix[0, :, :]
    return matrix




def compose_transform(R=None, t=None):
    xp = cuda.get_array_module(R, t)

    if R is None:
        Rs = xp.eye(3)[None]
    else:
        Rs = R[None]

    if t is None:
        ts = xp.zeros((1, 3))
    else:
        ts = t[None]

    with chainer.no_backprop_mode():
        Ts = compose_transform_function(Rs, ts).array
        T = Ts[0]

    return T