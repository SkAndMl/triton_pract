import torch
import triton
import triton.language as tl


@triton.jit
def _softmax_kernel(
    x_ptr,
    y_ptr,
    stride_xm,
    stride_xn,
    stride_ym,
    stride_yn,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    row_start = x_ptr + pid * stride_xm

    m = tl.full((), float("-inf"), tl.float32)
    l = tl.full((), 0.0, tl.float32)

    for col_start in range(0, n_cols, BLOCK_SIZE):
        col_offs = col_start + tl.arange(0, BLOCK_SIZE)
        mask = col_offs < n_cols

        x = tl.load(
            row_start + col_offs * stride_xn, mask=mask, other=-float("inf")
        ).to(tl.float32)
        m_new = tl.maximum(m, tl.max(x, axis=0))
        alpha = tl.exp(m - m_new)

        l = l * alpha + tl.sum(tl.exp(x - m_new), axis=0)
        m = m_new

    for col_start in range(0, n_cols, BLOCK_SIZE):
        col_offs = col_start + tl.arange(0, BLOCK_SIZE)
        mask = col_offs < n_cols

        x = tl.load(
            row_start + col_offs * stride_xn, mask=mask, other=-float("inf")
        ).to(tl.float32)
        y = tl.exp(x - m) / l

        tl.store(y_ptr + pid * stride_ym + col_offs * stride_yn, y, mask=mask)


def softmax_triton(x: torch.Tensor, BLOCK_SIZE=1024) -> torch.Tensor:

    y = torch.empty_like(x)
    M, N = x.shape
    grid = (M,)

    _softmax_kernel[grid](
        x,
        y,
        x.stride(0),
        x.stride(1),
        y.stride(0),
        y.stride(1),
        N,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return y


if __name__ == "__main__":
    shapes = [(3, 7), (1, 1024), (512, 1024), (64, 1031)]

    for m, n in shapes:
        x = torch.randn(m, n, dtype=torch.float16, device="cuda:0")
        out = softmax_triton(x)
        ref = torch.softmax(x, dim=-1)

        torch.testing.assert_close(out, ref, rtol=5e-3, atol=5e-3)

    print("passed!")
