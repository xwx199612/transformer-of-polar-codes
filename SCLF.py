import numpy as np

def bit_reverse(i, n):
    """将整数 i 的低 n 位做 bit-reversal"""
    rev = 0
    for j in range(n):
        rev <<= 1
        rev |= (i >> j) & 1
    return rev

def f_func(a, b):
    """SC 算子 f"""
    return np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))

def g_func(a, b, u):
    """SC 算子 g"""
    return b + (1 - 2*u) * a

def sc_decode(llr, info_set, N):
    """
    简易 SC 解码
    llr: shape (N,) 的 LLR 输入
    info_set: 已选好的信息位索引列表（升序），长度 = K
    N: 码长, 必须是 2 的幂
    返回: 解码后的 (u_hat, 信息位)
    """
    n = int(np.log2(N))
    # 1) 生成 bit-reversal mapping
    rev = np.array([bit_reverse(i, n) for i in range(N)], dtype=int)
    # 2) 初始化 LLR 树和 bit 决策表
    # llr_tree[level][i]: level 从 n down to 0, i 是当前节点索引
    llr_tree = [np.zeros(2**(n-level)) for level in range(n+1)]
    bit_tree = [np.zeros(2**(n-level), dtype=int) for level in range(n+1)]
    llr_tree[n] = llr.copy()
    
    def recurse(level, idx):
        """SC 递归——处理第 level 层、节点 idx"""
        if level == n:
            # 叶子节点：作决策
            pos = idx
            if pos in info_set:
                # 信息位：硬判决
                bit_tree[level][idx] = 1 if llr_tree[level][idx] < 0 else 0
            else:
                # 冻结位：固定为 0
                bit_tree[level][idx] = 0
            return bit_tree[level][idx]
        
        # 1) 左子树 f
        llr_tree[level][idx] = f_func(
            llr_tree[level+1][idx*2],
            llr_tree[level+1][idx*2+1]
        )
        u_left = recurse(level+1, idx*2)
        
        # 2) 右子树 g
        llr_tree[level][idx] = g_func(
            llr_tree[level+1][idx*2],
            llr_tree[level+1][idx*2+1],
            u_left
        )
        u_right = recurse(level+1, idx*2+1)
        
        # 3) 合并 partial sum
        bit_tree[level][idx] = u_left ^ u_right
        return bit_tree[level][idx]

    # 从根节点开始递归
    recurse(0, 0)
    # 最终在 level=0 的 bit_tree[0][0] 无用，只需 level=n 的 bits
    u_hat = bit_tree[n].copy()
    # 还原 bit-reversal
    u_hat = u_hat[rev]
    # 提取信息位
    info_hat = u_hat[info_set]
    return u_hat, info_hat

# 示例用法
if __name__ == "__main__":
    N = 8
    K = 4
    info_set = [0, 1, 2, 3]   # 举例：前 4 位为信息位
    # 构造一个简单 (8,4) Polar 码
    # 例如随意选一组 message + 0 冻结填入
    message = np.array([1,0,1,1], dtype=int)
    u = np.zeros(N, dtype=int)
    u[info_set] = message
    # 极化编码: u 乘以 F_N · B_N
    F = np.array([[1,0],[1,1]])
    F_N = F
    while F_N.shape[0] < N:
        F_N = np.kron(F_N, F)
    rev = np.array([bit_reverse(i, int(np.log2(N))) for i in range(N)])
    G = F_N[:, rev]
    x = (1 - 2*(u @ G % 2)).astype(float)  # BPSK: 0→+1,1→-1

    # 加 AWGN 噪声
    snr_db = 1.0
    sigma = np.sqrt(1 / (2 * R * 10**(snr_db / 10)))
    y = x + np.random.normal(scale=sigma, size=N)
    llr = 2*y / (sigma**2)

    u_hat, info_hat = sc_decode(llr, info_set, N)
    print("原始信息:", message)
    print("解码信息:", info_hat)