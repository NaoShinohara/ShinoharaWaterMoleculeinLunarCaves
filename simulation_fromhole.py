import os
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor

# 定数の定義
L = 100  # 孔直径[m]
z0 = 0  # 孔底z座標[m]
d = 60  # 天井z座標[m]
l = 3000  # 空洞長さ[m]
n = 10000  # 粒子数[個]


# x方向速度
def velocity_x(x, y, z, phi, psi):
    if z == z0 or -z0:
        return np.cos(phi) * np.cos(psi)
    elif z == d:
        return -np.cos(phi) * np.cos(psi)
    elif y == L / 2:
        return -np.cos(phi) * np.cos(psi)
    elif y == -L / 2:
        return np.cos(phi) * np.cos(psi)
    elif x == l:
        return -np.sin(psi)
    elif x == -l:
        return np.sin(psi)


# y方向速度
def velocity_y(x, y, z, phi, psi):
    if z == z0 or -z0:
        return np.sin(phi) * np.cos(psi)
    elif z == d:
        return -np.sin(phi) * np.cos(psi)
    elif y == L / 2:
        return -np.sin(psi)
    elif y == -L / 2:
        return np.sin(psi)
    elif x == l:
        return np.cos(phi) * np.cos(psi)
    elif x == -l:
        return -np.cos(phi) * np.cos(psi)


# z方向速度
def velocity_z(x, y, z, phi, psi):
    if z == z0 or -z0:
        return np.sin(psi)
    elif z == d:
        return -np.sin(psi)
    elif y == L / 2:
        return np.sin(phi) * np.cos(psi)
    elif y == -L / 2:
        return np.sin(phi) * np.cos(psi)
    elif x == l:
        return np.sin(phi) * np.cos(psi)
    elif x == -l:
        return np.sin(phi) * np.cos(psi)


def time_x(x, y, z, phi, psi):
    vx = velocity_x(x, y, z, phi, psi)
    if vx >= 0:
        return (l - x) / vx
    elif vx < 0:
        return (l + x) / abs(vx)


def time_y(x, y, z, phi, psi):
    vy = velocity_y(x, y, z, phi, psi)
    if vy >= 0:
        return ((L / 2) - y) / vy
    elif vy < 0:
        return ((L / 2) + y) / abs(vy)


def time_z(x, y, z, phi, psi):
    vz = velocity_z(x, y, z, phi, psi)
    if vz >= 0:
        return (d - z) / vz
        # print(f'Velocity is {velocity_z(x,z,psi)}')
    elif vz < 0:
        return z / abs(vz)
        # print(f'Velocity is {velocity_z(x,z,psi)}')


# 最小到達時間
def minimum_time(x, y, z, phi, psi):
    tmin = min(
        time_x(x, y, z, phi, psi), time_y(x, y, z, phi, psi), time_z(x, y, z, phi, psi)
    )
    return tmin


def distance(x, y, z, new_x, new_y, new_z):
    distance = np.sqrt((new_x - x) ** 2 + (new_y - y) ** 2 + (new_z - z) ** 2)
    return round(distance, 2)


def simulate_flight(x, y, z):
    count = 0  # 飛行回数/1粒子
    total_distance = 0  # 脱出までの距離/1粒子

    while not (z == d and np.sqrt(x** 2 + y**2) < L / 2):
        phi = np.random.uniform(0, 2 * np.pi)  # 偏角
        U = np.random.uniform(0, 1)  # 乱数
        psi = np.arcsin(np.sqrt(U))  # 仰角

        min_time = minimum_time(x, y, z, phi, psi)

        vel_x = velocity_x(x, y, z, phi, psi)
        vel_y = velocity_y(x, y, z, phi, psi)
        vel_z = velocity_z(x, y, z, phi, psi)

        new_x = x + vel_x * min_time
        new_y = y + vel_y * min_time
        new_z = z + vel_z * min_time

        dist = distance(new_x, new_y, new_z, x, y, z)  # 旧ポジから新ポジまでの距離

        total_distance += dist
        count += 1

        x = round(new_x, 2)
        y = round(new_y, 2)
        z = round(new_z, 2)

    return count, total_distance


def simulate_flight_with_random_init(_):
    r = np.random.uniform(0, L / 2)
    angle = np.random.uniform(0, 2 * np.pi)
    x = r * np.cos(angle)
    y = r * np.sin(angle)
    return simulate_flight(x, y, 0)


# 並列処理でシミュレーションを実行
def parallel_simulate_flights(seed):
    np.random.seed(seed)
    with ProcessPoolExecutor(max_workers=18) as executor:
        results = list(
            executor.map(simulate_flight_with_random_init, range(n))
        )
    # 結果の統合
    counts = [result[0] for result in results]  # 1番目の戻り値
    total_distances = [result[1] for result in results]  # 2番目の戻り値

    return counts, total_distances


if __name__ == "__main__":
    num = int(np.log10(n)+2)
    # プログラム自体の場所を取得
    base_dir = os.path.dirname(os.path.abspath(__file__))
    # 保存先を「プログラムの場所/数値m」にする
    output_dir = os.path.join(base_dir, f"{l}m")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"3D_{l}m_1stinc_10^{num}_Knu.csv")
    
    all_counts = []
    all_distances = []
    
    particles_per_loop = n // 100
    
    # 100ループ実行
    for loop in range(100):
        counts, total_distances = parallel_simulate_flights(loop)
        all_counts.extend(counts)
        all_distances.extend(total_distances)
    
        df = pd.DataFrame(
            {
                "Counts": all_counts,
                "total_distances": all_distances,
            }
        )

        df.to_csv(output_path, index=False)
        print(f"ループ {loop + 1}/100 完了: 現在の総粒子数 = {len(all_counts)}")