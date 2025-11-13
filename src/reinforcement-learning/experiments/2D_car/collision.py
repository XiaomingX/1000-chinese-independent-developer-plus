import numpy as np

# 计算两条线段的交点
def check_line_intersection():
    # 定义两条线段的端点
    p = np.array([0, 0])  # 线段1的起点
    r = np.array([1, 1])  # 线段1的方向向量
    q = np.array([0.1, 0.1])  # 线段2的起点
    s = np.array([0.1, 0.1])  # 线段2的方向向量

    # 判断两条线段是否共线
    if np.cross(r, s) == 0 and np.cross(q - p, r) == 0:
        # 共线时，计算 t0 和 t1，表示线段相交的比例
        t0 = np.dot(q - p, r) / np.dot(r, r)
        t1 = t0 + np.dot(s, r) / np.dot(r, r)
        print(f"t0: {t0}, t1: {t1}")
        
        # 判断线段是否重叠
        if ((np.dot(s, r) > 0 and 0 <= t1 - t0 <= 1) or 
            (np.dot(s, r) <= 0 and 0 <= t0 - t1 <= 1)):
            print('线段共线并且重叠')
        else:
            print('线段共线但不重叠')
    # 判断是否平行
    elif np.cross(r, s) == 0 and np.cross(q - p, r) != 0:
        print('两条线段平行')
    else:
        # 计算交点的参数 t 和 u，判断是否在有效范围内
        t = np.cross(q - p, s) / np.cross(r, s)
        u = np.cross(q - p, r) / np.cross(r, s)
        if 0 <= t <= 1 and 0 <= u <= 1:
            intersection_point = p + t * r
            print(f'交点: {intersection_point}')
        else:
            print('两条线段不相交')

# 计算点到线段的距离，并判断点的位置关系
def point_to_segment_distance():
    # 定义点和线段的端点
    p = np.array([-1, 1])  # 点
    a = np.array([0, 1])   # 线段的起点
    b = np.array([1, 0])   # 线段的终点
    
    # 计算线段 AB 的方向向量和点 A 到点 P 的向量
    ab = b - a
    ap = p - a
    
    # 计算点到线段的垂直距离
    distance = np.abs(np.cross(ab, ap) / np.linalg.norm(ab))
    print(f'点到线段的距离: {distance}')
    
    # 计算点到线段两端的夹角
    bp = p - b
    cos_theta1 = np.dot(ap, ab) / (np.linalg.norm(ap) * np.linalg.norm(ab))
    theta1 = np.arccos(cos_theta1)
    cos_theta2 = np.dot(bp, ab) / (np.linalg.norm(bp) * np.linalg.norm(ab))
    theta2 = np.arccos(cos_theta2)
    
    # 判断点的位置关系
    if np.pi / 2 <= (theta1 % (2 * np.pi)) <= 3 / 2 * np.pi:
        print('点在 A 之外')
    elif -np.pi / 2 <= (theta2 % (2 * np.pi)) <= np.pi / 2:
        print('点在 B 之外')
    else:
        print('点在 A 和 B 之间')

# 主函数，调用所有功能
def main():
    print("检查线段交点:")
    check_line_intersection()
    print("\n计算点到线段的距离和位置:")
    point_to_segment_distance()

if __name__ == '__main__':
    main()
