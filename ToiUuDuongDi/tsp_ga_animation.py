import os, json, random, math
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# ========== Cấu hình ==========
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MAP_PATH = os.path.join(SCRIPT_DIR, "map.jpg")
CALIBRATED_PATH = os.path.join(SCRIPT_DIR, "provinces_calibrated.json")

POP_SIZE = 100        # số cá thể trong quần thể
N_GENERATIONS = 100   # số thế hệ
MUTATION_RATE = 0.03  # tỉ lệ đột biến
N_CITIES = 35         # số tỉnh sẽ chọn ngẫu nhiên từ 63

# ========== CÀI ĐẶT FONT CHỮ ĐỂ HỖ TRỢ TIẾNG VIỆT ==========
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']  # Font hỗ trợ Unicode
plt.rcParams['axes.unicode_minus'] = False

# ========== Load dữ liệu tỉnh ==========
with open(CALIBRATED_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

# data là dict {tên: [x, y]} → chuyển về list [(name,x,y)]
all_provinces = [(name, xy[0], xy[1]) for name, xy in data.items()]

# Lấy ngẫu nhiên N_CITIES tỉnh từ 63 tỉnh
provinces = random.sample(all_provinces, N_CITIES)

# CỐ ĐỊNH ĐIỂM XUẤT PHÁT - lấy tỉnh đầu tiên làm điểm xuất phát cố định
start_city = provinces[0]
start_city_name = start_city[0]
start_city_index = 0  # Chỉ số của điểm xuất phát luôn là 0

names = [p[0] for p in provinces]
coords = np.array([(p[1], p[2]) for p in provinces])

# ========== Hàm hỗ trợ ==========
def distance(a, b):
    return math.hypot(a[0]-b[0], a[1]-b[1])

dist_matrix = np.zeros((N_CITIES, N_CITIES))
for i in range(N_CITIES):
    for j in range(N_CITIES):
        dist_matrix[i, j] = distance(coords[i], coords[j])

def route_length(route):
    # Đảm bảo điểm xuất phát cố định là vị trí đầu tiên
    return sum(dist_matrix[route[i], route[i+1]] for i in range(N_CITIES-1)) + dist_matrix[route[-1], route[0]]

# ========== Giải thuật di truyền ==========
def create_population():
    # Tạo quần thể với điểm xuất phát cố định ở vị trí đầu tiên
    population = []
    for _ in range(POP_SIZE):
        # Tạo route với điểm xuất phát cố định là 0
        route = [0]  # Luôn bắt đầu từ điểm xuất phát cố định
        # Thêm các thành phố còn lại (1 đến N_CITIES-1) theo thứ tự ngẫu nhiên
        other_cities = random.sample(range(1, N_CITIES), N_CITIES-1)
        route.extend(other_cities)
        population.append(route)
    return population

def crossover(p1, p2):
    # Lai ghép với điểm xuất phát cố định
    start, end = sorted(random.sample(range(1, N_CITIES), 2))  # Bỏ qua điểm xuất phát cố định
    child = [-1] * N_CITIES
    child[0] = 0  # Điểm xuất phát luôn cố định
    
    # Sao chép đoạn từ cha/mẹ
    child[start:end] = p1[start:end]
    
    # Điền các thành phố còn lại từ cha/mẹ thứ hai
    fill = [x for x in p2 if x not in child and x != 0]
    j = 0
    for i in range(N_CITIES):
        if child[i] == -1:
            child[i] = fill[j]
            j += 1
    return child

def mutate(route):
    if random.random() < MUTATION_RATE:
        # Đột biến nhưng không thay đổi điểm xuất phát (vị trí 0)
        i, j = random.sample(range(1, N_CITIES), 2)  # Chỉ đột biến các vị trí từ 1 trở đi
        route[i], route[j] = route[j], route[i]

def next_generation(pop):
    ranked = sorted(pop, key=route_length)
    new_pop = ranked[:20]  # elitism
    while len(new_pop) < POP_SIZE:
        p1, p2 = random.sample(ranked[:50], 2)
        child = crossover(p1, p2)
        mutate(child)
        new_pop.append(child)
    return new_pop

# ========== Load ảnh nền ==========
img = Image.open(MAP_PATH).convert("RGB")
W, H = img.size

# ========== THIẾT LẬP GIAO DIỆN ==========
# Tạo figure với kích thước phù hợp
fig = plt.figure(figsize=(16, 12))

# Chia layout thành 2 phần: bản đồ (70%) và thông tin (30%)
gs = plt.GridSpec(1, 2, width_ratios=[6, 4])

ax_map = plt.subplot(gs[0])  # Bản đồ chiếm 60%
ax_info = plt.subplot(gs[1])  # Thông tin chiếm 40%

# Hiển thị bản đồ
ax_map.imshow(img)
ax_map.set_xlim(0, W)
ax_map.set_ylim(H, 0)
ax_map.axis('off')
ax_map.set_title("BẢN ĐỒ HÀNH TRÌNH", fontsize=16, fontweight='bold', pad=10)

# Thiết lập phần thông tin bài toán
ax_info.axis('off')
ax_info.set_xlim(0, 1)
ax_info.set_ylim(1, 0)
ax_info.set_title("THÔNG TIN BÀI TOÁN", fontsize=14, fontweight='bold', pad=10)

# ========== Chạy GA + Vẽ ==========
pop = create_population()
best_route = pop[0]
best_len = route_length(best_route)

# Vẽ các điểm tỉnh và tên
colors = ['yellow'] * N_CITIES
# Điểm xuất phát sẽ có màu đỏ
colors[start_city_index] = 'red'

scatter = ax_map.scatter(coords[:,0], coords[:,1], c=colors, s=50, zorder=5)
line_plot, = ax_map.plot([], [], 'r-', lw=1.5)

# Thêm tên các tỉnh
for i, (name, x, y) in enumerate(provinces):
    ax_map.annotate(name, (x, y), xytext=(5, 5), textcoords='offset points', 
                   fontsize=8, bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))

# Hiển thị thông tin bài toán
description_text = f"""BÀI TOÁN NGƯỜI BÁN HÀNG RONG (TSP)

- Người bán hàng xuất phát từ: {start_city_name}
- Phải đi qua {N_CITIES-1} tỉnh thành khác
- Mỗi tỉnh chỉ được đến đúng 1 lần
- Cuối cùng quay lại điểm xuất phát
- Mục tiêu: Tìm lộ trình có tổng quãng đường ngắn nhất

THUẬT TOÁN DI TRUYỀN
- Số thế hệ: {N_GENERATIONS}
- Kích thước quần thể: {POP_SIZE}
- Tỉ lệ đột biến: {MUTATION_RATE*100}%"""

# HIỂN THỊ THÔNG TIN BÀI TOÁN
ax_info.text(0.05, 0.05, description_text, fontsize=10, verticalalignment='top', linespacing=1.4)

# Text cho kết quả tạm thời
temp_result = ax_info.text(0.07, 0.5, "Đang chạy thuật toán...", fontsize=9, 
                          verticalalignment='top', color='blue',
                          bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.5'))

for gen in range(N_GENERATIONS):
    pop = next_generation(pop)
    current_best = min(pop, key=route_length)
    curr_len = route_length(current_best)
    if curr_len < best_len:
        best_len = curr_len
        best_route = current_best

    # Cập nhật hình
    ordered_coords = coords[best_route + [best_route[0]]]
    line_plot.set_data(ordered_coords[:,0], ordered_coords[:,1])
    ax_map.set_title(f"Thế hệ {gen+1}/{N_GENERATIONS} — Độ dài tốt nhất: {best_len:.0f}", fontsize=12)
    
    # Cập nhật thông tin kết quả tạm thời
    if gen % 10 == 0 or gen == N_GENERATIONS - 1:
        temp_result.set_text(f"Thế hệ: {gen+1}/{N_GENERATIONS}\nĐộ dài tốt nhất: {best_len:.0f}")
    
    plt.pause(0.001)

# Xóa kết quả tạm thời
temp_result.set_text("")

# Hàm hiển thị lộ trình theo 2 cột với font nhỏ hơn
def create_route_text(route, names):
    """Tạo văn bản lộ trình hiển thị theo 2 cột"""
    text_lines = []
    num_cities = len(route)
    rows = (num_cities + 1) // 2  # Số dòng cần thiết (làm tròn lên)
    
    for i in range(rows):
        line = ""
        for j in range(2):
            idx = i + j * rows
            if idx < num_cities:
                marker = " (XP)" if idx == 0 else ""
                province_name = names[route[idx]]
                if len(province_name) > 10:
                    province_name = province_name[:8] + ".."
                line += f"{idx+1:2d}. {province_name:10s}{marker}   "
        text_lines.append(line)
    
    # Thêm điểm quay về
    text_lines.append(f"\n{num_cities+1:2d}. {names[route[0]]} (quay về)")
    return "\n".join(text_lines)

# Hiển thị kết quả cuối cùng
route_text = create_route_text(best_route, names)
final_text = f"""KẾT QUẢ CUỐI CÙNG

• Độ dài lộ trình: {best_len:.0f}
• Số thế hệ: {N_GENERATIONS}
• Tỉnh xuất phát: {start_city_name}

LỘ TRÌNH TỐI ƯU:
{route_text}"""

# HIỂN THỊ KẾT QUẢ
ax_info.text(0.07, 0.45, final_text, fontsize=9, 
            verticalalignment='top', color='green',
            bbox=dict(facecolor='white', alpha=0.9, boxstyle='round,pad=0.8'),
            family='DejaVu Sans')

# Hiển thị kết quả trong console
print("="*60)
print("KẾT QUẢ CUỐI CÙNG")
print("="*60)
print(f"Độ dài lộ trình: {best_len:.0f}")
print(f"Số thế hệ: {N_GENERATIONS}")
print(f"Tỉnh xuất phát: {start_city_name}")
print("\nLỘ TRÌNH TỐI ƯU:")
for i, city_idx in enumerate(best_route):
    marker = " (xuất phát)" if i == 0 else ""
    print(f"{i+1:2d}. {names[city_idx]}{marker}")
print(f"{len(best_route)+1:2d}. {names[best_route[0]]} (quay về)")
print("="*60)

# Điều chỉnh khoảng cách để tránh bị che
plt.tight_layout()
plt.subplots_adjust(bottom=0.1)
plt.show()