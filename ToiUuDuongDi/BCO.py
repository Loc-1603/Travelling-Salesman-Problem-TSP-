import os, json, random, math
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# ========== Cấu hình ==========
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MAP_PATH = os.path.join(SCRIPT_DIR, "map.jpg")
CALIBRATED_PATH = os.path.join(SCRIPT_DIR, "provinces_calibrated.json")

POP_SIZE = 100        # số ong trong đàn
N_GENERATIONS = 100   # số vòng lặp
N_SCOUTS = 30         # số ong trinh sát
N_FOLLOWERS = POP_SIZE - N_SCOUTS # số ong theo dõi
N_CITIES = 30         # số tỉnh chọn ngẫu nhiên từ 63

# ========== CÀI ĐẶT FONT CHỮ ==========
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ========== Load dữ liệu ==========
with open(CALIBRATED_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

all_provinces = [(name, xy[0], xy[1]) for name, xy in data.items()]
provinces = random.sample(all_provinces, N_CITIES)

start_city = provinces[0]
start_city_name = start_city[0]
start_city_index = 0

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
    return sum(dist_matrix[route[i], route[i+1]] for i in range(N_CITIES-1)) + dist_matrix[route[-1], route[0]]

def create_random_route():
    route = [0]  # xuất phát cố định
    other = random.sample(range(1, N_CITIES), N_CITIES-1)
    route.extend(other)
    return route

def local_search(route):
    """Tìm kiếm cục bộ bằng cách đảo chỗ 2 thành phố"""
    new_route = route[:]
    i, j = random.sample(range(1, N_CITIES), 2)
    new_route[i], new_route[j] = new_route[j], new_route[i]
    return new_route

# ========== BCO ==========
def bee_colony_iteration(pop):
    fitness = np.array([1.0 / route_length(r) for r in pop])  # fitness càng lớn càng tốt
    probs = fitness / fitness.sum()

    new_pop = []

    # Ong trinh sát (khám phá)
    for _ in range(N_SCOUTS):
        if random.random() < 0.5:
            new_pop.append(create_random_route())
        else:
            r = random.choice(pop)
            new_pop.append(local_search(r))

    # Ong theo dõi (khai thác)
    for _ in range(N_FOLLOWERS):
        parent = pop[np.random.choice(len(pop), p=probs)]
        new_pop.append(local_search(parent))

    return new_pop

# ========== Load ảnh nền ==========
img = Image.open(MAP_PATH).convert("RGB")
W, H = img.size

fig = plt.figure(figsize=(16, 12))
gs = plt.GridSpec(1, 2, width_ratios=[6, 4])
ax_map = plt.subplot(gs[0])
ax_info = plt.subplot(gs[1])

ax_map.imshow(img)
ax_map.set_xlim(0, W)
ax_map.set_ylim(H, 0)
ax_map.axis('off')
ax_map.set_title("BẢN ĐỒ HÀNH TRÌNH", fontsize=16, fontweight='bold', pad=10)

ax_info.axis('off')
ax_info.set_xlim(0, 1)
ax_info.set_ylim(1, 0)
ax_info.set_title("THÔNG TIN BÀI TOÁN", fontsize=14, fontweight='bold', pad=10)

# Quần thể ban đầu
pop = [create_random_route() for _ in range(POP_SIZE)]
best_route = min(pop, key=route_length)
best_len = route_length(best_route)
initial_len = best_len   # <--- Lưu lại độ dài ban đầu


# Vẽ tỉnh
colors = ['yellow'] * N_CITIES
colors[start_city_index] = 'red'
ax_map.scatter(coords[:,0], coords[:,1], c=colors, s=50, zorder=5)
line_plot, = ax_map.plot([], [], 'r-', lw=1.5)

for i, (name, x, y) in enumerate(provinces):
    ax_map.annotate(name, (x, y), xytext=(5, 5), textcoords='offset points', fontsize=8,
                   bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))

description_text = f"""BÀI TOÁN NGƯỜI BÁN HÀNG RONG (TSP)

- Xuất phát từ: {start_city_name}
- Qua {N_CITIES-1} tỉnh
- Mỗi tỉnh chỉ đi 1 lần
- Quay lại điểm xuất phát
- Mục tiêu: Tổng quãng đường ngắn nhất

THUẬT TOÁN BCO
- Số thế hệ: {N_GENERATIONS}
- Số ong trinh sát: {N_SCOUTS}
- Số ong theo dõi: {N_FOLLOWERS}"""

ax_info.text(0.05, 0.05, description_text, fontsize=10, verticalalignment='top', linespacing=1.4)

temp_result = ax_info.text(0.07, 0.5, "Đang chạy thuật toán...", fontsize=9,
                          verticalalignment='top', color='blue',
                          bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.5'))

# Chạy BCO
for gen in range(N_GENERATIONS):
    pop = bee_colony_iteration(pop)
    current_best = min(pop, key=route_length)
    curr_len = route_length(current_best)
    if curr_len < best_len:
        best_len = curr_len
        best_route = current_best

    ordered_coords = coords[best_route + [best_route[0]]]
    line_plot.set_data(ordered_coords[:,0], ordered_coords[:,1])
    ax_map.set_title(f"Thế hệ {gen+1}/{N_GENERATIONS} — Độ dài tốt nhất: {best_len:.0f}", fontsize=12)

    if gen % 10 == 0 or gen == N_GENERATIONS - 1:
        temp_result.set_text(f"Thế hệ: {gen+1}/{N_GENERATIONS}\nĐộ dài tốt nhất: {best_len:.0f}")

    plt.pause(0.001)

temp_result.set_text("")

# Hiển thị lộ trình cuối cùng
def create_route_text(route, names):
    text_lines = []
    rows = (len(route) + 1) // 2
    for i in range(rows):
        line = ""
        for j in range(2):
            idx = i + j * rows
            if idx < len(route):
                marker = " (XP)" if idx == 0 else ""
                province_name = names[route[idx]]
                if len(province_name) > 10:
                    province_name = province_name[:8] + ".."
                line += f"{idx+1:2d}. {province_name:10s}{marker}   "
        text_lines.append(line)
    text_lines.append(f"\n{len(route)+1:2d}. {names[route[0]]} (quay về)")
    return "\n".join(text_lines)

route_text = create_route_text(best_route, names)
improvement = ((initial_len - best_len) / initial_len) * 100  # <--- Tính cải thiện
final_text = f"""KẾT QUẢ CUỐI CÙNG

• Độ dài lộ trình ban đầu: {initial_len:.0f}
• Độ dài lộ trình tối ưu: {best_len:.0f}
• Quãng đường cải thiện: {improvement:.2f}%
• Số thế hệ: {N_GENERATIONS}
• Xuất phát: {start_city_name}

LỘ TRÌNH TỐI ƯU:
{route_text}"""

ax_info.text(0.07, 0.45, final_text, fontsize=9,
            verticalalignment='top', color='green',
            bbox=dict(facecolor='white', alpha=0.9, boxstyle='round,pad=0.8'),
            family='DejaVu Sans')

plt.tight_layout()
plt.subplots_adjust(bottom=0.1)
plt.show()
