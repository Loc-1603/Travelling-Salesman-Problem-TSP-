import os, json
from PIL import Image
import matplotlib.pyplot as plt

# --- Đường dẫn ---
MAP_PATH = r"D:\ToiUuDuongDi\map.jpg"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_JSON = os.path.join(SCRIPT_DIR, "provinces_calibrated.json")

# --- Danh sách 63 tỉnh ---
province_names = [
    "Hà Giang","Cao Bằng","Tuyên Quang","Bắc Kạn","Phú Thọ","Thái Nguyên","Lạng Sơn","Bắc Giang","Quảng Ninh",
    "Lai Châu","Lào Cai","Điện Biên","Yên Bái","Sơn La","Hòa Bình","Hà Nội","Vĩnh Phúc","Bắc Ninh","Hưng Yên","Hải Dương",
    "Hải Phòng","Hà Nam","Thái Bình","Nam Định","Ninh Bình","Thanh Hóa","Nghệ An","Hà Tĩnh","Quảng Bình","Quảng Trị",
    "Thừa Thiên - Huế","Đà Nẵng","Quảng Nam","Quảng Ngãi","Bình Định","Phú Yên","Khánh Hòa","Ninh Thuận","Bình Thuận","Kon Tum",
    "Gia Lai","Đắk Lắk","Đắk Nông","Lâm Đồng","Bình Phước","Tây Ninh","Bình Dương","Đồng Nai","Bà Rịa - Vũng Tàu","TP Hồ Chí Minh",
    "Long An","Tiền Giang","Bến Tre","Trà Vinh","Vĩnh Long","Đồng Tháp","An Giang","Cần Thơ","Hậu Giang","Sóc Trăng",
    "Bạc Liêu","Kiên Giang","Cà Mau"
]

# --- Load ảnh ---
if not os.path.exists(MAP_PATH):
    raise SystemExit(f"Không tìm thấy ảnh bản đồ: {MAP_PATH}")
from PIL import Image
img_pil = Image.open(MAP_PATH).convert("RGB")
W, H = img_pil.size
print(f"Map size: {W} x {H} (pixels)")

# --- Setup figure ---
MAX_INCH = 12.0
dpi = 100.0
fig_w = min(W / dpi, MAX_INCH)
fig_h = min(H / dpi, MAX_INCH)
if fig_w == MAX_INCH or fig_h == MAX_INCH:
    scale = min(MAX_INCH * dpi / W, MAX_INCH * dpi / H)
    fig_w = (W * scale) / dpi
    fig_h = (H * scale) / dpi

fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
ax.imshow(img_pil)
ax.set_xlim(0, W)
ax.set_ylim(H, 0)
ax.axis('off')
plt.subplots_adjust(top=0.95, bottom=0.02)

# --- State ---
points = {}
artists = []
index = 0

# --- Load dữ liệu cũ nếu có ---
if os.path.exists(OUT_JSON):
    with open(OUT_JSON, "r", encoding="utf-8") as f:
        old_points = json.load(f)
    # Vẽ lại lên bản đồ
    for i, name in enumerate(province_names):
        if name in old_points:
            x, y = old_points[name]
            m = ax.scatter([x], [y], c='red', s=40, zorder=5)
            t = ax.text(x+8, y-6, name, fontsize=9, color='white', zorder=6)
            artists.append((m, t))
            points[name] = [x, y]
            index = i+1  # cập nhật index đến tỉnh tiếp theo chưa đánh dấu
    print(f"Đã tải lại {len(points)} điểm từ file JSON")
else:
    print("Không tìm thấy dữ liệu cũ, bắt đầu mới")

title_text = ax.text(0.5, -0.02, "", ha='center', va='top', transform=ax.transAxes, fontsize=12)

def update_title():
    if index < len(province_names):
        title_text.set_text(f"Chọn vị trí cho: {province_names[index]} ({index+1}/{len(province_names)}) — Click để chọn, 'r' undo, 'q' lưu+thoát")
    else:
        title_text.set_text("Đã hoàn tất. Có thể nhấn 'r' để sửa lại vài tỉnh cuối, hoặc 'q' để thoát")

update_title()

# --- Events ---
def on_click(event):
    global index
    if event.inaxes != ax or event.xdata is None or event.ydata is None:
        return
    # Nếu đang giữ Ctrl thì bỏ qua
    if event.key == 'control':
        return
    if index < len(province_names):
        x, y = int(round(event.xdata)), int(round(event.ydata))
        name = province_names[index]
        points[name] = [x, y]
        m = ax.scatter([x], [y], c='red', s=40, zorder=5)
        t = ax.text(x+8, y-6, name, fontsize=9, color='white', zorder=6)
        artists.append((m, t))
        print(f"{name}: {x}, {y}")
        index += 1
        update_title()
        fig.canvas.draw()
        if index >= len(province_names):
            print("Đã đánh dấu đủ 63 tỉnh — bạn vẫn có thể nhấn 'r' để sửa lại vài tỉnh cuối trước khi 'q' để lưu")
            # không tự thoát

def on_key(event):
    global index
    if event.key == 'r':
        if artists and index > 0:
            m, t = artists.pop()
            m.remove(); t.remove()
            index -= 1
            del points[province_names[index]]
            update_title()
            fig.canvas.draw()
    elif event.key == 'q':
        save_and_exit()

def save_and_exit():
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(points, f, ensure_ascii=False, indent=2)
    print(f"Đã lưu {len(points)} tỉnh vào: {OUT_JSON}")
    plt.close(fig)

fig.canvas.mpl_connect("button_press_event", on_click)
fig.canvas.mpl_connect("key_press_event", on_key)

print("Hướng dẫn: Click chuột trái để chọn, giữ Ctrl sẽ bỏ qua. 'r' để undo tỉnh cuối, 'q' để lưu và thoát.")

plt.show()
