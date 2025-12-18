import cv2
import mediapipe as mp
import numpy as np
import time

# --- Cấu hình MediaPipe ---
mp_hands = mp.solutions.hands
# Chỉ cần phát hiện một bàn tay
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

# --- Cấu hình chương trình ---
PINCH_THRESHOLD = 50  # Khoảng cách tối đa (pixel) giữa ngón cái và ngón trỏ để được coi là "chụm"
BOX_SIZE = 80         # Kích thước của các hộp vuông
GRAB_DISTANCE = 50    # Khoảng cách tối đa (pixel) từ ngón tay đến hộp để bắt đầu "cầm"
VELOCITY_SCALE = 1.2  # Hệ số nhân vận tốc để làm cho cú ném nổi bật hơn
MAX_HISTORY_POINTS = 3 # Số điểm lịch sử để tính vận tốc

# --- Cấu hình Vật Lý ---
GRAVITY = 1.2         # Gia tốc trọng trường (pixel/frame^2)
FRICTION = 0.5        # Hệ số giảm tốc khi va chạm (hệ số đàn hồi)

# --- Lớp đại diện cho các vật thể (Hộp) ---
class Box:
    def __init__(self, x, y, size):
        self.x = x
        self.y = y
        self.size = size
        self.is_held = False # Trạng thái: đang được giữ hay không
        self.vx = 0          # Vận tốc theo trục X (ngang)
        self.vy = 0          # Vận tốc theo trục Y (dọc)

    def get_center(self):
        return (self.x + self.size // 2, self.y + self.size // 2)

    def is_inside(self, px, py):
        """Kiểm tra xem điểm (px, py) có nằm trong hộp không."""
        # Chuyển đổi tọa độ hộp sang int để kiểm tra va chạm
        x_int = int(self.x)
        y_int = int(self.y)
        return x_int < px < x_int + self.size and y_int < py < y_int + self.size

    # Cập nhật: Đã triển khai cơ chế va chạm hộp-hộp đầy đủ
    def update_physics(self, height, all_boxes):
        """Cập nhật vị trí hộp khi không được giữ (áp dụng trọng lực và kiểm tra va chạm)."""
        if not self.is_held:
            # 1. Áp dụng trọng lực và vận tốc
            self.vy += GRAVITY
            
            # Cập nhật vị trí tạm thời
            self.x += self.vx 
            self.y += self.vy
            
            # --- 2. Kiểm tra và giải quyết va chạm giữa các hộp (Collision Resolution) ---
            
            for other_box in all_boxes:
                if other_box is self or other_box.is_held:
                    continue 

                # Tính độ chênh lệch giữa các tâm (dx, dy)
                dx = (self.x + self.size / 2) - (other_box.x + other_box.size / 2)
                dy = (self.y + self.size / 2) - (other_box.y + other_box.size / 2)
                
                # Tổng kích thước nửa hộp (Vì cả hai hộp đều có kích thước là BOX_SIZE/2)
                combined_half_size = self.size 
                
                # Tính độ sâu chồng lấn
                overlap_x = combined_half_size - abs(dx)
                overlap_y = combined_half_size - abs(dy)
                
                # Va chạm xảy ra nếu có chồng lấn trên cả hai trục
                if overlap_x > 0 and overlap_y > 0:
                    
                    # Xác định Trục dịch chuyển tối thiểu (MTV)
                    if overlap_x < overlap_y:
                        # Giải quyết va chạm theo trục X
                        move_amount = overlap_x
                        if dx > 0: # Hộp self nằm bên phải, đẩy sang phải
                            self.x += move_amount
                            move_dir_x = 1
                        else:      # Hộp self nằm bên trái, đẩy sang trái
                            self.x -= move_amount
                            move_dir_x = -1
                            
                        # Giải quyết vận tốc X (phản xạ/nảy)
                        # Chỉ áp dụng phản lực nếu đang di chuyển về phía nhau
                        if self.vx * move_dir_x < 0:
                            self.vx *= -FRICTION
                        
                    else:
                        # Giải quyết va chạm theo trục Y
                        move_amount = overlap_y
                        if dy > 0: # Hộp self nằm bên dưới, đẩy xuống dưới
                            self.y += move_amount
                            move_dir_y = 1
                        else:      # Hộp self nằm bên trên, đẩy lên trên (tiếp đất)
                            self.y -= move_amount
                            move_dir_y = -1
                            
                        # Giải quyết vận tốc Y (phản xạ/nảy)
                        if self.vy * move_dir_y < 0:
                            self.vy *= -FRICTION
                        
                        # Dừng hẳn nếu vận tốc nhỏ và đang "tiếp đất" (đẩy lên)
                        if move_dir_y == -1 and abs(self.vy) < 1.5:
                            self.vy = 0
                            
            # --- 3. Kiểm tra va chạm BIÊN (sàn, trần, tường) ---
            
            landed = False
            
            # Kiểm tra va chạm sàn
            if self.y + self.size > height:
                self.y = height - self.size # Đặt lên sàn
                self.vy *= -FRICTION         # Nảy
                if abs(self.vy) < 1.5:
                    self.vy = 0
                landed = True
            
            # Kiểm tra va chạm trần
            if self.y < 0:
                self.y = 0
                self.vy *= -FRICTION
                
            # Kiểm tra va chạm tường (trái/phải)
            if self.x < 0 or self.x + self.size > width:
                self.vx *= -FRICTION # Đảo ngược vận tốc X và giảm tốc
                self.x = np.clip(self.x, 0, width - self.size) # Đảm bảo nằm trong biên
            
            # Áp dụng ma sát ngang (chỉ khi hộp đang nằm yên trên mặt đất hoặc hộp khác)
            if (landed or self.vy == 0) and abs(self.vx) > 0.1: 
                self.vx *= 0.95 
            elif abs(self.vx) <= 0.1:
                self.vx = 0

    def draw(self, image):
        """Vẽ hộp lên hình ảnh."""
        # Chuyển đổi tọa độ float (y) sang int để vẽ
        x_int = int(self.x)
        y_int = int(self.y)
        
        color = (0, 255, 0) if not self.is_held else (0, 0, 255) # Xanh lá khi chưa cầm, Đỏ khi đang cầm
        
        # Vẽ hình chữ nhật
        cv2.rectangle(image, (x_int, y_int), (x_int + self.size, y_int + self.size), color, 2)
        
        # Thêm chữ (Sử dụng tiếng Anh/tiệt/việt không dấu để tối ưu tương thích với cv2.putText)
        text = "HELD" if self.is_held else "BOX"
        text_color = (0, 0, 255) if self.is_held else (0, 255, 0)
        cv2.putText(image, text, (x_int + 5, y_int + self.size - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)

# Khởi tạo danh sách các hộp cố định
# Tọa độ sẽ được định nghĩa sau khi biết kích thước khung hình
boxes = []
# Biến lưu trữ hộp đang được giữ
held_box = None 
# Mảng lưu trữ lịch sử vị trí điểm chụm (grab_point)
# Mỗi phần tử là một tuple (x, y)
grab_history = [] 

# --- Hàm tiện ích tính khoảng cách Euclidean ---
def get_distance(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

# --- Hàm đếm ngón tay mở rộng ---
def count_extended_fingers(landmarks):
    count = 0
    # Y-axis tăng xuống dưới. Ngón tay mở (chỉ lên) khi Tip Y < PIP Y.
    
    # 4 ngón: Index, Middle, Ring, Pinky
    finger_tips = [mp_hands.HandLandmark.INDEX_FINGER_TIP, mp_hands.HandLandmark.MIDDLE_FINGER_TIP, 
                   mp_hands.HandLandmark.RING_FINGER_TIP, mp_hands.HandLandmark.PINKY_TIP]
    finger_pips = [mp_hands.HandLandmark.INDEX_FINGER_PIP, mp_hands.HandLandmark.MIDDLE_FINGER_PIP, 
                   mp_hands.HandLandmark.RING_FINGER_PIP, mp_hands.HandLandmark.PINKY_PIP]
    
    for tip, pip in zip(finger_tips, finger_pips):
        # Kiểm tra ngón trỏ, giữa, áp út, út: Tip (đầu ngón) nằm cao hơn PIP (khớp giữa)
        if landmarks.landmark[tip].y < landmarks.landmark[pip].y:
            count += 1

    # Ngón cái (Thumb) - kiểm tra Y-tip (4) so với Y-base (3). Ngón cái mở khi tip cao hơn base.
    # Ngón cái có thể được xem là duỗi khi x_tip nằm xa hơn x_ip (cho tay phải, lật hình)
    # Tuy nhiên, để đơn giản và phù hợp với cử chỉ giơ lên, ta vẫn dùng kiểm tra Y
    if landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y < landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].y:
        count += 1
            
    return count

# Khởi tạo Camera
cap = cv2.VideoCapture(0)

# Yêu cầu kích thước 1280x720
WIDTH = 1280
HEIGHT = 720
cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

# Lấy kích thước khung hình sau khi set (có thể khác nếu camera không hỗ trợ)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Định vị các hộp trên màn hình
# Khởi tạo 3 hộp ban đầu (sẽ rơi)
boxes.append(Box(width // 2 - BOX_SIZE * 1.5, height - BOX_SIZE, BOX_SIZE)) 
boxes.append(Box(width // 2 + BOX_SIZE * 0.5, height - BOX_SIZE, BOX_SIZE))  
boxes.append(Box(width // 2 - BOX_SIZE // 2, 50, BOX_SIZE))                  

# Biến theo dõi trạng thái cử chỉ 5 ngón tay (tạo hộp)
is_five_fingers_up_prev = False
# Biến theo dõi trạng thái cử chỉ 3 ngón tay (xóa hộp)
is_three_fingers_up_prev = False 

print(f"Bắt đầu chương trình nhận diện tay. Nhấn 'Q' để thoát.")
print(f"Kích thước khung hình thực tế: {width}x{height}")

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Không thể đọc từ camera.")
        continue

    # Lật hình ảnh theo chiều ngang để có cái nhìn phản chiếu tự nhiên (như gương)
    image = cv2.flip(image, 1)
    
    # Chuyển đổi màu từ BGR sang RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Xử lý hình ảnh với MediaPipe Hands
    results = hands.process(image_rgb)
    
    # Chuyển đổi màu lại sang BGR để hiển thị với OpenCV
    image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    
    # --- Áp dụng vật lý cho tất cả các hộp ---
    for box in boxes:
        # Nếu hộp không được giữ, cập nhật vật lý
        if not box.is_held:
            # Truyền danh sách boxes để kiểm tra va chạm chồng chất
            box.update_physics(height, boxes)
            
    # Cập nhật trạng thái cầm nắm và cử chỉ
    finger_count = 0
    pinch_dist = 0 # Khởi tạo giá trị mặc định cho pinch_dist
    is_five_fingers_up_current = False
    is_three_fingers_up_current = False
    
    # Lấy vị trí điểm chụm hiện tại (để tính vận tốc)
    current_grab_point = None 
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            
            # --- Đếm ngón tay ---
            finger_count = count_extended_fingers(hand_landmarks)
            
            if finger_count == 5:
                is_five_fingers_up_current = True
            elif finger_count == 20: 
                is_three_fingers_up_current = True
            
            # Lấy tọa độ pixel của ngón cái (4) và ngón trỏ (8)
            thumb_tip_x = int(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x * width)
            thumb_tip_y = int(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y * height)
            index_tip_x = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * width)
            index_tip_y = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * height)
            wrist_x = int(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x * width)
            wrist_y = int(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y * height)

            # Tính khoảng cách giữa ngón cái và ngón trỏ (Pinch distance)
            pinch_dist = get_distance((thumb_tip_x, thumb_tip_y), (index_tip_x, index_tip_y))
            
            is_pinching = pinch_dist < PINCH_THRESHOLD
            
            # --- Logic Tạo Hộp (5 ngón - Debouncing) ---
            if is_five_fingers_up_current and not is_five_fingers_up_prev:
                # Cử chỉ 5 ngón tay vừa được kích hoạt
                new_box_x = width // 2 - BOX_SIZE // 2
                new_box_y = 0 
                boxes.append(Box(new_box_x, new_box_y, BOX_SIZE))
                print(f"CREATED NEW BOX: Total boxes now {len(boxes)}")
                
                # Hiển thị thông báo tạo hộp trên màn hình trong chốc lát
                cv2.putText(image, "NEW BOX CREATED!", (width // 2 - 150, 100), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 255), 3)

            # --- Logic Xóa Hộp (3 ngón - Debouncing) ---
            if is_three_fingers_up_current and not is_three_fingers_up_prev and len(boxes) > 0:
                if held_box is boxes[-1]:
                    held_box.is_held = False
                    held_box = None
                
                boxes.pop() 
                print(f"DELETED BOX: Total boxes now {len(boxes)}")
                
                cv2.putText(image, "LAST BOX DELETED!", (width // 2 - 150, 150), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)


            # --- Logic Cầm (Grab) và Thả (Drop) ---
            
            # 1. Cử chỉ Cầm (Grab)
            if is_pinching:
                # Tính điểm chụm/cầm hiện tại
                grab_point_x = (thumb_tip_x + index_tip_x) // 2
                grab_point_y = (thumb_tip_y + index_tip_y) // 2
                current_grab_point = (grab_point_x, grab_point_y)
                
                # Cập nhật lịch sử điểm chụm
                grab_history.append(current_grab_point)
                if len(grab_history) > MAX_HISTORY_POINTS:
                    grab_history.pop(0) # Giữ cho mảng chỉ có 3 điểm gần nhất

                # Nếu chưa cầm hộp nào, tìm hộp gần nhất để cầm
                if held_box is None:
                    
                    cv2.circle(image, current_grab_point, 10, (255, 0, 0), cv2.FILLED)
                    
                    for box in boxes:
                        # Chỉ cho phép cầm hộp đang nằm yên
                        if box.vy == 0 and box.is_inside(grab_point_x, grab_point_y):
                            box.is_held = True
                            # Khi cầm, đặt vận tốc về 0
                            box.vx = 0
                            box.vy = 0 
                            held_box = box
                            break 
                            
                # Nếu đang cầm một hộp, di chuyển hộp đó
                else:
                    new_x = grab_point_x - BOX_SIZE // 2 
                    new_y = grab_point_y - BOX_SIZE // 2
                    
                    held_box.x = np.clip(new_x, 0, width - BOX_SIZE)
                    held_box.y = np.clip(new_y, 0, height - BOX_SIZE)
                    
                    cv2.line(image, (wrist_x, wrist_y), held_box.get_center(), (0, 255, 255), 3)

            # 2. Cử chỉ Thả (Drop)
            else:
                # Nếu vừa thả hộp ra
                if held_box is not None:
                    held_box.is_held = False
                    
                    # --- Logic NÉM (Throwing) ---
                    # Cần ít nhất 2 điểm lịch sử để tính vận tốc
                    if len(grab_history) >= 2:
                        # Điểm hiện tại (trước khi thả, là điểm cuối cùng trong lịch sử)
                        p_current = grab_history[-1] 
                        # Điểm trước đó (dùng điểm này để tính delta)
                        p_prev = grab_history[0] 
                        
                        # Tính sự thay đổi vị trí (tạm xem là vận tốc)
                        dx = p_current[0] - p_prev[0]
                        dy = p_current[1] - p_prev[1]
                        
                        # Áp dụng vận tốc ném
                        held_box.vx = dx * VELOCITY_SCALE
                        held_box.vy = dy * VELOCITY_SCALE
                        
                        cv2.putText(image, "THROW!", (width // 2 - 50, 200), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 165, 0), 3)
                        print(f"THREW BOX with V=({held_box.vx:.2f}, {held_box.vy:.2f})")
                        
                    # Dọn dẹp lịch sử điểm chụm
                    grab_history.clear()
                    
                    # Cập nhật hộp đang giữ
                    held_box = None
                
                # Nếu không cầm hộp, xóa lịch sử điểm chụm để tránh tính vận tốc sai 
                # khi người dùng bắt đầu cầm hộp mới
                grab_history.clear() 
            
            # Vẽ các điểm mốc tay và kết nối
            mp_drawing.draw_landmarks(
                image, 
                hand_landmarks, 
                mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(255, 255, 0), thickness=2, circle_radius=4), 
                mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=2)
            )
            
            # Vẽ một vòng tròn lớn hơn tại điểm ngón trỏ (điểm chính để thao tác)
            cv2.circle(image, (index_tip_x, index_tip_y), 15, (0, 165, 255), cv2.FILLED) 
            
    # Nếu không phát hiện tay, xóa lịch sử để tránh lỗi
    if not results.multi_hand_landmarks:
        grab_history.clear()
            
    # Cập nhật trạng thái cử chỉ trước đó
    is_five_fingers_up_prev = is_five_fingers_up_current
    is_three_fingers_up_prev = is_three_fingers_up_current 
    
    # --- Hiển thị thông tin cử chỉ ---
    # pinch_dist hiện đã được định nghĩa là 0 nếu không có tay
    cv2.putText(image, f"Fingers: {finger_count}/5", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(image, f"Pinch: {int(pinch_dist)}", (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # Hiển thị trạng thái cầm nắm hiện tại
    status_text = "DANG CAM" if held_box else "CHUA CAM"
    status_color = (0, 0, 255) if held_box else (0, 255, 0)
    
    cv2.putText(image, status_text, (width - 300, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, status_color, 3)
    
    # Ghi chú về UTF-8 (tiếng Việt có dấu)
    cv2.putText(image, "Luu y: cv2.putText KHONG ho tro dau TV", (width - 400, height - 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
            
    # --- Vẽ tất cả các hộp lên khung hình ---
    for box in boxes:
        box.draw(image)
        
    # --- Hiển thị khung hình ---
    cv2.imshow('Hand Gesture Grab & Drop (Nắm và Thả)', image)

    # Thoát khi nhấn 'q'
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# Dọn dẹp
cap.release()
cv2.destroyAllWindows()