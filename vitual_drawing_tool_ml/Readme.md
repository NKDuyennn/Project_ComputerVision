# **Vẽ Không Chạm – Công Cụ Vẽ Ảo Bằng Cử Chỉ Tay** 🎨✨  

Bạn đã bao giờ muốn vẽ mà không cần chạm vào giấy hay màn hình chưa? Trong bài viết này, mình sẽ giới thiệu một dự án cực kỳ thú vị – **"Vẽ Không Chạm"**, một công cụ vẽ ảo cho phép bạn vẽ trong không khí chỉ bằng cách di chuyển tay! Dự án này sử dụng sự kết hợp mạnh mẽ giữa **OpenCV** và **MediaPipe** để theo dõi cử chỉ tay và biến chúng thành những nét vẽ trên màn hình.  

---

## 🔧 **Công cụ và thư viện sử dụng:**  
📌 **Python3, NumPy, OpenCV, MediaPipe**  

- **OpenCV:** Đây là một thư viện mã nguồn mở cực kỳ phổ biến trong thị giác máy tính. Nhờ OpenCV, mình có thể dễ dàng thu và xử lý video từ webcam.  
- **MediaPipe:** Được phát triển bởi Google, MediaPipe giúp nhận diện và theo dõi bàn tay một cách chính xác theo thời gian thực. Trong dự án này, nó giúp xác định các điểm đặc trưng trên bàn tay để tạo hiệu ứng vẽ.  

---

## 🎯 **Cách hoạt động:**  
1️⃣ **Mở webcam và thu video** bằng OpenCV.  
2️⃣ **Xử lý từng khung hình**, sử dụng MediaPipe để nhận diện bàn tay.  
3️⃣ **Xác định vị trí ngón tay trỏ**, sau đó dùng dữ liệu này để vẽ lên màn hình.  

---

## 📌 **Thuật toán chi tiết:**  
✅ **Bước 1:** Đọc từng khung hình và chuyển sang hệ màu **HSV** để dễ phát hiện màu sắc.  
✅ **Bước 2:** Tạo **một vùng canvas riêng** và hiển thị các nút màu để chọn màu vẽ.  
✅ **Bước 3:** Điều chỉnh thông số của MediaPipe để **chỉ theo dõi một bàn tay duy nhất**.  
✅ **Bước 4:** Phát hiện các điểm đặc trưng trên bàn tay, xác định tọa độ của **ngón trỏ** và lưu lại để tạo hiệu ứng vẽ liên tục.  
✅ **Bước 5:** Vẽ các điểm đã lưu lên màn hình, tạo cảm giác như đang vẽ trực tiếp trong không khí!  

---

## 🚀 **Kết luận:**  
Chỉ với một chiếc webcam và vài dòng code, mình đã có thể biến không khí thành một tấm bảng vẽ ảo. Đây là một dự án thú vị giúp mình hiểu rõ hơn về **thị giác máy tính (Computer Vision)** và cách ứng dụng AI để tương tác với thế giới xung quanh. Nếu bạn cũng thích những dự án như thế này, hãy thử bắt tay vào làm ngay nhé! 😍🎨