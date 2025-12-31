#Giới thiệu đề tài 

*Bài toán:
Trong lĩnh vực marketing hiện đại, doanh nghiệp thường sở hữu một lượng lớn dữ liệu khách hàng bao gồm thông tin nhân khẩu học, hành vi mua sắm và mức độ tương tác với sản phẩm/dịch vụ. Tuy nhiên, việc khai thác hiệu quả nguồn dữ liệu này để hiểu rõ từng nhóm khách hàng vẫn là một thách thức lớn.
Bài toán phân cụm khách hàng (Customer Segmentation) nhằm mục đích chia tập khách hàng thành các nhóm khác nhau sao cho những khách hàng trong cùng một nhóm có đặc điểm và hành vi tương đồng, trong khi các nhóm khác nhau có sự khác biệt rõ rệt. Việc phân cụm được thực hiện khi không có nhãn sẵn, do đó đây là một bài toán học không giám sát (Unsupervised Learning).

*Mục tiêu của đề tài là:
Áp dụng kỹ thuật phân cụm dữ liệu để phân nhóm khách hàng dựa trên các đặc trưng liên quan đến thu nhập và hành vi tiêu dùng
Xác định các nhóm khách hàng tiềm năng và đặc điểm nổi bật của từng nhóm
Hỗ trợ doanh nghiệp xây dựng các chiến lược marketing phù hợp cho từng phân khúc khách hàng
Nâng cao hiệu quả tiếp cận khách hàng, tối ưu chi phí marketing và tăng doanh thu

#Dataset 
Nguồn data: kaggle, Dataset: Customer Segmentation Dataset

Link tải: https://www.kaggle.com/datasets/vishakhdapat/customer-segmentation-clustering

Mô tả cột:

Id: Mã định danh duy nhất cho mỗi cá thể trong tập dữ liệu.

Year_Birth: Năm sinh của cá nhân.

Education:  Mức độ học vấn cao nhất mà cá nhân đạt được.

Marital_Status: Tình trạng hôn nhân của cá nhân.

Income: Thu nhập hàng năm của cá nhân.

Kidhome: Số lượng trẻ nhỏ trong hộ gia đình.

Teenhome: Số lượng thanh thiếu niên trong hộ gia đình.

Dt_Customer: Ngày khách hàng được đăng ký lần đầu hoặc trở thành một phần của
cơ sở dữ liệu khách hàng của công ty.

Recency: Số ngày kể từ lần mua hàng hoặc tương tác gần nhất.

MntWines: Số tiền chi cho rượu vang.

MntFruits: Số tiền chi cho trái cây.

MntMeatProducts: Số tiền chi cho các sản phẩm thịt.

MntFishProducts: Số tiền chi cho các sản phẩm từ cá.

MntSweetProducts: Số tiền chi cho các sản phẩm đồ ngọt.

MntGoldProds: Số tiền chi cho các sản phẩm vàng.

NumDealsPurchases: Số lượng giao dịch mua hàng được thực hiện với mức giảm giá
hoặc nằm trong chương trình khuyến mãi.

NumWebPurchases: Số lượng giao dịch mua hàng được thực hiện thông qua trang web của công ty.

NumCatalogPurchases: Số lượng giao dịch mua hàng được thực hiện thông qua catalog.

NumStorePurchases: Số lượng giao dịch mua hàng được thực hiện tại các cửa hàng vật lý.

NumWebVisitsMonth: Số lượt truy cập vào trang web của công ty trong một tháng.

AcceptedCmp3: Biến nhị phân (1 hoặc 0) cho biết cá nhân đó có chấp nhận chiến
dịch tiếp thị thứ ba hay không.

AcceptedCmp4: Biến nhị phân (1 hoặc 0) cho biết cá nhân đó có chấp nhận chiến dịch tiếp thị thứ tư hay không.

AcceptedCmp5: Biến nhị phân (1 hoặc 0) cho biết cá nhân đó có chấp nhận chiến dịch tiếp thị thứ năm hay không.

AcceptedCmp1: Biến nhị phân (1 hoặc 0) cho biết cá nhân đó có chấp nhận chiến dịch tiếp thị đầu tiên hay không.

AcceptedCmp2: Biến nhị phân (1 hoặc 0) cho biết cá nhân đó có chấp nhận chiến dịch tiếp thị thứ hai hay không.

Complain: Biến nhị phân (1 hoặc 0) cho biết cá nhân đó có đưa ra khiếu nại hay không.

Z_CostContact: Chi phí cố định liên quan đến việc liên hệ với khách hàng.

Z_Revenue: Doanh thu cố định gắn liền với phản hồi thành công của chiến dịch.

Response: Biến nhị phân (1 hoặc 0) cho biết cá nhân đó có phản hồi lại chiến dịch tiếp thị hay không. 

#Pipeline (tiền xử lý → train → evaluate → inference)
Pipeline xử lý dữ liệu và xây dựng mô hình trong đề tài được thực hiện theo các bước sau:

Bước 1. Tiền xử lý dữ liệu (Preprocessing)
Đọc dữ liệu từ file CSV chứa thông tin khách hàng
Loại bỏ các cột không cần thiết(theo mục tiêu phân chia khách hàng theo nhân khẩu học và các hành vi mua sắm) như:
	ID
	Dt_Customer
	Z_CostContact
	Z_Revenue
	Các cột liên quan đến chiến dịch marketing và khiếu nại
Kiểm tra và xử lý giá trị thiếu trong tập dữ liệu
Loại bỏ các bản ghi trùng lặp
xử lý giá trị ngoại lai bằng IQR
Mã hoá dữ liệu phân loại bằng One-Hot Encoding 
Chỉ giữ lại các đặc trưng phục vụ cho phân cụm
Chuẩn hóa dữ liệu bằng StandardScaler nhằm đưa các đặc trưng về cùng thang đo

Bước 2. Huấn luyện mô hình (Training)
Sử dụng thuật toán K-Means Clustering
Thử nghiệm nhiều giá trị số cụm k khác nhau
Áp dụng Elbow Method, Silhouette Score, Davies–Bouldin Index để lựa chọn số cụm tối ưu
Huấn luyện mô hình K-Means với số cụm đã chọn

Bước 3. Đánh giá mô hình (Evaluation)
Đánh giá kết quả phân cụm thông qua:
Silhouette Score, Davies–Bouldin Index
Quan sát trực quan biểu đồ Elbow
Phân tích đặc điểm của từng cụm thông qua giá trị trung bình của các đặc trưng
Trực quan hóa kết quả phân cụm để đánh giá mức độ tách biệt giữa các nhóm khách hàng

Bước 4. Suy diễn / Dự đoán (Inference)
Gán nhãn cụm cho từng khách hàng trong tập dữ liệu
Sử dụng kết quả phân cụm để:
Phân tích hành vi khách hàng
Hỗ trợ phân khúc khách hàng cho mục đích kinh doanh
Kết quả đầu ra là tập dữ liệu đã được gắn nhãn cụm

#Mô hình sử dụng 

-K-Means Clustering

-Lý do lựa chọn:
	Phù hợp cho bài toán phân cụm không giám sát
	Dễ triển khai và hiệu quả với dữ liệu số
	Khả năng mở rộng tốt với tập dữ liệu lớn
	Dễ trực quan hóa kết quả

#Kết quả

*Chỉ số đánh giá
Elbow Method: xác định số cụm tối ưu ra k=4 là tốt nhất
Silhouette Score: Cho thấy mức độ phân tách giữa các cụm là tốt, đạt giá trị cao nhất tại k = 2 Silhouette=0.316,K=4 Silhouette=0.300
Davies–Bouldin Index(DBI): Cho thấy mức độ chặt chẽ và chồng lấn, đạt giá trị nhỏ nhất tại k = 4 DBI=1.1673

*Nhận xét
Dữ liệu được phân thành các nhóm khách hàng rõ ràng
Mỗi cụm thể hiện đặc điểm chi tiêu và hành vi khác nhau
Kết quả có thể ứng dụng trong chiến lược marketing và chăm sóc khách hàn
 
#Hướng dẫn chạy

*Cài môi trường trong file requirements

*Chạy code trong thư mục app:Chạy tuần tự các cell trong file KMeans_BTL.ipynb

*Chạy demo trong thư mục demo

Model, scaler và profile cluster được lưu:
	kmeans1.pkl
	scaler1.pkl
	cluster_profile1.pkl
	
Load model bằng joblib để dự đoán cluster cho khách hàng mới

#Cấu trúc thư mục dự án

Trong thư mục BTL_ML có:

	app/KMeans_BTL.ipynb
	
	data/customer_segmentation.csv
	
	demo/kmeans1.pkl
	
	demo/scaler1.pkl
	
	demo/cluster_profile1.pkl
	
	reports/BuiThiThanhTuyen_12523091_12423TN.docx
	
	slides/KMeans_btl.pptx
	
	README.md
	
	.gitignore

Tác giả: Bùi Thị Thanh Tuyền-12523091-12423TN  
