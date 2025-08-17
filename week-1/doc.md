# Các Thuật Toán Học Máy Cơ Bản: Linear Regression, K-Means, K-Nearest Neighbors
**Tài Liệu Tham Khảo**:
- [Bài 1: Giới thiệu về Machine Learning](https://machinelearningcoban.com/2016/12/26/introduce/)
- [Bài 2: Phân nhóm các thuật toán Machine Learning](https://machinelearningcoban.com/2016/12/27/categories/)
- [Bài 3: Linear Regression](https://machinelearningcoban.com/2016/12/28/linearregression/)
- [Bài 4: K-means Clustering](https://machinelearningcoban.com/2017/01/01/kmeans/)
- [Bài 6: K-nearest neighbors](https://machinelearningcoban.com/2017/01/08/knn/)

## 1. Giới Thiệu Về Machine Learning
### Lý Thuyết Chính
Machine Learning (ML) là lĩnh vực cho phép máy tính học mà không cần lập trình cụ thể, dựa trên dữ liệu. Các ứng dụng: Nhận diện khuôn mặt, gợi ý sản phẩm, xe tự lái.

Phân nhóm thuật toán ML dựa trên phương thức học:
- **Supervised Learning (Học có giám sát)**: Dự đoán dựa trên dữ liệu có nhãn (input-output). Ví dụ: Phân loại email spam.
- **Unsupervised Learning (Học không giám sát)**: Tìm cấu trúc ẩn trong dữ liệu không nhãn. Ví dụ: Phân cụm khách hàng.
- **Semi-supervised Learning**: Kết hợp dữ liệu có và không nhãn.
- **Reinforcement Learning**: Học qua thử nghiệm và phần thưởng.

Dựa trên chức năng: Classification (phân loại), Regression (hồi quy), Clustering (phân cụm), v.v.

### Code
```python
from sklearn.datasets import load_iris
iris = load_iris()
X = iris.data  # Features
y = iris.target  # Labels
print(X.shape, y.shape)  # (150, 4), (150,)
```

### Ưu Nhược Điểm
- Ưu: Giúp tự động hóa quyết định dựa trên dữ liệu lớn.
- Nhược: Cần dữ liệu chất lượng cao, có thể overfit nếu không cẩn thận.

## 2. Linear Regression (Hồi Quy Tuyến Tính)

### Lý Thuyết Chính
Linear Regression là thuật toán Supervised Learning để dự đoán giá trị liên tục dựa trên mối quan hệ tuyến tính. Phương trình: \( y \approx \hat{y} = w_0 + w_1 x_1 + \dots + w_n x_n \), với \( \mathbf{w} \) là hệ số, \( w_0 \) là bias.

Hàm mất mát (Loss Function): \( \mathcal{L}(\mathbf{w}) = \frac{1}{2} \sum_{i=1}^N (y_i - \hat{y}_i)^2 \), sử dụng Mean Squared Error (MSE).

Nghiệm: Sử dụng công thức đóng \( \mathbf{w} = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y} \), hoặc Gradient Descent để tối ưu.

### Code
Sử dụng Scikit-learn để dự đoán chiều dài cánh hoa từ chiều dài đài hoa trong Iris:
```python
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

iris = load_iris()
X = iris.data[:, 0].reshape(-1, 1)  # Sepal length
y = iris.data[:, 2]  # Petal length

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = LinearRegression()
model.fit(X_train, y_train)
print("Hệ số:", model.coef_, "Bias:", model.intercept_)
# Dự đoán: model.predict([[5.0]])
```

### Ưu Nhược Điểm
- Ưu: Đơn giản, dễ giải thích.
- Nhược: Giả định tuyến tính, nhạy cảm với outliers.

## 3. K-means Clustering (Phân Cụm K-means)

### Lý Thuyết Chính
K-means là thuật toán Unsupervised Learning để phân dữ liệu thành K cụm, dựa trên khoảng cách đến trung tâm cụm (centroid). Ý tưởng: Tối ưu hàm mất mát \( J = \sum_{i=1}^N \min_{k=1}^K \| \mathbf{x}_i - \mathbf{c}_k \|^2 \), với \( \mathbf{c}_k \) là centroid.

Các bước thuật toán:
1. Chọn K centroid ngẫu nhiên.
2. Gán mỗi điểm vào centroid gần nhất.
3. Cập nhật centroid là trung bình của cụm.
4. Lặp đến khi hội tụ.

### Code
Phân cụm Iris thành 3 cụm:
```python
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris

iris = load_iris()
X = iris.data

kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)
print("Centroids:", kmeans.cluster_centers_)
print("Labels:", kmeans.labels_)
```

### Ưu Nhược Điểm
- Ưu: Nhanh, dễ triển khai.
- Nhược: Cần chọn K thủ công, nhạy cảm với khởi tạo.

## 4. K-nearest Neighbors (KNN)

### Lý Thuyết Chính
KNN là thuật toán Supervised Learning "lười" (lazy learning), dự đoán dựa trên K láng giềng gần nhất. Khoảng cách thường dùng Euclidean: \( d(\mathbf{x}, \mathbf{y}) = \sqrt{\sum (x_i - y_i)^2} \).

Đối với Classification: Chọn nhãn phổ biến nhất trong K láng giềng. Regression: Trung bình giá trị.

### Code Ví Dụ
Phân loại Iris với KNN:
```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score

iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
print("Độ chính xác:", accuracy_score(y_test, y_pred))
```

### Ưu Nhược Điểm
- Ưu: Đơn giản, không cần huấn luyện.
- Nhược: Chậm với dữ liệu lớn, nhạy cảm với nhiễu.

## 5. Project: Phân Tích Bộ Dữ Liệu Iris
**Mô tả**: Sử dụng bộ dữ liệu Iris (150 mẫu hoa với 4 đặc trưng: chiều dài/rộng đài/cánh hoa, 3 lớp). Project bao gồm:
- **Linear Regression**: Dự đoán chiều dài cánh hoa từ chiều dài đài hoa.
- **K-means**: Phân cụm dữ liệu thành 3 cụm và so sánh với nhãn thật.
- **KNN**: Phân loại hoa mới dựa trên đặc trưng.

**Các Bước Thực Hiện**:
1. Load dữ liệu Iris.
2. Áp dụng từng thuật toán như code ví dụ trên.
3. Đánh giá: MSE cho Regression, Accuracy cho KNN, Silhouette score cho K-means.
4. Trình bày: Vẽ biểu đồ (sử dụng Matplotlib) để minh họa cụm hoặc đường hồi quy.

**Code**: [demo.ipynb](code/demo.ipynb)