# Các Thuật Toán Học Máy: PLA, Logistic Regression, Softmax Regression, Overfitting

**Tài Liệu Tham Khảo**:

- [Bài 9: Perceptron Learning Algorithm](https://machinelearningcoban.com/2017/01/21/perceptron/)
- [Bài 10: Logistic Regression](https://machinelearningcoban.com/2017/01/27/logisticregression/)
- [Bài 13: Softmax Regression](https://machinelearningcoban.com/2017/02/17/softmax/)
- [Bài 15: Overfitting](https://machinelearningcoban.com/2017/03/04/overfitting/)

## 1. Perceptron Learning Algorithm (PLA)

### Lý Thuyết Chính

Perceptron Learning Algorithm (PLA) là một thuật toán học có giám sát cơ bản trong lĩnh vực Machine Learning, được sử dụng cho bài toán phân loại nhị phân (binary classification). Thuật toán này được phát triển dựa trên mô hình Perceptron, một trong những mô hình học máy đầu tiên, được Frank Rosenblatt đề xuất năm 1958. PLA nhằm tìm một siêu phẳng (hyperplane) tách biệt hoàn hảo hai lớp dữ liệu, giả định rằng dữ liệu là tuyến tính phân biệt (linearly separable). Nếu dữ liệu không thỏa mãn điều kiện này, thuật toán có thể không hội tụ.

#### Mô hình Perceptron

Mô hình Perceptron được xây dựng dựa trên ý tưởng mô phỏng nơ-ron sinh học đơn giản. Đầu vào là vector đặc trưng \(\mathbf{x} = [x_1, x_2, \dots, x_n]\), được nhân với vector trọng số \(\mathbf{w} = [w_1, w_2, \dots, w_n]\), cộng với bias \(b\). Đầu ra được tính bằng hàm kích hoạt sign (hàm dấu):

$$ z = \mathbf{w}^T \mathbf{x} + b $$
$$ \hat{y} = \text{sign}(z) = \begin{cases} 
1 & \text{nếu } z \geq 0 \\
-1 & \text{nếu } z < 0 
\end{cases} $$

Siêu phẳng phân cách được định nghĩa bởi phương trình \(\mathbf{w}^T \mathbf{x} + b = 0\). Các điểm ở một bên siêu phẳng sẽ có dự đoán +1, bên kia là -1.

#### Quy tắc học của PLA

PLA sử dụng quy tắc học dựa trên lỗi: Thuật toán lặp qua các điểm dữ liệu và chỉ cập nhật trọng số khi có lỗi phân loại. Nếu điểm \(\mathbf{x}_i\) bị phân loại sai, tức là \( y_i \cdot (\mathbf{w}^T \mathbf{x}_i + b) \leq 0 \), thì cập nhật:

$$ \mathbf{w} \gets \mathbf{w} + \eta y_i \mathbf{x}_i $$
$$ b \gets b + \eta y_i $$

trong đó \(\eta\) là learning rate (thường chọn \(\eta = 1\) để đơn giản). Quá trình lặp lại cho đến khi tất cả các điểm được phân loại đúng hoặc đạt số vòng lặp tối đa.

#### Các bước thuật toán chi tiết

1. Khởi tạo \(\mathbf{w} = \mathbf{0}\), \(b = 0\).
2. Lặp qua tập huấn luyện (có thể theo thứ tự ngẫu nhiên để tránh chu kỳ).
3. Với mỗi điểm \(\mathbf{x}_i\):
   - Tính \( \hat{y}_i = \text{sign}(\mathbf{w}^T \mathbf{x}_i + b) \).
   - Nếu \(\hat{y}_i \neq y_i\) (tức lỗi), cập nhật trọng số và bias.
4. Dừng khi không còn lỗi hoặc đạt giới hạn lặp.

#### Định lý hội tụ

Nếu dữ liệu là tuyến tính phân biệt, PLA sẽ hội tụ sau hữu hạn bước. Tuy nhiên, nếu không, thuật toán có thể dao động vô hạn. Để khắc phục, có thể sử dụng Pocket Algorithm (lưu lại trọng số tốt nhất) hoặc thêm điều kiện dừng.

#### Ví dụ minh họa

Giả sử dữ liệu 2D với hai lớp: Lớp +1 ở phía trên đường thẳng \(x_1 + x_2 = 3\), lớp -1 ở dưới. PLA sẽ cập nhật dần để tìm đường thẳng phù hợp. Trong thực tế, với bộ dữ liệu Iris (setosa vs. versicolor), PLA có thể hội tụ nhanh vì dữ liệu gần như tuyến tính phân biệt.

### Code

Phân loại nhị phân hai lớp đầu tiên của bộ dữ liệu Iris (setosa vs. versicolor) với 2 đặc trưng:

```python
import numpy as np
from sklearn.datasets import load_iris

iris = load_iris()
X = iris.data[:100, :2]  # 2 đặc trưng: sepal length, width
y = iris.target[:100] * 2 - 1  # Chuyển nhãn thành -1 và 1

def perceptron(X, y, eta=1, max_iter=1000):
    w = np.zeros(X.shape[1])
    b = 0
    for _ in range(max_iter):
        misclassified = False
        for i in range(len(X)):
            if y[i] * (np.dot(w, X[i]) + b) <= 0:
                w += eta * y[i] * X[i]
                b += eta * y[i]
                misclassified = True
        if not misclassified:
            break
    return w, b

w, b = perceptron(X, y)
print("Trọng số:", w, "Bias:", b)
```

## 2. Logistic Regression

### Lý Thuyết Chính

Logistic Regression là một mô hình tuyến tính được sử dụng chủ yếu cho bài toán phân loại (classification), mặc dù tên gọi có chứa "regression". Nó khác với Linear Regression và Perceptron Learning Algorithm (PLA) ở activation function, cho phép đầu ra được thể hiện dưới dạng xác suất, phù hợp với các bài toán như dự đoán xác suất thi đỗ dựa trên thời gian ôn thi hoặc xác suất mưa dựa trên thông tin đo được.

#### Giới thiệu

- **So sánh với các mô hình tuyến tính khác:**
  - Linear Regression: \( f(\mathbf{x}) = \mathbf{w}^T \mathbf{x} \), đầu ra là giá trị thực không bị chặn.
  - PLA: \( f(\mathbf{x}) = \text{sgn}(\mathbf{w}^T\mathbf{x}) \), đầu ra là \(-1\) hoặc \(1\), phù hợp với binary classification nhưng yêu cầu dữ liệu linearly separable.
  - Logistic Regression: \( f(\mathbf{x}) = \theta(\mathbf{w}^T\mathbf{x}) \), với \(\theta\) là logistic function, đầu ra bị chặn trong \([0, 1]\), thể hiện xác suất.

- Ví dụ minh họa: Một nhóm 20 sinh viên với thời gian ôn thi từ 0 đến 6 giờ, kết quả thi đỗ hoặc trượt (dữ liệu không linearly separable, nên PLA không phù hợp, và Linear Regression cũng không hiệu quả do không bị chặn).

#### Mô hình Logistic Regression

Đầu ra của Logistic Regression được định nghĩa là:
\[
f(\mathbf{x}) = \theta(\mathbf{w}^T\mathbf{x})
\]
Trong đó:
- \(\mathbf{w}\) là vector trọng số.
- \(\mathbf{x}\) là vector đầu vào, thường được mở rộng với \(x_0 = 1\) để tính toán thuận tiện.
- \(\theta\) là activation function, thường là sigmoid function:
\[
\sigma(s) = \frac{1}{1 + e^{-s}}
\]
- Đặc điểm của sigmoid:
  - Giá trị nằm trong \((0, 1)\).
  - \(\lim_{s \rightarrow -\infty} \sigma(s) = 0\), \(\lim_{s \rightarrow +\infty} \sigma(s) = 1\).
  - Đạo hàm: \(\sigma'(s) = \sigma(s)(1 - \sigma(s))\), đơn giản và thuận lợi cho tối ưu.

Ngoài sigmoid, hàm tanh cũng được sử dụng:
\[
\text{tanh}(s) = \frac{e^s - e^{-s}}{e^s + e^{-s}}
\]
Với \(\text{tanh}(s) = 2\sigma(2s) - 1\), có thể đưa về khoảng \((0, 1)\).

#### Xây dựng và suy diễn hàm mất mát

Mục tiêu là tìm \(\mathbf{w}\) sao cho mô hình dự đoán xác suất phù hợp với dữ liệu. Giả sử:
- Xác suất điểm \(\mathbf{x}_i\) thuộc class 1 là \(P(y_i = 1 | \mathbf{x}_i; \mathbf{w}) = f(\mathbf{w}^T\mathbf{x}_i) = z_i\).
- Xác suất thuộc class 0 là \(P(y_i = 0 | \mathbf{x}_i; \mathbf{w}) = 1 - z_i\).

Với \(z_i = \sigma(\mathbf{w}^T\mathbf{x}_i)\), ta có:
\[
P(y_i | \mathbf{x}_i; \mathbf{w}) = z_i^{y_i}(1 - z_i)^{1 - y_i}
\]
- Khi \(y_i = 1\), biểu thức trở thành \(z_i\).
- Khi \(y_i = 0\), biểu thức trở thành \(1 - z_i\).

Với toàn bộ tập huấn luyện \(\mathbf{X} = [\mathbf{x}_1, \dots, \mathbf{x}_N]\), \(\mathbf{y} = [y_1, \dots, y_N]\), giả sử dữ liệu độc lập, likelihood function là:
\[
P(\mathbf{y} | \mathbf{X}; \mathbf{w}) = \prod_{i=1}^N P(y_i | \mathbf{x}_i; \mathbf{w}) = \prod_{i=1}^N z_i^{y_i}(1 - z_i)^{1 - y_i}
\]

Để tối ưu, ta sử dụng log-likelihood:
\[
\ell(\mathbf{w}) = \ln P(\mathbf{y} | \mathbf{X}; \mathbf{w}) = \sum_{i=1}^N \left[ y_i \ln z_i + (1 - y_i) \ln (1 - z_i) \right]
\]

Hàm mất mát (negative log-likelihood):
\[
\mathcal{L}(\mathbf{w}) = -\ell(\mathbf{w}) = -\sum_{i=1}^N \left[ y_i \ln z_i + (1 - y_i) \ln (1 - z_i) \right]
\]

#### Phương pháp tối ưu

- **Gradient Descent**: Cập nhật \(\mathbf{w}\) theo gradient của \(\mathcal{L}\).
- Đạo hàm của sigmoid: \(\frac{dz_i}{ds_i} = z_i(1 - z_i)\), với \(s_i = \mathbf{w}^T\mathbf{x}_i\).
- Gradient của \(\mathcal{L}\):
\[
\frac{\partial \mathcal{L}}{\partial \mathbf{w}} = \sum_{i=1}^N (z_i - y_i) \mathbf{x}_i
\]
- Cập nhật: \(\mathbf{w} \gets \mathbf{w} - \eta \frac{\partial \mathcal{L}}{\partial \mathbf{w}}\), với \(\eta\) là learning rate.

### Code

Phân loại nhị phân hai lớp đầu tiên của bộ dữ liệu Iris với 2 đặc trưng:

```python
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

iris = load_iris()
X = iris.data[:100, :2]  # 2 đặc trưng
y = iris.target[:100]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)
print("Độ chính xác:", accuracy_score(y_test, model.predict(X_test)))
print("Trọng số:", model.coef_, "Bias:", model.intercept_)
```

## 3. Softmax Regression

### Lý Thuyết Chính

Softmax Regression là một phương pháp mở rộng của Logistic Regression, được sử dụng cho bài toán phân loại đa lớp (multi-class classification). Nó khắc phục hạn chế của kỹ thuật one-vs-rest trong binary classifiers, đặc biệt là vấn đề tổng xác suất không nhất thiết bằng 1. Mặc dù tên gọi là "Regression", đây thực chất là một phương pháp phân loại, thường được áp dụng trong các mạng nơ-ron sâu (Deep Neural Networks).

#### Giới thiệu

- **Mối liên hệ với Logistic Regression**: Khi số lớp \(C = 2\), Softmax Regression trở thành Logistic Regression, với đầu ra là hàm sigmoid. Trong trường hợp \(C > 2\), Softmax Regression xử lý đồng thời tất cả các lớp, dựa trên mối quan hệ giữa các giá trị đầu vào.

#### Softmax Function

Softmax function là hàm kích hoạt chính trong Softmax Regression, đảm bảo đầu ra là xác suất, thỏa mãn các điều kiện: dương, tổng bằng 1, và đồng biến với đầu vào.

##### Công thức Softmax
Với đầu vào là vector \(\mathbf{z} = [z_1, z_2, \dots, z_C]\), đầu ra \(\mathbf{a} = [a_1, a_2, \dots, a_C]\) được tính bằng:
\[
a_i = \frac{\exp(z_i)}{\sum_{j=1}^C \exp(z_j)}, \quad \forall i = 1, 2, \dots, C
\]
- Ở đây, \(z_i = \mathbf{w}_i^T\mathbf{x}\), với \(\mathbf{w}_i\) là vector trọng số của lớp \(i\), \(\mathbf{x}\) là vector đầu vào (có thêm bias, tức \(\mathbf{x}\) có kích thước \((d+1)\)).
- Đầu ra \(a_i\) thể hiện xác suất để \(\mathbf{x}\) thuộc lớp \(i\), tức:
\[
P(y = i | \mathbf{x}; \mathbf{W}) = a_i
\]
- Softmax đảm bảo \(a_i > 0\), \(\sum_{i=1}^C a_i = 1\), và giữ thứ tự của \(z_i\) (nếu \(z_i > z_j\) thì \(a_i > a_j\)).

##### Phiên bản ổn định của Softmax
Để tránh hiện tượng tràn số (overflow) khi \(z_i\) quá lớn, công thức được biến đổi:
\[
a_i = \frac{\exp(z_i - c)}{\sum_{j=1}^C \exp(z_j - c)}
\]
Thường chọn \(c = \max_i z_i\). Phiên bản này được triển khai trong Python với hàm `softmax_stable`:
```python
def softmax_stable(Z):
    e_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))
    A = e_Z / e_Z.sum(axis=0)
    return A
```

##### Ví dụ:
- Nếu \(z_i\) bằng nhau (ví dụ \(z_1 = z_2 = z_3\)), thì \(a_i = 1/3\).
- Nếu \(z_1\) lớn nhất, \(a_1\) sẽ lớn nhất, nhưng không bao giờ bằng 1 hoặc 0 (tính chất "soft" của hàm).
- Khi \(z_i\) âm, \(a_i\) vẫn dương và giữ thứ tự.

#### Hàm mất mát và phương pháp tối ưu
##### One-hot coding
- Đầu ra thực sự \(\mathbf{y}\) được biểu diễn dưới dạng one-hot coding: vector có đúng 1 phần tử bằng 1 (vị trí tương ứng với lớp), các phần tử còn lại bằng 0.
- Đầu ra dự đoán là \(\mathbf{a} = \text{softmax}(\mathbf{W}^T\mathbf{x})\).

##### Cross Entropy
Hàm mất mát sử dụng cross entropy để đo khoảng cách giữa phân phối dự đoán \(\mathbf{a}\) và phân phối thực tế \(\mathbf{y}\):
\[
H(\mathbf{p}, \mathbf{q}) = -\sum_{i=1}^C p_i \log q_i
\]
Với toàn bộ tập huấn luyện, hàm mất mát là trung bình cross entropy:
\[
\mathcal{L}(\mathbf{W}) = \frac{1}{N} \sum_{n=1}^N H(\mathbf{y}_n, \mathbf{a}_n) = -\frac{1}{N} \sum_{n=1}^N \sum_{i=1}^C y_{n,i} \log a_{n,i}
\]

##### Tối ưu hóa
- **Gradient Descent**: Tính gradient của \(\mathcal{L}\) đối với \(\mathbf{W}\), sử dụng backpropagation trong mạng nơ-ron.
- Đạo hàm của softmax: \(\frac{\partial a_i}{\partial z_j} = a_i (\delta_{ij} - a_j)\), với \(\delta_{ij} = 1\) nếu \(i=j\), 0 nếu khác.
- Gradient: \(\frac{\partial \mathcal{L}}{\partial \mathbf{w}_j} = \frac{1}{N} \sum_{n=1}^N (a_{n,j} - y_{n,j}) \mathbf{x}_n\).
- Cập nhật: \(\mathbf{W} \gets \mathbf{W} - \eta \frac{\partial \mathcal{L}}{\partial \mathbf{W}}\).

#### Ví dụ minh họa
- Với dữ liệu 3 lớp (như Iris), Softmax tính xác suất cho từng lớp dựa trên đầu vào, và chọn lớp có xác suất cao nhất.
- Trong mạng nơ-ron, Softmax thường là lớp cuối cùng cho phân loại.

#### **Thông tin bổ sung để học**
- **So sánh với one-vs-rest**: Softmax xử lý tốt hơn vì xem xét mối quan hệ giữa các lớp.
- **Ứng dụng**: Trong Deep Learning (CNN, RNN), làm lớp output cho multi-class.
- **Mở rộng**: Kết hợp với regularization để tránh overfitting.

### Code
Phân loại 3 lớp của bộ dữ liệu Iris:

```python
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

iris = load_iris()
X = iris.data  # 4 đặc trưng
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression(multi_class='multinomial', max_iter=200)
model.fit(X_train, y_train)
print("Độ chính xác:", accuracy_score(y_test, model.predict(X_test)))
print("Trọng số:", model.coef_, "Bias:", model.intercept_)
```

## 4. Overfitting

### Lý Thuyết Chính
Overfitting là hiện tượng phổ biến trong Machine Learning, xảy ra khi mô hình học quá tốt trên dữ liệu huấn luyện (training data) nhưng hoạt động kém trên dữ liệu mới (test data hoặc validation data). Thay vì học các quy luật tổng quát, mô hình "ghi nhớ" cả nhiễu và chi tiết cụ thể của dữ liệu huấn luyện, dẫn đến thiếu khả năng tổng quát hóa (generalization).

#### Dấu hiệu
- Loss hoặc error thấp trên training data nhưng cao trên test/validation data.
- Accuracy cao trên train (gần 100%) nhưng thấp trên test.
- Đường cong learning curve: Training loss giảm mạnh, validation loss tăng sau một điểm.

#### Nguyên nhân
1. **Mô hình quá phức tạp**: Mô hình có quá nhiều tham số so với lượng dữ liệu (ví dụ: polynomial regression độ cao, decision tree không prune).
2. **Dữ liệu huấn luyện ít hoặc nhiễu**: Với dữ liệu nhỏ, mô hình dễ fit vào nhiễu thay vì pattern thực.
3. **Thiếu regularization**: Không có cơ chế phạt mô hình phức tạp.
4. **Không sử dụng cross-validation**: Chỉ train trên một tập, không kiểm tra trên validation.

#### Phương pháp phát hiện và khắc phục Overfitting

1. **Cross-Validation**: Chia dữ liệu thành k-fold, train trên k-1 fold, test trên fold còn lại. Ví dụ: k=5 hoặc k=10.
2. **Regularization**: Thêm phạt vào hàm mất mát để hạn chế trọng số lớn.
   - **L1 Regularization (Lasso)**: \(\mathcal{L}(\mathbf{w}) + \lambda \sum |w_i|\), khuyến khích trọng số bằng 0 (feature selection).
   - **L2 Regularization (Ridge)**: \(\mathcal{L}(\mathbf{w}) + \lambda \sum w_i^2\), phạt trọng số lớn nhưng giữ tất cả feature.
   - \(\lambda\) là hyperparameter, chọn bằng grid search hoặc cross-validation.

3. **Early Stopping**: Dừng huấn luyện khi validation loss bắt đầu tăng.
4. **Data Augmentation**: Tăng dữ liệu huấn luyện bằng cách biến đổi (rotate, flip hình ảnh).
5. **Dropout**: Trong neural networks, ngẫu nhiên bỏ qua một số nơ-ron trong huấn luyện.
6. **Ensemble Methods**: Kết hợp nhiều mô hình (bagging, boosting) để giảm variance.

#### **Ví dụ minh họa**
- Với polynomial regression: Fit polynomial độ 10 trên dữ liệu tuyến tính + nhiễu sẽ overfit, đường cong uốn lượn theo nhiễu. Sử dụng L2 regularization sẽ làm đường cong mượt mà hơn.
- Trong Iris dataset: Mô hình phức tạp có thể fit 100% train nhưng accuracy test giảm nếu có nhiễu.

### Code
Sử dụng L2 regularization trong Logistic Regression trên bộ dữ liệu Iris để minh họa chống overfitting:

```python
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression(multi_class='multinomial', C=0.1, max_iter=200)
model.fit(X_train, y_train)
print("Độ chính xác trên train:", accuracy_score(y_train, model.predict(X_train)))
print("Độ chính xác trên test:", accuracy_score(y_test, model.predict(X_test)))
```

## 5. Project: Phân Tích Bộ Dữ Liệu Iris

### Mô tả
Sử dụng bộ dữ liệu Iris (150 mẫu hoa với 4 đặc trưng: chiều dài/rộng đài/cánh hoa, 3 lớp) để áp dụng các thuật toán:
- **PLA**: Phân loại nhị phân (setosa vs. versicolor).
- **Logistic Regression**: Phân loại nhị phân với regularization.
- **Softmax Regression**: Phân loại 3 lớp.
- **Overfitting**: Đánh giá chênh lệch accuracy train/test.

### Các Bước Thực Hiện
1. Load dữ liệu Iris.
2. Áp dụng từng thuật toán như code trên.
3. Đánh giá: Accuracy cho phân loại, so sánh train/test để kiểm tra overfitting.
4. Trình bày: Vẽ biểu đồ decision boundary (nếu có) để trực quan hóa.

### Code
[Link tới demo.py](code/demo.py)