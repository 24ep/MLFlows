import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# กำหนดที่เก็บข้อมูลของ MLflow
mlflow.set_tracking_uri("file:///tmp/mlruns")  # บันทึกใน temporary directory

# โหลดข้อมูล
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)

# เริ่มต้นการติดตาม
mlflow.start_run()

# สร้างโมเดล
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# บันทึกโมเดลใน MLflow
mlflow.sklearn.log_model(model, "random_forest_model")

# บันทึกพารามิเตอร์และเมตริก
mlflow.log_param("n_estimators", 100)
mlflow.log_metric("accuracy", model.score(X_test, y_test))

# เสร็จสิ้นการติดตาม
mlflow.end_run()

