import sys
# 상위 디렉토리의 모듈 사용하기 위해 경로 추가
sys.path.append("../")  # to import module

# Handling Value Errors
# 로지스틱 회귀 모델에 사용되는 헬퍼 함수 임포트
from utils import denominator, log_prob

from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt


class LogisticRegression:

    # 클래수의 초기화 메서드 - 매개 변수 받아 모델 초기화
    def __init__(
        self,
        n_features: int = 2,         # 특성의 수
        is_train: bool = True,       # 학습 모드 여부
        learning_rate: float = 0.1,  # 학습률
        is_render: bool = True       # 그래픽 출력을 위한 렌더링 모드 여부
    ):

        # 클래스 내부 변수 '__n_features'에 입력된 'n_features'값 저장
        self.__n_features = n_features

        # Randomly init the weights
        # n_features + bias
        # 가중치 행렬 'W' 무작위로 초기화 - 특성의 수와 편향에 따라 크기 결정
        self.W = np.random.rand(n_features + 1)

        # Gradients
        # 그레이디언트 저장을 위한 변수들 초기화
        self.__model_grad = 0
        self.__sigmoid_grad = 0
        self.__binary_cross_entropy_grad = 0

        # Train parameters
        # 학습 관련 매개 변수들 초기화
        self.__is_train = is_train
        self.__learning_rate = learning_rate

        # Only for 2 features
        # 특성이 2개일 때만 그래픽 출력을 위한 렌더링 모드 활성화
        if n_features == 2:
            self.__is_render = is_render
        else:
            self.__is_render = False

        # epoch step number
        # 에포크 변수 초기화
        self.step = 0

        # Rendering
        # 그래픽 출력을 위한 초기 설정
        if self.__is_render:
            self.fig = plt.figure()
            plt.ion()  # Use interactive mode
            self.ax = self.fig.add_subplot(111)
            self.line = None
            
# -------------------> 여기부터 다시 시작
    @property
    def is_train(self) -> bool:
        return self.__is_train

    @is_train.setter
    def is_train(
        self,
        is_train: bool
    ) -> bool:
        self.__is_train = is_train

    @property
    def learning_rate(self) -> float:
        return self.__learning_rate

    @learning_rate.setter
    def learning_rate(
        self,
        learning_rate: float
    ) -> float:
        self.__learning_rate = learning_rate

    @property
    def is_render(self) -> bool:
        return self.__is_render

    @is_render.setter
    def is_render(
        self,
        is_render: bool
    ):
        self.__is_render = is_render

    def model(
        self,
        data: np.ndarray
    ) -> np.ndarray:

        # Get the number of data
        data_size = data.shape[0]
        ones = np.ones(data_size)  # For bias multiplication

        # Add column of ones to multiply bias
        X = np.column_stack((ones, data))

        # Save grads
        if self.__is_train:
            # (|N|, features)
            self.__model_grad = X

        # (|N|, )
        return np.matmul(X, self.W)

    def sigmoid(
        self,
        arr: np.ndarray
    ) -> np.ndarray:

        # Save grads
        if self.__is_train:
            # (|N|, )
            self.__sigmoid_grad = np.exp(-arr) / (1 + np.exp(-arr))**2

        # (|N|, )
        return 1 / (1 + np.exp(-arr))

    def binary_cross_entropy(
        self,
        Y_hat: np.ndarray,
        Y: np.ndarray
    ) -> np.ndarray:

        if self.__is_train:
            # (|N|, )
            self.__binary_cross_entropy_grad = \
                (Y_hat - Y) / denominator(Y_hat*(1 - Y_hat))

        # (|N|, )
        return -(Y*log_prob(Y_hat) + (1-Y)*log_prob(1-Y_hat))

    def batch_gradient_descent(
        self
    ):

        temp_grad = np.zeros(self.__n_features + 1)
        grad = np.zeros(self.__n_features + 1)
        batch_size = self.__model_grad.shape[0]

        for idx, model_grad in enumerate(self.__model_grad):
            temp_grad = np.copy(model_grad)
            temp_grad = self.__sigmoid_grad[idx] * temp_grad
            temp_grad = self.__binary_cross_entropy_grad[idx] * temp_grad
            grad = grad + temp_grad

        # Get average of the gradients
        grad = grad / batch_size

        # Update the weights
        self.W = self.W - self.__learning_rate * grad

    def forward(
        self,
        X: np.ndarray,
        Y: np.ndarray
    ) -> np.float64:

        Y_hat = self.model(X)
        Y_hat = self.sigmoid(Y_hat)
        L = self.binary_cross_entropy(Y_hat, Y)

        return np.mean(L)

    # If f = 0, sigmoid(f) = 0.5.
    # So, decision boundary is equation f = 0
    def decision_boundary(
        self,
        X: np.ndarray
    ) -> Tuple[list, list]:

        # Get min, max value from data
        min_value = np.min(X[:, 0]) - 1
        max_value = np.max(X[:, 0]) + 1
        line_range = np.arange(min_value, max_value, 0.1)

        x = [x for x in line_range]
        y = [-((self.W[0] + self.W[1]*x) / self.W[2]) \
            for x in line_range]

        return x, y

    def render_init(
        self,
        X: np.ndarray,
        Y: np.ndarray
    ):

        # Set lim
        x_min = np.min(X[:, 0]) - 1
        x_max = np.max(X[:, 0]) + 1
        y_min = np.min(X[:, 1]) - 1
        y_max = np.max(X[:, 1]) + 1
        self.ax.set_xlim(x_min, x_max)
        self.ax.set_ylim(y_min, y_max)

        # Plot data
        self.ax.scatter(X[Y == 0, 0], X[Y == 0, 1],
                        s=80,
                        color='red',
                        marker='*')
        self.ax.scatter(X[Y == 1, 0], X[Y == 1, 1],
                        s=80,
                        color='blue',
                        marker='D')

        x, y = self.decision_boundary(X)
        self.line, = self.ax.plot(x, y, color="black")

        plt.savefig(f"./results/{self.step}.png")
        plt.show()

    def render_update(
        self,
        X: np.ndarray,
    ):

        x, y = self.decision_boundary(X)

        # Update
        self.line.set_ydata(y)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

        plt.savefig(f"./results/{self.step}.png")

    def train(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        epochs: int = 10,
    ):

        if self.__is_render:
            self.render_init(X, Y)

        for epoch in range(epochs):
            L = self.forward(X, Y)
            self.batch_gradient_descent()
            self.step += 1

            if self.__is_render:
                self.render_update(X)

            # print training status
            resultStr = f"epoch:{epoch} loss: {round(L, 6)}"
            resultLen = len(resultStr)
            print(resultStr)
            print('-'*resultLen)

if __name__ == "__main__":
    # data = np.array([[1, 2], [3, 2], [10, 2.3]])
    X = np.array([[1, 0], [3, 2], [10, 2.3]])
    Y = np.array([0, 0, 1])

    linear_model = LogisticRegression()
    linear_model.train(X, Y, 200)
