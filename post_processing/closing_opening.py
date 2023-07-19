import cv2

class MorphOpenClose:
    def __init__(self, kernel_size=5, opening_iterations=2, closing_iterations=1):
        self.kernel_size = kernel_size
        self.opening_iterations = opening_iterations
        self.closing_iterations = closing_iterations
        self.kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (kernel_size, kernel_size))

    def opening_closing(self, input_file):
        # 입력 이미지를 그레이스케일로 읽기
        input_image = cv2.imread(input_file, cv2.IMREAD_GRAYSCALE)

        # 입력 이미지에 opening 수행
        opened_image = cv2.morphologyEx(input_image, cv2.MORPH_OPEN, self.kernel, iterations=self.opening_iterations)

        # 입력 이미지에 closing 수행
        closed_image = cv2.morphologyEx(opened_image, cv2.MORPH_CLOSE, self.kernel, iterations=self.closing_iterations)


        # 결과 이미지 반환
        return closed_image