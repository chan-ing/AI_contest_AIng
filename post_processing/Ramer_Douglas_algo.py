import cv2

class PolygonApproximator:
    def approximate_polygon(self, input_image, epsilon=0.01):

        # Ramer-Douglas-Peucker 알고리즘으로 다각형 근사화
        contours, _ = cv2.findContours(input_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 모든 다각형에 대해 근사화된 다각형 그리기
        output_image = input_image.copy()
        for contour in contours:
            approximated_contour = cv2.approxPolyDP(contour, epsilon, True)
            cv2.polylines(output_image, [approximated_contour], isClosed=True, color=(225, 225, 255), thickness=1)
            cv2.fillPoly(output_image, [approximated_contour], color=(225, 225, 255))  # 내부를 같은 색으로 채우기
        output_image_rgb = cv2.cvtColor(output_image, cv2.COLOR_GRAY2RGB)
        return output_image

# import cv2
#
# class PolygonApproximator:
#     def __init__(self, input_file):
#         # Read the input PNG file
#         self.class_image = cv2.imread(input_file)
#         self.class_gray = cv2.cvtColor(self.class_image, cv2.COLOR_BGR2GRAY)
#
#     def approximate_polygon(self, epsilon=0.01):
#         # Ramer-Douglas-Peucker 알고리즘으로 다각형 근사화
#         contours, _ = cv2.findContours(self.class_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#
#         # 모든 다각형에 대해 근사화된 다각형 그리기
#         output_image = self.class_image.copy()
#         for contour in contours:
#             approximated_contour = cv2.approxPolyDP(contour, epsilon, True)
#             cv2.polylines(output_image, [approximated_contour], isClosed=True, color=(0, 0, 255), thickness=1)
#             cv2.fillPoly(output_image, [approximated_contour], color=(0, 0, 255))  # Fill the interior with the same color
#
#         return output_image
#
# # # 클래스 사용 예시
# # input_file = "./test_mask_img/MASK_00086.png"
# # polygon_approximator = PolygonApproximator(input_file)
# # output_image = polygon_approximator.approximate_polygon()