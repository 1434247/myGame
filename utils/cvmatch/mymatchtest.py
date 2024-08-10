# SIFT 和 bf knn matcher
import cv2

img1 = cv2.imread("../技能.jpg", 0)
img2 = cv2.imread("../jh_screen_205.jpg", 0)

sift = cv2.SIFT_create()

kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

bf = cv2.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)

good = []
for m, n in matches:
    if m.distance < 0.5*n.distance:
        good.append([m])

img_res = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good,
                            None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

cv2.imshow("sift-bfknnmatches", img_res)
cv2.waitKey(0)
cv2.destroyAllWindows()
